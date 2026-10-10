"""Comprueba que config/datos.yaml se puede leer y que el lector rechaza valores imposibles.

Córrela después de editar el archivo:  pytest tests/test_datos_config.py

Solo las cuatro primeras pruebas leen el archivo del repositorio, y no revisan
fechas ni valores concretos: cambiar el periodo o un ajuste por otro válido no
rompe ninguna. Las demás prueban el lector con una configuración fija escrita
aquí (TEXTO_BASE).
"""

import dataclasses
from datetime import date

import pandas as pd
import pytest

from src.data.datos_config import SPLIT_NAMES, TRAMOS, Datos, DatosError, Simulacion, Tramo, load_datos
from src.paths import TZ

# Configuración fija para probar el lector. NO es el archivo del repositorio.
TEXTO_BASE = """\
meses_crudos: [202607, 202608]
periodo:
  inicio: 2026-07-01
  fin: 2026-08-31
tramos:
  entrenamiento: {inicio: 2026-07-01, fin: 2026-08-10}
  validacion:    {inicio: 2026-08-11, fin: 2026-08-20}
  prueba:        {inicio: 2026-08-21, fin: 2026-08-31}
union_de_ids:
  distancia_maxima_m: 1.0
estaciones_excluidas:
  prefijos: [HB, JC]
  ids: ["1234.56", "Shop Morgan"]
simulacion:
  dias_de_calentamiento: 2
  remuestreos: 500
  nivel_del_intervalo: 0.80
  semilla: 7
  bandas: [0, 0.5, 1]
  franja_noche: {inicio: "22:00", fin: "06:00"}
  sensibilidad_ocupacion_inicial: [0.4, 0.6]
"""

# El bloque de simulación de TEXTO_BASE, para las pruebas que lo quitan o lo cambian entero.
BLOQUE_SIMULACION = TEXTO_BASE[TEXTO_BASE.index("simulacion:"):]


def _archivo(tmp_path, texto):
    archivo = tmp_path / "datos.yaml"
    archivo.write_text(texto, encoding="utf-8")
    return archivo


def _editado(viejo, nuevo):
    """TEXTO_BASE con un pedazo cambiado, como lo dejaría quien edita el archivo a mano."""
    assert viejo in TEXTO_BASE, viejo
    return TEXTO_BASE.replace(viejo, nuevo, 1)


@pytest.fixture()
def datos():
    """El archivo del repositorio. Si está mal, muestra solo el mensaje, sin la traza de Python."""
    try:
        return load_datos()
    except DatosError as error:
        mensaje = str(error)
    pytest.fail(mensaje, pytrace=False)


# --------------------------------------------------------------------------- #
# El archivo del repositorio
# --------------------------------------------------------------------------- #

def test_el_archivo_del_repositorio_es_valido(datos):
    assert isinstance(datos, Datos)
    assert list(datos.tramos) == list(TRAMOS)
    assert datos.distancia_maxima_m >= 0


def test_los_tramos_del_repositorio_cubren_el_periodo_sin_huecos(datos):
    """Decisión 4: entrenamiento, validación y prueba, ordenados en el tiempo."""
    entrenamiento, validacion, prueba = (datos.tramos[nombre] for nombre in TRAMOS)
    assert entrenamiento.start == datos.periodo.start
    assert entrenamiento.end == validacion.start
    assert validacion.end == prueba.start
    assert prueba.end == datos.periodo.end


def test_los_meses_crudos_del_repositorio_cubren_el_periodo(datos):
    """Todo mes que toca el periodo debe estar en la lista de meses que se leen."""
    meses = pd.period_range(datos.periodo.inicio, datos.periodo.fin, freq="M").strftime("%Y%m")
    faltan = [mes for mes in meses if mes not in datos.meses_crudos]
    assert not faltan, f"config/datos.yaml: el periodo toca meses que no están en meses_crudos: {faltan}"


def test_el_archivo_del_repositorio_trae_los_ajustes_de_la_simulacion(datos):
    """El lector admite un archivo sin el bloque `simulacion`, pero el del repositorio lo necesita.

    Lo lee el simulador: sin ese bloque no hay calentamiento ni remuestreo que usar.
    """
    assert isinstance(datos.simulacion, Simulacion), "config/datos.yaml no trae el bloque simulacion"
    validacion = datos.tramos["validacion"]
    dias_de_validacion = (validacion.fin - validacion.inicio).days + 1
    # Tiene que quedar al menos un día de validación que sí cuente.
    assert 0 <= datos.simulacion.dias_de_calentamiento < dias_de_validacion


# --- De aquí en adelante ninguna prueba lee config/datos.yaml ---


# --------------------------------------------------------------------------- #
# Un archivo válido
# --------------------------------------------------------------------------- #

def test_un_archivo_valido_se_carga_con_sus_valores(tmp_path):
    datos = load_datos(_archivo(tmp_path, TEXTO_BASE))

    assert datos.meses_crudos == ("202607", "202608")  # texto AAAAMM, como el nombre del Parquet
    assert (datos.periodo.inicio, datos.periodo.fin) == (date(2026, 7, 1), date(2026, 8, 31))
    assert datos.tramos["validacion"] == Tramo("validacion", date(2026, 8, 11), date(2026, 8, 20))
    assert datos.distancia_maxima_m == 1.0 and isinstance(datos.distancia_maxima_m, float)
    assert datos.prefijos_excluidos == ("HB", "JC")
    assert datos.ids_excluidos == ("1234.56", "Shop Morgan")  # texto: un id nunca es un número
    assert datos.simulacion == Simulacion(
        dias_de_calentamiento=2, remuestreos=500, nivel_del_intervalo=0.8, semilla=7,
        bandas=(0.0, 0.5, 1.0), franja_noche=("22:00", "06:00"), sensibilidad_ocupacion_inicial=(0.4, 0.6),
    )
    assert isinstance(datos.simulacion.nivel_del_intervalo, float)


def test_el_fin_de_un_tramo_esta_incluido(tmp_path):
    """`fin` es el último día. `end` es el primer instante de después, en hora de Nueva York."""
    tramo = load_datos(_archivo(tmp_path, TEXTO_BASE)).tramos["validacion"]

    assert tramo.start == pd.Timestamp("2026-08-11 00:00", tz=TZ)
    assert tramo.end == pd.Timestamp("2026-08-21 00:00", tz=TZ)
    horas = pd.Series(pd.to_datetime(["2026-08-10 23:00", "2026-08-11 00:00", "2026-08-20 23:00", "2026-08-21 00:00"]))
    assert tramo.contains(horas.dt.tz_localize(TZ)).tolist() == [False, True, True, False]


def test_los_nombres_de_los_tramos_para_predictions():
    assert SPLIT_NAMES == {"validacion": "val", "prueba": "test"}
    assert set(SPLIT_NAMES) < set(TRAMOS)


def test_se_admiten_listas_de_excluidas_vacias(tmp_path):
    texto = _editado('  prefijos: [HB, JC]\n  ids: ["1234.56", "Shop Morgan"]', "  prefijos: []\n  ids: []")
    datos = load_datos(_archivo(tmp_path, texto))
    assert datos.prefijos_excluidos == () and datos.ids_excluidos == ()


def test_se_admiten_meses_crudos_de_mas(tmp_path):
    """La lista puede traer meses que el periodo no toca; lo que no puede es faltarle uno."""
    datos = load_datos(_archivo(tmp_path, _editado("[202607, 202608]", "[202606, 202607, 202608, 202609]")))
    assert datos.meses_crudos == ("202606", "202607", "202608", "202609")


def test_un_mes_de_en_medio_que_falta_se_rechaza(tmp_path):
    """Periodo de julio a septiembre sin agosto en la lista: agosto saldría entero en ceros."""
    texto = (
        _editado("[202607, 202608]", "[202607, 202609]")
        .replace("  fin: 2026-08-31\n", "  fin: 2026-09-29\n")
        .replace("fin: 2026-08-31}", "fin: 2026-09-29}")
    )
    with pytest.raises(DatosError) as error:
        load_datos(_archivo(tmp_path, texto))
    assert str(error.value).endswith("- meses_crudos: el periodo toca meses que no están en la lista: 202608")
    # Con agosto en la lista, el mismo archivo es válido.
    datos = load_datos(_archivo(tmp_path, texto.replace("[202607, 202609]", "[202607, 202608, 202609]")))
    assert datos.periodo.fin == date(2026, 9, 29)


# --------------------------------------------------------------------------- #
# Bloque de simulación
# --------------------------------------------------------------------------- #

def test_sin_bloque_de_simulacion_el_archivo_es_valido_y_el_ajuste_queda_vacio(tmp_path):
    """La limpieza y la línea base no usan ese bloque: un archivo sin él sigue sirviendo para ellas."""
    datos = load_datos(_archivo(tmp_path, _editado(BLOQUE_SIMULACION, "")))
    assert datos.simulacion is None
    assert datos.tramos["validacion"].inicio == date(2026, 8, 11)


def test_los_ajustes_de_la_simulacion_no_se_pueden_cambiar_ya_cargados(tmp_path):
    simulacion = load_datos(_archivo(tmp_path, TEXTO_BASE)).simulacion
    with pytest.raises(dataclasses.FrozenInstanceError):
        simulacion.remuestreos = 10


@pytest.mark.parametrize(
    "viejo, nuevo",
    [
        # (texto de TEXTO_BASE, otro valor que también es válido)
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: 0"),   # sin calentamiento
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: 9"),   # validación tiene 10 días: queda 1 que cuenta
        ("remuestreos: 500", "remuestreos: 100"),                   # el mínimo
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 0.51"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 0.999"),
        ("semilla: 7", "semilla: 0"),
    ],
)
def test_valores_de_la_simulacion_en_el_borde_que_si_se_admiten(tmp_path, viejo, nuevo):
    clave, valor = nuevo.split(": ")
    simulacion = load_datos(_archivo(tmp_path, _editado(viejo, nuevo))).simulacion
    assert getattr(simulacion, clave) == float(valor)


@pytest.mark.parametrize(
    "viejo, nuevo, mensaje",
    [
        # El tramo de validación de TEXTO_BASE tiene 10 días (11 a 20 de agosto).
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: 10",
         "simulacion.dias_de_calentamiento: debe ser un número entero mayor o igual a 0 y menor que los 10 días "
         "del tramo de validación"),
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: -1", "simulacion.dias_de_calentamiento: debe ser"),
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: 1.0", "simulacion.dias_de_calentamiento: debe ser"),
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: true", "simulacion.dias_de_calentamiento: debe ser"),
        ("dias_de_calentamiento: 2", 'dias_de_calentamiento: "2"', "simulacion.dias_de_calentamiento: debe ser"),
        ("dias_de_calentamiento: 2", "dias_de_calentamiento: uno", "simulacion.dias_de_calentamiento: debe ser"),
        ("  dias_de_calentamiento: 2\n", "", "simulacion.dias_de_calentamiento: debe ser"),
        ("remuestreos: 500", "remuestreos: 99",
         "simulacion.remuestreos: debe ser un número entero mayor o igual a 100"),
        ("remuestreos: 500", "remuestreos: 0", "simulacion.remuestreos: debe ser"),
        ("remuestreos: 500", "remuestreos: 2000.0", "simulacion.remuestreos: debe ser"),
        ("remuestreos: 500", "remuestreos: 2,000", "simulacion.remuestreos: debe ser"),
        ("remuestreos: 500", "remuestros: 500", "simulacion.remuestreos: debe ser"),  # clave mal escrita
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 0.5",
         "simulacion.nivel_del_intervalo: debe ser un número mayor que 0.5 y menor que 1"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 1", "simulacion.nivel_del_intervalo: debe ser"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 1.0", "simulacion.nivel_del_intervalo: debe ser"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 90", "(0.90 es un intervalo del 90 %)"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: 0,90", "simulacion.nivel_del_intervalo: debe ser"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: .nan", "simulacion.nivel_del_intervalo: debe ser"),
        ("nivel_del_intervalo: 0.80", "nivel_del_intervalo: true", "simulacion.nivel_del_intervalo: debe ser"),
        ("semilla: 7", "semilla: -1", "simulacion.semilla: debe ser un número entero mayor o igual a 0"),
        ("semilla: 7", "semilla: 1.5", "simulacion.semilla: debe ser"),
        ("semilla: 7", "semilla: false", "simulacion.semilla: debe ser"),
        ("  semilla: 7\n", "", "simulacion.semilla: debe ser"),
        # El bloque está, pero vacío o con otra forma (por ejemplo, con todas sus líneas comentadas).
        (BLOQUE_SIMULACION, "simulacion:\n", "simulacion: debe ser un bloque con dias_de_calentamiento"),
        (BLOQUE_SIMULACION, "simulacion: [1, 2000, 0.9, 7]\n", "simulacion: debe ser un bloque con"),
        (BLOQUE_SIMULACION, "simulacion: 2000\n", "simulacion: debe ser un bloque con"),
    ],
)
def test_un_bloque_de_simulacion_mal_escrito_se_rechaza_con_un_mensaje_claro(tmp_path, viejo, nuevo, mensaje):
    archivo = _archivo(tmp_path, _editado(viejo, nuevo))
    with pytest.raises(DatosError) as error:
        load_datos(archivo)
    assert mensaje in str(error.value), str(error.value)
    assert str(archivo) in str(error.value)


def test_el_calentamiento_se_compara_con_los_dias_del_tramo_de_validacion(tmp_path):
    """Con un tramo de validación más corto, el mismo calentamiento deja de caber."""
    # Entrenamiento hasta el 18 de agosto: validación queda en dos días (19 y 20).
    corto = (
        TEXTO_BASE
        .replace("fin: 2026-08-10}", "fin: 2026-08-18}")
        .replace("validacion:    {inicio: 2026-08-11", "validacion:    {inicio: 2026-08-19")
    )
    assert corto.count("2026-08-18") == 1 and corto.count("2026-08-19") == 1
    with pytest.raises(DatosError, match="menor que los 2 días del tramo de validación"):
        load_datos(_archivo(tmp_path, corto))
    datos = load_datos(_archivo(tmp_path, corto.replace("dias_de_calentamiento: 2", "dias_de_calentamiento: 1")))
    assert datos.simulacion.dias_de_calentamiento == 1


def test_con_los_tramos_mal_escritos_el_mensaje_del_calentamiento_no_inventa_un_tope(tmp_path):
    """Si los tramos no se pudieron leer, no se sabe cuántos días tiene validación: se dice solo lo que se sabe."""
    texto = _editado("  prueba:  ", "  test:    ").replace("dias_de_calentamiento: 2", "dias_de_calentamiento: -3")
    with pytest.raises(DatosError) as error:
        load_datos(_archivo(tmp_path, texto))
    assert "tramos: deben ser exactamente" in str(error.value)
    assert "simulacion.dias_de_calentamiento: debe ser un número entero mayor o igual a 0\n" in str(error.value) + "\n"
    assert "días del tramo de validación" not in str(error.value)


def test_el_mensaje_junta_todos_los_problemas_de_la_simulacion(tmp_path):
    texto = _editado(BLOQUE_SIMULACION, "simulacion: {dias_de_calentamiento: 99, remuestreos: 5}\n")
    with pytest.raises(DatosError) as error:
        load_datos(_archivo(tmp_path, texto))
    for clave in ("dias_de_calentamiento", "remuestreos", "nivel_del_intervalo", "semilla", "bandas",
                  "franja_noche", "sensibilidad_ocupacion_inicial"):
        assert f"simulacion.{clave}: debe" in str(error.value), clave


@pytest.mark.parametrize(
    "viejo, nuevo, mensaje",
    [
        ("bandas: [0, 0.5, 1]", "bandas: []", "simulacion.bandas: debe ser una lista"),
        ("bandas: [0, 0.5, 1]", "bandas: 0.5", "simulacion.bandas: debe ser una lista"),
        ("bandas: [0, 0.5, 1]", "bandas: [0, -1]", "simulacion.bandas: debe ser una lista"),
        ("bandas: [0, 0.5, 1]", "bandas: [0.5, 0.5]", "simulacion.bandas: debe ser una lista"),
        ("  bandas: [0, 0.5, 1]\n", "", "simulacion.bandas: debe ser una lista"),
        ('inicio: "22:00", fin: "06:00"', 'inicio: "22:00", fin: "22:00"', "simulacion.franja_noche: debe tener"),
        ('inicio: "22:00", fin: "06:00"', 'inicio: 22, fin: "06:00"', "simulacion.franja_noche: debe tener"),
        ('inicio: "22:00", fin: "06:00"', 'inicio: "25:00", fin: "06:00"', "simulacion.franja_noche: debe tener"),
        ("sensibilidad_ocupacion_inicial: [0.4, 0.6]", "sensibilidad_ocupacion_inicial: [0.4, 1.5]",
         "simulacion.sensibilidad_ocupacion_inicial: debe ser una lista"),
        ("  sensibilidad_ocupacion_inicial: [0.4, 0.6]\n", "",
         "simulacion.sensibilidad_ocupacion_inicial: debe ser una lista"),
    ],
)
def test_las_bandas_la_noche_y_la_sensibilidad_mal_escritas_se_rechazan(tmp_path, viejo, nuevo, mensaje):
    with pytest.raises(DatosError) as error:
        load_datos(_archivo(tmp_path, _editado(viejo, nuevo)))
    assert mensaje in str(error.value)


def test_la_sensibilidad_puede_ir_vacia_y_la_noche_puede_no_cruzar_la_medianoche(tmp_path):
    texto = _editado("sensibilidad_ocupacion_inicial: [0.4, 0.6]", "sensibilidad_ocupacion_inicial: []")
    texto = texto.replace('inicio: "22:00", fin: "06:00"', 'inicio: "00:00", fin: "05:00"')
    simulacion = load_datos(_archivo(tmp_path, texto)).simulacion
    assert simulacion.sensibilidad_ocupacion_inicial == () and simulacion.franja_noche == ("00:00", "05:00")


# --------------------------------------------------------------------------- #
# Archivos mal escritos: el mensaje dice qué revisar
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "viejo, nuevo, mensaje",
    [
        # (texto de TEXTO_BASE, cómo quedó mal escrito, qué debe decir el mensaje)
        # Hueco entre tramos: el 11 de agosto no es de nadie.
        ("validacion:    {inicio: 2026-08-11", "validacion:    {inicio: 2026-08-12",
         "'validacion' debe empezar el día siguiente al fin de 'entrenamiento'"),
        # Traslape: el 10 de agosto está en dos tramos.
        ("validacion:    {inicio: 2026-08-11", "validacion:    {inicio: 2026-08-10",
         "'validacion' debe empezar el día siguiente al fin de 'entrenamiento'"),
        ("prueba:        {inicio: 2026-08-21", "prueba:        {inicio: 2026-08-19",
         "'prueba' debe empezar el día siguiente al fin de 'validacion'"),
        # Los tramos no cubren el periodo.
        ("fin: 2026-08-31}", "fin: 2026-08-30}", "tramos: deben empezar y terminar igual que el periodo"),
        ("entrenamiento: {inicio: 2026-07-01", "entrenamiento: {inicio: 2026-07-02",
         "tramos: deben empezar y terminar igual que el periodo"),
        ("  fin: 2026-08-31\n", "  fin: 2026-09-30\n", "tramos: deben empezar y terminar igual que el periodo"),
        # Meses imposibles o escritos como texto.
        ("[202607, 202608]", "[202607, 202613]", "meses_crudos: debe ser una lista de meses AAAAMM"),
        ("[202607, 202608]", "[202012]", "meses_crudos: debe ser una lista de meses AAAAMM"),
        ("[202607, 202608]", "[2026-07]", "meses_crudos: debe ser una lista de meses AAAAMM"),
        ("[202607, 202608]", '["202607"]', "meses_crudos: debe ser una lista de meses AAAAMM"),
        ("[202607, 202608]", "[]", "meses_crudos: debe ser una lista de meses AAAAMM"),
        ("[202607, 202608]", "202607", "meses_crudos: debe ser una lista de meses AAAAMM"),
        # Un mes del periodo que no está en la lista: saldría como un mes de ceros.
        ("[202607, 202608]", "[202607]", "meses_crudos: el periodo toca meses que no están en la lista: 202608"),
        ("[202607, 202608]", "[202608]", "meses_crudos: el periodo toca meses que no están en la lista: 202607"),
        ("[202607, 202608]", "[202606, 202609]",
         "meses_crudos: el periodo toca meses que no están en la lista: 202607, 202608"),
        # Fechas entre comillas, con hora o al revés.
        ("  inicio: 2026-07-01\n", '  inicio: "2026-07-01"\n',
         "periodo: debe tener inicio y fin como fechas AAAA-MM-DD, sin comillas"),
        ("  inicio: 2026-07-01\n", "  inicio: 2026-07-01 00:00:00\n",
         "periodo: debe tener inicio y fin como fechas AAAA-MM-DD"),
        ("  inicio: 2026-07-01\n", "  inicio: 01/07/2026\n", "periodo: debe tener inicio y fin como fechas AAAA-MM-DD"),
        ("{inicio: 2026-08-11, fin: 2026-08-20}", '{inicio: "2026-08-11", fin: 2026-08-20}',
         "tramos.validacion: debe tener inicio y fin como fechas AAAA-MM-DD, sin comillas"),
        ("  inicio: 2026-07-01\n", "  inicio: 2026-09-01\n", "periodo: inicio no puede ser posterior a fin"),
        # Claves que faltan.
        ("periodo:\n  inicio: 2026-07-01\n  fin: 2026-08-31\n", "", "periodo: debe tener inicio y fin"),
        ("  fin: 2026-08-31\n", "", "periodo: debe tener inicio y fin"),
        ("meses_crudos: [202607, 202608]\n", "", "meses_crudos: debe ser una lista"),
        ("  prueba:        {inicio: 2026-08-21, fin: 2026-08-31}\n", "", "tramos: deben ser exactamente"),
        ("  prueba:  ", "  test:    ", "tramos: deben ser exactamente"),
        ("union_de_ids:\n  distancia_maxima_m: 1.0\n", "", "union_de_ids.distancia_maxima_m: debe ser un número"),
        ("  distancia_maxima_m: 1.0", "  distancia_m: 1.0", "union_de_ids.distancia_maxima_m: debe ser un número"),
        ("  prefijos: [HB, JC]\n", "", "estaciones_excluidas.prefijos: debe ser una lista"),
        ('  ids: ["1234.56", "Shop Morgan"]\n', "", "estaciones_excluidas.ids: debe ser una lista"),
        # Valores imposibles.
        ("distancia_maxima_m: 1.0", "distancia_maxima_m: -1", "union_de_ids.distancia_maxima_m"),
        ("distancia_maxima_m: 1.0", "distancia_maxima_m: uno", "union_de_ids.distancia_maxima_m"),
        ("distancia_maxima_m: 1.0", "distancia_maxima_m: true", "union_de_ids.distancia_maxima_m"),
        # 1000 metros o más ya no es "la misma coordenada": el tope es parte de la regla.
        ("distancia_maxima_m: 1.0", "distancia_maxima_m: 1000",
         "union_de_ids.distancia_maxima_m: debe ser un número de metros mayor o igual a 0 y menor que 1000"),
        # Un id sin comillas se lee como número y pierde su forma: se rechaza.
        ('ids: ["1234.56", "Shop Morgan"]', "ids: [1234.56]",
         "estaciones_excluidas.ids: debe ser una lista de textos entre comillas"),
        ("prefijos: [HB, JC]", "prefijos: HB", "estaciones_excluidas.prefijos: debe ser una lista"),
    ],
)
def test_un_archivo_mal_escrito_se_rechaza_con_un_mensaje_claro(tmp_path, viejo, nuevo, mensaje):
    archivo = _archivo(tmp_path, _editado(viejo, nuevo))
    with pytest.raises(DatosError) as error:
        load_datos(archivo)
    assert mensaje in str(error.value), str(error.value)
    assert str(archivo) in str(error.value)  # dice qué archivo hay que corregir


def test_el_mensaje_junta_todos_los_problemas(tmp_path):
    texto = _editado("[202607, 202608]", "[202613]").replace("distancia_maxima_m: 1.0", "distancia_maxima_m: -5")
    with pytest.raises(DatosError) as error:
        load_datos(_archivo(tmp_path, texto))
    assert "meses_crudos" in str(error.value) and "distancia_maxima_m" in str(error.value)


@pytest.mark.parametrize(
    "texto, mensaje",
    [
        ("- uno\n- dos\n", "debe ser un mapa de parámetros"),
        ("", "debe ser un mapa de parámetros"),
        ("periodo: {inicio: 2026-07-01\n", "está mal escrito y no se pudo leer"),
    ],
)
def test_un_archivo_que_no_es_un_mapa_se_rechaza(tmp_path, texto, mensaje):
    with pytest.raises(DatosError, match=mensaje):
        load_datos(_archivo(tmp_path, texto))
