"""Comprueba que config/datos.yaml se puede leer y que el lector rechaza valores imposibles.

Córrela después de editar el archivo:  pytest tests/test_datos_config.py

Solo las tres primeras pruebas leen el archivo del repositorio, y no revisan
fechas concretas: cambiar el periodo por otro válido no rompe ninguna. Las
demás prueban el lector con una configuración fija escrita aquí (TEXTO_BASE).
"""

from datetime import date

import pandas as pd
import pytest

from src.data.datos_config import SPLIT_NAMES, TRAMOS, Datos, DatosError, Tramo, load_datos
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
"""


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
