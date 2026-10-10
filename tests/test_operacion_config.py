"""Comprueba que config/operacion.yaml se puede leer y tiene valores posibles.

Córrela después de editar el archivo:  pytest tests/test_operacion_config.py

Solo las dos primeras pruebas leen el archivo del repositorio. Las demás prueban
el validador con una configuración fija escrita aquí abajo (TEXTO_BASE): así,
cambiar config/operacion.yaml por otros valores válidos (otras horas, otros
nombres de franja, una o tres franjas, otro reparto del presupuesto, otros
presupuestos, otro presupuesto de referencia, otro tipo para las estaciones
sin tipo) no rompe ninguna.
"""

import copy
import math

import pytest
import yaml

from src.contracts import STATION_TYPES
from src.sim.operacion import OperacionError, load_operacion, presupuesto_en_bicis, validate_operacion

# Configuración fija para probar el validador. NO es el archivo del repositorio.
TEXTO_BASE = """\
estado: provisional
confirmado_por: null
fecha_confirmacion: null
franjas:
  - nombre: manana
    momento_intervencion: "05:00"
    inicio: "06:00"
    fin: "12:00"
    parte_del_presupuesto: 0.5
  - nombre: tarde
    momento_intervencion: "14:00"
    inicio: "15:00"
    fin: "21:00"
    parte_del_presupuesto: 0.5
umbral_riesgo:
  bicis_minimas: 2
  lugares_libres_minimos: 2
presupuestos_por_estacion: [0, 0.5, 1.0]
presupuesto_de_referencia: 0.5
distancia_maxima_km: 3.0
tipo_si_falta: mixta
estrategia_por_tipo:
""" + "".join(
    f"  {tipo}:\n    prioridad: 2\n    ocupacion_respaldo:\n      manana: 0.5\n      tarde: 0.5\n"
    for tipo in STATION_TYPES
)
BASE = yaml.safe_load(TEXTO_BASE)


@pytest.fixture()
def cfg():
    """El archivo del repositorio. Si está mal, muestra solo el mensaje, sin la traza de Python."""
    try:
        return load_operacion()
    except OperacionError as error:
        mensaje = str(error)
    pytest.fail(mensaje, pytrace=False)


def test_el_archivo_del_repositorio_es_valido(cfg):
    assert validate_operacion(cfg) == []


def test_estan_los_parametros_de_las_cinco_preguntas(cfg):
    for parametro in (
        "franjas",
        "umbral_riesgo",
        "presupuestos_por_estacion",
        "presupuesto_de_referencia",
        "distancia_maxima_km",
        "estrategia_por_tipo",
        "tipo_si_falta",
    ):
        assert parametro in cfg
    assert all("parte_del_presupuesto" in franja for franja in cfg["franjas"])
    assert "presupuestos_diarios" not in cfg  # el nombre anterior, con otra unidad


# --- De aquí en adelante ninguna prueba lee config/operacion.yaml ---


def test_la_configuracion_de_prueba_es_valida():
    assert validate_operacion(BASE) == []


def _con_el_nombre_anterior(c):
    """Deja 'oficinas' con la clave como se llamaba antes: ocupacion_objetivo."""
    regla = c["estrategia_por_tipo"]["oficinas"]
    regla["ocupacion_objetivo"] = regla.pop("ocupacion_respaldo")


def _con_presupuestos_diarios(c):
    """Deja los presupuestos con el nombre y la unidad de antes: el total de bicis del día."""
    del c["presupuestos_por_estacion"]
    c["presupuestos_diarios"] = [0, 50, 100]


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c["franjas"][0].update(inicio="25:00"), "HH:MM"),
        (lambda c: c["franjas"][0].update(inicio=600), "HH:MM"),
        (lambda c: c["franjas"][0].update(momento_intervencion="07:00"), "momento_intervencion <= inicio < fin"),
        (lambda c: c["franjas"][1].update(momento_intervencion="10:00", inicio="11:00"), "se enciman"),
        (lambda c: c["umbral_riesgo"].update(bicis_minimas=-1), "bicis_minimas"),
        (lambda c: c.update(distancia_maxima_km=0), "distancia_maxima_km"),
        (lambda c: c["estrategia_por_tipo"].pop("mixta"), "los tipos deben ser exactamente"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"].update(prioridad=5), "prioridad"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"].update(manana=1.5), "entre 0 y 1"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"].pop("tarde"), "un valor por franja"),
        (lambda c: c["estrategia_por_tipo"].update(oficinas=None), "falta prioridad y ocupacion_respaldo"),
        # Un archivo que todavía use el nombre anterior no pasa: el mensaje pide la clave nueva.
        (_con_el_nombre_anterior, "oficinas.ocupacion_respaldo: debe tener un valor por franja"),
        (lambda c: c.update(estado="confirmado"), "confirmado_por"),
        (lambda c: c.update(estado="listo"), "estado"),
        (lambda c: c.update(distancia_maxima_km=float("nan")), "mayor que 0"),
        (lambda c: c.update(distancia_maxima_km=float("inf")), "mayor que 0"),
    ],
)
def test_detecta_valores_imposibles(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert any(texto_esperado in problem for problem in problems), problems


# --- Pregunta 2: umbral de riesgo ---


@pytest.mark.parametrize("campo", ["bicis_minimas", "lugares_libres_minimos"])
def test_el_umbral_de_riesgo_acepta_1_y_rechaza_0(campo):
    """Está en riesgo la estación con MENOS que el umbral: con 0 nunca lo estaría y todo el riesgo saldría en 0."""
    bueno = copy.deepcopy(BASE)
    bueno["umbral_riesgo"][campo] = 1
    assert validate_operacion(bueno) == []

    roto = copy.deepcopy(BASE)
    roto["umbral_riesgo"][campo] = 0
    problems = validate_operacion(roto)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"umbral_riesgo.{campo}: debe ser un entero mayor o igual a 1")
    assert "con 0 nunca lo estaría" in problems[0]
    assert 'Para "vacía" o "llena" escribe 1' in problems[0]


# --- Pregunta 3: presupuestos en bicis por estación por día ---


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c.update(presupuestos_por_estacion=[]), "presupuestos_por_estacion: debe ser una lista"),
        (lambda c: c.pop("presupuestos_por_estacion"), "presupuestos_por_estacion: debe ser una lista"),
        (lambda c: c.update(presupuestos_por_estacion=0.5), "presupuestos_por_estacion: debe ser una lista"),
        (lambda c: c.update(presupuestos_por_estacion=[0, -0.5, 0.5]), "números mayores o iguales a 0"),
        (lambda c: c.update(presupuestos_por_estacion=[0, "medio", 0.5]), "números mayores o iguales a 0"),
        (lambda c: c.update(presupuestos_por_estacion=[0, True, 0.5]), "números mayores o iguales a 0"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 0.5, float("inf")]), "números mayores o iguales a 0"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 0.5, float("nan")]), "números mayores o iguales a 0"),
        # Una lista escrita con guiones y un decimal con coma: YAML lee "0,5" como texto.
        (lambda c: c.update(presupuestos_por_estacion=[0, "0,5"]), "usa punto decimal: 0.5"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 0.5, 0.5]), "repetidos"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 1, 1.0, 0.5]), "repetidos"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 1.0, 0.5]), "de menor a mayor"),
        # Con la coma seguida de espacio no se puede saber qué se quiso escribir: va la pista.
        (lambda c: c.update(presupuestos_por_estacion=[0, 0, 5, 1]), "repetidos. Dentro de la lista la coma separa"),
        (lambda c: c.update(presupuestos_por_estacion=[0, 500, 1, 150]), "Si usaste separador de miles, quítalo"),
    ],
)
def test_detecta_una_lista_de_presupuestos_imposible(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert any(texto_esperado in problem for problem in problems), problems


def test_la_clave_anterior_de_presupuestos_se_rechaza_y_dice_la_nueva_y_su_unidad():
    """Era el total de bicis del día. Aceptarla con la unidad nueva daría presupuestos absurdos."""
    roto = copy.deepcopy(BASE)
    _con_presupuestos_diarios(roto)
    problems = validate_operacion(roto)
    assert len(problems) == 1, problems  # un solo mensaje: no se repite que falta la clave nueva
    assert problems[0].startswith("presupuestos_diarios: esta clave ya no se usa")
    assert "presupuestos_por_estacion" in problems[0]
    assert "bicis movidas por estación por día" in problems[0]
    assert "no el total de bicis del día" in problems[0]


def test_la_clave_anterior_se_rechaza_aunque_tambien_este_la_nueva():
    roto = copy.deepcopy(BASE)
    roto["presupuestos_diarios"] = [0, 50, 100]
    problems = validate_operacion(roto)
    assert len(problems) == 1 and problems[0].startswith("presupuestos_diarios: esta clave ya no se usa"), problems


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c.pop("presupuesto_de_referencia"), "presupuesto_de_referencia: debe ser un número mayor que 0"),
        # El 0 está en la lista, pero con 0 no se mueve nada: no sirve para recomendar.
        (lambda c: c.update(presupuesto_de_referencia=0), "presupuesto_de_referencia: debe ser un número mayor que 0"),
        (lambda c: c.update(presupuesto_de_referencia=-0.5), "presupuesto_de_referencia: debe ser un número mayor"),
        (lambda c: c.update(presupuesto_de_referencia=True), "presupuesto_de_referencia: debe ser un número mayor"),
        (lambda c: c.update(presupuesto_de_referencia=float("inf")), "presupuesto_de_referencia: debe ser un número"),
        (lambda c: c.update(presupuesto_de_referencia="0,5"), "usa punto decimal: 0.5"),
        (
            lambda c: c.update(presupuesto_de_referencia=0.75),
            "presupuesto_de_referencia: 0.75 no está en presupuestos_por_estacion [0, 0.5, 1.0]",
        ),
        # Quitar de la lista el valor de referencia también se detecta.
        (lambda c: c.update(presupuestos_por_estacion=[0, 1.0, 2.0]), "presupuesto_de_referencia: 0.5 no está en"),
    ],
)
def test_detecta_un_presupuesto_de_referencia_imposible(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert any(texto_esperado in problem for problem in problems), problems


def test_el_presupuesto_de_referencia_puede_escribirse_como_entero():
    """`1` y `1.0` son el mismo valor: escribir uno en la lista y el otro en la referencia es válido."""
    bueno = copy.deepcopy(BASE)
    bueno["presupuesto_de_referencia"] = 1
    assert validate_operacion(bueno) == []


def test_presupuesto_en_bicis_multiplica_por_las_estaciones_y_redondea():
    # El caso del YAML y de docs/reglas_operacion.md: las 2,315 estaciones simuladas el 10 de octubre de 2026.
    por_estacion = (0, 0.1, 0.25, 0.5, 1.0, 2.0)
    assert [presupuesto_en_bicis(valor, 2315) for valor in por_estacion] == [0, 232, 579, 1158, 2315, 4630]
    # Un medio exacto va al entero par: 1,157.5 queda en 1,158 y 1,156.5 queda en 1,156.
    assert presupuesto_en_bicis(0.5, 2315) == 1158
    assert presupuesto_en_bicis(0.5, 2313) == 1156
    assert presupuesto_en_bicis(0.5, 2300) == 1150
    assert presupuesto_en_bicis(0, 2300) == 0
    assert presupuesto_en_bicis(0.1, 2300) == 230   # 0.1 x 2300 da 230.00000000000003
    assert presupuesto_en_bicis(0.25, 41) == 10     # 10.25 baja a 10
    assert presupuesto_en_bicis(0.3, 9) == 3        # 2.7 sube a 3
    assert isinstance(presupuesto_en_bicis(1.0, 40), int)


# --- Pregunta 1: reparto del presupuesto del día entre las franjas ---


def _partes(*valores):
    """Cambio que deja las franjas de BASE con estas partes del presupuesto, en orden."""

    def cambio(c):
        for franja, valor in zip(c["franjas"], valores, strict=True):
            franja["parte_del_presupuesto"] = valor

    return cambio


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c["franjas"][0].pop("parte_del_presupuesto"), "franja 'manana': falta parte_del_presupuesto"),
        (_partes(0, 1), "franja 'manana': parte_del_presupuesto debe ser un número mayor que 0 y hasta 1"),
        (_partes(-0.5, 0.5), "franja 'manana': parte_del_presupuesto debe ser un número mayor que 0 y hasta 1"),
        (_partes(0.5, 1.5), "franja 'tarde': parte_del_presupuesto debe ser un número mayor que 0 y hasta 1"),
        (_partes(True, 0.5), "franja 'manana': parte_del_presupuesto debe ser un número"),
        (_partes(float("nan"), 0.5), "franja 'manana': parte_del_presupuesto debe ser un número"),
        (_partes("0,5", 0.5), "usa punto decimal: 0.5"),
        # Un porcentaje donde va una fracción.
        (_partes(50, 50), "si es un porcentaje, escríbelo como fracción: 50 % es 0.5"),
        (_partes("60 %", 0.4), "si es un porcentaje, escríbelo como fracción: 60 % es 0.6"),
        # Cada parte es válida, pero entre todas no reparten el día completo. El mensaje enseña la suma.
        (_partes(0.5, 0.4), "deben sumar 1 y suman 0.9 (manana 0.5 + tarde 0.4)"),
        (_partes(0.6, 0.5), "deben sumar 1 y suman 1.1 (manana 0.6 + tarde 0.5)"),
        (_partes(1, 1), "deben sumar 1 y suman 2 (manana 1 + tarde 1)"),
        (_partes(0.33, 0.33), "deben sumar 1 y suman 0.66"),
        (_partes(0.5, 0.49999), "deben sumar 1 y suman 0.99999"),
    ],
)
def test_detecta_un_reparto_del_presupuesto_imposible(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert any(texto_esperado in problem for problem in problems), problems


def test_si_una_parte_es_invalida_no_se_reporta_ademas_la_suma():
    """Con una parte mal escrita la suma no dice nada útil: se pide arreglar esa parte y ya."""
    roto = copy.deepcopy(BASE)
    _partes(50, 50)(roto)
    problems = validate_operacion(roto)
    assert len(problems) == 2 and not any("deben sumar" in problem for problem in problems), problems


def _con_tres_franjas(partes):
    """BASE con una tercera franja en la noche y estas tres partes del presupuesto."""
    cfg = copy.deepcopy(BASE)
    cfg["franjas"].append({"nombre": "noche", "momento_intervencion": "21:30", "inicio": "22:00", "fin": "23:59"})
    for franja, parte in zip(cfg["franjas"], partes, strict=True):
        franja["parte_del_presupuesto"] = parte
    for regla in cfg["estrategia_por_tipo"].values():
        regla["ocupacion_respaldo"]["noche"] = 0.5
    return cfg


@pytest.mark.parametrize(
    "partes",
    [(0.1, 0.2, 0.7), (0.33, 0.33, 0.34), (0.3333, 0.3333, 0.3334), (0.6, 0.3, 0.1)],
    ids=str,
)
def test_tres_partes_que_suman_uno_pasan(partes):
    assert validate_operacion(_con_tres_franjas(partes)) == []


@pytest.mark.parametrize("partes", [(0.29, 0.7, 0.01), (0.69, 0.01, 0.3), (0.102, 0.204, 0.694)], ids=str)
def test_la_holgura_perdona_el_redondeo_de_los_decimales_de_la_computadora(partes):
    """Estas partes suman 1 en papel y 0.9999999999999999 en la computadora: deben pasar."""
    assert math.fsum(partes) != 1  # si esto dejara de cumplirse, la prueba ya no probaría la holgura
    assert validate_operacion(_con_tres_franjas(partes)) == []


def test_tres_tercios_mal_redondeados_no_pasan():
    problems = validate_operacion(_con_tres_franjas((0.33, 0.33, 0.33)))
    assert problems == [
        "franjas: las parte_del_presupuesto deben sumar 1 y suman 0.99 (manana 0.33 + tarde 0.33 + noche 0.33). "
        "Ajusta una para que entre todas repartan el presupuesto completo del día"
    ]


def test_una_sola_franja_se_lleva_todo_el_presupuesto():
    cfg = copy.deepcopy(BASE)
    cfg["franjas"] = cfg["franjas"][:1]
    for regla in cfg["estrategia_por_tipo"].values():
        del regla["ocupacion_respaldo"]["tarde"]
    cfg["franjas"][0]["parte_del_presupuesto"] = 1  # también vale escrito 1.0
    assert validate_operacion(cfg) == []
    cfg["franjas"][0]["parte_del_presupuesto"] = 0.5
    assert any("deben sumar 1 y suman 0.5 (manana 0.5)" in problem for problem in validate_operacion(cfg))


# --- Pregunta 5: tipo que se aplica a una estación sin tipo ---


@pytest.mark.parametrize("tipo", STATION_TYPES)
def test_tipo_si_falta_acepta_cualquiera_de_los_tipos(tipo):
    bueno = copy.deepcopy(BASE)
    bueno["tipo_si_falta"] = tipo
    assert validate_operacion(bueno) == []


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c.pop("tipo_si_falta"), "(falta)"),
        (lambda c: c.update(tipo_si_falta=None), "(falta)"),
        (lambda c: c.update(tipo_si_falta="comercial"), "(dice 'comercial')"),
        (lambda c: c.update(tipo_si_falta="Mixta"), "(dice 'Mixta')"),
        (lambda c: c.update(tipo_si_falta=2), "(dice 2)"),
        (lambda c: c.update(tipo_si_falta=["mixta"]), "(dice ['mixta'])"),
    ],
)
def test_detecta_un_tipo_si_falta_imposible(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"tipo_si_falta: debe ser uno de {list(STATION_TYPES)}")
    assert texto_esperado in problems[0]


# --- Leer el archivo: errores de escritura ---


def _archivo(tmp_path, texto):
    archivo = tmp_path / "operacion.yaml"
    archivo.write_text(texto, encoding="utf-8")
    return archivo


def _editado(viejo, nuevo):
    """TEXTO_BASE con un pedazo cambiado, como lo dejaría quien edita el archivo a mano."""
    assert viejo in TEXTO_BASE, viejo
    return TEXTO_BASE.replace(viejo, nuevo, 1)


def test_un_archivo_valido_se_carga_tal_cual(tmp_path):
    assert load_operacion(_archivo(tmp_path, TEXTO_BASE)) == BASE


def test_un_archivo_invalido_no_se_carga(tmp_path):
    with pytest.raises(OperacionError):
        load_operacion(_archivo(tmp_path, "estado: provisional\n"))


def test_un_archivo_con_la_clave_anterior_no_se_carga(tmp_path):
    texto = _editado("presupuestos_por_estacion: [0, 0.5, 1.0]", "presupuestos_diarios: [0, 50, 100]")
    with pytest.raises(OperacionError, match="Ahora se llama presupuestos_por_estacion y cambió la unidad"):
        load_operacion(_archivo(tmp_path, texto))


def test_un_archivo_con_un_umbral_en_0_no_se_carga(tmp_path):
    texto = _editado("  lugares_libres_minimos: 2", "  lugares_libres_minimos: 0")
    with pytest.raises(OperacionError, match="lugares_libres_minimos: debe ser un entero mayor o igual a 1"):
        load_operacion(_archivo(tmp_path, texto))


@pytest.mark.parametrize(
    "viejo, nuevo, donde",
    [
        # (texto de TEXTO_BASE, cómo quedó mal escrito, dónde manda a revisar el mensaje)
        ("  bicis_minimas: 2", "\tbicis_minimas: 2", "Revisa cerca de la línea 16."),
        ("  lugares_libres_minimos: 2", "   lugares_libres_minimos: 2", "Revisa cerca de la línea 17."),
        ("distancia_maxima_km: 3.0", "distancia_maxima_km:3.0", "Revisa de la línea 20 a la 21,"),
        # En la línea 1, PyYAML solo nota el problema al llegar a la 2: por eso dice "cerca de".
        ("estado: provisional", "estado:provisional", "Revisa cerca de la línea 2."),
        ('inicio: "06:00"', 'inicio: "06:00', "Revisa de la línea 5 a la 8,"),
        ("[0, 0.5, 1.0]", "[0, 0.5, 1.0", "Revisa de la línea 18 a la 19,"),
        ("parte_del_presupuesto: 0.5", "parte_del_presupuesto:0.5", "Revisa de la línea 9 a la 10,"),
        ("presupuesto_de_referencia: 0.5", "presupuesto_de_referencia:0.5", "Revisa de la línea 19 a la 20,"),
        ("tipo_si_falta: mixta", "tipo_si_falta:mixta", "Revisa de la línea 21 a la 22,"),
    ],
    ids=[
        "tabulador",
        "un_espacio_de_mas",
        "falta_espacio_tras_dos_puntos",
        "falta_espacio_en_la_primera_linea",
        "comilla_sin_cerrar",
        "corchete_sin_cerrar",
        "falta_espacio_en_la_parte_del_presupuesto",
        "falta_espacio_en_el_presupuesto_de_referencia",
        "falta_espacio_en_tipo_si_falta",
    ],
)
def test_un_archivo_mal_escrito_se_explica_sin_traza_de_yaml(tmp_path, viejo, nuevo, donde):
    # El error de PyYAML (en inglés) no debe salir: se convierte en OperacionError con la línea.
    with pytest.raises(OperacionError) as error:
        load_operacion(_archivo(tmp_path, _editado(viejo, nuevo)))
    mensaje = str(error.value)
    assert "está mal escrito" in mensaje and donde in mensaje, mensaje


@pytest.mark.parametrize(
    "viejo, nuevo, texto_esperado",
    [
        # Dentro de una lista entre corchetes la coma separa valores: 0,5 es 0 y 5; 1,150 es 1 y 150.
        # La lista está en la línea 18 de TEXTO_BASE.
        ("[0, 0.5, 1.0]", "[0, 0,5, 1.0]", r"'0,5' \(línea 18\) se lee como dos valores \(0 y 5\)"),
        ("[0, 0.5, 1.0]", "[0, 0,5, 1.0]", r"Si es un decimal, usa punto: 0\.5\."),
        # Este se leería como 0, 0.5, 1 y 5: una lista válida. Sin la revisión pasaría sin aviso.
        ("[0, 0.5, 1.0]", "[0, 0.5, 1,5]", r"'1,5' \(línea 18\) se lee como dos valores \(1 y 5\)"),
        ("[0, 0.5, 1.0]", "[0, 500, 1,150]", r"'1,150' .* se lee como dos valores \(1 y 150\).* separador de miles"),
        ("[0, 0.5, 1.0]", "[0, 0.5, 1,000]", r"'1,000' .* se lee como dos valores \(1 y 000\)"),
        ("[0, 0.5, 1.0]", "[0,5, 1.0]", r"sepáralos con coma y espacio: 0, 5"),
        # Fuera de una lista YAML lo lee como texto y el validador dice cómo escribirlo.
        ("distancia_maxima_km: 3.0", "distancia_maxima_km: 3,0", r"usa punto decimal: 3\.0"),
        ("manana: 0.5", "manana: 0,30", r"usa punto decimal: 0\.30"),
        ("presupuesto_de_referencia: 0.5", "presupuesto_de_referencia: 0,5", r"usa punto decimal: 0\.5"),
        ("parte_del_presupuesto: 0.5", "parte_del_presupuesto: 0,5", r"usa punto decimal: 0\.5"),
        ("[0, 0.5, 1.0]", "0,5", r"presupuestos_por_estacion: debe ser una lista .*usa punto decimal: 0\.5"),
    ],
    ids=[
        "decimal_en_la_lista",
        "decimal_en_la_lista_como_arreglarlo",
        "decimal_que_dejaria_una_lista_valida",
        "miles_1,150",
        "miles_1,000",
        "valores_pegados",
        "coma_en_distancia",
        "coma_en_ocupacion",
        "coma_en_referencia",
        "coma_en_parte_del_presupuesto",
        "coma_en_vez_de_lista",
    ],
)
def test_un_numero_escrito_con_coma_se_explica(tmp_path, viejo, nuevo, texto_esperado):
    with pytest.raises(OperacionError, match=texto_esperado):
        load_operacion(_archivo(tmp_path, _editado(viejo, nuevo)))


@pytest.mark.parametrize(
    "lista",
    ["[0, 0.5, 1.0, 1.5]", "[0,0.5,1.0]", "[0.5]", "[ 0 , 0.5 , 1 ]", "[0, 0.25,\n  0.5]"],
    ids=["mas_valores", "sin_espacios_con_punto", "un_solo_valor", "espacios_de_mas", "en_dos_lineas"],
)
def test_una_lista_bien_escrita_no_se_confunde_con_un_numero_con_coma(tmp_path, lista):
    cfg = load_operacion(_archivo(tmp_path, _editado("[0, 0.5, 1.0]", lista)))
    assert cfg["presupuestos_por_estacion"] == yaml.safe_load(lista)


def test_una_clave_repetida_no_se_acepta_en_silencio(tmp_path):
    # PyYAML se quedaría con la última (300) sin avisar.
    archivo = _archivo(tmp_path, TEXTO_BASE + "distancia_maxima_km: 300\n")
    linea = len(TEXTO_BASE.splitlines()) + 1
    with pytest.raises(OperacionError, match=f"'distancia_maxima_km' está escrito dos veces .*línea {linea}\\b"):
        load_operacion(archivo)
