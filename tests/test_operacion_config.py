"""Comprueba que config/operacion.yaml se puede leer y tiene valores posibles.

Córrela después de editar el archivo:  pytest tests/test_operacion_config.py

Solo las dos primeras pruebas leen el archivo del repositorio. Las demás prueban
el validador con una configuración fija escrita aquí abajo (TEXTO_BASE): así,
cambiar config/operacion.yaml por otros valores válidos (otras horas, otros
nombres de franja, una o tres franjas, otros presupuestos) no rompe ninguna.
"""

import copy

import pytest
import yaml

from src.contracts import STATION_TYPES
from src.sim.operacion import OperacionError, load_operacion, validate_operacion

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
  - nombre: tarde
    momento_intervencion: "14:00"
    inicio: "15:00"
    fin: "21:00"
umbral_riesgo:
  bicis_minimas: 2
  lugares_libres_minimos: 2
presupuestos_diarios: [0, 50, 100]
distancia_maxima_km: 3.0
estrategia_por_tipo:
""" + "".join(
    f"  {tipo}:\n    prioridad: 2\n    ocupacion_objetivo:\n      manana: 0.5\n      tarde: 0.5\n"
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


def test_estan_los_cinco_parametros(cfg):
    for parametro in (
        "franjas",
        "umbral_riesgo",
        "presupuestos_diarios",
        "distancia_maxima_km",
        "estrategia_por_tipo",
    ):
        assert parametro in cfg


# --- De aquí en adelante ninguna prueba lee config/operacion.yaml ---


def test_la_configuracion_de_prueba_es_valida():
    assert validate_operacion(BASE) == []


@pytest.mark.parametrize(
    "cambio, texto_esperado",
    [
        (lambda c: c["franjas"][0].update(inicio="25:00"), "HH:MM"),
        (lambda c: c["franjas"][0].update(inicio=600), "HH:MM"),
        (lambda c: c["franjas"][0].update(momento_intervencion="07:00"), "momento_intervencion <= inicio < fin"),
        (lambda c: c["franjas"][1].update(momento_intervencion="10:00", inicio="11:00"), "se enciman"),
        (lambda c: c["umbral_riesgo"].update(bicis_minimas=-1), "bicis_minimas"),
        (lambda c: c.update(presupuestos_diarios=[]), "presupuestos_diarios"),
        (lambda c: c.update(presupuestos_diarios=[0, 50, 50]), "repetidos"),
        (lambda c: c.update(distancia_maxima_km=0), "distancia_maxima_km"),
        (lambda c: c["estrategia_por_tipo"].pop("mixta"), "los tipos deben ser exactamente"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"].update(prioridad=5), "prioridad"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"]["ocupacion_objetivo"].update(manana=1.5), "entre 0 y 1"),
        (lambda c: c["estrategia_por_tipo"]["oficinas"]["ocupacion_objetivo"].pop("tarde"), "un valor por franja"),
        (lambda c: c.update(estado="confirmado"), "confirmado_por"),
        (lambda c: c.update(estado="listo"), "estado"),
        (lambda c: c.update(presupuestos_diarios=[0, 100, 50]), "de menor a mayor"),
        (lambda c: c.update(distancia_maxima_km=float("nan")), "mayor que 0"),
        (lambda c: c.update(distancia_maxima_km=float("inf")), "mayor que 0"),
    ],
)
def test_detecta_valores_imposibles(cambio, texto_esperado):
    roto = copy.deepcopy(BASE)
    cambio(roto)
    problems = validate_operacion(roto)
    assert any(texto_esperado in problem for problem in problems), problems


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


@pytest.mark.parametrize(
    "viejo, nuevo, donde",
    [
        # (texto de TEXTO_BASE, cómo quedó mal escrito, dónde manda a revisar el mensaje)
        ("  bicis_minimas: 2", "\tbicis_minimas: 2", "Revisa cerca de la línea 14."),
        ("  lugares_libres_minimos: 2", "   lugares_libres_minimos: 2", "Revisa cerca de la línea 15."),
        ("distancia_maxima_km: 3.0", "distancia_maxima_km:3.0", "Revisa de la línea 17 a la 18,"),
        # En la línea 1, PyYAML solo nota el problema al llegar a la 2: por eso dice "cerca de".
        ("estado: provisional", "estado:provisional", "Revisa cerca de la línea 2."),
        ('inicio: "06:00"', 'inicio: "06:00', "Revisa de la línea 5 a la 8,"),
        ("[0, 50, 100]", "[0, 50, 100", "Revisa de la línea 16 a la 17,"),
    ],
    ids=[
        "tabulador",
        "un_espacio_de_mas",
        "falta_espacio_tras_dos_puntos",
        "falta_espacio_en_la_primera_linea",
        "comilla_sin_cerrar",
        "corchete_sin_cerrar",
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
        # Con separador de miles YAML lee dos valores: 1,500 es 1 y 500; 1,000 es 1 y 0.
        ("[0, 50, 100]", "[0, 50, 100, 1,500]", "de menor a mayor. Si usaste separador de miles"),
        ("[0, 50, 100]", "[0, 50, 100, 1,000]", "repetidos. Si usaste separador de miles"),
        ("distancia_maxima_km: 3.0", "distancia_maxima_km: 3,0", "usa punto decimal: 3.0"),
        ("manana: 0.5", "manana: 0,30", "usa punto decimal: 0.30"),
    ],
    ids=["miles_1,500", "miles_1,000", "coma_en_distancia", "coma_en_ocupacion"],
)
def test_un_numero_escrito_con_coma_se_explica(tmp_path, viejo, nuevo, texto_esperado):
    with pytest.raises(OperacionError, match=texto_esperado):
        load_operacion(_archivo(tmp_path, _editado(viejo, nuevo)))


def test_una_clave_repetida_no_se_acepta_en_silencio(tmp_path):
    # PyYAML se quedaría con la última (300) sin avisar.
    archivo = _archivo(tmp_path, TEXTO_BASE + "distancia_maxima_km: 300\n")
    linea = len(TEXTO_BASE.splitlines()) + 1
    with pytest.raises(OperacionError, match=f"'distancia_maxima_km' está escrito dos veces .*línea {linea}\\b"):
        load_operacion(archivo)
