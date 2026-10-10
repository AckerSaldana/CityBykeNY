"""Prueba de humo del dashboard: las cinco páginas abren y respetan las reglas.

Usa AppTest de Streamlit, que ejecuta las páginas sin navegador. Las tablas
pequeñas se construyen con data/sample en una carpeta temporal y app/datos.py
se apunta ahí: ninguna prueba toca data/processed.

    pytest tests/test_app_smoke.py
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"

# Orden y títulos del menú. Las rutas son relativas a app/Inicio.py.
PAGES = {
    "Inicio": "vistas/inicio.py",
    "Mapa de demanda": "vistas/mapa_de_demanda.py",
    "Estaciones y rutas": "vistas/estaciones_y_rutas.py",
    "Pronóstico por estación": "vistas/pronostico_por_estacion.py",
    "Confiabilidad": "vistas/confiabilidad.py",
}
LINE = "Resultados de simulación con datos de validación"
# Esa línea va solo en las páginas con resultados de la simulación o del pronóstico.
# Las que describen viajes dicen de qué meses son (aquí, de los datos de muestra).
PAGES_WITH_LINE = ("Inicio", "Pronóstico por estación", "Confiabilidad")
CONTEXT_PAGE = "vistas/contexto.py"
BUILD = "python scripts/build_app_tables.py"

# Nombres que no deben verse en pantalla: columnas de los contratos, columnas
# internas de las tablas de la app y nombres de archivo.
FORBIDDEN = [
    "station_id", "ts_hour", "departures", "arrivals", "net_flow", "station_type", "capacity",
    "dep_q", "arr_q", "window_start", "window_end", "start_inventory_target", "bikes_to_move",
    "risk_level", "feasible", "risk_station_hours", "ci_low", "ci_high", "start_station",
    "end_station", "station_hour", "policy_curve", "baseline", "lgbm", "historical",
    "estacion_id", "nivel_riesgo", "pct_evitado", "flujo_neto", "tipo_dia", "sin_dato",
    ".parquet", ".json", ".csv", "datos leídos", "pendiente", "nan", "none",
    "inventario_antes", "hora_inicio", "tipo_estacion", "pct_min", "media_global", "model",
]
# Más de un decimal en pantalla (los números van con datos.num y datos.pct).
LONG_DECIMALS = r"\d\.\d{2,}"


@pytest.fixture(scope="session")
def tablas(tmp_path_factory):
    """Las tablas de la app, construidas una sola vez con los datos de muestra."""
    carpeta = tmp_path_factory.mktemp("app_tablas")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture()
def datos(monkeypatch, tablas):
    """El módulo app/datos.py (el mismo que importan las páginas), leyendo de la carpeta temporal."""
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", tablas)
    return module


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    if page != PAGES["Inicio"]:
        at = at.switch_page(page).run()
    return at


def visible_text_of(at: AppTest) -> str:
    """Todo el texto de la página, en minúsculas."""
    partes = []
    for clase in ("title", "header", "subheader", "markdown", "caption", "info", "warning",
                  "error", "success", "text", "metric", "selectbox", "radio", "slider",
                  "select_slider", "multiselect", "checkbox", "toggle", "button", "tabs", "expander"):
        for elemento in getattr(at, clase, []):
            for campo in ("label", "value", "body", "delta", "options"):
                valor = getattr(elemento, campo, None)
                if valor is not None:
                    partes.append(str(valor))
    for tabla in at.dataframe:
        partes.append(" ".join(map(str, tabla.value.columns)))
    return " ".join(partes).lower()


@pytest.mark.parametrize("page", PAGES.values(), ids=PAGES.keys())
def test_la_pagina_abre_sin_errores_y_con_la_linea_discreta(datos, page):
    at = open_page(page)

    assert [str(e.value) for e in at.exception] == []
    assert not at.error
    assert len(at.title) == 1
    titulo = next(t for t, ruta in PAGES.items() if ruta == page)
    assert (titulo in at.title[0].value) == (titulo != "Inicio")  # se abrió esa página y no otra
    lineas = [e.value for e in at.caption if LINE in e.value]
    if titulo in PAGES_WITH_LINE:
        assert len(lineas) == 1
    else:
        assert lineas == []
        lineas = [e.value for e in at.caption if e.value.startswith("Viajes de ejemplo de ")]
        assert len(lineas) == 1 and "Viajes reales" not in lineas[0]
        assert re.search(r"de \w+ a \w+ de \d{4}", lineas[0])  # los meses salen del periodo de los datos
    # Los datos de muestra son sintéticos y la línea lo dice (nunca se presentan como reales).
    assert "sintéticos" in lineas[0]
    assert not at.main.warning


def test_la_linea_de_viajes_sale_del_periodo(datos, monkeypatch):
    assert datos.periodo_en_meses("2026-07-01", "2026-09-29") == "julio a septiembre de 2026"
    assert datos.periodo_en_meses("2026-07-01", "2026-07-31") == "julio de 2026"
    assert datos.periodo_en_meses("2025-12-01", "2026-01-31") == "diciembre de 2025 a enero de 2026"
    reales = {**datos.indicadores(), "sinteticos": False, "periodo": {"inicio": "2026-07-01", "fin": "2026-09-29"}}
    monkeypatch.setattr(datos, "indicadores", lambda: reales)
    assert datos.linea_discreta("viajes") == "Viajes reales de julio a septiembre de 2026"
    assert datos.linea_discreta() == LINE
    monkeypatch.setattr(datos, "indicadores", lambda: {**reales, "periodo": {}})
    assert datos.linea_discreta("viajes") == "Viajes reales del periodo analizado"


def test_la_pagina_de_contexto_abre_con_la_linea_de_viajes(datos):
    """La sexta página: con los datos de muestra no hay tabla de contexto y lo dice sin fallar."""
    at = open_page(CONTEXT_PAGE)

    assert [str(e.value) for e in at.exception] == []
    assert not at.error
    assert not any(LINE in e.value for e in at.caption)
    assert any(e.value.startswith("Viajes de ejemplo") for e in at.caption)
    texto = visible_text_of(at)
    assert ".parquet" not in texto and not re.search(r"\bnone\b|\bnan\b", texto)


@pytest.mark.parametrize("page", PAGES.values(), ids=PAGES.keys())
def test_no_se_ven_nombres_de_columnas_ni_de_archivos(datos, page):
    texto = visible_text_of(open_page(page))
    for nombre in FORBIDDEN:
        if nombre in ("nan", "none"):  # valores vacíos de Python que se colaron al texto
            patron = rf"\b{nombre}\b"
        elif nombre.startswith(".") or " " in nombre:
            patron = re.escape(nombre)
        else:
            patron = rf"\b{re.escape(nombre)}"
        if nombre == "model":  # la clave interna; "modelo" sí es una palabra de la pantalla
            patron = r"\bmodel\b"
        assert not re.search(patron, texto), f"{page}: se ve '{nombre}'"
    assert "traceback" not in texto
    assert not re.search(LONG_DECIMALS, texto), f"{page}: número con más de un decimal"


def test_el_menu_tiene_las_cinco_paginas_con_acentos():
    codigo = (APP / "Inicio.py").read_text(encoding="utf-8")
    titulos = re.findall(r'st\.Page\("([^"]+)", title="([^"]+)"', codigo)
    assert [(titulo, ruta) for ruta, titulo in titulos][: len(PAGES)] == list(PAGES.items())  # la sexta es la sección de contexto
    assert "st.navigation" in codigo
    for ruta in PAGES.values():
        assert (APP / ruta).exists(), ruta
    assert not (APP / "pages").exists()  # con pages/ Streamlit armaría otro menú


@pytest.mark.parametrize("page", PAGES.values(), ids=PAGES.keys())
def test_si_faltan_las_tablas_hay_mensaje_amable(datos, monkeypatch, tmp_path, page):
    monkeypatch.setattr(datos, "CARPETA", tmp_path / "no_existe")
    at = open_page(page)

    assert not at.exception
    assert any(BUILD in e.value for e in at.info)


def test_formato_de_numeros_y_sin_dato(datos):
    assert datos.num(15017726) == "15,017,726"
    assert datos.num(4757.666, 1) == "4,757.7"
    assert datos.num(1.23456, 5) == "1.2"  # nunca más de un decimal
    assert datos.pct(0.5394) == "53.9 %"
    assert datos.rango_pct(0.31, 0.54) == "31 % a 54 %"
    for vacio in (None, float("nan")):
        assert datos.num(vacio) == datos.pct(vacio) == datos.texto(vacio) == "Sin dato"
    assert datos.etiqueta(datos.TIPOS_ESTACION, None) == "Sin dato"
    assert datos.etiqueta(datos.POLITICAS, "none") == "Sin intervención"
    assert [datos.POLITICAS[p] for p in ("none", "historical", "model")] == [
        "Sin intervención", "Regla histórica", "Modelo"]
    assert datos.accion(5, 6) == "Llevar 5 bicis antes de las 06:00"
    assert datos.accion(-1, 15) == "Retirar 1 bici antes de las 15:00"
    assert datos.accion(0, 6) == "Sin traslado"
    assert datos.fecha("2026-09-02") == "miércoles 2 de septiembre de 2026"


def test_los_lectores_devuelven_lo_que_prometen(datos):
    ind = datos.indicadores()
    est = datos.estaciones()
    assert ind["estaciones"] == len(est)

    mapa = datos.demanda_mapa(8, "laborable", "miembros")
    assert {"nombre", "lat", "lon", "salidas", "llegadas", "flujo_neto"} <= set(mapa.columns)
    todos = datos.demanda_mapa(8, "laborable", "todos")
    assert (mapa["salidas"] <= todos["salidas"] + 0.01).all()
    assert mapa["flujo_neto"].tolist() == todos["flujo_neto"].tolist()  # las llegadas no se separan por usuario
    assert datos.demanda_mapa(None, "fin_de_semana")["salidas"].sum() > todos["salidas"].sum() * 0  # día completo

    assert len(datos.top_estaciones(15)) == 15 and len(datos.top_rutas(15)) == 15
    assert not datos.top_rutas(15, con_ida_y_vuelta=False)["misma_estacion"].any()

    opciones = datos.opciones_de_estacion()
    estacion_id = next(iter(opciones))
    pron = datos.pronostico_estacion(estacion_id)
    assert list(pron.columns) == ["ts", "fecha", "hora", "modelo", "q10", "q50", "q90", "observado"]
    assert set(pron["modelo"]) == {"baseline", "lgbm"}
    assert pron["fecha"].nunique() == ind["validacion"]["dias"]
    assert datos.pronostico_estacion("no existe").empty
    rec = datos.recomendaciones_estacion(estacion_id)
    assert set(rec["estacion_id"]) <= {estacion_id}
    assert datos.estacion(estacion_id)["nombre"] and datos.estacion("no existe") == {}

    assert set(datos.curva()["politica"]) == {"none", "historical", "model"}
    assert set(datos.metricas("bloque")["grupo"]) == set(datos.BLOQUES)
    assert set(datos.metricas("volumen")["grupo"]) <= set(datos.VOLUMENES)
    assert set(datos.riesgo_niveles("tipo_estacion")["grupo"]) <= set(datos.TIPOS_ESTACION)
    assert list(datos.sensibilidad().columns)[:2] == ["inventario_inicial", "politica"]
    assert isinstance(datos.limitaciones(), list)


def test_los_hallazgos_se_arman_con_las_cifras(datos, monkeypatch):
    ind = datos.indicadores()
    frases = datos.hallazgos()
    assert len(frases) == 4 and all(isinstance(f, str) and f.endswith(".") for f in frases)
    assert datos.num(ind["referencia"]["presupuesto"]) in frases[0]

    # Si cambia una cifra, cambia la frase: no hay números escritos a mano.
    otro = {**ind, "referencia": {**ind["referencia"], "presupuesto": 987654}}
    monkeypatch.setattr(datos, "indicadores", lambda: otro)
    assert "987,654" in datos.hallazgos()[0]

    # El cuarto: empleos alrededor de las estaciones de oficinas contra las residenciales.
    con_contexto = {**ind, "contexto": {"por_tipo": {"oficinas": {"proporcion_empleos": 0.871},
                                                     "residencial": {"proporcion_empleos": 0.294}}}}
    monkeypatch.setattr(datos, "indicadores", lambda: con_contexto)
    assert "87 %" in datos.hallazgos()[3] and "29 %" in datos.hallazgos()[3]

    # Sin simulación, las frases dicen "Sin dato" en lugar de fallar.
    vacio = {**ind, "referencia": {"presupuesto": None, "politicas": {}},
             "sensibilidad": {"filas": [], "rango": {}},
             "comparacion": {"modelo_pronostica_mejor": None, "modelo_evita": None},
             "presupuesto_umbral": {"politicas": {}}}
    monkeypatch.setattr(datos, "indicadores", lambda: vacio)
    assert all("Sin dato" in f for f in datos.hallazgos())


# --------------------------------------------------------------------------- #
# Reglas de la carpeta app/
# --------------------------------------------------------------------------- #

def test_la_configuracion_de_streamlit_esta_junto_a_inicio():
    """En app/.streamlit se lee desde cualquier carpeta. En la raíz, solo al abrir la app desde la raíz."""
    config = tomllib.loads((APP / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert config["server"]["showEmailPrompt"] is False
    # Para la demostración no se vigilan los archivos: no aparece el aviso de "Source file changed".
    assert config["server"]["fileWatcherType"] == "none"
    assert config["server"]["runOnSave"] is False
    assert config["browser"]["gatherUsageStats"] is False


def test_la_app_no_importa_nada_de_src_ni_lee_las_tablas_grandes():
    """app/ es independiente y solo datos.py lee archivos (las tablas pequeñas)."""
    for path in sorted(APP.rglob("*.py")):
        codigo = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(codigo)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] != "src", f"{path.name} importa {module}"
        if path.name != "datos.py":
            for lector in ("read_parquet", "read_table", "duckdb", "read_csv"):
                assert lector not in codigo, f"{path.name} lee archivos por su cuenta ({lector})"


def test_la_app_no_habla_de_viajes_que_los_datos_no_registran():
    """Decisión 9: los datos solo registran demanda atendida. La frase prohibida no aparece en app/."""
    forbidden = " ".join(["viajes", "recuperados"])  # armada aquí para no escribirla
    for path in sorted(APP.rglob("*")):
        if path.suffix in (".py", ".md"):
            text = " ".join(path.read_text(encoding="utf-8").lower().split())
            assert forbidden not in text, path
