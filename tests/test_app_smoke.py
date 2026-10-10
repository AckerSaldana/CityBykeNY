"""Prueba de humo del dashboard: abre sin errores y avisa de los datos sintéticos.

Usa AppTest de Streamlit, que ejecuta las páginas sin navegador. Comprueba que
el código corre y qué textos produce. No comprueba cómo se ve en pantalla.

Ninguna prueba escribe en data/processed: las que necesitan archivos "reales"
apuntan las carpetas de app/datos.py a carpetas temporales.

    pytest tests/test_app_smoke.py
"""

import ast
import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"

# None es la página de inicio. Las demás rutas son relativas a app/Inicio.py.
PAGES = [
    None,
    "pages/1_Mapa_de_demanda.py",
    "pages/2_Ranking_de_estaciones_y_rutas.py",
    "pages/3_Pronostico_por_estacion.py",
]
PAGE_IDS = ["inicio", "mapa", "ranking", "pronostico"]
# Todas las vistas que hay hoy en app/pages, leídas del disco. Una vista nueva
# entra sola a la prueba del aviso, sin tocar este archivo.
ALL_PAGES = [None] + sorted(f"pages/{path.name}" for path in (APP / "pages").glob("*.py"))
TABLES = ["stations", "station_hour", "predictions", "recommendations", "policy_curve"]
WARNING = "DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"
MAKE_SAMPLE = "python scripts/make_sample_data.py"


@pytest.fixture()
def datos(monkeypatch):
    """El módulo app/datos.py: el mismo objeto que importan las páginas."""
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    return module


@pytest.fixture()
def processed(datos, monkeypatch, tmp_path):
    """Hace que la app busque los datos reales en una carpeta temporal vacía."""
    folder = tmp_path / "processed"
    folder.mkdir()
    monkeypatch.setattr(datos, "DATA_PROCESSED", folder)
    return folder


def open_page(page: str | None) -> AppTest:
    """Abre la app como `streamlit run app/Inicio.py` y, si se pide, cambia de vista."""
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    if page is not None:
        at = at.switch_page(page).run()
    return at


def errors_of(at: AppTest) -> list[str]:
    """Textos de las excepciones que quedaron en pantalla (lista vacía = ninguna)."""
    return [str(element.value) for element in at.exception]


def write_stations(path: Path, station_id: str) -> None:
    """Escribe una tabla de estaciones de una fila, con las columnas del contrato."""
    pd.DataFrame(
        {
            "station_id": [station_id],
            "name": ["Estación de prueba"],
            "lat": [40.75],
            "lon": [-73.98],
            "capacity": [20],
            "station_type": ["mixta"],
        }
    ).to_parquet(path)


# --------------------------------------------------------------------------- #
# La app abre y avisa
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_abre_sin_errores_y_avisa_que_los_datos_son_sinteticos(datos, processed, page):
    at = open_page(page)

    assert errors_of(at) == []
    assert not at.error
    assert len(at.title) == 1
    assert "Pendiente" in [element.value for element in at.subheader]
    assert any("filas leídas" in element.value for element in at.markdown)

    # Sin datos reales, las cinco tablas salen de data/sample y el aviso las nombra.
    assert datos.synthetic_tables() == TABLES
    text = at.main.warning[0].value
    assert WARNING in text
    for table in TABLES:
        assert f"`{table}`" in text
    assert WARNING in at.sidebar.warning[0].value


def test_abre_con_lo_que_hay_hoy_en_el_repositorio(datos):
    """Sin carpetas temporales: data/processed y data/sample tal como están."""
    for page in PAGES:
        at = open_page(page)
        assert errors_of(at) == [], page
        assert not at.error, page
        # El aviso se ve exactamente cuando alguna tabla sale de la muestra.
        assert bool(at.main.warning) == bool(datos.synthetic_tables()), page


@pytest.mark.parametrize("page", ALL_PAGES, ids=lambda page: Path(page).stem if page else "inicio")
def test_toda_vista_de_app_pages_abre_y_muestra_el_aviso(datos, processed, page):
    """Regla para cualquier vista, también las nuevas: abre sin error y trae el aviso."""
    at = open_page(page)
    name = page or "Inicio.py"

    assert errors_of(at) == [], name
    assert not at.error, f"{name} muestra un mensaje de error"
    assert any(WARNING in element.value for element in at.main.warning), (
        f"{name} no llama a datos.show_synthetic_warning()"
    )
    assert any(WARNING in element.value for element in at.sidebar.warning), name


def test_un_enlace_a_una_vista_que_ya_no_existe_avisa_sin_tumbar_inicio(datos, processed, monkeypatch, tmp_path):
    # Simula que los archivos de app/pages cambiaron de nombre: los enlaces de
    # Inicio los buscan en una carpeta donde no están.
    monkeypatch.setattr(datos, "ROOT", tmp_path)

    at = open_page(None)

    assert errors_of(at) == []  # ningún error de Python en pantalla
    assert len(at.error) == 3  # un mensaje por cada enlace que no encontró su vista
    for element in at.error:
        assert "app/pages/" in element.value
        assert "app/Inicio.py" in element.value
    # La página no se corta en "Vistas": sigue la nota de lenguaje honesto.
    assert any("demanda atendida" in element.value for element in at.info)


def test_el_selector_de_estacion_filtra_las_tablas(datos, processed):
    at = open_page("pages/3_Pronostico_por_estacion.py")
    last = datos.load_stations()["station_id"].iloc[-1]

    at.selectbox[0].select(last).run()

    assert errors_of(at) == []
    assert at.selectbox[0].value == last
    assert any(f"({last})" in element.value for element in at.markdown)


# --------------------------------------------------------------------------- #
# data/processed tiene prioridad sobre data/sample
# --------------------------------------------------------------------------- #

def test_el_lector_prefiere_processed_sobre_sample(datos, processed, monkeypatch, tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    monkeypatch.setattr(datos, "DATA_SAMPLE", sample)

    # Solo existe la muestra: se usa la muestra y se declara sintética.
    write_stations(sample / "stations.parquet", "MUESTRA-1")
    assert datos.find_file("stations") == sample / "stations.parquet"
    assert datos.load_stations()["station_id"].tolist() == ["MUESTRA-1"]
    assert datos.synthetic_tables() == ["stations"]

    # Llega el archivo real: se usa ese y la tabla deja de ser sintética.
    write_stations(processed / "stations.parquet", "REAL-1")
    assert datos.find_file("stations") == processed / "stations.parquet"
    assert datos.load_stations()["station_id"].tolist() == ["REAL-1"]
    assert datos.synthetic_tables() == []


def test_el_aviso_solo_nombra_las_tablas_que_siguen_en_la_muestra(datos, processed):
    # Copia de la muestra en la carpeta temporal: solo para simular que llegó un archivo.
    shutil.copy(SAMPLE / "stations.parquet", processed / "stations.parquet")

    at = open_page(None)

    assert errors_of(at) == []
    assert datos.synthetic_tables() == TABLES[1:]
    text = at.main.warning[0].value
    assert WARNING in text
    assert "`stations`" not in text
    for table in TABLES[1:]:
        assert f"`{table}`" in text


def test_el_aviso_explica_la_mezcla_de_tablas_reales_y_sinteticas(datos, processed):
    """Las estaciones de la muestra (SIM-...) no existen en los datos reales: los cruces salen vacíos."""
    mixed = "tablas reales y tablas sintéticas mezcladas"

    # Todo sale de la muestra: no hay mezcla y el aviso no habla de ella.
    assert mixed not in open_page(None).main.warning[0].value

    # Llega una sola tabla real: el aviso explica por qué los cruces salen vacíos.
    shutil.copy(SAMPLE / "stations.parquet", processed / "stations.parquet")
    for page in PAGES:
        text = open_page(page).main.warning[0].value
        assert mixed in text, page
        assert "el resultado sale vacío" in text, page


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_sin_aviso_cuando_todas_las_tablas_son_reales(datos, processed, page):
    # Copia de la muestra en la carpeta temporal: solo para simular datos reales.
    for table in TABLES:
        shutil.copy(SAMPLE / f"{table}.parquet", processed / f"{table}.parquet")

    at = open_page(page)

    assert errors_of(at) == []
    assert not at.error
    assert datos.synthetic_tables() == []
    assert not at.warning


# --------------------------------------------------------------------------- #
# Si falta un archivo: mensaje amable, no un error de Python
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_si_faltan_los_archivos_hay_mensaje_amable(datos, monkeypatch, tmp_path, page):
    monkeypatch.setattr(datos, "DATA_PROCESSED", tmp_path / "no_existe_processed")
    monkeypatch.setattr(datos, "DATA_SAMPLE", tmp_path / "no_existe_sample")

    at = open_page(page)

    assert errors_of(at) == []
    assert len(at.error) == 1
    assert MAKE_SAMPLE in at.error[0].value
    # La página se detiene ahí: no llega a la lista de pendientes.
    assert "Pendiente" not in [element.value for element in at.subheader]


def test_si_falta_una_sola_tabla_solo_fallan_las_vistas_que_la_usan(datos, processed, monkeypatch, tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    monkeypatch.setattr(datos, "DATA_SAMPLE", sample)
    for table in TABLES:
        if table != "predictions":
            shutil.copy(SAMPLE / f"{table}.parquet", sample / f"{table}.parquet")
    assert datos.missing_tables() == ["predictions"]

    # La vista 1 no usa el pronóstico: abre normal.
    at = open_page("pages/1_Mapa_de_demanda.py")
    assert errors_of(at) == []
    assert not at.error
    # Falta un archivo, pero ninguna tabla es real: el aviso no habla de mezcla.
    assert "mezcladas" not in at.main.warning[0].value

    # La vista 3 sí lo usa: muestra el mensaje y dice qué archivo falta.
    at = open_page("pages/3_Pronostico_por_estacion.py")
    assert errors_of(at) == []
    assert "`predictions.parquet`" in at.error[0].value
    assert MAKE_SAMPLE in at.error[0].value


def test_fuera_de_streamlit_el_archivo_faltante_da_un_error_claro(datos, monkeypatch, tmp_path):
    """En un notebook no hay página que detener: se lanza un error con el mismo mensaje."""
    monkeypatch.setattr(datos, "DATA_PROCESSED", tmp_path / "no_existe_processed")
    monkeypatch.setattr(datos, "DATA_SAMPLE", tmp_path / "no_existe_sample")

    with pytest.raises(FileNotFoundError, match="make_sample_data"):
        datos.load_stations()


# --------------------------------------------------------------------------- #
# Horas en el eje de una gráfica
# --------------------------------------------------------------------------- #

def test_hora_para_grafica_da_la_misma_hora_de_nueva_york_sin_zona(datos, processed):
    ts_hour = datos.load_station_hour()["ts_hour"]

    chart_hour = datos.hora_para_grafica(ts_hour)

    assert str(ts_hour.dt.tz) == "America/New_York"  # la tabla conserva su zona
    assert chart_hour.dt.tz is None  # el eje de la gráfica ya no la trae
    clock = "%Y-%m-%d %H:%M"
    assert chart_hour.dt.strftime(clock).tolist() == ts_hour.dt.strftime(clock).tolist()


# --------------------------------------------------------------------------- #
# Reglas de la carpeta app/
# --------------------------------------------------------------------------- #

def test_la_configuracion_de_streamlit_esta_junto_a_inicio():
    """En app/.streamlit se lee desde cualquier carpeta. En la raíz, solo al abrir la app desde la raíz."""
    config = tomllib.loads((APP / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert config["server"]["showEmailPrompt"] is False
    assert config["browser"]["gatherUsageStats"] is False


def test_la_app_no_importa_nada_de_src():
    """app/ es independiente: sigue abriendo aunque cambie el resto del repositorio."""
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] != "src", f"{path.name} importa {module}"


def test_la_app_no_habla_de_viajes_que_los_datos_no_registran():
    """Decisión 9: los datos solo registran demanda atendida. La frase prohibida no aparece en app/."""
    forbidden = " ".join(["viajes", "recuperados"])  # armada aquí para no escribirla
    for path in sorted(APP.rglob("*")):
        if path.suffix in (".py", ".md"):
            text = " ".join(path.read_text(encoding="utf-8").lower().split())
            assert forbidden not in text, path
