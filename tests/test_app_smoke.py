"""Prueba de humo del dashboard: abre sin errores, avisa de los datos sintéticos
y dice qué tablas son resultados de una simulación.

Usa AppTest de Streamlit, que ejecuta las páginas sin navegador. Comprueba que
el código corre y qué textos produce. No comprueba cómo se ve en pantalla.

Ninguna prueba escribe en data/processed: las que necesitan archivos "reales"
apuntan las carpetas de app/datos.py a carpetas temporales.

    pytest tests/test_app_smoke.py
"""

import ast
import json
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
RANKING = PAGES[2]  # la vista 2: la única vista que usa la tabla routes
FORECAST = PAGES[3]  # la vista 3: la única vista que usa la tabla predictions
# Todas las vistas que hay hoy en app/pages, leídas del disco. Una vista nueva
# entra sola a la prueba del aviso, sin tocar este archivo.
ALL_PAGES = [None] + sorted(f"pages/{path.name}" for path in (APP / "pages").glob("*.py"))
# Las seis tablas, en el mismo orden que TABLES de app/datos.py.
TABLES = ["stations", "station_hour", "predictions", "recommendations", "policy_curve", "routes"]
WARNING = "DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"
# Nota de una vista cuyas tablas son todas reales mientras otras siguen sintéticas.
REAL_ONLY = "Esta vista usa solo tablas reales"
MAKE_SAMPLE = "python scripts/make_sample_data.py"
# Las dos tablas que salen de la simulación de inventario, y su nota.
SIM_TABLES = ["recommendations", "policy_curve"]
SIM_NOTE = "RESULTADOS DE UNA SIMULACIÓN: no son mediciones"


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


@pytest.fixture()
def loaded(datos, monkeypatch):
    """Lista de las tablas que una página lee DE VERDAD, no las que declara.

    Se llena sola: cada lectura de app/datos.py pasa por datos._load. Hay que
    pasarla a open_page(), que la deja solo con las tablas de la página pedida.
    """
    tables: list[str] = []
    original = datos._load

    def spy(table: str) -> pd.DataFrame:
        tables.append(table)
        return original(table)

    monkeypatch.setattr(datos, "_load", spy)
    return tables


def open_page(page: str | None, loaded: list[str] | None = None) -> AppTest:
    """Abre la app como `streamlit run app/Inicio.py` y, si se pide, cambia de vista.

    `loaded` es la lista del fixture del mismo nombre. Al terminar tiene solo
    las tablas que leyó `page`.
    """
    if loaded is not None:
        loaded.clear()
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    if page is not None:
        if loaded is not None:
            loaded.clear()  # lo que leyó Inicio no cuenta para la vista
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


def copy_sample(folder: Path, *, skip: str | None = None) -> None:
    """Copia a `folder` las tablas de la muestra, menos `skip` si se indica."""
    for table in TABLES:
        if table != skip:
            shutil.copy(SAMPLE / f"{table}.parquet", folder / f"{table}.parquet")


def write_report(folder: Path, rules: str = "provisional", forecast: str = "baseline") -> None:
    """Escribe en `folder` un reporte de simulación con solo los dos datos que lee la app."""
    (folder / "reports").mkdir(exist_ok=True)
    report = {"reglas": {"estado": rules}, "pronostico": {"modelo_que_alimenta_model": forecast}}
    (folder / "reports" / "simulacion.json").write_text(json.dumps(report), encoding="utf-8")


def sim_notes_of(at: AppTest) -> list[str]:
    """Notas de simulación que hay en el cuerpo de la página (lista vacía = ninguna)."""
    return [element.value for element in at.main.info if SIM_NOTE in element.value]


def data_check_of(at: AppTest) -> dict[str, str]:
    """Sección "Datos leídos": nombre de la tabla -> su renglón, con filas y origen."""
    text = next(element.value for element in at.markdown if "filas leídas" in element.value)
    return {line.split("`")[1]: line for line in text.splitlines()}


def visible_text_of(at: AppTest) -> str:
    """Todo el texto corrido de la página, en minúsculas y en un solo renglón."""
    elements = [*at.markdown, *at.caption, *at.info]
    return " ".join(" ".join(element.value.lower().split()) for element in elements)


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

    # Sin datos reales, las seis tablas salen de data/sample y el aviso las nombra.
    assert datos.synthetic_tables() == TABLES
    text = at.main.warning[0].value
    assert WARNING in text
    for table in TABLES:
        assert f"`{table}`" in text
    # El aviso del menú de la izquierda también las nombra.
    side = at.sidebar.warning[0].value
    assert WARNING in side
    for table in TABLES:
        assert f"`{table}`" in side
    # Nadie dice "solo tablas reales" cuando todo sale de la muestra.
    assert not any(REAL_ONLY in element.value for element in at.info)


def test_abre_con_lo_que_hay_hoy_en_el_repositorio(datos, loaded):
    """Sin carpetas temporales: data/processed y data/sample tal como están."""
    synthetic = set(datos.synthetic_tables())
    for page in PAGES:
        at = open_page(page, loaded)
        assert errors_of(at) == [], page
        assert not at.error, page
        reads_synthetic = bool(synthetic & set(loaded))
        # El aviso completo se ve exactamente cuando la página lee alguna tabla
        # de la muestra.
        assert bool(at.main.warning) == reads_synthetic, page
        assert bool(at.sidebar.warning) == reads_synthetic, page
        # Si la página lee solo tablas reales y quedan sintéticas en otras
        # vistas, lo dice con la nota corta. Si todo es real, no dice nada.
        note = any(REAL_ONLY in element.value for element in at.main.info)
        assert note == (bool(synthetic) and not reads_synthetic), page
        # La nota de simulación se ve exactamente cuando la página lee una
        # tabla simulada que ya es real.
        reads_simulated = bool((set(SIM_TABLES) & set(loaded)) - synthetic)
        assert bool(sim_notes_of(at)) == reads_simulated, page
        assert any(SIM_NOTE in element.value for element in at.sidebar.info) == reads_simulated, page


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


@pytest.mark.parametrize("only", TABLES)
@pytest.mark.parametrize("page", ALL_PAGES, ids=lambda page: Path(page).stem if page else "inicio")
def test_la_vista_que_lee_una_tabla_sintetica_muestra_el_aviso_completo(datos, processed, loaded, page, only):
    """Regla para cualquier vista, también las nuevas, cuando solo UNA tabla sigue en la muestra.

    Si la vista lee esa tabla, muestra el aviso completo en el cuerpo y en el
    menú. No vale declarar otra lista: cuenta lo que la vista lee de verdad.
    """
    # Copias de la muestra en la carpeta temporal: simulan que llegó todo menos `only`.
    copy_sample(processed, skip=only)
    at = open_page(page, loaded)
    name = page or "Inicio.py"

    assert errors_of(at) == [], name
    assert not at.error, name
    assert datos.synthetic_tables() == [only]
    warns = any(WARNING in element.value and f"`{only}`" in element.value for element in at.main.warning)
    notes = any(REAL_ONLY in element.value and f"`{only}`" in element.value for element in at.main.info)
    if only in loaded:
        assert warns, (
            f"{name} lee la tabla sintética `{only}` y no muestra el aviso completo. "
            f"Agrega \"{only}\" a la lista de datos.show_synthetic_warning([...]) de esa vista."
        )
        assert any(
            WARNING in element.value and f"`{only}`" in element.value for element in at.sidebar.warning
        ), f"{name}: falta el aviso en el menú de la izquierda"
        assert not any(REAL_ONLY in element.value for element in at.info), (
            f"{name} dice que usa solo tablas reales, pero lee la tabla sintética `{only}`"
        )
    else:
        # La vista no lee la tabla sintética: aviso completo o nota corta, nunca nada.
        assert warns or notes, f"{name} no llama a datos.show_synthetic_warning()"


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_sin_las_tablas_simuladas_solo_la_vista_del_mapa_dice_que_usa_tablas_reales(datos, processed, loaded, page):
    """Reales stations, station_hour, predictions y routes; de la muestra las dos tablas simuladas.

    Es lo que hay en una máquina donde ya se corrió la limpieza y la línea
    base, pero todavía no la simulación (python scripts/run_simulation.py).
    """
    pending = ["recommendations", "policy_curve"]
    for table in TABLES:
        if table not in pending:
            # Copia de la muestra en la carpeta temporal: solo para simular que llegó el archivo.
            shutil.copy(SAMPLE / f"{table}.parquet", processed / f"{table}.parquet")

    at = open_page(page, loaded)

    assert errors_of(at) == []
    assert not at.error
    assert datos.synthetic_tables() == pending
    if page == "pages/1_Mapa_de_demanda.py":
        # Sus dos tablas son reales: nota corta en el cuerpo y en el menú, sin aviso.
        assert sorted(set(loaded)) == ["station_hour", "stations"]
        assert not at.warning
        for note in (at.main.info[0].value, at.sidebar.info[0].value):
            assert REAL_ONLY in note
            assert WARNING not in note
            for table in pending:
                assert f"`{table}`" in note
    else:
        # Inicio y las vistas 2 y 3 leen al menos una tabla sintética: aviso completo.
        assert set(loaded) & set(pending)
        for text in (at.main.warning[0].value, at.sidebar.warning[0].value):
            assert WARNING in text
            for table in pending:
                assert f"`{table}`" in text
        assert "mezcladas" in at.main.warning[0].value
        assert not any(REAL_ONLY in element.value for element in at.info)
    # Las tablas simuladas vienen de la muestra: manda el aviso de datos
    # sintéticos y la nota de simulación no sale en ninguna vista.
    assert sim_notes_of(at) == []
    assert not any(SIM_NOTE in element.value for element in at.sidebar.info)


@pytest.mark.parametrize(
    ("used", "shows_warning"),
    [
        ('["stations", "station_hour"]', False),  # todas reales: nota corta
        ('["stations", "recommendations"]', True),  # una es sintética
        ("", True),  # sin lista: se supone que lee las seis tablas
        ("[]", True),  # lista vacía: igual que sin lista
        ('["estaciones"]', True),  # nombre mal escrito: ante la duda, aviso completo
    ],
    ids=["todas_reales", "una_sintetica", "sin_lista", "lista_vacia", "nombre_mal_escrito"],
)
def test_ante_la_duda_el_aviso_completo(datos, processed, used, shows_warning):
    """show_synthetic_warning(lista): solo calla el aviso si TODAS las tablas de la lista son reales."""
    copy_sample(processed, skip="recommendations")  # la única sintética: recommendations

    at = AppTest.from_string(f"import datos\ndatos.show_synthetic_warning({used})\n", default_timeout=30).run()

    assert errors_of(at) == []
    assert bool(at.main.warning) == shows_warning
    assert bool(at.sidebar.warning) == shows_warning
    assert any(REAL_ONLY in element.value for element in at.main.info) != shows_warning
    assert any(REAL_ONLY in element.value for element in at.sidebar.info) != shows_warning


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


def test_la_vista_de_pronostico_dice_que_lo_observado_solo_se_compara_en_validacion(datos, processed):
    """Decisión 4: la prueba final se usa una sola vez. La vista 3 lo deja escrito en pantalla."""
    at = open_page(FORECAST)
    text = visible_text_of(at)

    assert "decisión 4" in text
    assert "solo en los días de validación" in text
    assert "sin lo observado al lado y sin calcular su error" in text
    assert "la simulación de inventario tampoco se corre sobre la prueba final" in text
    # La prueba de lectura cuenta aparte las filas de validación de la estación elegida.
    predictions = datos.load_predictions()
    station = predictions["station_id"] == at.selectbox[0].value
    rows = int((station & (predictions["split"] == "val")).sum())
    assert 0 < rows < int(station.sum())  # la estación tiene filas de validación y de prueba final
    assert f"({rows} de validación)" in text


# --------------------------------------------------------------------------- #
# La tabla de rutas (viajes entre pares de estaciones)
# --------------------------------------------------------------------------- #

def test_el_modulo_de_datos_conoce_las_seis_tablas(datos):
    assert list(datos.TABLES) == TABLES
    for table in TABLES:
        assert callable(getattr(datos, f"load_{table}")), f"falta datos.load_{table}()"


def test_inicio_lee_las_seis_tablas_y_la_vista_de_ranking_lee_las_rutas(datos, processed):
    rows = len(datos.load_routes())

    assert list(data_check_of(open_page(None))) == TABLES

    check = data_check_of(open_page(RANKING))
    assert "routes" in check, "la vista 2 no lee la tabla routes"
    assert f"{rows} filas leídas" in check["routes"]


def test_las_rutas_se_pueden_cruzar_con_las_estaciones(datos, processed):
    """Lo que necesita la vista de ranking para mostrar nombres en lugar de identificadores."""
    routes = datos.load_routes()
    stations = set(datos.load_stations()["station_id"])

    assert list(routes.columns) == ["start_station_id", "end_station_id", "trips"]
    assert set(routes["start_station_id"]) <= stations
    assert set(routes["end_station_id"]) <= stations
    assert (routes["trips"] >= 1).all()
    # Los viajes que empiezan y terminan en la misma estación llegan a la vista.
    assert (routes["start_station_id"] == routes["end_station_id"]).any()


def test_la_vista_de_ranking_aclara_que_sus_rutas_no_son_de_camiones(datos, processed):
    """routes son viajes de usuarios. Las rutas de rebalanceo quedan fuera por ahora."""
    text = visible_text_of(open_page(RANKING))

    assert "no son rutas de camiones" in text, (
        "La vista 2 debe decir que sus rutas son viajes de usuarios, no rutas de camiones"
    )
    assert "quedan fuera por ahora" in text


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


@pytest.mark.parametrize("table", TABLES)
def test_cada_una_de_las_seis_tablas_prefiere_processed_sobre_sample(datos, processed, monkeypatch, tmp_path, table):
    sample = tmp_path / "sample"
    sample.mkdir()
    monkeypatch.setattr(datos, "DATA_SAMPLE", sample)
    load = getattr(datos, f"load_{table}")  # cada tabla se pide con datos.load_<nombre>()
    full = pd.read_parquet(SAMPLE / f"{table}.parquet")

    # Solo existe la muestra: se usa la muestra y se declara sintética.
    shutil.copy(SAMPLE / f"{table}.parquet", sample / f"{table}.parquet")
    assert datos.find_file(table) == sample / f"{table}.parquet"
    assert len(load()) == len(full)
    assert datos.synthetic_tables() == [table]

    # Llega el archivo "real" (aquí, las dos primeras filas de la muestra): se usa ese.
    full.head(2).to_parquet(processed / f"{table}.parquet")
    assert datos.find_file(table) == processed / f"{table}.parquet"
    assert len(load()) == 2
    assert datos.synthetic_tables() == []


@pytest.mark.parametrize("real", TABLES)
def test_el_aviso_solo_nombra_las_tablas_que_siguen_en_la_muestra(datos, processed, real):
    # Copia de la muestra en la carpeta temporal: solo para simular que llegó un archivo.
    shutil.copy(SAMPLE / f"{real}.parquet", processed / f"{real}.parquet")
    pending = [table for table in TABLES if table != real]

    at = open_page(None)

    assert errors_of(at) == []
    assert datos.synthetic_tables() == pending
    text = at.main.warning[0].value
    assert WARNING in text
    assert f"`{real}`" not in text
    for table in pending:
        assert f"`{table}`" in text


def test_si_solo_routes_sigue_en_la_muestra_el_aviso_nombra_solo_esa_tabla(datos, processed):
    # Copias de la muestra en la carpeta temporal: simulan que llegó todo menos las rutas.
    copy_sample(processed, skip="routes")

    at = open_page(RANKING)

    assert errors_of(at) == []
    assert not at.error
    assert datos.synthetic_tables() == ["routes"]
    text = at.main.warning[0].value
    assert WARNING in text
    assert "`routes`" in text
    for table in TABLES[:-1]:
        assert f"`{table}`" not in text
    assert "mezcladas" in text  # hay tablas reales y una sintética
    # La prueba de lectura dice de dónde salió cada tabla.
    check = data_check_of(at)
    assert "SINTÉTICA, de `data/sample`" in check["routes"]
    assert "real, de `data/processed`" in check["stations"]


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
    copy_sample(processed)

    at = open_page(page)

    assert errors_of(at) == []
    assert not at.error
    assert datos.synthetic_tables() == []
    assert not at.warning


# --------------------------------------------------------------------------- #
# Tablas que salen de la simulación: reales, pero no son mediciones
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (None, ["recommendations", "policy_curve"]),
        ("pages/1_Mapa_de_demanda.py", []),
        ("pages/2_Ranking_de_estaciones_y_rutas.py", ["recommendations", "policy_curve"]),
        ("pages/3_Pronostico_por_estacion.py", ["recommendations"]),
    ],
    ids=PAGE_IDS,
)
def test_con_las_seis_tablas_reales_la_nota_de_simulacion_sale_donde_se_leen_tablas_simuladas(
    datos, processed, page, expected
):
    """Sin tablas sintéticas ya no hay aviso. Las tablas simuladas siguen sin ser mediciones: se dice."""
    # Copias de la muestra en la carpeta temporal: solo para simular datos reales.
    copy_sample(processed)
    write_report(processed, rules="provisional", forecast="baseline")

    at = open_page(page)

    assert errors_of(at) == []
    assert not at.error
    assert not at.warning  # el aviso de datos sintéticos no sale: nada viene de la muestra
    notes = sim_notes_of(at)
    side = [element.value for element in at.sidebar.info if SIM_NOTE in element.value]
    if not expected:
        assert notes == [] and side == []  # la vista no lee tablas simuladas
        return
    assert len(notes) == 1 and len(side) == 1
    for text in (notes[0], side[0]):
        for table in SIM_TABLES:
            assert (f"`{table}`" in text) == (table in expected), table
    text = " ".join(notes[0].lower().split())
    assert "días de validación" in text
    assert "la prueba final no se simula (decisión 4)" in text
    assert "son **provisionales**" in text  # reglas de operación, según el reporte
    assert "línea base" in text  # lo que alimentó a la política "model", según el reporte
    # Lenguaje honesto (decisión 9).
    assert "*riesgo simulado*" in text
    assert "*riesgo simulado evitado*" in text
    assert "viajes bloqueados de la simulación son simulados: no son demanda insatisfecha medida" in text


@pytest.mark.parametrize("page", ALL_PAGES, ids=lambda page: Path(page).stem if page else "inicio")
def test_toda_vista_que_lee_una_tabla_simulada_real_muestra_la_nota(datos, processed, loaded, page):
    """Regla para cualquier vista, también las nuevas. Cuenta lo que la vista lee de verdad."""
    copy_sample(processed)
    at = open_page(page, loaded)
    name = page or "Inicio.py"

    assert errors_of(at) == [], name
    read = [table for table in SIM_TABLES if table in loaded]
    notes = sim_notes_of(at)
    if not read:
        assert notes == [], f"{name} muestra la nota de simulación sin leer ninguna tabla simulada"
        return
    assert len(notes) == 1, (
        f"{name} lee tablas simuladas ({', '.join(read)}) y no muestra la nota. Agrega "
        "datos.show_simulation_note(tables_used) después de datos.show_synthetic_warning(tables_used)."
    )
    for table in read:
        assert f"`{table}`" in notes[0], (
            f"{name} lee la tabla simulada `{table}` y la nota no la nombra. "
            f"Agrega \"{table}\" a la lista de tablas de esa vista."
        )
    assert any(SIM_NOTE in element.value for element in at.sidebar.info), (
        f"{name}: falta la nota en el menú de la izquierda"
    )


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_con_las_tablas_simuladas_en_la_muestra_manda_el_aviso_de_datos_sinteticos(datos, processed, page):
    """Todo sale de data/sample: aviso de datos sintéticos y ninguna nota de simulación."""
    at = open_page(page)

    assert errors_of(at) == []
    assert WARNING in at.main.warning[0].value
    assert sim_notes_of(at) == []
    assert not any(SIM_NOTE in element.value for element in at.sidebar.info)


@pytest.mark.parametrize("only", SIM_TABLES)
def test_la_nota_de_simulacion_no_nombra_la_tabla_que_sigue_en_la_muestra(datos, processed, only):
    """Una tabla simulada real y la otra sintética: la nota es de la real y el aviso de la sintética."""
    copy_sample(processed, skip=only)  # la única sintética: `only`
    other = next(table for table in SIM_TABLES if table != only)

    # La vista 2 lee las dos: aviso por la sintética, nota por la real.
    at = open_page(RANKING)
    assert errors_of(at) == []
    assert f"`{only}`" in at.main.warning[0].value
    notes = sim_notes_of(at)
    assert len(notes) == 1
    assert f"`{other}`" in notes[0]
    assert f"`{only}`" not in notes[0]

    # La vista 3 solo lee recommendations.
    at = open_page(FORECAST)
    assert errors_of(at) == []
    if only == "recommendations":
        # Es sintética: aviso completo y ninguna nota de simulación.
        assert f"`{only}`" in at.main.warning[0].value
        assert sim_notes_of(at) == []
        assert not any(SIM_NOTE in element.value for element in at.sidebar.info)
    else:
        # Es real: nota de simulación, sin aviso. Además dice que usa solo tablas reales.
        assert not at.warning
        assert len(sim_notes_of(at)) == 1
        assert "`recommendations`" in sim_notes_of(at)[0]
        assert any(REAL_ONLY in element.value for element in at.main.info)


@pytest.mark.parametrize(
    ("used", "named"),
    [
        ('["stations", "station_hour"]', []),  # no lee tablas simuladas: sin nota
        ('["stations", "recommendations"]', ["recommendations"]),
        ('["policy_curve"]', ["policy_curve"]),
        ("", SIM_TABLES),  # sin lista: se supone que lee las seis tablas
        ("[]", SIM_TABLES),  # lista vacía: igual que sin lista
        ('["recomendaciones"]', SIM_TABLES),  # nombre mal escrito: ante la duda, la nota completa
    ],
    ids=["sin_simuladas", "recommendations", "policy_curve", "sin_lista", "lista_vacia", "nombre_mal_escrito"],
)
def test_ante_la_duda_la_nota_de_simulacion(datos, processed, used, named):
    """show_simulation_note(lista): nombra las tablas simuladas reales de la lista."""
    copy_sample(processed)

    at = AppTest.from_string(f"import datos\ndatos.show_simulation_note({used})\n", default_timeout=30).run()

    assert errors_of(at) == []
    assert not at.warning
    notes = sim_notes_of(at)
    side = [element.value for element in at.sidebar.info if SIM_NOTE in element.value]
    assert len(notes) == len(side) == (1 if named else 0)
    for text in notes + side:
        for table in SIM_TABLES:
            assert (f"`{table}`" in text) == (table in named), table


@pytest.mark.parametrize(
    ("report", "present", "absent"),
    [
        (("provisional", "baseline"), ["son **provisionales**", "línea base", "no es un modelo entrenado"], ["LightGBM"]),
        (("confirmado", "lgbm"), ["ya estaban confirmadas", "LightGBM"], ["provisionales", "línea base"]),
        (("provisional", "ninguno"), ["son **provisionales**", "**no se simuló**"], ["línea base", "se alimentó"]),
        (("otro_estado", "otro_modelo"), ["`otro_estado`", "`otro_modelo`"], ["son **provisionales**", "línea base"]),
        (None, ["No se pudo leer el reporte", "no se sabe"], ["son **provisionales**", "línea base", "LightGBM"]),
        ("roto", ["No se pudo leer el reporte", "no se sabe"], ["son **provisionales**", "línea base", "LightGBM"]),
    ],
    ids=["provisional_y_linea_base", "confirmado_y_lgbm", "model_no_se_simulo", "valores_desconocidos", "sin_reporte", "reporte_roto"],
)
def test_la_nota_dice_lo_que_trae_el_reporte_de_la_corrida(datos, processed, report, present, absent):
    """El estado de las reglas y el pronóstico de "model" no vienen en las tablas: salen del reporte.

    Si alguien confirma las reglas o llega LightGBM y vuelve a simular, la
    nota cambia sola. Si no hay reporte, dice que no se sabe: no lo supone.
    """
    copy_sample(processed)
    if report == "roto":
        (processed / "reports").mkdir()
        (processed / "reports" / "simulacion.json").write_text("{ esto no es JSON", encoding="utf-8")
    elif report is not None:
        write_report(processed, rules=report[0], forecast=report[1])

    at = AppTest.from_string("import datos\ndatos.show_simulation_note()\n", default_timeout=30).run()

    assert errors_of(at) == []
    text = sim_notes_of(at)[0]
    for phrase in present:
        assert phrase in text, phrase
    for phrase in absent:
        assert phrase not in text, phrase
    # Lo que no depende del reporte está siempre.
    assert "*riesgo simulado evitado*" in text
    assert "no son demanda insatisfecha medida" in text


def test_la_nota_cuenta_los_dias_de_la_tabla_no_los_trae_escritos(datos, processed):
    """Si alguien vuelve a simular con otros días, la nota cambia sola."""
    copy_sample(processed)
    full = pd.read_parquet(SAMPLE / "recommendations.parquet")
    days = full["window_start"].dt.normalize()
    script = "import datos\ndatos.show_simulation_note()\n"

    text = sim_notes_of(AppTest.from_string(script, default_timeout=30).run())[0]
    assert days.nunique() > 2
    assert f"`recommendations` cubre {days.nunique()} días: {datos.periodo_en_texto(days.min(), days.max())}." in text

    # Otra corrida "real": solo los dos primeros días.
    first_two = sorted(days.unique())[:2]
    full[days.isin(first_two)].to_parquet(processed / "recommendations.parquet")
    text = sim_notes_of(AppTest.from_string(script, default_timeout=30).run())[0]
    assert f"`recommendations` cubre 2 días: {datos.periodo_en_texto(first_two[0], first_two[1])}." in text


def test_las_fechas_se_escriben_en_espanol(datos):
    first, last = pd.Timestamp("2026-09-02", tz="America/New_York"), pd.Timestamp("2026-09-14", tz="America/New_York")

    assert datos.fecha_en_texto(first) == "2 de septiembre de 2026"
    assert datos.periodo_en_texto(first, last) == "del 2 al 14 de septiembre de 2026"
    # Si el periodo cruza de mes, cada fecha lleva su mes.
    other = pd.Timestamp("2026-10-01", tz="America/New_York")
    assert datos.periodo_en_texto(last, other) == "del 14 de septiembre de 2026 al 1 de octubre de 2026"
    assert datos.periodo_en_texto(first, first) == "el 2 de septiembre de 2026"  # un solo día


def test_la_prueba_de_lectura_cuenta_lo_que_traen_las_tablas_simuladas(datos, processed):
    """Estaciones, días, franjas, políticas y presupuestos salen de la tabla, no del código."""
    recommendations = datos.load_recommendations()
    policy_curve = datos.load_policy_curve()
    days = recommendations["window_start"].dt.normalize()

    check = data_check_of(open_page(RANKING))

    line = check["recommendations"]
    assert "Una fila por estación, día, franja horaria y política" in line
    assert f"{recommendations['station_id'].nunique()} estaciones" in line
    assert f"{days.nunique()} días ({datos.periodo_en_texto(days.min(), days.max())})" in line
    assert f"{recommendations['window_start'].dt.hour.nunique()} franjas por día" in line
    assert "3 políticas" in line
    line = check["policy_curve"]
    assert "Una fila por política y presupuesto diario" in line
    budgets = ", ".join(str(budget) for budget in sorted(policy_curve["budget"].unique()))
    assert f"{policy_curve['budget'].nunique()} presupuestos ({budgets} bicis al día" in line
    # Las tablas que no son simuladas siguen con su renglón de siempre.
    assert check["stations"].endswith("(SINTÉTICA, de `data/sample`).")

    # Con otra tabla, la frase cambia: no hay cifras escritas a mano.
    fewer = recommendations[recommendations["policy"] != "model"]
    assert "2 políticas" in datos.describe_simulated("recommendations", fewer)
    assert "1 presupuestos (0 bicis al día" in datos.describe_simulated(
        "policy_curve", policy_curve[policy_curve["budget"] == 0]
    )
    # Una tabla sin filas no rompe la página, y las otras tablas no llevan frase.
    assert datos.describe_simulated("recommendations", recommendations.head(0)) == ""
    assert datos.describe_simulated("stations", datos.load_stations()) == ""


def test_las_paginas_abren_aunque_las_recomendaciones_reales_no_traigan_filas(datos, processed):
    copy_sample(processed)
    pd.read_parquet(SAMPLE / "recommendations.parquet").head(0).to_parquet(processed / "recommendations.parquet")

    for page in (None, RANKING, FORECAST):
        at = open_page(page)
        assert errors_of(at) == [], page
        assert not at.error, page
        assert len(sim_notes_of(at)) == 1, page


def test_la_vista_de_pronostico_dice_cuando_una_estacion_no_se_simulo(datos, processed):
    """Con tablas reales, "0 de recomendaciones" quiere decir que la estación quedó fuera de la simulación."""
    copy_sample(processed)
    stations = pd.read_parquet(SAMPLE / "stations.parquet")
    full = pd.read_parquet(SAMPLE / "recommendations.parquet")
    # Dos estaciones quedan fuera de la "simulación": una sin capacidad y una con capacidad.
    without = stations.loc[stations["capacity"].isna(), "station_id"].iloc[0]
    with_capacity = stations.loc[stations["capacity"].notna(), "station_id"].iloc[0]
    out = [without, with_capacity]
    full[~full["station_id"].isin(out)].to_parquet(processed / "recommendations.parquet")
    simulated = next(station for station in stations["station_id"] if station not in out)
    not_simulated = "no se simuló"

    at = open_page(FORECAST)

    # Una estación simulada: se cuentan sus días y políticas, y no sale el mensaje.
    at.selectbox[0].select(simulated).run()
    assert errors_of(at) == []
    rows = full[full["station_id"] == simulated]
    days = rows["window_start"].dt.normalize().nunique()
    text = visible_text_of(at)
    assert f"cubren {days} días y 3 políticas" in text
    assert "hay que elegir un día y una política" in text
    assert not_simulated not in text

    # Una estación que quedó fuera: se dice, con cuántas están así.
    for station in out:
        at.selectbox[0].select(station).run()
        assert errors_of(at) == []
        text = visible_text_of(at)
        assert "0 de recomendaciones" in text
        assert "esta estación no se simuló: por eso no tiene recomendaciones" in text
        assert "las estaciones sin capacidad registrada" in text
        assert "las que no tuvieron viajes en los días simulados" in text
        # Las cuentas salen de las tablas: 2 estaciones fuera, 1 de ellas sin capacidad.
        assert f"no se simularon 2 de las {len(stations)} estaciones (1 sin capacidad registrada)" in text


def test_con_tablas_mezcladas_la_vista_de_pronostico_no_dice_que_la_estacion_no_se_simulo(datos, processed):
    """Con estaciones de la muestra y recomendaciones "reales", el cero viene de la mezcla: lo explica el aviso."""
    full = pd.read_parquet(SAMPLE / "recommendations.parquet")
    station = datos.load_stations()["station_id"].iloc[0]
    full[full["station_id"] != station].to_parquet(processed / "recommendations.parquet")

    at = open_page(FORECAST)
    at.selectbox[0].select(station).run()

    assert errors_of(at) == []
    assert datos.synthetic_tables() == [table for table in TABLES if table != "recommendations"]
    text = visible_text_of(at)
    assert "0 de recomendaciones" in text
    assert "no se simuló" not in text
    assert "mezcladas" in at.main.warning[0].value


def test_el_reporte_real_de_la_simulacion_trae_lo_que_lee_la_app(datos):
    """Si la simulación cambia el nombre de esos dos datos en su reporte, la nota diría "no se sabe"."""
    report = ROOT / "data" / "processed" / "reports" / "simulacion.json"
    if not report.is_file():
        pytest.skip("no hay reporte real de la simulación en data/processed/reports")

    run = datos.simulation_run()

    assert run["rules"] in ("provisional", "confirmado"), run
    # "ninguno": la política "model" no se simuló porque no había filas "lgbm" en predictions.
    assert run["forecast"] in ("lgbm", "ninguno"), run
    assert run["forecast"] in datos.FORECAST_TEXT, run


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
    for table in TABLES:
        assert f"`{table}.parquet`" in at.error[0].value
    # La página se detiene ahí: no llega a la lista de pendientes.
    assert "Pendiente" not in [element.value for element in at.subheader]


def test_si_falta_una_sola_tabla_solo_fallan_las_vistas_que_la_usan(datos, processed, monkeypatch, tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    monkeypatch.setattr(datos, "DATA_SAMPLE", sample)
    copy_sample(sample, skip="predictions")
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


def test_si_falta_solo_routes_el_mensaje_dice_como_generarla(datos, processed, monkeypatch, tmp_path):
    """Una muestra generada antes de que existiera routes.parquet: hay que volver a generarla."""
    sample = tmp_path / "sample"
    sample.mkdir()
    monkeypatch.setattr(datos, "DATA_SAMPLE", sample)
    copy_sample(sample, skip="routes")
    assert datos.missing_tables() == ["routes"]

    # Las vistas 1 y 3 no usan las rutas: abren normal.
    for page in ("pages/1_Mapa_de_demanda.py", "pages/3_Pronostico_por_estacion.py"):
        at = open_page(page)
        assert errors_of(at) == [], page
        assert not at.error, page

    # Inicio y la vista 2 sí las usan: muestran el mensaje y dicen qué archivo falta.
    for page in (None, RANKING):
        at = open_page(page)
        assert errors_of(at) == [], page
        assert len(at.error) == 1, page
        assert "`routes.parquet`" in at.error[0].value, page
        assert "`stations.parquet`" not in at.error[0].value, page
        assert MAKE_SAMPLE in at.error[0].value, page


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


def test_los_textos_de_la_app_ya_no_describen_las_tablas_simuladas_como_antes():
    """Frases que eran ciertas antes de que existiera la simulación y hoy confunden."""
    stale = [
        "por estación, franja horaria y política",  # falta el día: hay una fila por día
        "conviene que la estación empiece la franja",  # start_inventory_target no es un inventario ideal
        "sintéticas todavía",  # recommendations y policy_curve ya salen de la simulación
        "hasta que exista el simulador",
        "cuando existan datos reales",
        "qué convenía hacer",  # bikes_to_move es lo que la política movió, no lo que convenía
    ]
    for path in sorted(APP.rglob("*")):
        if path.suffix in (".py", ".md"):
            text = " ".join(path.read_text(encoding="utf-8").lower().split())
            for phrase in stale:
                assert phrase not in text, f"{path.name}: {phrase}"
    # El documento de la carpeta dice qué son las dos columnas que más se malinterpretan.
    leeme = " ".join((APP / "LEEME.md").read_text(encoding="utf-8").split())
    assert "Una fila por estación, día, franja horaria y política" in leeme
    assert "No es el inventario ideal" in leeme
    assert "el presupuesto no alcanzó" in leeme
    assert "no se simul" in leeme


def test_la_app_no_habla_de_viajes_que_los_datos_no_registran():
    """Decisión 9: los datos solo registran demanda atendida. La frase prohibida no aparece en app/."""
    forbidden = " ".join(["viajes", "recuperados"])  # armada aquí para no escribirla
    for path in sorted(APP.rglob("*")):
        if path.suffix in (".py", ".md"):
            text = " ".join(path.read_text(encoding="utf-8").lower().split())
            assert forbidden not in text, path
