"""La página "Mapa de demanda" abre, tiene sus tres selectores y responde a cada uno.

Igual que tests/test_app_smoke.py: AppTest ejecuta la página sin navegador y
app/datos.py se apunta a una carpeta de tablas. Se prueba con las tablas
construidas de data/sample y, si existen, con las reales de data/processed/app.

    pytest tests/test_app_vista_mapa.py
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REALES = ROOT / "data" / "processed" / "app"
PAGINA = "vistas/mapa_de_demanda.py"


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_mapa")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch):
    """El módulo app/datos.py leyendo de la carpeta que toca."""
    if request.param == "muestra":
        carpeta = request.getfixturevalue("tablas_de_muestra")
    else:
        if not (REALES / "indicadores.json").exists():
            pytest.skip("no hay tablas reales en data/processed/app")
        carpeta = REALES
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    return at.switch_page(PAGINA).run()


def sin_errores(at: AppTest) -> None:
    assert [str(e.value) for e in at.exception] == []
    assert not at.error
    assert not at.main.warning


def puntos_del_mapa(at: AppTest) -> list[dict]:
    """Los puntos que la página mandó al mapa."""
    (mapa,) = at.get("deck_gl_json_chart")
    return json.loads(mapa.proto.json)["layers"][0]["data"]


def test_abre_con_el_mapa_y_los_tres_selectores(datos):
    at = abrir()

    sin_errores(at)
    assert len(at.title) == 1 and "Mapa de demanda" in at.title[0].value
    assert at.selectbox(key="mapa_hora").label == "Hora del día"
    assert at.radio(key="mapa_tipo_dia").label == "Tipo de día"
    assert at.radio(key="mapa_usuario").label == "Tipo de usuario"
    assert at.selectbox(key="mapa_hora").value == 8  # abre en un día laborable a las 08:00
    assert at.radio(key="mapa_tipo_dia").value == "laborable"
    assert any("08:00" in s.value for s in at.subheader)
    assert len(at.selectbox(key="mapa_hora").options) == 25
    assert "Todo el día" in at.selectbox(key="mapa_hora").options
    assert list(at.radio(key="mapa_tipo_dia").options) == list(datos.TIPOS_DIA.values())
    assert list(at.radio(key="mapa_usuario").options) == list(datos.USUARIOS.values())

    puntos = puntos_del_mapa(at)
    assert len(puntos) == len(datos.demanda_mapa(8, "laborable", "todos"))
    assert len(at.get("vega_lite_chart")) == 1  # demanda por hora del día
    notas = " ".join(c.value for c in at.caption)
    assert "demanda atendida" in notas and "no separan las llegadas" in notas

    # Una frase en lugar de dos cifras sueltas, con las cuentas de la selección.
    flujo = datos.demanda_mapa(8, "laborable", "todos")["flujo_neto"]
    se_vacian, se_llenan = int((flujo <= -0.05).sum()), int((flujo >= 0.05).sum())
    frase = next(m.value for m in at.markdown if "a llenarse" in m.value)
    assert f"{datos.num(se_vacian)} estaciones tienden a vaciarse y {datos.num(se_llenan)} a llenarse" in frase
    assert [m.label for m in at.metric] == ["Salidas en esa hora"]


def test_el_mapa_no_se_acerca_con_la_rueda_y_la_vista_es_fija(datos):
    def mapa(at):
        (m,) = at.get("deck_gl_json_chart")
        return json.loads(m.proto.json)

    at = abrir()
    deck = mapa(at)
    (vista,) = deck["views"]
    assert vista["@@type"] == "MapView" and vista["controller"]["scrollZoom"] is False
    # La vista sale de todas las estaciones: no cambia con la hora ni con el tipo de día.
    est = datos.estaciones()
    inicial = deck["initialViewState"]
    assert inicial["latitude"] == pytest.approx(est["lat"].median())
    assert inicial["longitude"] == pytest.approx(est["lon"].median())
    at = at.selectbox(key="mapa_hora").set_value(3).run()
    at = at.radio(key="mapa_tipo_dia").set_value("fin_de_semana").run()
    assert mapa(at)["initialViewState"] == inicial


def test_cada_selector_vuelve_a_dibujar_sin_errores(datos):
    at = abrir()

    for hora in (8, 17, 3, None):
        at = at.selectbox(key="mapa_hora").set_value(hora).run()
        sin_errores(at)
    for tipo_dia in ("fin_de_semana", "laborable"):
        at = at.radio(key="mapa_tipo_dia").set_value(tipo_dia).run()
        sin_errores(at)
    for usuario in ("miembros", "ocasionales", "todos"):
        at = at.radio(key="mapa_usuario").set_value(usuario).run()
        sin_errores(at)
    assert len(at.get("deck_gl_json_chart")) == 1


def test_el_tipo_de_usuario_solo_cambia_las_salidas(datos):
    at = abrir()
    todos = {(p["lon"], p["lat"], p["nombre"]): p for p in puntos_del_mapa(at)}
    at = at.radio(key="mapa_usuario").set_value("miembros").run()
    miembros = {(p["lon"], p["lat"], p["nombre"]): p for p in puntos_del_mapa(at)}

    assert miembros.keys() == todos.keys()
    for clave, punto in miembros.items():
        for campo in ("llegadas", "flujo", "estado", "color"):
            assert punto[campo] == todos[clave][campo]
    assert any(miembros[c]["salidas"] != todos[c]["salidas"] for c in todos)


def test_el_globo_no_muestra_valores_vacios_ni_claves_internas(datos):
    at = abrir()
    colores = {tuple(p["color"][:3]) for p in puntos_del_mapa(at)}
    esperados = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
                 for c in (datos.COLOR_SE_VACIA, datos.COLOR_SE_LLENA)}
    assert esperados <= colores

    etiquetas_de_tipo = set(datos.TIPOS_ESTACION.values())
    for punto in puntos_del_mapa(at):
        for campo in ("nombre", "salidas", "llegadas", "flujo", "estado", "capacidad", "tipo"):
            valor = punto[campo]
            assert isinstance(valor, str) and valor.strip()
            assert valor.lower() not in ("nan", "none", "sin_dato")
        assert punto["tipo"] in etiquetas_de_tipo
