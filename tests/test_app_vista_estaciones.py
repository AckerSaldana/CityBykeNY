"""Prueba de la página "Estaciones y rutas": abre, tiene sus dos gráficas y el interruptor funciona.

Igual que tests/test_app_smoke.py: las tablas se construyen con data/sample en
una carpeta temporal. Si existen las tablas reales (data/processed/app), la
página también se abre con ellas, sin modificarlas.

    pytest tests/test_app_vista_estaciones.py
"""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REALES = ROOT / "data" / "processed" / "app"
PAGINA = "vistas/estaciones_y_rutas.py"
PROHIBIDO = ["estacion_id", "origen_id", "destino_id", "misma_estacion", "origen_lat", "salidas_dia",
             ".parquet", ".json", "datos leídos", "pendiente"]


@pytest.fixture(scope="module")
def de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_estaciones")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch):
    """app/datos.py leyendo de las tablas de muestra o de las reales."""
    if request.param == "reales":
        if not (REALES / "indicadores.json").exists():
            pytest.skip("no están construidas las tablas reales")
        carpeta = REALES
    else:
        carpeta = request.getfixturevalue("de_muestra")
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    return at.switch_page(PAGINA).run()


def graficas(at: AppTest) -> list:
    return [*at.get("vega_lite_chart"), *at.get("arrow_vega_lite_chart")]  # el nombre cambia con la versión


def revisar(at: AppTest) -> None:
    assert [str(e.value) for e in at.exception] == []
    assert not at.error and not at.main.warning
    assert len(at.title) == 1 and "Estaciones y rutas" in at.title[0].value
    assert len(graficas(at)) == 2
    texto = " ".join(str(getattr(e, campo, "") or "")
                     for clase in ("markdown", "caption", "subheader", "metric", "toggle", "info")
                     for e in getattr(at, clase)
                     for campo in ("label", "value", "body")).lower()
    for nombre in PROHIBIDO:
        assert nombre not in texto, nombre
    assert not re.search(r"\b(nan|none)\b", texto)


def test_abre_con_dos_graficas_y_el_interruptor_vuelve_a_dibujar(datos):
    at = abrir()
    revisar(at)
    assert len(at.toggle) == 1 and at.toggle[0].value is False  # por defecto, sin viajes a la misma estación
    assert datos.fecha(datos.indicadores()["periodo"]["inicio"], con_dia=False) in at.markdown[0].value

    at.toggle[0].set_value(True).run()
    revisar(at)
    at.toggle[0].set_value(False).run()
    revisar(at)


def test_las_cifras_salen_de_las_tablas(datos):
    at = abrir()
    cifras = {m.label: m.value for m in at.metric}
    assert cifras["Viajes analizados"] == datos.num(datos.indicadores()["viajes_analizados"])
    assert cifras["Estaciones"] == datos.num(len(datos.estaciones()))
