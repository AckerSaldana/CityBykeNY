"""La página Inicio: cuatro indicadores, la curva de riesgo contra presupuesto y tres hallazgos.

Se abre igual que en tests/test_app_smoke.py, con las tablas hechas de
data/sample en una carpeta temporal y, si existen, con las reales de
data/processed/app (solo se leen).

    pytest tests/test_app_vista_inicio.py
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


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_inicio")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch, tablas_de_muestra):
    """app/datos.py leyendo de las tablas de muestra o de las reales."""
    if request.param == "reales":
        if not (REALES / "indicadores.json").exists():
            pytest.skip("no hay tablas reales en data/processed/app")
        carpeta = REALES
    else:
        carpeta = tablas_de_muestra
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    return AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()


def test_inicio_muestra_indicadores_curva_y_hallazgos(datos):
    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert not at.error and not at.main.warning
    assert len(at.title) == 1

    # Cuatro indicadores, con las cifras de los datos y sin valores vacíos de Python.
    ind = datos.indicadores()
    assert [m.label for m in at.metric] == [
        "Viajes analizados", "Estaciones", "Riesgo simulado evitado", "Error del pronóstico (WAPE)"]
    valores = [m.value for m in at.metric]
    assert valores[0] == datos.num(ind["viajes_analizados"])
    assert valores[1] == datos.num(ind["estaciones"])
    assert valores[2] == datos.pct(ind["referencia"]["politicas"]["model"]["pct_evitado"])
    assert valores[3] == datos.pct(ind["pronostico"]["lgbm"]["salidas"]["wape"])
    assert datos.pct(ind["pronostico"]["lgbm"]["salidas"]["cobertura"]) in " ".join(c.value for c in at.caption)

    # Una gráfica: la curva con las tres políticas, sus nombres en pantalla y el presupuesto de referencia.
    graficas = at.get("vega_lite_chart")
    assert len(graficas) == 1
    spec = json.loads(graficas[0].proto.spec)
    texto_spec = json.dumps(spec, ensure_ascii=False)
    for nombre in ("Sin intervención", "Regla histórica", "Modelo",
                   "Bicis movidas por día", "Horas de estación en riesgo por día"):
        assert nombre in texto_spec
    assert f"Presupuesto de referencia: {datos.num(ind['referencia']['presupuesto'])} bicis" in texto_spec

    # Tres hallazgos, los mismos que arma datos.hallazgos().
    frases = datos.hallazgos()
    textos = [m.value for m in at.markdown]
    assert len(frases) == 3
    for frase in frases:
        assert sum(frase in t for t in textos) == 1


def test_inicio_no_se_rompe_si_falta_la_simulacion(datos, monkeypatch):
    """Sin resultados de simulación, la página dice "Sin dato" en lugar de fallar."""
    ind = datos.indicadores()
    vacio = {**ind, "referencia": {"presupuesto": None, "politicas": {}}, "pronostico": {},
             "sensibilidad": {"filas": [], "rango": {}},
             "comparacion": {"modelo_pronostica_mejor": None, "modelo_evita": None},
             "presupuesto_umbral": {"politicas": {}}}
    monkeypatch.setattr(datos, "indicadores", lambda: vacio)
    monkeypatch.setattr(datos, "curva", lambda: datos._leer("curva").iloc[0:0])

    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert len(at.metric) == 4
    assert [m.value for m in at.metric][2:] == ["Sin dato", "Sin dato"]
    assert len(at.get("vega_lite_chart")) == 0
    assert any("Sin dato" in e.value for e in at.info)
