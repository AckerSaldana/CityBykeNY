"""Página "Pronóstico por estación": abre, tiene selectores, gráfica y tarjetas de acción.

Igual que tests/test_app_smoke.py: AppTest de Streamlit, con las tablas pequeñas
construidas con data/sample en una carpeta temporal. Si existen las tablas
reales (data/processed/app), la página también se abre con ellas.

    pytest tests/test_app_vista_pronostico.py
"""

import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REALES = ROOT / "data" / "processed" / "app"
PAGINA = "vistas/pronostico_por_estacion.py"
ACCION = re.compile(r"Llevar [\d,]+ bicis? antes de las \d\d:00|Retirar [\d,]+ bicis? antes de las \d\d:00|Sin traslado")


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_pronostico")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch, tablas_de_muestra):
    """app/datos.py leyendo de la carpeta temporal (muestra) o de las tablas reales si existen."""
    if request.param == "reales":
        if not (REALES / "indicadores.json").exists():
            pytest.skip("No están construidas las tablas reales (data/processed/app)")
        carpeta = REALES
    else:
        carpeta = tablas_de_muestra
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
    assert len(at.title) == 1


def textos(at: AppTest) -> str:
    return " ".join(str(e.value) for clase in ("markdown", "caption", "info") for e in getattr(at, clase))


def graficas(at: AppTest) -> list:
    return list(at.get("vega_lite_chart")) + list(at.get("arrow_vega_lite_chart"))


def selector(at: AppTest, etiqueta: str):
    return next(s for s in at.selectbox if s.label == etiqueta)


def dia_simulado(datos, estacion_id):
    """Un día con pronóstico y con simulación para la estación (None si no hay)."""
    pron = set(pd.to_datetime(datos.pronostico_estacion(estacion_id)["fecha"]))
    rec = set(pd.to_datetime(datos.recomendaciones_estacion(estacion_id)["fecha"]))
    return min(pron & rec, default=min(rec, default=None))


def test_abre_con_selectores_grafica_y_acciones(datos):
    at = abrir()
    sin_errores(at)

    assert [s.label for s in at.selectbox] == ["Estación", "Día"]
    assert at.radio[0].options == ["Salidas", "Llegadas"]
    assert len(graficas(at)) == 1

    # Los días se muestran como fechas en español y las estaciones por nombre.
    dia = selector(at, "Día")
    assert all(re.match(r"^(lunes|martes|miércoles|jueves|viernes|sábado|domingo) \d+ de \w+ de \d{4}$", o)
               for o in dia.options)
    opciones = datos.opciones_de_estacion()
    assert selector(at, "Estación").options == list(opciones.values())

    # En un día simulado hay una tarjeta por franja y política, con el texto de la acción.
    estacion_id = next(e for e in opciones if dia_simulado(datos, e) is not None)
    selector(at, "Estación").set_value(estacion_id)
    at = at.run()
    selector(at, "Día").set_value(dia_simulado(datos, estacion_id))
    at = at.run()
    sin_errores(at)
    rec = datos.recomendaciones_estacion(estacion_id)
    del_dia = rec[(rec["fecha"] == dia_simulado(datos, estacion_id)) & rec["politica"].isin(["model", "historical"])]
    assert len(ACCION.findall(textos(at))) == len(del_dia) > 0
    assert textos(at).count("Riesgo simulado:") == len(del_dia)
    for politica in ("Modelo", "Regla histórica"):
        assert f"**{politica}**" in textos(at)


def test_cambiar_estacion_dia_y_objetivo_no_falla(datos):
    at = abrir()
    opciones = list(datos.opciones_de_estacion())
    for estacion_id in opciones[:3] + opciones[-2:]:
        selector(at, "Estación").set_value(estacion_id)
        at = at.run()
        sin_errores(at)
        dias = datos.pronostico_estacion(estacion_id)["fecha"].drop_duplicates().sort_values().tolist()
        for dia in (dias[0], dias[-1]):
            selector(at, "Día").set_value(pd.Timestamp(dia))
            at = at.run()
            sin_errores(at)
            assert len(graficas(at)) == 1
    at.radio[0].set_value("llegadas")
    at = at.run()
    sin_errores(at)
    assert len(graficas(at)) == 1
    assert "Llegadas por hora" in [s.value for s in at.subheader]


def test_estacion_sin_dato_sin_historia_o_sin_simular(datos):
    """Capacidad o tipo desconocidos dicen "Sin dato"; lo no simulado y lo sin historia, una línea."""
    est = datos.estaciones()
    est = est[est["tiene_pronostico"]]
    casos = {
        "sin_capacidad": est[est["capacidad"].isna()],
        "sin_historia": est[est["sin_historia"]],
        "sin_simular": est[~est["simulada"]],
    }
    if all(c.empty for c in casos.values()):
        pytest.skip("Estas tablas no tienen estaciones con datos faltantes")
    for caso, filas in casos.items():
        if filas.empty:
            continue
        at = abrir()
        selector(at, "Estación").set_value(filas["estacion_id"].iloc[0])
        at = at.run()
        sin_errores(at)
        if caso == "sin_capacidad":
            assert next(m for m in at.metric if m.label == "Capacidad").value == "Sin dato"
        if caso == "sin_historia":
            assert "promedio del sistema" in textos(at)
        if caso == "sin_simular":
            assert "no se simuló" in textos(at)
            assert not ACCION.search(textos(at))
        assert not re.search(r"\b(nan|none)\b", textos(at).lower())


def test_presupuesto_insuficiente_se_dice_en_una_frase(datos):
    """Riesgo alto sin mover bicis: la tarjeta dice que el presupuesto no alcanzó."""
    for estacion_id in list(datos.opciones_de_estacion())[:40]:
        rec = datos.recomendaciones_estacion(estacion_id)
        casos = rec[(rec["politica"] == "model") & (rec["nivel_riesgo"] == "alto") & (rec["bicis"] == 0)]
        if not casos.empty:
            break
    else:
        pytest.skip("Ninguna de las primeras estaciones quedó en riesgo alto sin traslado")
    at = abrir()
    selector(at, "Estación").set_value(estacion_id)
    at = at.run()
    selector(at, "Día").set_value(pd.Timestamp(casos["fecha"].iloc[0]))
    at = at.run()
    sin_errores(at)
    assert "El presupuesto no alcanzó para esta estación." in textos(at)
