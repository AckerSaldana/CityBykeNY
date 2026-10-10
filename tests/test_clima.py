"""Clima y demanda: comparación lluvia contra seco y sección del dashboard.

Sin red: horas sintéticas diminutas.

    pytest tests/test_clima.py
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data import clima as cl


def _hours():
    # 2026-07-06 es lunes; 2026-07-11 es sábado. Hora 8, seis días laborables: 3 secos (100) y 3 con lluvia (60).
    rows = []
    for i, day in enumerate(pd.date_range("2026-07-06", periods=5).tolist() + [pd.Timestamp("2026-07-13")]):
        wet = i >= 3
        rows.append((day + pd.Timedelta(hours=8), 60.0 if wet else 100.0, 1.0 if wet else 0.0, 20.0))
        # Hora 9: solo una hora con lluvia, no alcanza para comparar.
        rows.append((day + pd.Timedelta(hours=9), 10.0 if i == 0 else 50.0, 0.5 if i == 0 else 0.05, 21.0))
    # Fin de semana, hora 8: 3 secos (200) y 3 con lluvia (150).
    for i, day in enumerate(["2026-07-11", "2026-07-12", "2026-07-18", "2026-07-19", "2026-07-25", "2026-07-26"]):
        wet = i % 2 == 0
        rows.append((pd.Timestamp(day) + pd.Timedelta(hours=8), 150.0 if wet else 200.0, 0.1 if wet else 0.0, 25.0))
    demand = pd.DataFrame(rows, columns=["local_hour", "departures", "precipitation", "temperature"])
    return cl.join_hours(demand[["local_hour", "departures"]],
                         demand[["local_hour", "precipitation", "temperature"]])


def test_rain_threshold_and_day_type():
    h = _hours()
    assert h.loc[h.precipitation == 0.05, "rain"].eq(False).all()  # bajo el umbral no es lluvia
    assert h.loc[h.precipitation == 0.1, "rain"].all()  # el umbral sí cuenta
    assert set(h.loc[h.local_hour.dt.day == 11, "day_type"]) == {"fin_de_semana"}


def test_compare_rain_dry():
    cells = cl.compare_rain_dry(_hours()).set_index(["tipo_dia", "hora"])
    lab = cells.loc[("laborable", 8)]
    assert (lab.demanda_seco, lab.demanda_lluvia, lab.horas_seco, lab.horas_lluvia) == (100, 60, 3, 3)
    assert lab.diferencia == pytest.approx(-0.4) and lab.comparable
    assert not cells.loc[("laborable", 9), "comparable"]  # una sola hora con lluvia
    assert cells.loc[("fin_de_semana", 8), "diferencia"] == pytest.approx(-0.25)
    total = cl.weighted_difference(cells.reset_index())
    # Ponderado por la demanda en seco: (100 * -0.4 + 200 * -0.25) / 300
    assert total["diferencia"] == pytest.approx(-0.3)
    assert (total["celdas"], total["horas_lluvia"], total["horas_seco"]) == (2, 6, 6)


def _seccion(ruta):
    import sys
    from pathlib import Path

    app = str(Path.cwd() / "app") if (Path.cwd() / "app").exists() else None
    if app and app not in sys.path:
        sys.path.insert(0, app)
    import seccion_clima

    seccion_clima.mostrar(ruta)


def test_section_runs_with_table(tmp_path, monkeypatch):
    monkeypatch.chdir(cl.ROOT)
    cells = cl.compare_rain_dry(_hours())
    cells["umbral_lluvia_mm"] = cl.RAIN_MM
    ruta = tmp_path / "clima_demanda.parquet"
    cells.to_parquet(ruta, index=False)
    at = AppTest.from_function(_seccion, args=(str(ruta),)).run(timeout=60)
    assert not at.exception
    texto = " ".join(m.value for m in at.markdown)
    assert "30.0 % menos" in texto
    todo = texto + " ".join(c.value for c in at.caption)
    assert "observado" in todo and "parquet" not in todo and "demanda_seco" not in todo


def test_section_without_table(tmp_path, monkeypatch):
    monkeypatch.chdir(cl.ROOT)
    at = AppTest.from_function(_seccion, args=(str(tmp_path / "no_existe.parquet"),)).run(timeout=60)
    assert not at.exception
    assert len(at.info) == 1
