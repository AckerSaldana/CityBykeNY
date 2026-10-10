"""Contexto de estaciones: suma a 300 m, proporción y sección del dashboard.

Sin red: manzanas y estaciones sintéticas diminutas.

    pytest tests/test_station_context.py
"""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data import station_context as sc

# Un grado de latitud mide unos 111,195 m en la esfera que usa haversine.
M_POR_GRADO = 2 * np.pi * sc.RADIO_TIERRA_M / 360


def _norte(metros: float) -> float:
    return 40.75 + metros / M_POR_GRADO


@pytest.fixture
def stations():
    return pd.DataFrame({"station_id": ["a", "b", "c"],
                         "lat": [40.75, _norte(2000), _norte(4000)],
                         "lon": [-73.99, -73.99, -73.99]})


@pytest.fixture
def blocks():
    # Alrededor de "a": dos manzanas dentro (100 y 290 m) y una fuera (310 m).
    # Alrededor de "b": una manzana dentro, sin empleos ni residentes. "c": ninguna.
    return pd.DataFrame({
        "block_id": ["1", "2", "3", "4"],
        "lat": [_norte(100), _norte(-290), _norte(310), _norte(2050)],
        "lon": [-73.99] * 4,
        "empleos": [30, 10, 1000, 0],
        "residentes": [5, 15, 1000, 0],
    })


def test_haversine_known_distance():
    assert sc.haversine_m(40.75, -73.99, _norte(300), -73.99) == pytest.approx(300, abs=0.01)
    # Un grado de longitud a 40.75 de latitud se encoge por el coseno.
    esperado = M_POR_GRADO * np.cos(np.radians(40.75))
    assert sc.haversine_m(40.75, -73.99, 40.75, -72.99) == pytest.approx(esperado, rel=1e-4)


def test_sum_within_300m_and_proportion(stations, blocks):
    out = sc.sum_within(stations, blocks).set_index("station_id")
    assert out.loc["a", "empleos_300m"] == 40          # 30 + 10; la de 310 m no entra
    assert out.loc["a", "residentes_300m"] == 20       # 5 + 15
    assert out.loc["a", "proporcion_empleos"] == pytest.approx(40 / 60)
    # Manzana cercana con cero y cero: sumas 0 y proporción nula.
    assert out.loc["b", "empleos_300m"] == 0 and out.loc["b", "residentes_300m"] == 0
    assert pd.isna(out.loc["b", "proporcion_empleos"])
    # Sin manzanas cerca: no hay dato (nulo, no cero).
    assert pd.isna(out.loc["c", "empleos_300m"]) and pd.isna(out.loc["c", "proporcion_empleos"])


def test_count_within(stations, blocks):
    assert sc.count_within(stations, blocks).tolist() == [2, 1, 0]
    assert sc.count_within(stations, blocks.iloc[0:0]).tolist() == [0, 0, 0]


def test_latest_file_reads_listing():
    html = '<a href="ny_wac_S000_JT00_2021.csv.gz">x</a> ny_wac_S000_JT00_2023.csv.gz ny_wac_SA01_JT00_2024.csv.gz'
    assert sc.latest_file(html, "ny", "wac") == ("ny_wac_S000_JT00_2023.csv.gz", 2023)
    with pytest.raises(RuntimeError):
        sc.latest_file(html, "ny", "rac")


def test_correlations_and_medians():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, None])
    y = pd.Series([1.0, 4.0, 9.0, 16.0, 5.0])
    r, n = sc.spearman(x, y)
    assert n == 4 and r == pytest.approx(1.0)
    r, n = sc.pearson(x, y)
    assert n == 4 and r == pytest.approx(np.corrcoef([1, 2, 3, 4], [1, 4, 9, 16])[0, 1])
    df = pd.DataFrame({"station_type": ["oficinas", "oficinas", "mixta"],
                       "empleos_300m": [10, 30, 5], "residentes_300m": [1, 3, 5],
                       "proporcion_empleos": [0.9, 0.7, 0.5]})
    med = sc.medians_by_type(df).set_index("station_type")
    assert med.loc["oficinas", "estaciones"] == 2 and med.loc["oficinas", "empleos_mediana"] == 20
    assert med.loc["oficinas", "proporcion_mediana"] == pytest.approx(0.8)


def _seccion(ruta):
    import sys
    from pathlib import Path

    app = str(Path.cwd() / "app") if (Path.cwd() / "app").exists() else None
    if app and app not in sys.path:
        sys.path.insert(0, app)
    import seccion_contexto

    seccion_contexto.mostrar(ruta)


def test_section_runs_with_table(tmp_path, monkeypatch):
    monkeypatch.chdir(sc.ROOT)
    rng = np.random.default_rng(0)
    n = 40
    tabla = pd.DataFrame({
        "estacion": [f"Estación {i}" for i in range(n)],
        "tipo": rng.choice(["oficinas", "residencial", "recreativa", "mixta"], n),
        "proporcion_empleos": rng.uniform(0, 1, n),
        "am_indice": rng.uniform(-0.5, 0.5, n),
        "empleos": pd.array(rng.integers(0, 5000, n), dtype="Int64"),
        "residentes": pd.array(rng.integers(0, 5000, n), dtype="Int64"),
        "anio_datos": 2023,
    })
    tabla.loc[0, "proporcion_empleos"] = None
    ruta = tmp_path / "contexto_estaciones.parquet"
    tabla.to_parquet(ruta, index=False)
    at = AppTest.from_function(_seccion, args=(str(ruta),), default_timeout=30).run()
    assert not at.exception
    texto = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "39 estaciones" in texto
    assert "no población total" in texto and "2023" in texto and "ruido" in texto
    for prohibido in ["proporcion_empleos", "am_indice", ".parquet", "station_"]:
        assert prohibido not in texto


def test_section_without_table_shows_one_message(tmp_path, monkeypatch):
    monkeypatch.chdir(sc.ROOT)
    at = AppTest.from_function(_seccion, args=(str(tmp_path / "no_existe.parquet"),), default_timeout=30).run()
    assert not at.exception
    assert len(at.info) == 1 and ".parquet" not in at.info[0].value
