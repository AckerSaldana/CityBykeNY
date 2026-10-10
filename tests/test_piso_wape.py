"""Piso de azar del WAPE: los dos pronósticos ingenuos, con un caso que se puede seguir a mano."""

import pandas as pd
import pytest

from src.models.metrics import FinalTestError
from src.models.piso_wape import NAIVE_MEAN, NAIVE_SHUFFLE, compute, global_mean_forecast, shuffled_forecast
from src.paths import TZ

FRANJAS = ("noche",) * 6 + ("manana",) * 8 + ("tarde",) * 8 + ("noche",) * 2


def _hours(day: str, hours, stations, values) -> pd.DataFrame:
    rows = [
        (station, pd.Timestamp(f"{day} {hour:02d}:00", tz=TZ), float(value), float(value))
        for hour in hours for station, value in zip(stations, values)
    ]
    frame = pd.DataFrame(rows, columns=["station_id", "ts_hour", "departures", "arrivals"])
    frame["ts_hour"] = pd.DatetimeIndex(frame["ts_hour"]).as_unit("us")
    return frame


def _case():
    stations = ["A", "B", "C", "D"]
    training = _hours("2026-08-03", [8, 9], stations, [0, 2, 4, 10])      # media 4
    validation = _hours("2026-09-02", [8, 15], stations, [1, 3, 4, 8])    # suma real por hora 16
    baseline = validation[["station_id", "ts_hour"]].copy()
    baseline["model"], baseline["split"] = "baseline", "val"
    for prefix in ("dep", "arr"):
        for q, shift in (("q10", -1.0), ("q50", 0.0), ("q90", 1.0)):
            baseline[f"{prefix}_{q}"] = validation["departures"] + shift   # la línea base acierta todo
    return training, validation, baseline


def test_la_media_global_es_un_solo_numero_y_su_wape_se_sigue_a_mano():
    training, validation, baseline = _case()
    rows, means = global_mean_forecast(baseline, training)
    assert means == {"dep": 4.0, "arr": 4.0} and set(rows["dep_q50"]) == {4.0}
    metrics = compute(baseline, validation, training, 7, FRANJAS)["metricas"]
    general = metrics[(metrics["group_by"] == "general") & (metrics["target"] == "dep")].set_index("model")
    assert general.loc["baseline", "wape"] == 0.0
    # |1-4| + |3-4| + |4-4| + |8-4| = 8 por hora, entre 16 de demanda real: 0.5.
    assert general.loc[NAIVE_MEAN, "wape"] == pytest.approx(0.5)
    bands = metrics[(metrics["group_by"] == "franja") & (metrics["model"] == NAIVE_MEAN) & (metrics["target"] == "arr")]
    assert bands["group"].tolist() == ["manana", "tarde"] and bands["n"].tolist() == [4, 4]


def test_la_permutacion_reparte_dentro_de_cada_hora_y_se_repite_con_la_semilla():
    _, validation, baseline = _case()
    first, again = (shuffled_forecast(baseline, seed) for seed in (7, 7))
    assert first.equals(again) and set(first["model"]) == {NAIVE_SHUFFLE}
    for _, hour in first.groupby("ts_hour"):
        assert sorted(hour["dep_q50"]) == [1.0, 3.0, 4.0, 8.0]   # los mismos valores, en otras estaciones
        assert (hour["dep_q50"] == hour["arr_q50"]).all()        # salidas y llegadas se mueven juntas
    assert (first["dep_q10"] == first["dep_q90"]).all()


def test_el_piso_no_califica_la_prueba_final():
    training, _, baseline = _case()
    late = baseline.assign(ts_hour=baseline["ts_hour"] + pd.Timedelta(days=20))   # cae en la prueba final
    actual = _hours("2026-09-22", [8, 15], ["A", "B", "C", "D"], [1, 3, 4, 8])
    with pytest.raises(FinalTestError):
        compute(late, actual, training, 7, FRANJAS)
