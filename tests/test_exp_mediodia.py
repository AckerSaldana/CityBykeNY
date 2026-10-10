"""La corrección de mediodía (experimento) no usa datos posteriores a la hora de intervención."""
import importlib.util
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.paths import ROOT, TZ
from src.sim.engine import Franja
from src.sim.policies import ForecastInfo

spec = importlib.util.spec_from_file_location("exp_mediodia", ROOT / "notebooks" / "a" / "exp_mediodia.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def test_la_correccion_no_cambia_si_cambia_lo_observado_desde_la_hora_de_intervencion():
    days = (date(2026, 9, 2), date(2026, 9, 3))
    hours = pd.date_range("2026-09-02", periods=48, freq="h", tz=TZ)
    network = SimpleNamespace(size=2, index=pd.Index(["A", "B"], dtype="object"))
    franjas = (Franja("manana", "05:00", "06:00", "12:00", 0.5), Franja("tarde", "14:00", "15:00", "21:00", 0.5))
    reglas = SimpleNamespace(franjas=franjas)
    rng = np.random.default_rng(0)
    keys = pd.DataFrame({"station_id": np.repeat(["A", "B"], 48), "ts_hour": np.tile(hours, 2)})
    predictions = keys.assign(dep_q10=1.0, dep_q50=3.0, dep_q90=6.0, arr_q10=1.0, arr_q50=2.0, arr_q90=5.0)
    observed = keys.assign(departures=rng.integers(0, 9, 96), arrivals=rng.integers(0, 9, 96))
    base = ForecastInfo(paths={(d, f.nombre): (np.ones(2, bool), None, None, None) for d in days for f in franjas})

    original = exp.corrected_forecast(base, predictions, observed, network, reglas, days)
    later = observed.copy()
    after = later["ts_hour"].dt.hour >= 14          # de la intervención de la tarde en adelante, los dos días
    later.loc[after, ["departures", "arrivals"]] = 999
    changed = exp.corrected_forecast(base, predictions, later, network, reglas, days)
    for day in days:
        for a, b in zip(original.paths[(day, "tarde")][1:], changed.paths[(day, "tarde")][1:]):
            np.testing.assert_array_equal(a, b)

    # Y sí depende de la mañana: si no, la prueba no probaría nada.
    earlier = observed.copy()
    earlier.loc[~after, "departures"] = 999
    moved = exp.corrected_forecast(base, predictions, earlier, network, reglas, days)
    assert not np.array_equal(original.paths[(days[0], "tarde")][1], moved.paths[(days[0], "tarde")][1])
