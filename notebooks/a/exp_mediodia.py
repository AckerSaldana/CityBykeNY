"""EXPERIMENTO (solo validación): política modelo con corrección de mediodía.

En la intervención que no es la primera del día (hoy, la de la tarde), el
pronóstico de LightGBM de cada estación se multiplica por
    (observado en la mañana + K) / (pronosticado en la mañana + K)
con las horas completas de ESE día anteriores a la intervención. Se hace por
separado para salidas y para llegadas. K suaviza las estaciones con pocos
viajes. La primera ronda del día no cambia.

Uso:  python notebooks/a/exp_mediodia.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.contracts import QUANTILE_COLUMNS
from src.data.datos_config import load_datos
from src.paths import DATA_PROCESSED
from src.sim import simulate as S
from src.sim.engine import Reglas, build_calendar, build_network, daily_budget, initial_inventory, prepare_events, simulate
from src.sim.operacion import load_operacion
from src.sim.policies import ForecastInfo, ForecastPolicy, NoIntervention, _local, stations_with_history

K = 5.0      # viajes de suavizado (valor del experimento, sin ajustar)
BAND = 0.5   # la banda ya elegida en validación


def _cube(frame, column, network, days):
    """Valores de una columna en [día, hora, estación]; lo que falta queda en 0, con su conteo."""
    n = network.size
    station = network.index.get_indexer(frame["station_id"].astype("object"))
    local = _local(frame["ts_hour"])
    day = ((local.dt.normalize() - pd.Timestamp(days[0])) // pd.Timedelta(days=1)).to_numpy()
    keep = (station >= 0) & (day >= 0) & (day < len(days))
    cell = (day[keep].astype(np.int64) * 24 + local.dt.hour.to_numpy()[keep]) * n + station[keep]
    size, shape = len(days) * 24 * n, (len(days), 24, n)
    values = np.bincount(cell, weights=frame[column].to_numpy("float64")[keep], minlength=size).reshape(shape)
    return values, np.bincount(cell, minlength=size).reshape(shape)


def corrected_forecast(base: ForecastInfo, predictions, observed, network, reglas, days, k=K) -> ForecastInfo:
    """Copia de `base` con las franjas que no son la primera del día corregidas.

    `observed` son filas de station_hour. De cada día solo se leen las horas
    completas anteriores a la hora de intervención de la franja.
    """
    q = {c: _cube(predictions, c, network, days)[0] for c in QUANTILE_COLUMNS}
    dep, _ = _cube(observed, "departures", network, days)
    arr, _ = _cube(observed, "arrivals", network, days)
    first = min(reglas.franjas, key=lambda f: f.intervencion_min)
    paths = dict(base.paths)
    for franja in reglas.franjas:
        if franja is first:
            continue
        before = list(range(franja.intervencion_min // 60))   # horas completas ya transcurridas
        hours = list(franja.hours)
        for d, day in enumerate(days):
            r_dep = (dep[d][before].sum(axis=0) + k) / (q["dep_q50"][d][before].sum(axis=0) + k)
            r_arr = (arr[d][before].sum(axis=0) + k) / (q["arr_q50"][d][before].sum(axis=0) + k)
            a10, a50, a90 = (r_arr * q[f"arr_q{p}"][d][hours] for p in (10, 50, 90))
            d10, d50, d90 = (r_dep * q[f"dep_q{p}"][d][hours] for p in (10, 50, 90))
            known = base.paths[(day, franja.nombre)][0]
            path = np.cumsum(a50 - d50, axis=0)
            down = np.sqrt(np.cumsum((a50 - a10) ** 2 + (d90 - d50) ** 2, axis=0))
            up = np.sqrt(np.cumsum((a90 - a50) ** 2 + (d50 - d10) ** 2, axis=0))
            paths[(day, franja.nombre)] = (known, path, down, up)
    return ForecastInfo(paths=paths)


def main():
    datos = load_datos()
    tramo = datos.tramos["validacion"]
    reglas = Reglas.from_operacion(load_operacion())
    stations, trips, history, forecasts, frozen = S.read_inputs(DATA_PROCESSED, datos, tramo)
    aj = datos.simulacion
    calendar = build_calendar(tramo.inicio, tramo.fin, reglas.franjas, int(aj.dias_de_calentamiento), tuple(aj.franja_noche))
    chosen, _ = S.select_stations(stations, trips, calendar.start_us, calendar.end_us)
    network = build_network(chosen)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    occupancy, _ = S.observed_occupancy(network, frozen.table)
    start = initial_inventory(network, occupancy)
    with_history = stations_with_history(history, network, before=tramo.start)
    info = {n: ForecastInfo.from_predictions(forecasts[n], network, reglas, calendar.days, with_history) for n in ("historical", "model")}
    # Lo observado en validación (demanda atendida). La corrección solo lee horas anteriores a cada intervención.
    table = pq.read_table(DATA_PROCESSED / "station_hour.parquet", filters=[("ts_hour", ">=", tramo.start), ("ts_hour", "<", tramo.end)]).to_pandas()
    info["model_mediodia"] = corrected_forecast(info["model"], forecasts["model"], table, network, reglas, calendar.days)
    neighbors = network.neighbors_within(reglas.distancia_maxima_km)
    budget = daily_budget(reglas.presupuesto_de_referencia, network.size)
    runs = {"none": simulate(network, events, calendar, reglas, NoIntervention(), 0, start)}
    for name, source in info.items():
        policy = ForecastPolicy("model" if name != "historical" else "historical", network, reglas, source, BAND, neighbors)
        runs[name] = simulate(network, events, calendar, reglas, policy, budget, start)
    print(f"Validación, {len(calendar.evaluated)} días evaluados, {budget:,} bicis al día, banda {BAND}, K = {K:g}")
    print("política          | riesgo/día | " + " | ".join(calendar.labels))
    days = {}
    for name, run in runs.items():
        hours = run.risk_hours[calendar.evaluated][:, :, 0]
        days[name] = hours.sum(axis=1)
        print(f"{name:17s} | {days[name].mean():10.1f} | " + " | ".join(f"{v:8.1f}" for v in hours.mean(axis=0)))
    diff = days["model"] - days["model_mediodia"]
    print("mejora diaria contra modelo (positivo = menos riesgo):", np.round(diff, 1).tolist())
    print(f"días en que mejora: {(diff > 0).sum()} de {len(diff)}; media {diff.mean():.1f}")


if __name__ == "__main__":
    main()
