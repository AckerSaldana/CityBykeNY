"""Variables del modelo LightGBM que salen de station_hour (decisión 5: sin fuga).

Tres constructores registrados. Las variables de una fila cuyo día local de
Nueva York es D usan solo filas de días ANTERIORES a D, o solo la fecha y la
hora de la propia fila (calendario). tests/test_no_leakage.py los revisa a los
tres sin hacer nada más.

`rezagos_misma_hora`, para salidas (dep) y llegadas (arr), siempre a la misma
hora del reloj de Nueva York:

- {dep,arr}_rez_1d, _rez_2d, _rez_7d, _rez_14d: el valor 1, 2, 7 y 14 días de
  calendario antes.
- {dep,arr}_media_7d y _media_28d: promedio de los 7 y de los 28 días previos.
- {dep,arr}_q10_28d y _q90_28d: cuantiles 10 % y 90 % de los 28 días previos.
- {dep,arr}_media_dia_4s, _min_dia_4s y _max_dia_4s: promedio, mínimo y máximo
  del mismo día de la semana en las 4 semanas previas (7, 14, 21 y 28 días
  antes).
- dias_con_historia_28d: cuántos de los 28 días previos tienen fila a esa hora.

`totales_dias_previos`, para salidas y llegadas:

- {dep,arr}_total_1d y _total_7d: total del día de calendario anterior y del
  de 7 días antes (suma de todas las filas de ese día).
- {dep,arr}_total_media_7d: promedio de los totales de los 7 días previos.

`calendario`: hora, dia_semana (0 = lunes), fin_de_semana y festivo (festivo
federal de Estados Unidos, también si cae en fin de semana). Solo mira ts_hour.

Reglas comunes, las mismas de src/features/lags.py:

- Los rezagos se buscan por fecha y hora local, nunca por posición ni restando
  24 horas.
- El día de 25 horas la 01:00 aparece dos veces: como origen se usa el
  promedio de las dos. El día de 23 horas la 02:00 no existe: queda vacío.
- Sin historia el valor queda vacío (NaN). Los promedios, cuantiles, mínimos y
  máximos usan los días que sí tienen fila y quedan vacíos si no hay ninguno.

Las variables fijas de cada estación (promedios del tramo de ajuste,
coordenadas, capacidad) no van aquí: necesitan saber qué tramo es la historia.
Están en src/features/estaticas.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import KEY_COLUMNS, register
from src.models.metrics import federal_holidays
from src.paths import TZ

PREFIXES = ("dep", "arr")
SOURCE_COLUMNS = {"dep": "departures", "arr": "arrivals"}

LAGS = (1, 2, 7, 14)           # días de calendario antes, misma hora
SHORT_WINDOW, WINDOW = 7, 28   # días previos de los promedios y cuantiles
SAME_WEEKDAY = (7, 14, 21, 28)
Q_LOW, Q_HIGH = 0.10, 0.90

LAG_COLUMNS = [
    f"{prefix}_{name}"
    for prefix in PREFIXES
    for name in (
        *(f"rez_{days}d" for days in LAGS),
        "media_7d", "media_28d", "q10_28d", "q90_28d",
        "media_dia_4s", "min_dia_4s", "max_dia_4s",
    )
] + ["dias_con_historia_28d"]
TOTAL_COLUMNS = [f"{prefix}_{name}" for prefix in PREFIXES for name in ("total_1d", "total_7d", "total_media_7d")]
CALENDAR_COLUMNS = ["hora", "dia_semana", "fin_de_semana", "festivo"]

_STATION_CHUNK = 256  # estaciones por bloque al calcular las ventanas: acota la memoria


def local_clock(station_hour: pd.DataFrame) -> pd.Series:
    """Fecha y hora del reloj de Nueva York de cada fila, sin zona."""
    ts = station_hour["ts_hour"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"ts_hour debe traer zona horaria (tipo recibido: {ts.dtype}). Localízala a {TZ}.")
    return ts.dt.tz_convert(TZ).dt.tz_localize(None)


def _positions(station_hour: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Número de estación, de día (0 = primer día de la tabla) y hora local de cada fila."""
    local = local_clock(station_hour)
    day = local.dt.normalize()
    station = pd.factorize(station_hour["station_id"])[0]
    days = (day - day.min()).dt.days.to_numpy() if len(day) else np.zeros(0, dtype=np.int64)
    n_stations = int(station.max()) + 1 if len(station) else 0
    n_days = int(days.max()) + 1 if len(days) else 0
    return station, days.astype(np.int64), local.dt.hour.to_numpy().astype(np.int64), n_stations, n_days


def _shift_days(table: np.ndarray, days: int) -> np.ndarray:
    """La misma tabla (estación, día, ...) vista `days` días después: en el día d queda el valor de d - days."""
    out = np.full_like(table, np.nan)
    if days < table.shape[1]:
        out[:, days:] = table[:, : table.shape[1] - days]
    return out


def _mean(stack: np.ndarray) -> np.ndarray:
    """Promedio a lo largo del primer eje sin contar los vacíos. Todo vacío da vacío."""
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _quantile(ordered: np.ndarray, count: np.ndarray, q: float) -> np.ndarray:
    """Cuantil con interpolación lineal (el de numpy) de los `count` primeros valores, ya ordenados, del primer eje."""
    position = q * np.maximum(count - 1, 0)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, np.maximum(count - 1, 0))
    below = np.take_along_axis(ordered, low[None], axis=0)[0]
    above = np.take_along_axis(ordered, high[None], axis=0)[0]
    value = below + (above - below) * (position - low)
    return np.where(count > 0, value, np.nan)


@register("rezagos_misma_hora")
def rezagos_misma_hora(station_hour: pd.DataFrame) -> pd.DataFrame:
    """Rezagos y resúmenes de los días previos a la misma hora local, por estación."""
    station, day, hour, n_stations, n_days = _positions(station_hour)
    out = station_hour[KEY_COLUMNS].copy()
    cell = (station * n_days + day) * 24 + hour
    size = n_stations * n_days * 24
    rows_per_cell = np.bincount(cell, minlength=size)

    history_days = np.zeros(len(out))
    for prefix in PREFIXES:
        # Un valor por estación, día y hora local (promedio si la hora se repite); vacío donde no hay fila.
        total = np.bincount(cell, weights=station_hour[SOURCE_COLUMNS[prefix]].to_numpy(dtype=float), minlength=size)
        cube = np.where(rows_per_cell > 0, total / np.maximum(rows_per_cell, 1), np.nan).reshape(n_stations, n_days, 24)
        columns = {name: np.empty(len(out)) for name in LAG_COLUMNS if name.startswith(prefix)}
        for start in range(0, n_stations, _STATION_CHUNK):
            block = cube[start : start + _STATION_CHUNK]
            rows = (station >= start) & (station < start + _STATION_CHUNK)
            at = (station[rows] - start, day[rows], hour[rows])
            # previous[k - 1] = el valor k días de calendario antes, k = 1..28.
            previous = np.stack([_shift_days(block, days) for days in range(1, WINDOW + 1)])
            for days in LAGS:
                columns[f"{prefix}_rez_{days}d"][rows] = previous[days - 1][at]
            columns[f"{prefix}_media_7d"][rows] = _mean(previous[:SHORT_WINDOW])[at]
            columns[f"{prefix}_media_28d"][rows] = _mean(previous)[at]
            count = (~np.isnan(previous)).sum(axis=0)
            ordered = np.sort(previous, axis=0)  # los vacíos quedan al final
            columns[f"{prefix}_q10_28d"][rows] = _quantile(ordered, count, Q_LOW)[at]
            columns[f"{prefix}_q90_28d"][rows] = _quantile(ordered, count, Q_HIGH)[at]
            weekday = previous[[days - 1 for days in SAME_WEEKDAY]]
            columns[f"{prefix}_media_dia_4s"][rows] = _mean(weekday)[at]
            columns[f"{prefix}_min_dia_4s"][rows] = np.fmin.reduce(weekday, axis=0)[at]
            columns[f"{prefix}_max_dia_4s"][rows] = np.fmax.reduce(weekday, axis=0)[at]
            history_days[rows] = count[at]
        for name, values in columns.items():
            out[name] = values
    out["dias_con_historia_28d"] = history_days
    return out[KEY_COLUMNS + LAG_COLUMNS]


@register("totales_dias_previos")
def totales_dias_previos(station_hour: pd.DataFrame) -> pd.DataFrame:
    """Totales por estación de los días de calendario previos (todas las horas del día)."""
    station, day, _, n_stations, n_days = _positions(station_hour)
    out = station_hour[KEY_COLUMNS].copy()
    cell = station * n_days + day
    size = n_stations * n_days
    has_rows = np.bincount(cell, minlength=size) > 0
    for prefix in PREFIXES:
        total = np.bincount(cell, weights=station_hour[SOURCE_COLUMNS[prefix]].to_numpy(dtype=float), minlength=size)
        daily = np.where(has_rows, total, np.nan).reshape(n_stations, n_days)  # vacío si ese día no tiene filas
        previous = np.stack([_shift_days(daily, days) for days in range(1, SHORT_WINDOW + 1)])
        out[f"{prefix}_total_1d"] = previous[0][station, day]
        out[f"{prefix}_total_7d"] = previous[SHORT_WINDOW - 1][station, day]
        out[f"{prefix}_total_media_7d"] = _mean(previous)[station, day]
    return out[KEY_COLUMNS + TOTAL_COLUMNS]


@register("calendario")
def calendario(station_hour: pd.DataFrame) -> pd.DataFrame:
    """Hora local, día de la semana, fin de semana y festivo federal de la propia fila."""
    local = local_clock(station_hour)
    out = station_hour[KEY_COLUMNS].copy()
    day = local.dt.normalize()
    out["hora"] = local.dt.hour.to_numpy(dtype=float)
    out["dia_semana"] = local.dt.dayofweek.to_numpy(dtype=float)
    out["fin_de_semana"] = (local.dt.dayofweek >= 5).to_numpy(dtype=float)
    if len(day):
        festivos = pd.DatetimeIndex(list(federal_holidays(day.min().year, day.max().year)))
        out["festivo"] = day.isin(festivos).to_numpy(dtype=float)
    else:
        out["festivo"] = np.zeros(0)
    return out[KEY_COLUMNS + CALENDAR_COLUMNS]
