"""Constructor de REFERENCIA: rezagos de días anteriores.

OJO: este archivo es un EJEMPLO, no el conjunto de variables del proyecto.
Existe para dos cosas: que la prueba de fuga (tests/test_no_leakage.py) tenga
un constructor correcto que revisar, y que sirva de plantilla para escribir
los siguientes. Las variables reales del modelo se deciden aparte.

Variables que produce (todas vienen de días ANTERIORES, nunca del mismo día):

- dep_lag_1d, arr_lag_1d: salidas y llegadas a la misma hora local, un día
  de calendario antes.
- dep_lag_7d, arr_lag_7d: lo mismo, siete días de calendario antes.
- dep_media_7d, arr_media_7d: promedio de la misma hora local en los siete
  días de calendario anteriores.

Cómo se manejan los cambios de horario:

- "Un día antes" es el día de calendario anterior a la misma hora del reloj
  de Nueva York. No se restan 24 horas: el día del cambio de horario eso
  caería en otra hora del reloj.
- El día de 25 horas (noviembre) la 01:00 aparece dos veces. Como dato de
  origen se usa el promedio de las dos, para que siga valiendo "una hora".
- El día de 23 horas (marzo) la 02:00 no existe. Quien la busque como rezago
  recibe vacío (NaN).

Si no hay historia, el valor queda vacío (NaN). El promedio de siete días usa
los días que sí tienen fila y queda vacío solo si no hay ninguno. Atención: si
la tabla no trae fila para las horas sin viajes, ese promedio ignora esas
horas en lugar de contarlas como cero (decisión abierta en CLAUDE.md).
"""

from __future__ import annotations

import pandas as pd

from src.features.registry import KEY_COLUMNS, register
from src.paths import TZ

FEATURE_COLUMNS = ["dep_lag_1d", "dep_lag_7d", "arr_lag_1d", "arr_lag_7d", "dep_media_7d", "arr_media_7d"]


@register("lags_dias_previos")
def lags_dias_previos(station_hour: pd.DataFrame) -> pd.DataFrame:
    """Rezagos de 1 y 7 días y promedio de los 7 días previos, por estación y hora local."""
    ts = station_hour["ts_hour"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"ts_hour debe traer zona horaria (tipo recibido: {ts.dtype}). Localízala a {TZ}.")

    # Día y hora del reloj de Nueva York de cada fila.
    local = ts.dt.tz_convert(TZ).dt.tz_localize(None)
    rows = pd.DataFrame(
        {
            "station_id": station_hour["station_id"].to_numpy(),
            "day": local.dt.normalize().to_numpy(),
            "hour": local.dt.hour.to_numpy(),
            "dep": station_hour["departures"].to_numpy(dtype=float),
            "arr": station_hour["arrivals"].to_numpy(dtype=float),
        }
    )

    # Origen de los rezagos: UN valor por estación, día y hora local. Agrupar
    # antes de buscar evita filas duplicadas por la 01:00 repetida de noviembre.
    source = rows.groupby(["station_id", "day", "hour"])[["dep", "arr"]].mean()

    def days_before(days: int) -> pd.DataFrame:
        """Para cada fila, el valor de la misma estación y hora `days` días de calendario antes."""
        wanted = pd.MultiIndex.from_arrays(
            [rows["station_id"], rows["day"] - pd.Timedelta(days=days), rows["hour"]]
        )
        return source.reindex(wanted).reset_index(drop=True)  # lo que no existe queda en NaN

    previous = {days: days_before(days) for days in range(1, 8)}

    out = station_hour[KEY_COLUMNS].copy()
    for prefix in ("dep", "arr"):
        out[f"{prefix}_lag_1d"] = previous[1][prefix].to_numpy()
        out[f"{prefix}_lag_7d"] = previous[7][prefix].to_numpy()
        week = pd.concat([previous[days][prefix] for days in range(1, 8)], axis=1)
        out[f"{prefix}_media_7d"] = week.mean(axis=1).to_numpy()  # ignora los días sin dato
    return out[KEY_COLUMNS + FEATURE_COLUMNS]
