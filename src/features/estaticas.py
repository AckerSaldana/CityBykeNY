"""Variables fijas de cada estación para el modelo LightGBM.

No son constructores registrados: no reciben solo station_hour, necesitan que
quien llama les diga cuáles filas son la HISTORIA (el tramo con el que se
ajusta el modelo). Quien llama pasa solo esas filas; del tramo que se predice
no entra ninguna (decisión 5). Lo prueba tests/test_lgbm.py.

Variables:

- lat, lon y capacidad: de stations.parquet. La capacidad es la de GBFS de hoy
  (ver CLAUDE.md) y puede venir vacía.
- {dep,arr}_media_estacion: promedio por hora de la estación en la historia.
- {dep,arr}_media_estacion_hora: promedio de la estación a esa hora local.
- {dep,arr}_media_estacion_tipo_hora: lo mismo, separando lunes a viernes de
  sábado y domingo.

Una estación sin filas en la historia queda con sus promedios vacíos (NaN).

Aviso: para las filas de la propia historia, estos promedios incluyen el valor
de la fila (uno entre decenas). Por eso no se usa el promedio por día de la
semana y hora, que solo tendría un valor por semana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.historia import PREFIXES, SOURCE_COLUMNS, local_clock

STATION_COLUMNS = ["lat", "lon", "capacidad"]
MEAN_COLUMNS = [
    f"{prefix}_{name}"
    for prefix in PREFIXES
    for name in ("media_estacion", "media_estacion_hora", "media_estacion_tipo_hora")
]
STATIC_COLUMNS = STATION_COLUMNS + MEAN_COLUMNS


def station_statics(rows: pd.DataFrame, history: pd.DataFrame, stations: pd.DataFrame | None) -> pd.DataFrame:
    """Variables fijas para cada fila de `rows` (mismo orden e índice), calculadas solo con `history`.

    De `rows` solo se leen station_id y ts_hour. `history` son las filas de
    station_hour del tramo de ajuste. `stations` sigue el contrato stations;
    con None, lat, lon y capacidad quedan vacías.
    """
    local = local_clock(rows)
    keys = pd.DataFrame(
        {
            "station_id": rows["station_id"].to_numpy(),
            "hour": local.dt.hour.to_numpy(),
            "weekend": (local.dt.dayofweek >= 5).to_numpy(),
        }
    )
    past_local = local_clock(history)
    past = pd.DataFrame(
        {
            "station_id": history["station_id"].to_numpy(),
            "hour": past_local.dt.hour.to_numpy(),
            "weekend": (past_local.dt.dayofweek >= 5).to_numpy(),
            **{prefix: history[SOURCE_COLUMNS[prefix]].to_numpy(dtype=float) for prefix in PREFIXES},
        }
    )
    out = pd.DataFrame(index=rows.index)

    if stations is None:
        for column in STATION_COLUMNS:
            out[column] = np.nan
    else:
        info = stations.set_index("station_id")
        out["lat"] = info["lat"].astype(float).reindex(keys["station_id"]).to_numpy()
        out["lon"] = info["lon"].astype(float).reindex(keys["station_id"]).to_numpy()
        out["capacidad"] = info["capacity"].astype("Float64").reindex(keys["station_id"]).to_numpy(dtype=float, na_value=np.nan)

    levels = {
        "media_estacion": ["station_id"],
        "media_estacion_hora": ["station_id", "hour"],
        "media_estacion_tipo_hora": ["station_id", "weekend", "hour"],
    }
    for name, by in levels.items():
        means = past.groupby(by, sort=False)[list(PREFIXES)].mean()
        wanted = keys[by[0]] if len(by) == 1 else pd.MultiIndex.from_frame(keys[by])
        found = means.reindex(wanted)
        for prefix in PREFIXES:
            out[f"{prefix}_{name}"] = found[prefix].to_numpy()
    return out[STATIC_COLUMNS]
