"""Clima OBSERVADO por hora (Open-Meteo) y su relación con la demanda del sistema.

Es una descripción, no una entrada del modelo: el clima observado no se conoce
al momento de pronosticar (la regla del equipo es que un modelo solo puede usar
clima pronosticado). Lo que sale de aquí es asociación, no causa.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "clima"
PROCESSED = ROOT / "data" / "processed"

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 40.7549, -73.9840  # Midtown Manhattan (Bryant Park), punto central pedido
VARIABLES = ["temperature_2m", "precipitation"]
TIMEZONE = "America/New_York"
RAIN_MM = 0.1  # una hora "con lluvia" tiene precipitación >= 0.1 mm en esa hora
MIN_HOURS = 3  # horas mínimas de cada lado para comparar una celda


def download_weather(start: str, end: str, force: bool = False) -> Path:
    """Baja el clima por hora y lo guarda tal cual (JSON) en data/raw/clima."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"open_meteo_{start}_{end}.json"
    if path.exists() and not force:
        return path
    r = requests.get(ENDPOINT, timeout=60, params={
        "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES), "timezone": TIMEZONE})
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data or not data["hourly"].get("time"):
        raise RuntimeError(f"Open-Meteo no devolvió horas: {str(data)[:200]}")
    path.write_text(json.dumps(data))
    return path


def load_weather(path: Path) -> tuple[pd.DataFrame, dict]:
    """Tabla con hora local (sin zona), temperatura y precipitación; y los metadatos de la respuesta."""
    data = json.loads(Path(path).read_text())
    h = data["hourly"]
    df = pd.DataFrame({"local_hour": pd.to_datetime(h["time"]),
                       "temperature": pd.to_numeric(h["temperature_2m"], errors="coerce"),
                       "precipitation": pd.to_numeric(h["precipitation"], errors="coerce")})
    meta = {k: data.get(k) for k in ["latitude", "longitude", "elevation", "timezone", "hourly_units"]}
    return df, meta


def system_demand(start: str, end: str, path: Path | None = None) -> pd.DataFrame:
    """Salidas de todo el sistema por hora local de Nueva York."""
    import duckdb

    path = path or PROCESSED / "station_hour.parquet"
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    df = con.execute(
        f"""SELECT CAST(ts_hour AS TIMESTAMP) AS local_hour, SUM(departures) AS departures
            FROM '{path}'
            WHERE CAST(ts_hour AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
            GROUP BY 1 ORDER BY 1""").df()
    df["local_hour"] = pd.to_datetime(df["local_hour"]).dt.tz_localize(None) if getattr(
        df["local_hour"].dt, "tz", None) is not None else pd.to_datetime(df["local_hour"])
    df["departures"] = df["departures"].astype(float)
    return df


def join_hours(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Une demanda y clima por hora local y marca lluvia, hora del día y tipo de día."""
    df = demand.merge(weather, on="local_hour", how="inner")
    df = df[df["precipitation"].notna()].copy()
    df["rain"] = df["precipitation"] >= RAIN_MM
    df["hour_of_day"] = df["local_hour"].dt.hour
    df["day_type"] = np.where(df["local_hour"].dt.dayofweek >= 5, "fin_de_semana", "laborable")
    return df


def compare_rain_dry(hours: pd.DataFrame) -> pd.DataFrame:
    """Por celda (hora del día, tipo de día): demanda media con lluvia y en seco.

    `diferencia` es (lluvia - seco) / seco. `comparable` pide al menos MIN_HOURS
    horas de cada lado.
    """
    g = hours.groupby(["day_type", "hour_of_day", "rain"])["departures"].agg(["mean", "count"]).unstack("rain")
    out = pd.DataFrame({
        "tipo_dia": g.index.get_level_values("day_type"),
        "hora": g.index.get_level_values("hour_of_day").astype(int),
        "demanda_seco": g["mean"].get(False, pd.Series(np.nan, index=g.index)).to_numpy(),
        "demanda_lluvia": g["mean"].get(True, pd.Series(np.nan, index=g.index)).to_numpy(),
        "horas_seco": g["count"].get(False, pd.Series(0, index=g.index)).fillna(0).astype(int).to_numpy(),
        "horas_lluvia": g["count"].get(True, pd.Series(0, index=g.index)).fillna(0).astype(int).to_numpy(),
    })
    out["diferencia"] = (out["demanda_lluvia"] - out["demanda_seco"]) / out["demanda_seco"].where(
        out["demanda_seco"] > 0)
    out["comparable"] = (out["horas_seco"] >= MIN_HOURS) & (out["horas_lluvia"] >= MIN_HOURS) & out[
        "diferencia"].notna()
    return out.sort_values(["tipo_dia", "hora"]).reset_index(drop=True)


def weighted_difference(cells: pd.DataFrame) -> dict:
    """Diferencia relativa promedio, ponderada por la demanda en seco de cada celda comparable."""
    c = cells[cells["comparable"]]
    if c.empty:
        return {"diferencia": None, "celdas": 0, "horas_lluvia": 0, "horas_seco": 0}
    w = c["demanda_seco"]
    return {"diferencia": float((c["diferencia"] * w).sum() / w.sum()),
            "celdas": int(len(c)), "celdas_totales": int(len(cells)),
            "celdas_con_menos_demanda": int((c["diferencia"] < 0).sum()),
            "horas_lluvia": int(c["horas_lluvia"].sum()), "horas_seco": int(c["horas_seco"].sum())}


def summary_by_day_type(cells: pd.DataFrame) -> dict:
    return {t: weighted_difference(cells[cells["tipo_dia"] == t]) for t in sorted(cells["tipo_dia"].unique())}


def temperature_correlation(hours: pd.DataFrame) -> dict:
    """Correlación simple (Pearson) entre temperatura y demanda, y la misma quitando la media de cada celda."""
    h = hours[hours["temperature"].notna()]
    if len(h) < 3:
        return {"simple": None, "dentro_de_celda": None, "horas": int(len(h))}
    simple = float(np.corrcoef(h["temperature"], h["departures"])[0, 1])
    grp = h.groupby(["day_type", "hour_of_day"])
    dt = h["temperature"] - grp["temperature"].transform("mean")
    dd = h["departures"] - grp["departures"].transform("mean")
    inside = float(np.corrcoef(dt, dd)[0, 1]) if dt.std() > 0 and dd.std() > 0 else None
    return {"simple": simple, "dentro_de_celda": inside, "horas": int(len(h))}


def daily_test_errors(weather: pd.DataFrame, start: str, end: str,
                      predictions: Path | None = None, station_hour: Path | None = None) -> pd.DataFrame:
    """Por día del tramo de prueba: lluvia y WAPE de salidas de cada modelo. Solo describe; no recalifica nada."""
    import duckdb

    predictions = predictions or PROCESSED / "predictions.parquet"
    station_hour = station_hour or PROCESSED / "station_hour.parquet"
    con = duckdb.connect()
    con.execute("SET TimeZone='America/New_York'")
    err = con.execute(
        f"""SELECT CAST(p.ts_hour AS DATE) AS fecha, p.model AS modelo,
                   SUM(ABS(s.departures - p.dep_q50)) AS error_abs, SUM(s.departures) AS salidas
            FROM '{predictions}' p JOIN '{station_hour}' s USING (station_id, ts_hour)
            WHERE p.split = 'test' AND CAST(p.ts_hour AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
            GROUP BY 1, 2 ORDER BY 1, 2""").df()
    err["wape"] = err["error_abs"] / err["salidas"]
    wide = err.pivot(index="fecha", columns="modelo", values="wape").add_prefix("wape_")
    wide["salidas"] = err.groupby("fecha")["salidas"].first()
    wide = wide.reset_index()
    wide["fecha"] = pd.to_datetime(wide["fecha"])
    w = weather.assign(fecha=weather["local_hour"].dt.normalize())
    rain = w.groupby("fecha").agg(horas_lluvia=("precipitation", lambda x: int((x >= RAIN_MM).sum())),
                                  horas_lluvia_dia=("precipitation", lambda x: 0),
                                  precipitacion_mm=("precipitation", "sum"),
                                  temperatura_media=("temperature", "mean")).reset_index()
    day = w[w["local_hour"].dt.hour.between(7, 21)]
    rain["horas_lluvia_dia"] = rain["fecha"].map(
        day.groupby("fecha")["precipitation"].apply(lambda x: int((x >= RAIN_MM).sum()))).fillna(0).astype(int)
    out = wide.merge(rain, on="fecha", how="left")
    out["fin_de_semana"] = out["fecha"].dt.dayofweek >= 5
    return out
