"""Contexto de cada estación: empleos y trabajadores residentes a 300 metros.

Fuente: LODES 8 de la Oficina del Censo (LEHD), estado de Nueva York.
- WAC: empleos contados en la manzana donde está el trabajo.
- RAC: trabajadores contados en la manzana donde viven. Son personas con
  empleo, no población total.
- Tabla de geografía (xwalk): latitud y longitud del punto interior de cada
  manzana censal de 2020.
Se usan todos los empleos (JT00), todos los segmentos (S000) y la columna C000.
El Censo agrega ruido a propósito para proteger la confidencialidad.

El patrón de uso (tipo de estación e índices) se calcula corriendo
notebooks/b/tipo_estacion.sql, que usa solo el tramo de entrenamiento.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/{estado}/"
LODES_DIR = ROOT / "data" / "raw" / "lodes"
SQL_TIPO = ROOT / "notebooks" / "b" / "tipo_estacion.sql"
RADIO_M = 300.0
RADIO_TIERRA_M = 6_371_000.0
AGENTE = {"User-Agent": "Mozilla/5.0 (CityBykeNY, uso academico)"}
TIPOS = ["oficinas", "residencial", "recreativa", "mixta"]


# ---------------------------------------------------------------- descarga

def latest_file(listing_html: str, estado: str, clase: str) -> tuple[str, int]:
    """Nombre y año del archivo más reciente de la clase ('wac' o 'rac') en un listado."""
    patron = rf"{estado}_{clase}_S000_JT00_(\d{{4}})\.csv\.gz"
    anios = sorted({int(a) for a in re.findall(patron, listing_html)})
    if not anios:
        raise RuntimeError(f"El listado no trae archivos {clase} S000 JT00 de {estado}.")
    return f"{estado}_{clase}_S000_JT00_{anios[-1]}.csv.gz", anios[-1]


def download_lodes(estado: str = "ny", destino: Path = LODES_DIR) -> dict:
    """Lee los listados del servidor, elige el año más reciente y descarga lo que falte."""
    import requests

    destino.mkdir(parents=True, exist_ok=True)
    base = LODES_URL.format(estado=estado)
    usados = {}
    for clase in ("wac", "rac"):
        listado = requests.get(base + f"{clase}/", headers=AGENTE, timeout=60)
        listado.raise_for_status()
        nombre, anio = latest_file(listado.text, estado, clase)
        usados[clase] = {"archivo": nombre, "anio": anio, "url": base + f"{clase}/" + nombre}
    listado = requests.get(base, headers=AGENTE, timeout=60)
    listado.raise_for_status()
    nombre = f"{estado}_xwalk.csv.gz"
    if nombre not in listado.text:
        raise RuntimeError(f"El listado de {base} no trae {nombre}.")
    usados["xwalk"] = {"archivo": nombre, "anio": None, "url": base + nombre}

    for info in usados.values():
        ruta = destino / info["archivo"]
        if not ruta.exists():
            parcial = ruta.with_suffix(".parcial")
            with requests.get(info["url"], headers=AGENTE, timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(parcial, "wb") as f:
                    for trozo in r.iter_content(1 << 20):
                        f.write(trozo)
            parcial.rename(ruta)
        info["ruta"] = ruta
        info["bytes"] = ruta.stat().st_size
    return usados


# ---------------------------------------------------------------- manzanas

def load_blocks(wac: Path, rac: Path, xwalk: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Manzanas dentro de bbox (lat_min, lat_max, lon_min, lon_max) con empleos y residentes."""
    geo = pd.read_csv(xwalk, usecols=["tabblk2020", "blklatdd", "blklondd"], dtype={"tabblk2020": str})
    geo = geo.rename(columns={"tabblk2020": "block_id", "blklatdd": "lat", "blklondd": "lon"})
    lat0, lat1, lon0, lon1 = bbox
    geo = geo[geo.lat.between(lat0, lat1) & geo.lon.between(lon0, lon1)]
    w = pd.read_csv(wac, usecols=["w_geocode", "C000"], dtype={"w_geocode": str})
    r = pd.read_csv(rac, usecols=["h_geocode", "C000"], dtype={"h_geocode": str})
    w = w.rename(columns={"w_geocode": "block_id", "C000": "empleos"})
    r = r.rename(columns={"h_geocode": "block_id", "C000": "residentes"})
    out = geo.merge(w, on="block_id", how="left").merge(r, on="block_id", how="left")
    out[["empleos", "residentes"]] = out[["empleos", "residentes"]].fillna(0).astype("int64")
    return out.reset_index(drop=True)


def station_bbox(stations: pd.DataFrame, margen_grados: float = 0.01) -> tuple[float, float, float, float]:
    """Caja que encierra a las estaciones, con un margen (0.01 grados son unos 850 m o más)."""
    return (stations.lat.min() - margen_grados, stations.lat.max() + margen_grados,
            stations.lon.min() - margen_grados, stations.lon.max() + margen_grados)


# ---------------------------------------------------------------- suma a 300 m

def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en metros sobre la esfera. Acepta arreglos que se puedan combinar."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_M * np.arcsin(np.sqrt(a))


def within_matrix(stations: pd.DataFrame, points: pd.DataFrame, radio_m: float = RADIO_M, lote: int = 200):
    """Por lotes de estaciones, entrega (inicio, matriz booleana estación x punto) dentro del radio."""
    plat, plon = points.lat.to_numpy(float), points.lon.to_numpy(float)
    slat, slon = stations.lat.to_numpy(float), stations.lon.to_numpy(float)
    for i in range(0, len(stations), lote):
        d = haversine_m(slat[i:i + lote, None], slon[i:i + lote, None], plat[None, :], plon[None, :])
        yield i, d <= radio_m


def sum_within(stations: pd.DataFrame, blocks: pd.DataFrame, radio_m: float = RADIO_M) -> pd.DataFrame:
    """Suma empleos y residentes de las manzanas cuyo punto interior queda dentro del radio.

    Una estación sin ninguna manzana dentro del radio queda con nulos (no con
    cero): no hay dato para ella. proporcion_empleos es nula si ambas sumas son 0.
    """
    n = len(stations)
    empleos, residentes, manzanas = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    e, r = blocks.empleos.to_numpy(float), blocks.residentes.to_numpy(float)
    if len(blocks):
        for i, dentro in within_matrix(stations, blocks, radio_m):
            empleos[i:i + len(dentro)] = dentro @ e
            residentes[i:i + len(dentro)] = dentro @ r
            manzanas[i:i + len(dentro)] = dentro.sum(axis=1)
    out = pd.DataFrame({"station_id": stations.station_id.astype(str).to_numpy(),
                        "empleos_300m": empleos, "residentes_300m": residentes, "manzanas_300m": manzanas})
    sin_dato = out.manzanas_300m == 0
    out.loc[sin_dato, ["empleos_300m", "residentes_300m"]] = np.nan
    total = out.empleos_300m + out.residentes_300m
    out["proporcion_empleos"] = (out.empleos_300m / total).where(total > 0)
    out[["empleos_300m", "residentes_300m"]] = out[["empleos_300m", "residentes_300m"]].astype("Int64")
    return out


def count_within(stations: pd.DataFrame, points: pd.DataFrame, radio_m: float = RADIO_M) -> np.ndarray:
    """Cuántos puntos (lat, lon) quedan dentro del radio de cada estación."""
    out = np.zeros(len(stations), dtype=int)
    if len(points):
        for i, dentro in within_matrix(stations, points, radio_m):
            out[i:i + len(dentro)] = dentro.sum(axis=1)
    return out


# ---------------------------------------------------------------- patrón de uso

def usage_pattern(sql: Path = SQL_TIPO, raiz: Path = ROOT) -> pd.DataFrame:
    """Corre la consulta del tipo de estación tal cual (solo tramo de entrenamiento)."""
    import os

    import duckdb

    anterior = os.getcwd()
    os.chdir(raiz)  # la consulta lee data/processed/station_hour.parquet con ruta relativa
    try:
        con = duckdb.connect()
        sentencias = [s for s in sql.read_text(encoding="utf-8").split(";") if re.sub(r"--.*", "", s).strip()]
        for s in sentencias[:-1]:
            con.execute(s)
        df = con.execute(sentencias[-1]).df()
    finally:
        os.chdir(anterior)
    df["station_id"] = df["station_id"].astype(str)
    return df


# ---------------------------------------------------------------- comparación

def pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 3:
        return float("nan"), n
    return float(np.corrcoef(x[ok].astype(float), y[ok].astype(float))[0, 1]), n


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Pearson sobre los rangos (empates con rango promedio)."""
    ok = x.notna() & y.notna()
    return pearson(x[ok].astype(float).rank(), y[ok].astype(float).rank())


def medians_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana de empleos, residentes y proporción por tipo, con cuántas estaciones entran."""
    con_dato = df[df.empleos_300m.notna()]
    g = con_dato.groupby("station_type")
    out = pd.DataFrame({
        "estaciones": g.size(),
        "empleos_mediana": g.empleos_300m.median(),
        "residentes_mediana": g.residentes_300m.median(),
        "proporcion_mediana": g.proporcion_empleos.median(),
        "con_proporcion": g.proporcion_empleos.count(),
    })
    return out.reindex([t for t in TIPOS if t in out.index]).reset_index()


# ---------------------------------------------------------------- OpenStreetMap (opcional)

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]  # espejo, si el principal no responde


def osm_points(bbox: tuple[float, float, float, float], timeout: int = 120) -> pd.DataFrame:
    """Una consulta a Overpass: parques, atracciones turísticas y universidades (punto central).

    Prueba el servidor principal y, si falla, un espejo. El servidor que
    respondió queda en `.attrs["servidor"]`.
    """
    import requests

    lat0, lat1, lon0, lon1 = bbox
    caja = f"({lat0:.5f},{lon0:.5f},{lat1:.5f},{lon1:.5f})"
    consulta = (f"[out:json][timeout:{timeout}];("
                f'nwr["leisure"="park"]{caja};nwr["tourism"="attraction"]{caja};'
                f'nwr["amenity"="university"]{caja};);out center tags;')
    datos, error = None, None
    for url in OVERPASS_URLS:
        try:
            # Overpass rechaza (406) el agente tipo navegador: se identifica el proyecto.
            r = requests.post(url, data={"data": consulta},
                              headers={"User-Agent": "CityBykeNY-research/1.0"}, timeout=timeout + 30)
            r.raise_for_status()
            datos, servidor = r.json(), url
            break
        except Exception as e:  # se prueba el siguiente servidor
            error = e
    if datos is None:
        raise error
    filas = []
    for el in datos.get("elements", []):
        centro = el.get("center", el)
        if "lat" not in centro:
            continue
        tags = el.get("tags", {})
        clase = ("parque" if tags.get("leisure") == "park" else
                 "atraccion" if tags.get("tourism") == "attraction" else "universidad")
        filas.append({"clase": clase, "lat": centro["lat"], "lon": centro["lon"]})
    out = pd.DataFrame(filas, columns=["clase", "lat", "lon"])
    out.attrs["servidor"] = servidor
    return out
