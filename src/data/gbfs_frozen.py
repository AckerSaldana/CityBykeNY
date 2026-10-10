"""Capacidad e inventario inicial CONGELADOS de la captura de GBFS.

La captura de GBFS sigue creciendo y la capacidad que reporta el feed cambia
entre fotos. Si la limpieza leyera la captura directamente, cada
reconstrucción podría cambiar capacidades y, con ellas, la simulación. Por eso
`scripts/freeze_gbfs.py` resume la captura UNA vez en un archivo pequeño que
sí va en Git, `config/gbfs_congelado.csv`, y de ahí leen la limpieza (la
capacidad) y el simulador (el inventario inicial).

Qué trae el archivo, una fila por `short_name` (el id de los viajes):

- `capacity`: la capacidad más frecuente entre las fotos capturadas. Cada
  estado de `station_information` pesa el número de fotos de `station_status`
  tomadas mientras estuvo vigente (desde su captura hasta la del estado
  siguiente). Si dos capacidades empatan, queda la más grande.
- `ocupacion_inicial`: promedio de `num_bikes_available` entre esa capacidad,
  sobre las fotos cuya hora local de Nueva York (`feed_last_updated`) es la
  hora en que empieza la simulación, recortado a [0, 1]. Solo cuentan las
  fotos en que la estación está instalada (`is_installed`). Vacío si la
  capacidad es 0 o si no hay ninguna foto así.
- `fotos`: cuántas fotos entraron en ese promedio.

El encabezado (líneas que empiezan con `#`) dice de qué fechas es la captura.
OJO: la captura es de octubre de 2026, no de los días que se simulan: la
ocupación es la de una madrugada de octubre, usada como supuesto.

`num_bikes_available` cuenta bicis clásicas y eléctricas listas para usarse;
las descompuestas no entran.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.paths import CONFIG, TZ

FROZEN_FILE = CONFIG / "gbfs_congelado.csv"
COLUMNS = ("short_name", "capacity", "ocupacion_inicial", "fotos")
# Hora local de Nueva York en que empieza la simulación: el primer instante de un tramo es la medianoche
# de su primer día (`Tramo.start`), así que la primera hora es la de 00:00 a 00:59.
START_HOUR = 0
# Datos de la captura que van en el encabezado, en este orden.
META_KEYS = (
    "captura_desde_utc",
    "captura_hasta_utc",
    "fotos_de_estado",
    "estados_de_informacion",
    "hora_local_del_inventario",
    "fotos_en_esa_hora",
)


class GbfsFrozenError(ValueError):
    """La captura no alcanza para congelar, o el archivo congelado no se puede leer."""


@dataclass(frozen=True)
class Frozen:
    """El archivo congelado ya leído."""

    table: pd.DataFrame   # short_name (str), capacity (Int64), ocupacion_inicial (float), fotos (int)
    meta: dict            # datos de la captura que trae el encabezado


def _utc(stamp) -> str:
    return pd.Timestamp(stamp).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def freeze(gbfs_dir: str | Path, hour: int = START_HOUR) -> Frozen:
    """Resume la captura de `gbfs_dir` (la carpeta con station_information/ y station_status/).

    Solo lee. No toca la captura ni escribe nada.
    """
    gbfs_dir = Path(gbfs_dir)
    info_files = sorted((gbfs_dir / "station_information").glob("info_*.parquet"))
    status_files = sorted((gbfs_dir / "station_status").glob("date=*/status_*.parquet"))
    if not info_files:
        raise GbfsFrozenError(
            f"No hay ninguna foto de station_information en {gbfs_dir / 'station_information'}. "
            "Corre: python scripts/capture_gbfs.py --una-vez"
        )
    if not status_files:
        raise GbfsFrozenError(
            f"No hay ninguna foto de station_status en {gbfs_dir / 'station_status'}. "
            "Corre: python scripts/capture_gbfs.py --una-vez"
        )

    infos = []
    for path in info_files:
        frame = pq.read_table(path, columns=["snapshot_ts", "station_id", "short_name", "capacity"]).to_pandas()
        if len(frame):
            infos.append(frame)
    if not infos:
        raise GbfsFrozenError("Las fotos de station_information están vacías.")
    info_times = np.array([frame["snapshot_ts"].iloc[0].value for frame in infos], dtype=np.int64)
    if (np.diff(info_times) < 0).any():
        raise GbfsFrozenError("Las fotos de station_information no quedaron en orden de captura.")

    photos = []
    for path in status_files:
        frame = pq.read_table(
            path, columns=["snapshot_ts", "feed_last_updated", "station_id", "num_bikes_available", "is_installed"]
        ).to_pandas()
        if len(frame):
            photos.append(frame)
    if not photos:
        raise GbfsFrozenError("Las fotos de station_status están vacías.")
    photo_times = np.array([frame["snapshot_ts"].iloc[0].value for frame in photos], dtype=np.int64)
    # Estado de información vigente en cada foto: el último capturado antes o al mismo tiempo. Una foto
    # anterior al primer estado usa el primero.
    state_of = np.clip(np.searchsorted(info_times, photo_times, side="right") - 1, 0, None)
    weight = np.bincount(state_of, minlength=len(infos))

    # Capacidad más frecuente, pesando cada estado por sus fotos; en un empate, la más grande.
    votes = []
    for frame, w in zip(infos, weight):
        if w == 0:
            continue
        part = frame.loc[frame["short_name"].notna() & frame["capacity"].notna(), ["short_name", "capacity"]].copy()
        part["short_name"] = part["short_name"].str.strip()
        part = part.drop_duplicates("short_name")  # no pasa hoy; si pasa, queda la primera
        part["peso"] = int(w)
        votes.append(part)
    votes = pd.concat(votes, ignore_index=True)
    votes["capacity"] = votes["capacity"].astype("int64")
    tally = votes.groupby(["short_name", "capacity"], as_index=False)["peso"].sum()
    tally = tally.sort_values(["short_name", "peso", "capacity"], ascending=[True, False, False], kind="stable")
    capacity = tally.drop_duplicates("short_name").set_index("short_name")["capacity"]

    # Ocupación en la hora en que empieza la simulación.
    zone = ZoneInfo(TZ)
    ratios, in_hour = [], 0
    for frame, state in zip(photos, state_of):
        local = frame["feed_last_updated"].iloc[0].tz_convert(zone)
        if local.hour != hour:
            continue
        in_hour += 1
        bridge = infos[state].loc[infos[state]["short_name"].notna(), ["station_id", "short_name"]].copy()
        bridge["short_name"] = bridge["short_name"].str.strip()
        bridge = bridge.drop_duplicates("station_id")
        rows = frame.loc[frame["is_installed"].fillna(False) & frame["num_bikes_available"].notna()]
        rows = rows.merge(bridge, on="station_id", how="inner").drop_duplicates("short_name")
        places = rows["short_name"].map(capacity)
        keep = places.notna() & (places > 0)
        ratios.append(
            pd.DataFrame(
                {
                    "short_name": rows.loc[keep, "short_name"],
                    "ratio": rows.loc[keep, "num_bikes_available"].astype("float64") / places[keep].astype("float64"),
                }
            )
        )
    table = pd.DataFrame({"short_name": capacity.index.astype(str), "capacity": capacity.to_numpy()})
    if ratios and sum(len(part) for part in ratios):
        by_station = pd.concat(ratios, ignore_index=True).groupby("short_name")["ratio"].agg(["mean", "size"])
        table["ocupacion_inicial"] = table["short_name"].map(by_station["mean"]).clip(0.0, 1.0).round(6)
        table["fotos"] = table["short_name"].map(by_station["size"]).fillna(0).astype("int64")
    else:
        table["ocupacion_inicial"] = np.nan
        table["fotos"] = 0
    table = table.sort_values("short_name", kind="stable").reset_index(drop=True)
    table["capacity"] = table["capacity"].astype("Int64")
    meta = {
        "captura_desde_utc": _utc(pd.Timestamp(photo_times.min(), tz="UTC")),
        "captura_hasta_utc": _utc(pd.Timestamp(photo_times.max(), tz="UTC")),
        "fotos_de_estado": int(len(photos)),
        "estados_de_informacion": int(len(infos)),
        "hora_local_del_inventario": int(hour),
        "fotos_en_esa_hora": int(in_hour),
    }
    return Frozen(table=table[list(COLUMNS)], meta=meta)


def render(frozen: Frozen) -> str:
    """El texto del archivo congelado: encabezado con `#` y una fila por estación."""
    meta = frozen.meta
    hour = int(meta["hora_local_del_inventario"])
    lines = [
        "# Capacidad e inventario inicial CONGELADOS de la captura de GBFS.",
        "# Lo escribe scripts/freeze_gbfs.py. No se edita a mano: cambiarlo cambia la limpieza y la simulación.",
        f"# Captura del {meta['captura_desde_utc']} al {meta['captura_hasta_utc']} (UTC): "
        f"{meta['fotos_de_estado']} fotos de station_status y {meta['estados_de_informacion']} estados de "
        "station_information.",
        "# OJO: la captura NO es de los días que se simulan.",
        "# capacity: la más frecuente entre las fotos (si empatan, la más grande).",
        f"# ocupacion_inicial: promedio de num_bikes_available / capacity en las {meta['fotos_en_esa_hora']} "
        f"fotos de las {hour:02d}:00 a las {hour:02d}:59, hora de Nueva York (solo con la estación instalada), "
        "entre 0 y 1.",
        "# fotos: cuántas fotos entraron en ese promedio.",
    ]
    lines += [f"# {key}: {meta[key]}" for key in META_KEYS]
    body = frozen.table.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    return "\n".join(lines) + "\n" + body


def load_frozen(path: str | Path = FROZEN_FILE) -> Frozen:
    """Lee el archivo congelado. Los ids son texto: `6131.1` y `6131.10` son estaciones distintas."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GbfsFrozenError(
            f"No se pudo abrir {path}: {error}. Para crearlo corre: python scripts/freeze_gbfs.py"
        ) from None
    meta: dict = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        key, _, value = line[1:].partition(":")
        if key.strip() in META_KEYS:
            value = value.strip()
            meta[key.strip()] = int(value) if value.isdigit() else value
    try:
        table = pd.read_csv(
            io.StringIO(text), comment="#", dtype={"short_name": "str"}, keep_default_na=False, na_values=[""]
        )
    except (ValueError, pd.errors.ParserError) as error:
        raise GbfsFrozenError(f"{path} no se pudo leer como CSV: {error}") from None
    missing = [column for column in COLUMNS if column not in table.columns]
    if missing:
        raise GbfsFrozenError(f"A {path} le faltan columnas: {missing}.")
    table = table[list(COLUMNS)].copy()
    problems = []
    if table["short_name"].isna().any() or table["short_name"].duplicated().any():
        problems.append("short_name debe venir completo y sin repetidos")
    capacity = pd.to_numeric(table["capacity"], errors="coerce")
    if capacity.isna().any() or (capacity < 0).any() or (capacity != np.floor(capacity)).any():
        problems.append("capacity debe ser un entero mayor o igual a 0")
    occupancy = pd.to_numeric(table["ocupacion_inicial"], errors="coerce")
    if (occupancy.isna() != table["ocupacion_inicial"].isna()).any() or ((occupancy < 0) | (occupancy > 1)).any():
        problems.append("ocupacion_inicial debe estar entre 0 y 1, o ir vacía")
    if problems:
        raise GbfsFrozenError(f"{path} no es válido:\n- " + "\n- ".join(problems))
    table["short_name"] = table["short_name"].astype(str)
    table["capacity"] = capacity.astype("int64").astype("Int64")
    table["ocupacion_inicial"] = occupancy.astype("float64")
    table["fotos"] = pd.to_numeric(table["fotos"], errors="coerce").fillna(0).astype("int64")
    return Frozen(table=table, meta=meta)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de scripts/freeze_gbfs.py. Devuelve 0 si escribió el archivo y 1 si no."""
    import argparse

    from src.paths import DATA_RAW, ROOT

    parser = argparse.ArgumentParser(
        prog="freeze_gbfs.py",
        add_help=False,
        usage="python scripts/freeze_gbfs.py [opciones]",
        description=(
            "Resume la captura de GBFS en config/gbfs_congelado.csv: capacidad más frecuente e inventario "
            "inicial observado por estación. Solo lee la captura."
        ),
    )
    options = parser.add_argument_group("opciones")
    options.add_argument("--gbfs", metavar="DIR", default=str(DATA_RAW / "gbfs"),
                         help="carpeta de la captura (por defecto data/raw/gbfs)")
    options.add_argument("--salida", metavar="FILE", default=str(FROZEN_FILE),
                         help="archivo que se escribe (por defecto config/gbfs_congelado.csv)")
    options.add_argument("--reemplazar", action="store_true",
                         help="permite pisar un archivo que ya existe. Cambia la limpieza y la simulación")
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    args = parser.parse_args(argv)
    out = Path(args.salida).expanduser()
    if out.exists() and not args.reemplazar:
        print(
            f"ERROR. {out} ya existe. Está congelado a propósito: volver a escribirlo cambia las capacidades y "
            "los resultados de la simulación. Si de verdad quieres rehacerlo, agrega --reemplazar."
        )
        return 1
    try:
        frozen = freeze(Path(args.gbfs).expanduser())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(frozen), encoding="utf-8")
        again = load_frozen(out)  # se lee de vuelta: lo escrito debe ser lo calculado
    except (GbfsFrozenError, OSError) as error:
        print(f"ERROR. {error}")
        return 1
    same = again.table["short_name"].tolist() == frozen.table["short_name"].tolist() and (
        again.table["capacity"].astype("int64").to_numpy() == frozen.table["capacity"].astype("int64").to_numpy()
    ).all()
    if not same:
        print(f"ERROR. {out} no se leyó igual que como se escribió.")
        return 1
    table, meta = frozen.table, frozen.meta
    shown = out.resolve().relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out
    print(f"Captura del {meta['captura_desde_utc']} al {meta['captura_hasta_utc']} (UTC): "
          f"{meta['fotos_de_estado']} fotos de estado, {meta['estados_de_informacion']} estados de información.")
    print(f"Estaciones: {len(table):,}. Con capacidad mayor que 0: {int((table['capacity'] > 0).sum()):,}. "
          f"Con ocupación inicial: {int(table['ocupacion_inicial'].notna().sum()):,} "
          f"({meta['fotos_en_esa_hora']} fotos de las {meta['hora_local_del_inventario']:02d}:00, hora de Nueva York).")
    print(f"Ocupación inicial media: {table['ocupacion_inicial'].mean():.3f}.")
    print("La captura no es de los días que se simulan: es un supuesto declarado.")
    print(f"Escrito: {shown}")
    print("Ahora corre: python scripts/build_tables.py")
    return 0
