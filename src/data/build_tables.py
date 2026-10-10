"""Limpieza de los viajes y tablas reales: stations, station_hour y routes.

Lee los meses crudos de data/raw/trips que indica config/datos.yaml y escribe
en data/processed las tres tablas de contrato, una tabla interna de viajes y
un reporte de la limpieza:

    stations.parquet, station_hour.parquet, routes.parquet
    trips.parquet           (interna, para el simulador; NO es un contrato)
    reports/limpieza.json   (para programas)
    reports/limpieza.md     (para personas)

trips.parquet trae una fila por viaje limpio que sale o llega dentro del
periodo, ordenadas por started_at. Sus columnas están en TRIPS_SCHEMA:

    ride_id            texto
    started_at         instante con zona America/New_York
    ended_at           instante con zona America/New_York
    start_station_id   texto, el id que quedó después de la limpieza
    end_station_id     texto, el id que quedó después de la limpieza
    start_lat, start_lng, end_lat, end_lng   decimales, tal como vienen en el
                       archivo crudo (pueden ir vacíos)

Las coordenadas son las del viaje, sin tocar: una punta sin estación, o con una
estación que no entra a las tablas, conserva las suyas. El simulador las usa
para saber cerca de qué estación empezó o terminó un viaje así.

Una punta va vacía (nula) si el viaje no trae estación, si la estación está
excluida o si es un id sin ningún viaje dentro del periodo (no es estación de
las tablas). Una punta cuya fecha cae fuera del periodo conserva su fecha y su
estación: quien la lee decide qué hacer con ella. Sale de la misma tabla
intermedia que station_hour y routes, así que contar sus viajes por estación y
hora, o por par de estaciones, reproduce esas dos tablas. Eso se comprueba en
cada corrida leyendo de vuelta el archivo ya escrito.

Reglas, en orden. Cada una deja su conteo en el reporte.

1. Se unen los meses. Un mismo ride_id repetido se cuenta una sola vez. Se
   quitan los viajes sin fecha o que terminan antes de empezar.
   Límite conocido: en el cambio de horario de otoño, un viaje que empieza en
   la primera 01:xx y termina en la repetida trae ended_at menor que
   started_at y la limpieza lo quita como fecha imposible (queda contado en
   "terminan antes de empezar"). El periodo actual no tiene cambio de horario;
   si se agrega noviembre, hay que arreglarlo antes.
2. A los ids y nombres de estación se les quitan los espacios de las orillas.
   Un id vacío es un id que falta. Los ids son TEXTO de principio a fin.
3. Decisión 15: dos escrituras de un mismo id (2231.1 y 2231.10, o 5308.04 y
   5308.04_) se unen solo si tienen el mismo nombre y la misma coordenada.
4. Las estaciones excluidas de la configuración no entran a ninguna tabla.
5. Periodo en días locales de Nueva York. Una salida cuenta en la hora de
   started_at y una llegada en la hora de ended_at, cada una con su fecha.
6. station_hour es densa (decisión 12): una fila por hora, con ceros, entre la
   primera y la última hora con viajes de cada estación.
7. stations lleva una fila por estación de station_hour. La capacidad sale de
   config/gbfs_congelado.csv (la capacidad más frecuente de la captura de
   GBFS, congelada con scripts/freeze_gbfs.py). Es la de la captura, no la
   del periodo. Como el archivo está congelado, reconstruir no la cambia.
8. routes cuenta los viajes entre pares de estaciones.

Antes de escribir se hacen comprobaciones (las sumas cuadran con los viajes,
toda fila cruda queda explicada, la hora de tres viajes reales coincide con
un cálculo aparte en Python y trips.parquet reproduce station_hour y routes).
Si una falla, no se escribe nada.

Los datos solo registran demanda ATENDIDA: un cero en station_hour no siempre
es demanda cero (decisión 9).

Uso:  python scripts/build_tables.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.contracts import TS, ContractError, to_contract_table, write_contract
from src.data.datos_config import Datos, DatosError, load_datos
from src.data.gbfs_frozen import FROZEN_FILE, GbfsFrozenError, load_frozen
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE, DATOS_YAML, REPORTS, ROOT, TZ

TABLES = ("stations", "station_hour", "routes")      # las tres tablas de contrato que salen de aquí
# Regla de `station_type` (decisión 13): la consulta de logística. Lee station_hour del tramo de
# entrenamiento y devuelve una fila por estación con su tipo. `main` la aplica; ver `apply_station_types`.
TIPO_ESTACION_SQL = ROOT / "notebooks" / "b" / "tipo_estacion.sql"
# Así nombra la consulta a station_hour: se cambia por el archivo recién construido.
_TIPO_SQL_FUENTE = "read_parquet('data/processed/station_hour.parquet')"
_TIPO_SQL_FECHA = r"TIMESTAMPTZ\s+'(\d{4}-\d{2}-\d{2}) 00:00:00'\s+AS\s+%s\b"
SIN_TIPO = "sin tipo"
REPORT_FOLDER = REPORTS.relative_to(DATA_PROCESSED)  # "reports", dentro de la carpeta de salida
REPORT_NAME = "limpieza"
WORK_FOLDER = "_tmp_build_tables"                    # temporal dentro de la salida; se borra al terminar

# Tabla interna de viajes para el simulador (trips.parquet). NO es uno de los
# seis contratos de src/contracts.py: solo la escribe este módulo y solo la lee
# el simulador. Quien la lea puede comparar su esquema contra TRIPS_SCHEMA.
TRIPS_NAME = "trips"
TRIPS_SCHEMA = pa.schema(
    [
        ("ride_id", pa.string()),
        ("started_at", TS),                 # instante de salida, hora local de NY
        ("ended_at", TS),                   # instante de llegada, hora local de NY
        ("start_station_id", pa.string()),  # nulo: sin estación, excluida o sin viajes en el periodo
        ("end_station_id", pa.string()),    # nulo: sin estación, excluida o sin viajes en el periodo
        ("start_lat", pa.float64()),        # coordenadas crudas del viaje; pueden ir vacías
        ("start_lng", pa.float64()),
        ("end_lat", pa.float64()),
        ("end_lng", pa.float64()),
    ]
)
# Filas por grupo del Parquet. Como va ordenado por started_at, leer unos
# pocos días con un filtro sobre esa columna solo abre los grupos que tocan.
TRIPS_ROWS_PER_GROUP = 500_000

# Para el reporte: racha de horas seguidas sin ningún viaje a partir de la cual
# un cero es sospechoso (estación cerrada o fuera de servicio, no demanda cero).
LONG_ZERO_RUN_HOURS = 72

HOUR_US = 3_600_000_000  # microsegundos de una hora

# Columnas del Parquet crudo que usa la limpieza y la familia de tipo que deben traer.
RAW_COLUMNS = {
    "ride_id": "texto",
    "started_at": "fecha",
    "ended_at": "fecha",
    "start_station_name": "texto",
    "start_station_id": "texto",
    "end_station_name": "texto",
    "end_station_id": "texto",
    "start_lat": "decimal",
    "start_lng": "decimal",
    "end_lat": "decimal",
    "end_lng": "decimal",
    "member_casual": "texto",
}

# Lo que se recorta en las orillas: todo lo que str.strip() de Python considera espacio
# (también el espacio duro U+00A0), para que coincida con el recorte del short_name de GBFS.
_EDGES_RE = r"^[\s\p{Z}\x{0b}\x{1c}-\x{1f}\x{85}]+|[\s\p{Z}\x{0b}\x{1c}-\x{1f}\x{85}]+$"


class BuildError(Exception):
    """La limpieza no pudo terminar o una comprobación falló. El mensaje dice por qué."""


# --------------------------------------------------------------------------- #
# Piezas pequeñas, sin DuckDB (se prueban solas)
# --------------------------------------------------------------------------- #

def hour_sql(column: str) -> str:
    """SQL del inicio de la hora de una fecha local sin zona, como instante.

    Es la única forma que se usa para construir ts_hour. Nunca un CAST a
    TIMESTAMPTZ a secas: usaría la zona de la computadora y movería cada hora.
    """
    return f"date_trunc('hour', {column}) AT TIME ZONE '{TZ}'"


def instant_sql(column: str) -> str:
    """SQL de una fecha local sin zona como instante, sin recortarla a la hora.

    Es la única forma que se usa para started_at y ended_at de trips.parquet.
    Igual que en `hour_sql`: nunca un CAST a TIMESTAMPTZ a secas.
    """
    return f"{column} AT TIME ZONE '{TZ}'"


def local_instant(naive: datetime) -> datetime:
    """Un momento local de Nueva York (sin zona) como instante en UTC.

    Python puro (zoneinfo), sin DuckDB: sirve para comprobar lo que hace el SQL.
    En un cambio de horario hay dos lecturas posibles y se toma la más tardía,
    igual que DuckDB: la 01:xx repetida de otoño es la segunda (-05:00) y la
    02:xx que no existe en primavera se lee como 03:xx (-04:00).
    """
    naive = naive.replace(tzinfo=None)
    zone = ZoneInfo(TZ)
    return max(naive.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc) for fold in (0, 1))


def local_hour_instant(naive: datetime) -> datetime:
    """Inicio de la hora de un momento local de Nueva York, como instante en UTC.

    Es `local_instant` de la hora en punto: la 01:xx repetida de otoño cae en
    la segunda 01:00 (-05:00) y la 02:xx que no existe en primavera cae en las
    03:00 (-04:00).
    """
    return local_instant(naive.replace(minute=0, second=0, microsecond=0))


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def epoch_us(instant: datetime) -> int:
    """Microsegundos desde 1970 de un instante con zona. Entero exacto, sin pasar por decimales."""
    return (instant - _EPOCH) // timedelta(microseconds=1)


def normalise_name(name: str | None) -> str | None:
    """Nombre para comparar: sin espacios de sobra y sin distinguir mayúsculas."""
    if name is None:
        return None
    return " ".join(name.split()).casefold()


def candidate_form(station_id: str) -> str:
    """Forma común de un id para detectar variantes. No es el id final.

    Quita los guiones bajos del final y completa a dos decimales un id con un
    solo decimal: "2231.1" y "2231.10" dan "2231.10"; "5308.04_" da "5308.04".
    """
    text = station_id.rstrip("_")
    if re.fullmatch(r"\d+\.\d", text):
        text += "0"
    return text


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos (fórmula de haversine)."""
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def exclusion_rule(station_id: str, datos: Datos) -> str | None:
    """Regla de la configuración que excluye a un id (ya recortado), o None."""
    for prefix in datos.prefijos_excluidos:
        if station_id.startswith(prefix):
            return f"prefijo {prefix}"
    if station_id in datos.ids_excluidos:
        return f"id {station_id}"
    return None


@dataclass(frozen=True)
class RawId:
    """Un id de estación tal como viene en los viajes, ya recortado."""

    raw_id: str
    departures: int      # salidas dentro del periodo
    arrivals: int        # llegadas dentro del periodo
    name: str | None     # nombre más frecuente dentro del periodo
    lat: float | None    # coordenada más frecuente dentro del periodo
    lon: float | None

    @property
    def trip_ends(self) -> int:
        return self.departures + self.arrivals


@dataclass
class IdPlan:
    """Qué pasa con cada id crudo."""

    station_of: dict[str, str | None]  # id crudo -> id final (None si no entra a las tablas)
    state_of: dict[str, str]           # "ok", "excluida" o "inactiva" (sin viajes en el periodo)
    groups: list[dict]                 # grupos de variantes, para el reporte
    excluded: list[dict]               # estaciones excluidas con viajes en el periodo, por regla


def _why_not_merged(variant: RawId, anchor: RawId, max_m: float) -> tuple[float | None, list[str]]:
    """Distancia a la variante que queda y motivos para NO unir (lista vacía = se une)."""
    reasons = []
    if variant.name is None or anchor.name is None:
        reasons.append("falta el nombre")
    elif normalise_name(variant.name) != normalise_name(anchor.name):
        reasons.append("nombre distinto")
    distance = None
    if None in (variant.lat, variant.lon, anchor.lat, anchor.lon):
        reasons.append("faltan coordenadas")
    else:
        distance = haversine_m(variant.lat, variant.lon, anchor.lat, anchor.lon)
        if distance > max_m:
            reasons.append(f"coordenadas a {distance:.1f} m (máximo {max_m:g} m)")
    return distance, reasons


def plan_ids(raw_ids: list[RawId], datos: Datos) -> IdPlan:
    """Aplica las reglas 3 y 4: excluye estaciones y une variantes de un mismo id.

    Las variantes se unen solo si coinciden nombre y coordenadas (decisión 15).
    Queda el id con más salidas y llegadas en el periodo; en un empate, el que
    ordena al final ("2231.10" le gana a "2231.1"). Cada variante se compara
    contra el id que queda: la que no coincide sigue como estación aparte con
    su propio id. Un id sin gemelo nunca se cambia. Si dos escrituras que no
    se unieron coinciden entre sí, lanza BuildError: ese caso se revisa a mano.
    """
    if len({r.raw_id for r in raw_ids}) != len(raw_ids):
        raise BuildError("Error interno: hay ids crudos repetidos al planear las estaciones.")
    station_of: dict[str, str | None] = {}
    state_of: dict[str, str] = {}
    by_rule: dict[str, list[RawId]] = defaultdict(list)
    families: dict[str, list[RawId]] = defaultdict(list)
    for r in sorted(raw_ids, key=lambda r: r.raw_id):
        rule = exclusion_rule(r.raw_id, datos)
        if rule:
            station_of[r.raw_id], state_of[r.raw_id] = None, "excluida"
            if r.trip_ends:
                by_rule[rule].append(r)
        elif r.trip_ends == 0:
            station_of[r.raw_id], state_of[r.raw_id] = None, "inactiva"
        else:
            station_of[r.raw_id], state_of[r.raw_id] = r.raw_id, "ok"
            families[candidate_form(r.raw_id)].append(r)

    groups = []
    for form in sorted(families):
        variants = families[form]
        if len(variants) < 2:
            continue  # sin gemelo: el id se queda exactamente como está
        anchor = max(variants, key=lambda r: (r.trip_ends, r.raw_id))
        rows = []
        for variant in variants:
            distance, reasons = (None, []) if variant is anchor else _why_not_merged(
                variant, anchor, datos.distancia_maxima_m
            )
            if variant is not anchor and not reasons:
                station_of[variant.raw_id] = anchor.raw_id
            rows.append(
                {
                    "id": variant.raw_id,
                    "nombre": variant.name,
                    "lat": variant.lat,
                    "lon": variant.lon,
                    "salidas": variant.departures,
                    "llegadas": variant.arrivals,
                    "es_el_id_que_queda": variant is anchor,
                    "distancia_m": None if distance is None else round(distance, 1),
                    "se_une": variant is not anchor and not reasons,
                    "motivo": " y ".join(reasons) or None,
                }
            )
        # Con tres escrituras o más, dos que no se unieron al id que queda pueden coincidir
        # entre sí. La regla no cubre ese caso (no ocurre en los datos de hoy): se detiene.
        apart = [v for v in variants if v is not anchor and station_of[v.raw_id] == v.raw_id]
        for i, one in enumerate(apart):
            for other in apart[i + 1:]:
                if not _why_not_merged(one, other, datos.distancia_maxima_m)[1]:
                    raise BuildError(
                        f"Los ids {one.raw_id} y {other.raw_id} tienen el mismo nombre y la misma coordenada, "
                        f"pero ninguno coincide con {anchor.raw_id}, el id con más viajes de su grupo. "
                        "La unión de variantes no cubre ese caso: hay que revisarlo a mano."
                    )
        groups.append(
            {
                "forma_comun": form,
                "id_que_queda": anchor.raw_id,
                "unido": all(row["se_une"] or row["es_el_id_que_queda"] for row in rows),
                "variantes": rows,
            }
        )

    excluded = [
        {
            "regla": rule,
            "estaciones": len(found),
            "salidas": sum(r.departures for r in found),
            "llegadas": sum(r.arrivals for r in found),
            "ids": [
                {"id": r.raw_id, "nombre": r.name, "salidas": r.departures, "llegadas": r.arrivals} for r in found
            ],
        }
        for rule, found in sorted(by_rule.items())
    ]
    return IdPlan(station_of, state_of, groups, excluded)


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #

def _short(path: Path | str) -> str:
    """Ruta relativa a la raíz del repositorio, para mensajes y reportes más cortos."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _clean(column: str) -> str:
    """SQL que recorta las orillas de un texto y deja nulo el que queda vacío."""
    return f"nullif(regexp_replace({column}, '{_EDGES_RE}', '', 'g'), '')"


def _in_period(column: str) -> str:
    """SQL: la fecha local cae dentro del periodo (fin exclusivo). Usa $ini y $fin."""
    return f"({column} >= $ini AND {column} < $fin)"


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _was_interrupted(error: Exception) -> bool:
    # DuckDB convierte Ctrl+C en un RuntimeError "Query interrupted".
    if isinstance(error, duckdb.InterruptException):
        return True
    return type(error) is RuntimeError and str(error).strip().lower().startswith("query interrupted")


def check_out_dir(out_dir: Path) -> None:
    """La salida no puede ser data/raw ni data/sample (ni estar dentro)."""
    resolved = Path(out_dir).resolve()
    for folder, holds in ((DATA_RAW, "las descargas crudas"), (DATA_SAMPLE, "los datos sintéticos de muestra")):
        if resolved.is_relative_to(folder.resolve()):
            raise BuildError(
                f"La carpeta de salida no puede ser {_short(folder)} ni estar dentro: ahí van solo {holds}."
            )


def raw_files(datos: Datos, raw_dir: Path) -> dict[str, Path]:
    """Archivo de cada mes de la configuración. Falla si falta alguno o no trae las columnas."""
    files = {month: raw_dir / f"{month}.parquet" for month in datos.meses_crudos}
    missing = [month for month, path in files.items() if not path.is_file()]
    if missing:
        raise BuildError(
            f"Faltan meses crudos en {raw_dir}: {', '.join(missing)}.\n"
            f"Descárgalos con: python scripts/download_trips.py {' '.join(missing)}"
        )
    problems = []
    for month, path in files.items():
        schema = pq.read_schema(path)
        for column, kind in RAW_COLUMNS.items():
            if column not in schema.names:
                problems.append(f"{month}: falta la columna {column}")
                continue
            kind_found = schema.field(column).type
            ok = {
                "texto": pa.types.is_string(kind_found)
                or pa.types.is_large_string(kind_found)
                or pa.types.is_string_view(kind_found),
                "fecha": pa.types.is_timestamp(kind_found) and kind_found.tz is None,
                "decimal": pa.types.is_floating(kind_found),
            }[kind]
            if not ok:
                problems.append(f"{month}: {column} es {kind_found} y debe ser {kind}")
    if problems:
        raise BuildError(
            "Los Parquet crudos no tienen el formato de scripts/download_trips.py (ids como texto, "
            "fechas sin zona):\n- " + "\n- ".join(problems)
        )
    return files


def load_capacity(gbfs_file: Path | None, datos: Datos) -> tuple[dict[str, int | None], dict]:
    """Capacidad por short_name del archivo congelado de GBFS (config/gbfs_congelado.csv).

    Devuelve (capacidades, datos del archivo). Sin archivo, las capacidades van
    vacías y `foto` es None. Un archivo que existe y no se puede leer es un error.
    """
    path = Path(gbfs_file) if gbfs_file else None
    if path is None or not path.exists():
        return {}, {"foto": None, "archivo": _short(path) if path else None}
    try:
        frozen = load_frozen(path)
    except GbfsFrozenError as error:
        raise BuildError(str(error)) from error
    capacity = {
        str(name): int(value) for name, value in zip(frozen.table["short_name"], frozen.table["capacity"])
    }
    meta = frozen.meta
    until = meta.get("captura_hasta_utc")
    taken = datetime.fromisoformat(until.replace("Z", "+00:00")) if isinstance(until, str) else None
    info = {
        "foto": path.name,  # el nombre del archivo congelado
        "archivo": _short(path),
        "captura_desde_utc": meta.get("captura_desde_utc"),
        "captura_hasta_utc": until,
        "fotos_de_estado": meta.get("fotos_de_estado"),
        "estados_de_informacion": meta.get("estados_de_informacion"),
        # Días (locales de Nueva York) entre el último día del periodo y el fin de la captura.
        "dias_despues_del_periodo": (
            (taken.astimezone(ZoneInfo(TZ)).date() - datos.periodo.fin).days if taken else None
        ),
        "estaciones_en_el_archivo": len(capacity),
    }
    return capacity, info


# --------------------------------------------------------------------------- #
# Pasos en DuckDB
# --------------------------------------------------------------------------- #

def _connect(work: Path) -> duckdb.DuckDBPyConnection:
    """Base temporal en disco, con la zona de Nueva York en la sesión."""
    connection = duckdb.connect(str(work / "build.duckdb"))
    try:
        connection.execute(f"SET TimeZone = '{TZ}'")
        connection.execute(f"SET temp_directory = {_sql_text(work / 'duckdb_tmp')}")
    except Exception:
        connection.close()
        raise
    return connection


def _load_trips(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> dict:
    """Reglas 1 y 2: une los meses, recorta textos, quita duplicados y fechas inválidas.

    Deja la tabla `trips` y devuelve los conteos.
    """
    con.execute(
        """
        CREATE TABLE loaded (
            source_month VARCHAR, ride_id VARCHAR, started_at TIMESTAMP, ended_at TIMESTAMP,
            start_id VARCHAR, start_name VARCHAR, start_lat DOUBLE, start_lng DOUBLE,
            end_id VARCHAR, end_name VARCHAR, end_lat DOUBLE, end_lng DOUBLE, member_casual VARCHAR
        )
        """
    )
    def trimmed(*columns: str) -> str:
        """SQL: cuántos valores cambian al recortarlos (sin contar los que quedan vacíos)."""
        return " + ".join(f"count(*) FILTER (WHERE {_clean(c)} <> {c})" for c in columns)

    def blank(*columns: str) -> str:
        """SQL: cuántos valores traen solo espacios."""
        return " + ".join(f"count(*) FILTER (WHERE {c} IS NOT NULL AND {_clean(c)} IS NULL)" for c in columns)

    per_file = []
    hygiene = {"ids_con_espacios": 0, "ids_vacios": 0, "nombres_con_espacios": 0}
    for month, path in files.items():
        # Conteo de filas y de textos sucios, directo sobre el archivo.
        rows, with_spaces, empty, names = con.execute(
            f"""
            SELECT count(*),
                   {trimmed('start_station_id', 'end_station_id')},
                   {blank('start_station_id', 'end_station_id')},
                   {trimmed('start_station_name', 'end_station_name')}
            FROM read_parquet($path)
            """,
            {"path": str(path)},
        ).fetchone()
        per_file.append({"mes": month, "archivo": _short(path), "filas": rows})
        hygiene["ids_con_espacios"] += with_spaces
        hygiene["ids_vacios"] += empty
        hygiene["nombres_con_espacios"] += names
        con.execute(
            f"""
            INSERT INTO loaded
            SELECT $month, ride_id, started_at, ended_at,
                   {_clean('start_station_id')}, {_clean('start_station_name')}, start_lat, start_lng,
                   {_clean('end_station_id')}, {_clean('end_station_name')}, end_lat, end_lng, member_casual
            FROM read_parquet($path)
            """,
            {"month": month, "path": str(path)},
        )
    read = sum(f["filas"] for f in per_file)
    loaded = con.execute("SELECT count(*) FROM loaded").fetchone()[0]

    # Duplicados contados aparte, para comprobar después el paso que los quita.
    repeated_ids, extra_rows = con.execute(
        """
        SELECT count(*), coalesce(sum(n - 1), 0)
        FROM (SELECT count(*) AS n FROM loaded WHERE ride_id IS NOT NULL GROUP BY ride_id HAVING count(*) > 1)
        """
    ).fetchone()
    # De cada ride_id queda una sola fila: la del mes más antiguo y, si empatan,
    # la primera al ordenar por el resto de las columnas (orden fijo).
    con.execute(
        """
        CREATE TABLE trips AS
        SELECT * FROM loaded
        QUALIFY ride_id IS NULL OR row_number() OVER (
            PARTITION BY ride_id
            ORDER BY source_month, started_at, ended_at, start_id, end_id, member_casual,
                     start_name, end_name, start_lat, start_lng, end_lat, end_lng
        ) = 1
        """
    )
    con.execute("DROP TABLE loaded")
    after_dedup = con.execute("SELECT count(*) FROM trips").fetchone()[0]

    no_date, backwards = con.execute(
        """
        SELECT count(*) FILTER (WHERE started_at IS NULL OR ended_at IS NULL),
               count(*) FILTER (WHERE ended_at < started_at)
        FROM trips
        """
    ).fetchone()
    con.execute("DELETE FROM trips WHERE started_at IS NULL OR ended_at IS NULL OR ended_at < started_at")
    kept = con.execute("SELECT count(*) FROM trips").fetchone()[0]
    return {
        "archivos": per_file,
        "leidas": read,
        "cargadas": loaded,
        "ride_id_repetidos": repeated_ids,
        "duplicadas_quitadas": loaded - after_dedup,
        "duplicadas_contadas_aparte": int(extra_rows),
        "sin_fecha": no_date,
        "terminan_antes_de_empezar": backwards,
        "fechas_invalidas_quitadas": after_dedup - kept,
        "conservadas": kept,
        "texto": hygiene,
    }


def _raw_ids(con: duckdb.DuckDBPyConnection, period: dict) -> list[RawId]:
    """Cada id crudo con sus salidas, llegadas, nombre y coordenada dentro del periodo.

    Deja la tabla `end_counts` (conteos por id, nombre y coordenada), que
    después sirve para el nombre y la coordenada de las estaciones ya unidas.
    """
    con.execute(
        f"""
        CREATE TABLE end_counts AS
        SELECT side, id, name, lat, lon, count(*) AS n
        FROM (
            SELECT 'start' AS side, start_id AS id, start_name AS name, start_lat AS lat, start_lng AS lon
            FROM trips WHERE start_id IS NOT NULL AND {_in_period('started_at')}
            UNION ALL
            SELECT 'end', end_id, end_name, end_lat, end_lng
            FROM trips WHERE end_id IS NOT NULL AND {_in_period('ended_at')}
        )
        GROUP BY ALL
        """,
        period,
    )
    counts = {
        raw_id: (departures, arrivals)
        for raw_id, departures, arrivals in con.execute(
            """
            SELECT id, coalesce(sum(n) FILTER (WHERE side = 'start'), 0)::BIGINT,
                   coalesce(sum(n) FILTER (WHERE side = 'end'), 0)::BIGINT
            FROM end_counts GROUP BY id
            """
        ).fetchall()
    }
    # Todos los ids de los viajes, también los que solo aparecen fuera del
    # periodo. Esos quedan con cero viajes y no serán estaciones de las tablas.
    every_id = con.execute(
        """
        SELECT start_id FROM trips WHERE start_id IS NOT NULL
        UNION
        SELECT end_id FROM trips WHERE end_id IS NOT NULL
        """
    ).fetchall()
    attributes = _top_attributes(con, "SELECT id AS key, name, lat, lon, n FROM end_counts")
    raw_ids = []
    for (raw_id,) in every_id:
        departures, arrivals = counts.get(raw_id, (0, 0))
        name, lat, lon = attributes.get(raw_id, (None, None, None))
        raw_ids.append(RawId(raw_id, departures, arrivals, name, lat, lon))
    return raw_ids


def _top_attributes(con: duckdb.DuckDBPyConnection, source_sql: str) -> dict[str, tuple]:
    """Nombre y coordenada más frecuentes de cada `key` de `source_sql`.

    Empates: el nombre que va primero en orden alfabético y la coordenada
    menor. Así el resultado no cambia entre corridas.
    """
    names = dict(
        con.execute(
            f"""
            SELECT key, name
            FROM (SELECT key, name, sum(n) AS n FROM ({source_sql}) WHERE name IS NOT NULL GROUP BY key, name)
            QUALIFY row_number() OVER (PARTITION BY key ORDER BY n DESC, name) = 1
            """
        ).fetchall()
    )
    coordinates = {
        key: (lat, lon)
        for key, lat, lon in con.execute(
            f"""
            SELECT key, lat, lon
            FROM (
                SELECT key, lat, lon, sum(n) AS n FROM ({source_sql})
                WHERE lat IS NOT NULL AND lon IS NOT NULL GROUP BY key, lat, lon
            )
            QUALIFY row_number() OVER (PARTITION BY key ORDER BY n DESC, lat, lon) = 1
            """
        ).fetchall()
    }
    return {
        key: (names.get(key), *coordinates.get(key, (None, None))) for key in set(names) | set(coordinates)
    }


def _resolve(con: duckdb.DuckDBPyConnection, plan: IdPlan, period: dict) -> None:
    """Deja la tabla `resolved`: cada viaje con su estación final en cada punta.

    Estado de cada punta: 'ok', 'excluida', 'inactiva' (id sin viajes dentro
    del periodo), 'sin_estacion' (id vacío) o 'sin_mapa' (no debe ocurrir).
    """
    ids = sorted(plan.station_of)
    id_map = pa.table(
        {
            "raw_id": pa.array(ids, pa.string()),
            "station_id": pa.array([plan.station_of[i] for i in ids], pa.string()),
            "state": pa.array([plan.state_of[i] for i in ids], pa.string()),
        }
    )
    con.register("id_map_arrow", id_map)
    con.execute("CREATE TABLE id_map AS SELECT * FROM id_map_arrow")
    con.unregister("id_map_arrow")
    con.execute(
        f"""
        CREATE TABLE resolved AS
        SELECT t.ride_id, t.source_month, t.started_at, t.ended_at, t.member_casual,
               {_in_period('t.started_at')} AS start_in,
               {_in_period('t.ended_at')} AS end_in,
               CASE WHEN t.start_id IS NULL THEN 'sin_estacion' ELSE coalesce(s.state, 'sin_mapa') END AS start_state,
               s.station_id AS start_station,
               CASE WHEN t.end_id IS NULL THEN 'sin_estacion' ELSE coalesce(e.state, 'sin_mapa') END AS end_state,
               e.station_id AS end_station,
               t.start_lat, t.start_lng, t.end_lat, t.end_lng
        FROM trips t
        LEFT JOIN id_map s ON s.raw_id = t.start_id
        LEFT JOIN id_map e ON e.raw_id = t.end_id
        """,
        period,
    )


def _station_hour_table(con: duckdb.DuckDBPyConnection) -> pa.Table:
    """Reglas 5 y 6: salidas y llegadas por estación y hora, con la tabla ya densa.

    Deja también la tabla `active_hours` (solo las horas con algún viaje).
    """
    con.execute(
        f"""
        CREATE TABLE active_hours AS
        WITH dep AS (
            SELECT start_station AS station_id, {hour_sql('started_at')} AS ts_hour,
                   count(*) AS departures,
                   count(*) FILTER (WHERE member_casual = 'member') AS departures_member,
                   count(*) FILTER (WHERE member_casual = 'casual') AS departures_casual
            FROM resolved WHERE start_in AND start_state = 'ok'
            GROUP BY 1, 2
        ), arr AS (
            SELECT end_station AS station_id, {hour_sql('ended_at')} AS ts_hour, count(*) AS arrivals
            FROM resolved WHERE end_in AND end_state = 'ok'
            GROUP BY 1, 2
        )
        SELECT coalesce(d.station_id, a.station_id) AS station_id,
               coalesce(d.ts_hour, a.ts_hour) AS ts_hour,
               coalesce(d.departures, 0) AS departures,
               coalesce(a.arrivals, 0) AS arrivals,
               coalesce(d.departures_member, 0) AS departures_member,
               coalesce(d.departures_casual, 0) AS departures_casual
        FROM dep d FULL JOIN arr a ON a.station_id = d.station_id AND a.ts_hour = d.ts_hour
        """
    )
    # La rejilla de horas se arma en instantes (microsegundos desde 1970), no
    # en hora de reloj: el día de 25 horas da 25 filas y el de 23 da 23.
    return con.execute(
        f"""
        WITH span AS (
            SELECT station_id, epoch_us(min(ts_hour)) AS first_us, epoch_us(max(ts_hour)) AS last_us
            FROM active_hours GROUP BY station_id
        ), grid AS (
            SELECT station_id, make_timestamptz(unnest(range(first_us, last_us + 1, {HOUR_US}))) AS ts_hour
            FROM span
        )
        SELECT g.station_id, g.ts_hour,
               coalesce(a.departures, 0) AS departures,
               coalesce(a.arrivals, 0) AS arrivals,
               coalesce(a.arrivals, 0) - coalesce(a.departures, 0) AS net_flow,
               coalesce(a.departures_member, 0) AS departures_member,
               coalesce(a.departures_casual, 0) AS departures_casual
        FROM grid g LEFT JOIN active_hours a ON a.station_id = g.station_id AND a.ts_hour = g.ts_hour
        ORDER BY g.station_id, g.ts_hour
        """
    ).to_arrow_table()


def _routes_table(con: duckdb.DuckDBPyConnection) -> pa.Table:
    """Regla 8: viajes que salen dentro del periodo, con estación en las dos puntas."""
    return con.execute(
        """
        SELECT start_station AS start_station_id, end_station AS end_station_id, count(*) AS trips
        FROM resolved
        WHERE start_in AND start_state = 'ok' AND end_state = 'ok'
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).to_arrow_table()


def _write_trips(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Escribe la tabla interna de viajes en `path` y devuelve su tamaño en bytes.

    Sale de `resolved`, la misma tabla de la que salen station_hour y routes:
    un viaje entra si su salida o su llegada cae dentro del periodo, con la
    estación final de cada punta (nula si no entra a las tablas). Va ordenada
    por el instante de salida; el resto del orden solo sirve para que dos
    corridas den el mismo archivo.

    Se escribe por partes, sin armar la tabla completa en memoria (son millones
    de filas). Lo escribe pyarrow y no DuckDB para que el archivo guarde la
    zona de las fechas: DuckDB las dejaría en UTC.
    """
    reader = con.execute(
        f"""
        SELECT r.ride_id,
               {instant_sql('r.started_at')} AS started_at,
               {instant_sql('r.ended_at')} AS ended_at,
               r.start_station AS start_station_id,
               r.end_station AS end_station_id,
               r.start_lat, r.start_lng, r.end_lat, r.end_lng
        FROM resolved r
        WHERE r.start_in OR r.end_in
        ORDER BY 2, 3, 1, 4, 5, 6, 7, 8, 9  -- por posición: el instante de salida ya con zona, no la fecha local
        """
    ).to_arrow_reader(TRIPS_ROWS_PER_GROUP)
    with pq.ParquetWriter(
        path, TRIPS_SCHEMA, compression="zstd", use_dictionary=["start_station_id", "end_station_id"]
    ) as writer:
        for batch in reader:
            # DuckDB entrega las fechas con la zona de su sesión; el instante es
            # el mismo y aquí solo se le pone la etiqueta de Nueva York.
            writer.write_batch(batch.cast(TRIPS_SCHEMA), row_group_size=TRIPS_ROWS_PER_GROUP)
    return path.stat().st_size


def _trips_sides(con: duckdb.DuckDBPyConnection) -> dict:
    """Para el reporte: qué trae cada punta de los viajes que van a trips.parquet."""
    sides = {}
    for side, label in (("start", "inicio"), ("end", "fin")):
        found = dict(
            con.execute(
                f"SELECT {side}_state, count(*) FROM resolved WHERE start_in OR end_in GROUP BY 1"
            ).fetchall()
        )
        sides[label] = {
            "con_estacion": found.pop("ok", 0),
            "sin_estacion": found.pop("sin_estacion", 0),
            "estacion_excluida": found.pop("excluida", 0),
            "estacion_sin_viajes_en_el_periodo": found.pop("inactiva", 0),
            "sin_clasificar": sum(found.values()),  # debe ser cero
        }
    return sides


def _read_back_trips(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    period: dict,
    stations: pa.Table,
    station_hour: pa.Table,
    routes: pa.Table,
    tz_trips: list[dict],
) -> dict:
    """Lee de vuelta el archivo de viajes ya escrito y lo compara con las tres tablas.

    Todo se cuenta sobre el archivo, no sobre la tabla de la que salió: si la
    escritura cambiara una hora, un id o una fila, aquí se nota. Tres cosas se
    calculan aparte, sin el SQL que armó el archivo:

    - los bordes del periodo, como instantes, con zoneinfo en Python;
    - la hora de cada viaje, dividiendo su instante entre una hora (sin
      funciones de fecha, que dependen de la zona de la sesión);
    - el instante de salida y de llegada de los viajes de la comprobación de
      zona horaria, con `local_instant`.
    """
    schema = pq.read_schema(path)
    wrong_columns = sum(
        1
        for position, expected in enumerate(TRIPS_SCHEMA)
        if position >= len(schema) or schema.field(position).name != expected.name
        or schema.field(position).type != expected.type
    ) + max(0, len(schema) - len(TRIPS_SCHEMA))
    if wrong_columns:
        # Con otras columnas el resto de las comprobaciones no se puede ni leer.
        raise BuildError(
            f"{TRIPS_NAME}.parquet no quedó con las columnas acordadas. No se escribió nada.\n"
            f"- esperado: {', '.join(f'{f.name} ({f.type})' for f in TRIPS_SCHEMA)}\n"
            f"- obtenido: {', '.join(f'{f.name} ({f.type})' for f in schema)}"
        )

    edges = {"ini_us": epoch_us(local_instant(period["ini"])), "fin_us": epoch_us(local_instant(period["fin"]))}
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW written AS
        SELECT ride_id, epoch_us(started_at) AS started_us, epoch_us(ended_at) AS ended_us,
               start_station_id, end_station_id
        FROM read_parquet({_sql_text(path)})
        """
    )
    in_period = "({0} >= $ini_us AND {0} < $fin_us)"
    rows, no_ride_id, repeated, with_start, with_end, no_station, start_in, end_in, first_us, last_us = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE ride_id IS NULL),
               count(ride_id) - count(DISTINCT ride_id),
               count(start_station_id),
               count(end_station_id),
               count(*) FILTER (WHERE start_station_id IS NULL AND end_station_id IS NULL),
               count(*) FILTER (WHERE {in_period.format('started_us')}),
               count(*) FILTER (WHERE {in_period.format('ended_us')}),
               min(started_us), max(started_us)
        FROM written
        """,
        edges,
    ).fetchone()

    con.register("stations_check", stations)
    con.register("station_hour_check", station_hour)
    con.register("routes_check", routes)
    try:
        unknown_ids = con.execute(
            """
            SELECT count(*) FROM (
                SELECT start_station_id AS id FROM written WHERE start_station_id IS NOT NULL
                UNION
                SELECT end_station_id FROM written WHERE end_station_id IS NOT NULL
            )
            WHERE id NOT IN (SELECT station_id FROM stations_check)
            """
        ).fetchone()[0]

        def cells_that_differ(station: str, stamp: str, column: str) -> int:
            """Horas de estación donde el conteo del archivo no es el de station_hour."""
            return con.execute(
                f"""
                WITH in_file AS (
                    SELECT {station} AS station_id, {stamp} // {HOUR_US} AS hour, count(*) AS n
                    FROM written
                    WHERE {station} IS NOT NULL AND {in_period.format(stamp)}
                    GROUP BY 1, 2
                ), in_table AS (
                    SELECT station_id, epoch_us(ts_hour) // {HOUR_US} AS hour, {column} AS n
                    FROM station_hour_check WHERE {column} > 0
                )
                SELECT count(*) FROM in_file f FULL JOIN in_table t USING (station_id, hour)
                WHERE f.n IS DISTINCT FROM t.n
                """,
                edges,
            ).fetchone()[0]

        departures_differ = cells_that_differ("start_station_id", "started_us", "departures")
        arrivals_differ = cells_that_differ("end_station_id", "ended_us", "arrivals")
        routes_differ = con.execute(
            f"""
            WITH in_file AS (
                SELECT start_station_id, end_station_id, count(*) AS n
                FROM written
                WHERE start_station_id IS NOT NULL AND end_station_id IS NOT NULL
                  AND {in_period.format('started_us')}
                GROUP BY 1, 2
            )
            SELECT count(*) FROM in_file f FULL JOIN routes_check r USING (start_station_id, end_station_id)
            WHERE f.n IS DISTINCT FROM r.trips
            """,
            edges,
        ).fetchone()[0]
    finally:
        for name in ("stations_check", "station_hour_check", "routes_check"):
            con.unregister(name)

    # Orden: se recorre el archivo en el orden en que quedó escrito.
    out_of_order, previous = 0, None
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(columns=["started_at"]):
            stamps = batch.column(0).cast(pa.int64())
            if not len(stamps):
                continue
            if previous is not None and stamps[0].as_py() < previous:
                out_of_order += 1
            out_of_order += int(pc.sum(pc.less(stamps.slice(1), stamps.slice(0, len(stamps) - 1))).as_py() or 0)
            previous = stamps[-1].as_py()

    def as_text(microseconds: int | None, zone: str = "UTC") -> str | None:
        if microseconds is None:
            return None
        return (_EPOCH + timedelta(microseconds=microseconds)).astimezone(ZoneInfo(zone)).isoformat()

    # Zona horaria: los mismos viajes de `_timezone_check`, ahora con el instante
    # exacto de la salida y de la llegada tal como quedaron en el archivo.
    ride_ids = [t["ride_id"] for t in tz_trips]
    as_loaded, in_file = {}, defaultdict(list)
    if ride_ids:  # sin viajes que revisar no hay consulta (y `_timezone_check` ya lo marcó como problema)
        for ride_id, started_at, ended_at in con.execute(
            "SELECT ride_id, started_at, ended_at FROM resolved WHERE ride_id IN (SELECT unnest($ids))",
            {"ids": ride_ids},
        ).fetchall():
            as_loaded[ride_id] = (started_at, ended_at)
        for ride_id, started_us, ended_us in con.execute(
            "SELECT ride_id, started_us, ended_us FROM written WHERE ride_id IN (SELECT unnest($ids))",
            {"ids": ride_ids},
        ).fetchall():
            in_file[ride_id].append((started_us, ended_us))
    zone_trips = []
    for ride_id in ride_ids:
        started_at, ended_at = as_loaded[ride_id]
        expected = (epoch_us(local_instant(started_at)), epoch_us(local_instant(ended_at)))
        found = in_file.get(ride_id, [])
        zone_trips.append(
            {
                "ride_id": ride_id,
                "started_at": started_at.isoformat(sep=" ", timespec="milliseconds"),
                "ended_at": ended_at.isoformat(sep=" ", timespec="milliseconds"),
                "inicio_utc": as_text(expected[0]),
                "fin_utc": as_text(expected[1]),
                "inicio_en_el_archivo_utc": as_text(found[0][0]) if found else None,
                "fin_en_el_archivo_utc": as_text(found[0][1]) if found else None,
                "coincide": found == [expected],  # una sola fila en el archivo, con los dos instantes
            }
        )
    con.execute("DROP VIEW written")

    return {
        "columnas_distintas": wrong_columns,
        "filas": rows,
        "sin_ride_id": no_ride_id,
        "ride_id_repetidos": repeated,
        "con_estacion_de_inicio": with_start,
        "con_estacion_de_fin": with_end,
        "sin_estacion_en_ninguna_punta": no_station,
        "con_la_salida_dentro_del_periodo": start_in,
        "con_la_llegada_dentro_del_periodo": end_in,
        "primera_salida": as_text(first_us, TZ),
        "ultima_salida": as_text(last_us, TZ),
        "ids_que_no_estan_en_stations": unknown_ids,
        "horas_donde_las_salidas_no_dan_station_hour": departures_differ,
        "horas_donde_las_llegadas_no_dan_station_hour": arrivals_differ,
        "pares_que_no_dan_routes": routes_differ,
        "filas_fuera_de_orden": out_of_order,
        "zona_horaria": zone_trips,
    }


def _stations_table(
    con: duckdb.DuckDBPyConnection, plan: IdPlan, raw_ids: list[RawId], capacity: dict[str, int | None]
) -> tuple[pa.Table, dict]:
    """Regla 7: una fila por estación, con nombre, coordenada y capacidad de GBFS."""
    attributes = _top_attributes(
        con,
        "SELECT m.station_id AS key, c.name, c.lat, c.lon, c.n "
        "FROM end_counts c JOIN id_map m ON m.raw_id = c.id WHERE m.station_id IS NOT NULL",
    )
    volume: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in raw_ids:
        station = plan.station_of[r.raw_id]
        if station is not None:
            volume[station][0] += r.departures
            volume[station][1] += r.arrivals

    rows, no_name, no_coordinates, unmatched = [], [], [], []
    with_capacity = zero_capacity = no_capacity_value = 0
    for station in sorted(volume):
        name, lat, lon = attributes.get(station, (None, None, None))
        if name is None:
            no_name.append(station)
            name = station  # el contrato no admite nombre vacío: se usa el id
        if lat is None or lon is None:
            no_coordinates.append(station)
        value = None
        if station not in capacity:
            unmatched.append(
                {"id": station, "nombre": name, "salidas": volume[station][0], "llegadas": volume[station][1]}
            )
        elif capacity[station] is None:
            no_capacity_value += 1
        elif capacity[station] <= 0:
            zero_capacity += 1  # 0 es "no instalada hoy", no una capacidad real
        else:
            value = int(capacity[station])
            with_capacity += 1
        rows.append((station, name, lat, lon, value))
    if no_coordinates:
        raise BuildError(
            f"{len(no_coordinates)} estaciones no traen coordenadas en ningún viaje del periodo y "
            f"stations no admite lat/lon vacíos. Primeros casos: {no_coordinates[:10]}"
        )
    table = pa.table(
        {
            "station_id": pa.array([r[0] for r in rows], pa.string()),
            "name": pa.array([r[1] for r in rows], pa.string()),
            "lat": pa.array([r[2] for r in rows], pa.float64()),
            "lon": pa.array([r[3] for r in rows], pa.float64()),
            "capacity": pa.array([r[4] for r in rows], pa.int64()),
            # Decisión 13: el tipo lo pone después `apply_station_types`, con la consulta de logística.
            "station_type": pa.array([None] * len(rows), pa.string()),
        }
    )
    unmatched.sort(key=lambda u: (-(u["salidas"] + u["llegadas"]), u["id"]))
    report = {
        "estaciones": len(rows),
        "con_capacidad": with_capacity,
        "capacidad_cero_hoy": zero_capacity,  # capacidad 0 en el archivo congelado
        "en_gbfs_sin_dato_de_capacidad": no_capacity_value,
        "no_estan_en_gbfs": len(unmatched),
        "lista_no_estan_en_gbfs": unmatched,
        "sin_nombre_se_uso_el_id": no_name,
    }
    return table, report


def _accounting(con: duckdb.DuckDBPyConnection, period: dict) -> dict:
    """Qué pasó con cada punta de cada viaje conservado, y los totales esperados.

    Los totales esperados se cuentan directo sobre `trips` e `id_map`, sin
    pasar por `resolved` ni por las tablas agregadas: así las sumas de las
    tablas se comprueban contra un conteo aparte.
    """
    sides = {}
    for side, label in (("start", "inicio"), ("end", "fin")):
        found = dict(
            con.execute(
                f"""
                SELECT CASE WHEN NOT {side}_in THEN 'fuera_del_periodo' ELSE {side}_state END, count(*)
                FROM resolved GROUP BY 1
                """
            ).fetchall()
        )
        sides[label] = {
            "contadas": found.pop("ok", 0),
            "sin_estacion": found.pop("sin_estacion", 0),
            "excluidas": found.pop("excluida", 0),
            "fuera_del_periodo": found.pop("fuera_del_periodo", 0),
            # Lo que sigue debe ser cero: un id con un viaje en el periodo siempre es estación.
            "sin_clasificar": sum(found.values()),
        }
    route_ends = dict(
        con.execute(
            "SELECT end_state, count(*) FROM resolved WHERE start_in AND start_state = 'ok' GROUP BY 1"
        ).fetchall()
    )
    expected_departures = con.execute(
        f"""
        SELECT count(*) FROM trips t JOIN id_map m ON m.raw_id = t.start_id
        WHERE m.station_id IS NOT NULL AND {_in_period('t.started_at')}
        """,
        period,
    ).fetchone()[0]
    expected_arrivals = con.execute(
        f"""
        SELECT count(*) FROM trips t JOIN id_map m ON m.raw_id = t.end_id
        WHERE m.station_id IS NOT NULL AND {_in_period('t.ended_at')}
        """,
        period,
    ).fetchone()[0]
    expected_routes = con.execute(
        f"""
        SELECT count(*) FROM trips t
        JOIN id_map s ON s.raw_id = t.start_id
        JOIN id_map e ON e.raw_id = t.end_id
        WHERE s.station_id IS NOT NULL AND e.station_id IS NOT NULL AND {_in_period('t.started_at')}
        """,
        period,
    ).fetchone()[0]
    name_without_id = con.execute(
        f"""
        SELECT count(*) FROM trips
        WHERE end_id IS NULL AND end_name IS NOT NULL AND {_in_period('ended_at')}
        """,
        period,
    ).fetchone()[0]
    # Lo que debe traer trips.parquet, contado también sobre `trips` e `id_map`.
    either = f"({_in_period('t.started_at')} OR {_in_period('t.ended_at')})"
    expected_trips, expected_start_in, expected_end_in = con.execute(
        f"""
        SELECT count(*) FILTER (WHERE {either}),
               count(*) FILTER (WHERE {_in_period('t.started_at')}),
               count(*) FILTER (WHERE {_in_period('t.ended_at')})
        FROM trips t
        """,
        period,
    ).fetchone()
    expected_with_station = {
        side: con.execute(
            f"""
            SELECT count(*) FROM trips t JOIN id_map m ON m.raw_id = t.{side}_id
            WHERE m.station_id IS NOT NULL AND {either}
            """,
            period,
        ).fetchone()[0]
        for side in ("start", "end")
    }
    return {
        "trips_esperados": {
            "filas": expected_trips,
            "con_la_salida_dentro_del_periodo": expected_start_in,
            "con_la_llegada_dentro_del_periodo": expected_end_in,
            "con_estacion_de_inicio": expected_with_station["start"],
            "con_estacion_de_fin": expected_with_station["end"],
        },
        "puntas": sides,
        "filas_resueltas": con.execute("SELECT count(*) FROM resolved").fetchone()[0],
        "salidas_esperadas": expected_departures,
        "llegadas_esperadas": expected_arrivals,
        "rutas_esperadas": expected_routes,
        # Viajes que salen de una estación válida dentro del periodo, según su punta de fin.
        "salidas_segun_su_fin": {
            "con_estacion_de_fin": route_ends.get("ok", 0),
            "sin_estacion_de_fin": route_ends.get("sin_estacion", 0),
            "fin_en_estacion_excluida": route_ends.get("excluida", 0),
            "fin_en_estacion_sin_viajes_en_el_periodo": route_ends.get("inactiva", 0),
        },
        "llegadas_con_nombre_pero_sin_id": name_without_id,
    }


def _zero_runs(con: duckdb.DuckDBPyConnection) -> dict:
    """Horas en cero que caen en rachas largas sin ningún viaje en la estación."""
    runs, hours, stations = con.execute(
        f"""
        SELECT count(*), coalesce(sum(gap), 0)::BIGINT, count(DISTINCT station_id)
        FROM (
            SELECT station_id,
                   (epoch_us(lead(ts_hour) OVER (PARTITION BY station_id ORDER BY ts_hour)) - epoch_us(ts_hour))
                       // {HOUR_US} - 1 AS gap
            FROM active_hours
        )
        WHERE gap >= {LONG_ZERO_RUN_HOURS}
        """
    ).fetchone()
    return {"horas_minimas_de_la_racha": LONG_ZERO_RUN_HOURS, "rachas": runs, "filas": hours, "estaciones": stations}


def _timezone_check(con: duckdb.DuckDBPyConnection, station_hour: pa.Table) -> tuple[list[dict], list[str]]:
    """Regla 11: la hora que da DuckDB contra un cálculo aparte en Python.

    Toma el viaje con el menor ride_id de cada mes crudo (con salida contada)
    y hace dos comparaciones contra `local_hour_instant`:

    - el instante que DuckDB le da a ese viaje;
    - las salidas por hora de su estación, en todo ese día local, tal como
      quedaron en station_hour. Si la tabla estuviera corrida una o dos horas,
      el perfil del día no coincidiría.
    """
    picked = con.execute(
        f"""
        SELECT source_month, ride_id, started_at, start_station, epoch_us({hour_sql('started_at')})
        FROM resolved
        WHERE start_in AND start_state = 'ok' AND ride_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY source_month ORDER BY ride_id, started_at) = 1
        ORDER BY source_month
        """
    ).fetchall()
    trips, problems = [], []
    if not picked:
        problems.append("no hay ningún viaje con salida contada para comprobar la zona horaria")
    hour_type = station_hour.schema.field("ts_hour").type
    for month, ride_id, started_at, station, hour_us in picked:
        expected = local_hour_instant(started_at)
        from_duckdb = datetime.fromtimestamp(hour_us // 1_000_000, timezone.utc)

        # Salidas de esa estación en ese día local, hora por hora, contadas en Python.
        day = started_at.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day + timedelta(days=1)
        stamps = con.execute(
            """
            SELECT started_at FROM resolved
            WHERE start_in AND start_state = 'ok' AND start_station = $station
              AND started_at >= $day AND started_at < $next_day
            """,
            {"station": station, "day": day, "next_day": next_day},
        ).fetchall()
        in_python = Counter(local_hour_instant(stamp) for (stamp,) in stamps)
        rows = station_hour.filter(
            pc.and_(
                pc.equal(station_hour["station_id"], station),
                pc.and_(
                    pc.greater_equal(station_hour["ts_hour"], pa.scalar(local_hour_instant(day), type=hour_type)),
                    pc.less(station_hour["ts_hour"], pa.scalar(local_hour_instant(next_day), type=hour_type)),
                ),
            )
        )
        in_table = {
            hour.astimezone(timezone.utc): departures
            for hour, departures in zip(rows["ts_hour"].to_pylist(), rows["departures"].to_pylist())
            if departures
        }
        same_instant = from_duckdb == expected
        same_day = in_table == dict(in_python)
        trips.append(
            {
                "mes": month,
                "ride_id": ride_id,
                "started_at": started_at.isoformat(sep=" ", timespec="milliseconds"),
                "estacion": station,
                "hora_nueva_york": expected.astimezone(ZoneInfo(TZ)).isoformat(),
                "hora_utc": expected.isoformat(),
                "hora_duckdb_utc": from_duckdb.isoformat(),
                "salidas_de_esa_hora_en_station_hour": in_table.get(expected, 0),
                "salidas_de_esa_hora_contadas_en_python": in_python.get(expected, 0),
                "horas_con_salidas_ese_dia_en_station_hour": len(in_table),
                "horas_con_salidas_ese_dia_contadas_en_python": len(in_python),
                "coincide": same_instant and same_day,
            }
        )
        if not same_instant:
            problems.append(
                f"zona horaria: el viaje {ride_id} ({started_at}) debe caer en {expected.isoformat()} "
                f"y DuckDB lo puso en {from_duckdb.isoformat()}"
            )
        elif not same_day:
            problems.append(
                f"zona horaria: las salidas por hora de la estación {station} el día {day.date()} en "
                f"station_hour no coinciden con el conteo aparte en Python (viaje {ride_id})"
            )
    return trips, problems


# --------------------------------------------------------------------------- #
# Comprobaciones
# --------------------------------------------------------------------------- #

def _sum(table: pa.Table, column: str) -> int:
    return int(pc.sum(table[column]).as_py() or 0)


def run_checks(
    rows: dict, accounting: dict, stations: pa.Table, station_hour: pa.Table, routes: pa.Table
) -> list[dict]:
    """Regla 10: comprobaciones que deben cumplirse en cada corrida.

    Devuelve una lista de {nombre, esperado, obtenido, cumple}. No lanza
    errores: quien la llama decide.
    """
    checks = []

    def check(name: str, expected, found) -> None:
        checks.append({"nombre": name, "esperado": expected, "obtenido": found, "cumple": expected == found})

    kept = rows["conservadas"]
    check("filas leídas de los archivos = filas cargadas", rows["leidas"], rows["cargadas"])
    check(
        "duplicados quitados = duplicados contados aparte",
        rows["duplicadas_contadas_aparte"],
        rows["duplicadas_quitadas"],
    )
    check(
        "filas leídas = duplicadas + fechas inválidas + conservadas",
        rows["leidas"],
        rows["duplicadas_quitadas"] + rows["fechas_invalidas_quitadas"] + kept,
    )
    check("cada viaje conservado aparece una sola vez al asignar estaciones", kept, accounting["filas_resueltas"])
    for label, side in accounting["puntas"].items():
        check(
            f"puntas de {label}: conservadas = contadas + sin estación + excluidas + fuera del periodo",
            kept,
            side["contadas"] + side["sin_estacion"] + side["excluidas"] + side["fuera_del_periodo"],
        )
        check(f"puntas de {label} sin clasificar", 0, side["sin_clasificar"])
    check(
        "salidas contadas al clasificar = viajes del periodo con estación de inicio válida",
        accounting["salidas_esperadas"],
        accounting["puntas"]["inicio"]["contadas"],
    )
    check(
        "llegadas contadas al clasificar = viajes que terminan en el periodo con estación de fin válida",
        accounting["llegadas_esperadas"],
        accounting["puntas"]["fin"]["contadas"],
    )
    check(
        "suma de departures = viajes del periodo con estación de inicio válida",
        accounting["salidas_esperadas"],
        _sum(station_hour, "departures"),
    )
    check(
        "suma de arrivals = viajes que terminan en el periodo con estación de fin válida",
        accounting["llegadas_esperadas"],
        _sum(station_hour, "arrivals"),
    )
    check(
        "suma de routes.trips = viajes del periodo con las dos estaciones válidas",
        accounting["rutas_esperadas"],
        _sum(routes, "trips"),
    )
    by_type = pc.add(station_hour["departures_member"], station_hour["departures_casual"])
    check(
        "filas donde miembros + casuales no da el total de salidas",
        0,
        int(pc.sum(pc.not_equal(by_type, station_hour["departures"])).as_py() or 0),
    )
    known = set(stations["station_id"].to_pylist())
    in_hours = set(pc.unique(station_hour["station_id"]).to_pylist())
    in_routes = set(pc.unique(routes["start_station_id"]).to_pylist()) | set(
        pc.unique(routes["end_station_id"]).to_pylist()
    )
    check("estaciones de station_hour que no están en stations", 0, len(in_hours - known))
    check("estaciones de stations sin filas en station_hour", 0, len(known - in_hours))
    check("estaciones de routes que no están en stations", 0, len(in_routes - known))
    return checks


def trips_checks(accounting: dict, found: dict) -> list[dict]:
    """Comprobaciones de trips.parquet: lo que se leyó de vuelta del archivo contra lo esperado.

    `found` es lo que devuelve `_read_back_trips`. Mismo formato que `run_checks`.
    """
    expected = accounting["trips_esperados"]
    checks = []

    def check(name: str, wanted, got) -> None:
        checks.append({"nombre": f"trips: {name}", "esperado": wanted, "obtenido": got, "cumple": wanted == got})

    check("columnas con nombre o tipo distinto al acordado", 0, found["columnas_distintas"])
    check(
        "filas del archivo = viajes conservados con la salida o la llegada dentro del periodo",
        expected["filas"],
        found["filas"],
    )
    check(
        "filas con la salida dentro del periodo = viajes conservados que empiezan en el periodo",
        expected["con_la_salida_dentro_del_periodo"],
        found["con_la_salida_dentro_del_periodo"],
    )
    check(
        "filas con la llegada dentro del periodo = viajes conservados que terminan en el periodo",
        expected["con_la_llegada_dentro_del_periodo"],
        found["con_la_llegada_dentro_del_periodo"],
    )
    check(
        "filas con estación de inicio = conteo aparte sobre los viajes",
        expected["con_estacion_de_inicio"],
        found["con_estacion_de_inicio"],
    )
    check(
        "filas con estación de fin = conteo aparte sobre los viajes",
        expected["con_estacion_de_fin"],
        found["con_estacion_de_fin"],
    )
    check("filas con un ride_id que ya salió en otra fila", 0, found["ride_id_repetidos"])
    check("ids de estación que no están en stations", 0, found["ids_que_no_estan_en_stations"])
    check(
        "horas de estación donde las salidas del archivo no dan station_hour.departures",
        0,
        found["horas_donde_las_salidas_no_dan_station_hour"],
    )
    check(
        "horas de estación donde las llegadas del archivo no dan station_hour.arrivals",
        0,
        found["horas_donde_las_llegadas_no_dan_station_hour"],
    )
    check("pares de estaciones donde el archivo no da routes.trips", 0, found["pares_que_no_dan_routes"])
    check("filas fuera de orden por started_at", 0, found["filas_fuera_de_orden"])
    check(
        "viajes de la revisión de zona horaria cuya salida o llegada no cae en el instante calculado aparte",
        0,
        sum(1 for trip in found["zona_horaria"] if not trip["coincide"]),
    )
    return checks


# --------------------------------------------------------------------------- #
# Flujo completo
# --------------------------------------------------------------------------- #

def _merge_summary(groups: list[dict]) -> dict:
    """Totales de la unión de variantes para el reporte."""
    merged = [g for g in groups if g["unido"]]
    apart = [g for g in groups if not g["unido"]]

    def ends(found: list[dict]) -> int:
        return sum(v["salidas"] + v["llegadas"] for g in found for v in g["variantes"])

    relabelled = [v for g in groups for v in g["variantes"] if v["se_une"]]
    return {
        "grupos_encontrados": len(groups),
        "grupos_unidos": len(merged),
        "grupos_no_unidos": len(apart),
        "puntas_en_grupos_unidos": ends(merged),
        "puntas_en_grupos_no_unidos": ends(apart),
        # Puntas que estaban bajo la escritura que desaparece y cambiaron de id.
        "salidas_que_cambiaron_de_id": sum(v["salidas"] for v in relabelled),
        "llegadas_que_cambiaron_de_id": sum(v["llegadas"] for v in relabelled),
        "ids_que_cambiaron": len(relabelled),
        "lista_no_unidos": apart,
        "lista_unidos": merged,
    }


def _config_report(datos: Datos, config_path: Path | None) -> dict:
    def days(tramo) -> dict:
        return {"inicio": tramo.inicio.isoformat(), "fin": tramo.fin.isoformat()}

    return {
        "archivo": _short(config_path) if config_path else None,
        "meses_crudos": list(datos.meses_crudos),
        "periodo": days(datos.periodo),
        "tramos": {name: days(tramo) for name, tramo in datos.tramos.items()},
        "union_de_ids": {"distancia_maxima_m": datos.distancia_maxima_m},
        "estaciones_excluidas": {"prefijos": list(datos.prefijos_excluidos), "ids": list(datos.ids_excluidos)},
    }


def _write_text(path: Path, text: str) -> None:
    """Escribe con nombre temporal y renombra: nunca queda un reporte a medias."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def apply_station_types(
    stations: pa.Table, station_hour_path: Path, sql_path: Path | None, datos: Datos
) -> tuple[pa.Table, dict]:
    """Llena `station_type` con la consulta de logística (notebooks/b/tipo_estacion.sql).

    La consulta se corre tal cual, en una conexión aparte de DuckDB, cambiando solo el archivo que lee
    (`data/processed/station_hour.parquet`) por el `station_hour` recién construido. Una estación que la
    consulta no devuelve queda sin tipo (nulo).

    La consulta trae escritas las fechas del tramo de ENTRENAMIENTO. Si no son las de config/datos.yaml no se
    aplica (todas las estaciones quedan sin tipo) y el reporte dice por qué: así el tipo nunca se calcula con
    filas de validación ni de la prueba final (decisiones 4 y 8).

    Devuelve la tabla y el resumen que va en limpieza.json (`tipo_de_estacion`).
    """
    report = {"consulta": None, "aplicada": False, "motivo": None, "por_tipo": {}, "sin_tipo": stations.num_rows}
    if sql_path is None:
        report["motivo"] = "no se pidió aplicar ninguna consulta"
        return stations, report
    report["consulta"] = _short(sql_path)
    if not Path(sql_path).exists():
        report["motivo"] = f"no existe {_short(sql_path)}"
        return stations, report
    sql = Path(sql_path).read_text(encoding="utf-8")
    train = datos.tramos["entrenamiento"]
    expected = (train.inicio.isoformat(), (train.fin + timedelta(days=1)).isoformat())
    found = tuple(
        (match.group(1) if (match := re.search(_TIPO_SQL_FECHA % name, sql)) else None) for name in ("t0", "t1")
    )
    report["tramo_de_la_consulta"] = {"desde": found[0], "antes_de": found[1]}
    if found != expected:
        report["motivo"] = (
            f"las fechas de la consulta ({found[0]} a {found[1]}, sin incluir el último) no son las del tramo de "
            f"entrenamiento de config/datos.yaml ({expected[0]} a {expected[1]}, sin incluir el último)"
        )
        return stations, report
    if _TIPO_SQL_FUENTE not in sql:
        raise BuildError(
            f"{_short(sql_path)} ya no lee station_hour con {_TIPO_SQL_FUENTE}: no se puede apuntar al "
            "station_hour recién construido. Ajusta la consulta o `_TIPO_SQL_FUENTE` en src/data/build_tables.py."
        )
    sql = sql.replace(_TIPO_SQL_FUENTE, f"read_parquet({_sql_text(str(station_hour_path))})")
    con = duckdb.connect()
    try:
        cursor = con.execute(sql)
        names = [column[0] for column in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        con.close()
    kinds = {row["station_id"]: row["station_type"] for row in rows}
    if len(kinds) != len(rows):
        raise BuildError(f"{_short(sql_path)} devolvió una estación más de una vez.")
    values = [kinds.get(station) for station in stations["station_id"].to_pylist()]
    table = stations.set_column(
        stations.schema.get_field_index("station_type"), "station_type", pa.array(values, pa.string())
    )
    counts = Counter(SIN_TIPO if value is None else value for value in values)
    report.update(
        aplicada=True,
        estaciones_que_devolvio_la_consulta=len(rows),
        de_ellas_fuera_de_stations=len(set(kinds) - set(stations["station_id"].to_pylist())),
        por_tipo={kind: count for kind, count in sorted(counts.items()) if kind != SIN_TIPO},
        sin_tipo=counts.get(SIN_TIPO, 0),
    )
    return table, report


def build(
    datos: Datos,
    raw_dir: Path,
    gbfs_file: Path | None,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    tipo_sql: Path | None = None,
    log=print,
) -> dict:
    """Corre la limpieza completa y escribe las tres tablas, trips.parquet y el reporte.

    Devuelve el reporte (el mismo contenido de limpieza.json). Lanza
    BuildError o ContractError si algo no cuadra; en ese caso no cambia
    ninguna tabla ni el reporte que ya hubiera en `out_dir`.
    """
    started = time.time()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    check_out_dir(out_dir)
    files = raw_files(datos, raw_dir)
    capacity, gbfs_info = load_capacity(gbfs_file, datos)
    period = {
        "ini": datetime.combine(datos.periodo.inicio, datetime.min.time()),
        "fin": datetime.combine(datos.periodo.fin + timedelta(days=1), datetime.min.time()),
    }

    work = out_dir / WORK_FOLDER
    shutil.rmtree(work, ignore_errors=True)  # restos de una corrida interrumpida
    work.mkdir(parents=True)
    con = None
    try:
        con = _connect(work)
        log(f"Leyendo {len(files)} meses de {_short(raw_dir)}...")
        rows = _load_trips(con, files)
        log(f"  {rows['leidas']:,} filas leídas, {rows['conservadas']:,} conservadas ({time.time() - started:.0f} s).")

        raw_ids = _raw_ids(con, period)
        plan = plan_ids(raw_ids, datos)
        _resolve(con, plan, period)
        log(f"  Estaciones asignadas ({time.time() - started:.0f} s).")

        station_hour = to_contract_table(_station_hour_table(con), "station_hour")
        routes = to_contract_table(_routes_table(con), "routes")
        stations, capacity_report = _stations_table(con, plan, raw_ids, capacity)
        stations = to_contract_table(stations, "stations")
        log(f"  Tablas armadas ({time.time() - started:.0f} s).")

        # trips.parquet se escribe ya, en la carpeta temporal, porque sus
        # comprobaciones leen de vuelta el archivo. De ahí no sale si algo falla.
        trips_path = work / f"{TRIPS_NAME}.parquet"
        trips_bytes = _write_trips(con, trips_path)
        log(f"  Viajes escritos en la carpeta temporal ({time.time() - started:.0f} s). Comprobando...")

        accounting = _accounting(con, period)
        checks = run_checks(rows, accounting, stations, station_hour, routes)
        tz_trips, tz_problems = _timezone_check(con, station_hour)
        trips_found = _read_back_trips(con, trips_path, period, stations, station_hour, routes, tz_trips)
        checks += trips_checks(accounting, trips_found)
        trips_sides = _trips_sides(con)
        zero_runs = _zero_runs(con)
        failed = [
            f"{c['nombre']}: se esperaba {c['esperado']:,} y se obtuvo {c['obtenido']:,}"
            for c in checks
            if not c["cumple"]
        ] + tz_problems
        if failed:
            raise BuildError("Comprobaciones que fallaron. No se escribió nada:\n- " + "\n- ".join(failed))

        # Las tres tablas se escriben primero en la carpeta temporal, donde ya
        # está trips.parquet. Solo si las tres cumplen su contrato, y el
        # reporte se pudo armar, pasan las cuatro a la carpeta de salida. Así
        # nunca quedan tablas de una corrida con el reporte de otra.
        sizes = {"station_hour": write_contract(station_hour, "station_hour", work / "station_hour.parquet").stat().st_size}
        # El tipo de estación sale del station_hour ya escrito (solo su tramo de entrenamiento).
        stations, type_report = apply_station_types(stations, work / "station_hour.parquet", tipo_sql, datos)
        stations = to_contract_table(stations, "stations")
        tables = {"stations": stations, "station_hour": station_hour, "routes": routes}
        for name in ("stations", "routes"):
            sizes[name] = write_contract(tables[name], name, work / f"{name}.parquet").stat().st_size
        trips_report = {
            "archivo": _short(out_dir / f"{TRIPS_NAME}.parquet"),
            "es_contrato": False,
            "columnas": {f.name: str(f.type) for f in TRIPS_SCHEMA},
            "filas": trips_found["filas"],
            "bytes": trips_bytes,
            "sin_ride_id": trips_found["sin_ride_id"],
            "primera_salida": trips_found["primera_salida"],
            "ultima_salida": trips_found["ultima_salida"],
            "puntas": {
                label: {
                    **trips_sides[label],
                    "fuera_del_periodo": trips_found["filas"] - trips_found[f"con_la_{within}_dentro_del_periodo"],
                }
                for label, within in (("inicio", "salida"), ("fin", "llegada"))
            },
            "sin_estacion_en_ninguna_punta": trips_found["sin_estacion_en_ninguna_punta"],
            "zona_horaria": trips_found["zona_horaria"],
        }
        report = _report(
            datos, config_path, raw_dir, out_dir, period, rows, accounting, plan, capacity_report, gbfs_info,
            tables, sizes, zero_runs, tz_trips, checks, started, trips_report,
        )
        report["tipo_de_estacion"] = type_report
        report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        report_md = render_markdown(report)
        for name in (*TABLES, TRIPS_NAME):
            os.replace(work / f"{name}.parquet", out_dir / f"{name}.parquet")
        reports = out_dir / REPORT_FOLDER
        reports.mkdir(parents=True, exist_ok=True)
        _write_text(reports / f"{REPORT_NAME}.json", report_json)
        _write_text(reports / f"{REPORT_NAME}.md", report_md)
    except Exception as error:
        if _was_interrupted(error):
            raise KeyboardInterrupt from None
        if isinstance(error, duckdb.Error):
            raise BuildError(f"DuckDB falló durante la limpieza: {str(error)[:600]}") from error
        raise
    finally:
        if con is not None:
            con.close()  # sin esto el proceso se queda colgado tras un Ctrl+C
        shutil.rmtree(work, ignore_errors=True)
    return report


def _report(
    datos, config_path, raw_dir, out_dir, period, rows, accounting, plan, capacity_report, gbfs_info,
    tables, sizes, zero_runs, tz_trips, checks, started, trips_report,
) -> dict:
    """Junta todos los conteos de la corrida. Es el contenido de limpieza.json."""
    stations, station_hour, routes = (tables[name] for name in TABLES)
    total = station_hour.num_rows
    zeros = int(pc.sum(pc.equal(pc.add(station_hour["departures"], station_hour["arrivals"]), 0)).as_py() or 0)
    loops = routes.filter(pc.equal(routes["start_station_id"], routes["end_station_id"]))
    now = datetime.now(timezone.utc)
    return {
        "generado_utc": now.isoformat(timespec="seconds"),
        "generado_nueva_york": now.astimezone(ZoneInfo(TZ)).isoformat(timespec="seconds"),
        "segundos": round(time.time() - started, 1),
        "config": _config_report(datos, config_path),
        "entradas": {"carpeta_de_viajes": _short(raw_dir), "archivos": rows["archivos"], "gbfs": gbfs_info},
        "filas": {k: v for k, v in rows.items() if k not in ("archivos", "texto")},
        "texto": rows["texto"],
        "periodo": {
            "primer_instante_local": period["ini"].isoformat(sep=" "),
            "primer_instante_fuera_local": period["fin"].isoformat(sep=" "),
            "zona": TZ,
        },
        "puntas": accounting["puntas"],
        "salidas_segun_su_fin": accounting["salidas_segun_su_fin"],
        "llegadas_con_nombre_pero_sin_id": accounting["llegadas_con_nombre_pero_sin_id"],
        "union_de_ids": _merge_summary(plan.groups),
        "estaciones_excluidas": plan.excluded,
        "capacidad": capacity_report,
        "tablas": {
            name: {"archivo": _short(out_dir / f"{name}.parquet"), "filas": tables[name].num_rows, "bytes": sizes[name]}
            for name in TABLES
        },
        "station_hour": {
            "filas": total,
            "estaciones": stations.num_rows,
            "salidas": _sum(station_hour, "departures"),
            "llegadas": _sum(station_hour, "arrivals"),
            "salidas_de_miembros": _sum(station_hour, "departures_member"),
            "salidas_de_casuales": _sum(station_hour, "departures_casual"),
            "filas_con_viajes": total - zeros,
            "filas_en_cero": zeros,
            "proporcion_en_cero": round(zeros / total, 6),
            "primera_hora": pc.min(station_hour["ts_hour"]).as_py().isoformat(),
            "ultima_hora": pc.max(station_hour["ts_hour"]).as_py().isoformat(),
            "ceros_en_rachas_largas": zero_runs,
        },
        "routes": {
            "pares": routes.num_rows,
            "viajes": _sum(routes, "trips"),
            "pares_misma_estacion": loops.num_rows,
            "viajes_misma_estacion": _sum(loops, "trips"),
        },
        # Tabla interna para el simulador. No es uno de los seis contratos.
        "trips": trips_report,
        "zona_horaria": tz_trips,
        "comprobaciones": checks,
    }


# --------------------------------------------------------------------------- #
# Reporte para personas
# --------------------------------------------------------------------------- #

def _n(value: int) -> str:
    return f"{value:,}"


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "0%"


def _size(size_bytes: int) -> str:
    """Tamaño de un archivo en MB (o en KB si es pequeño)."""
    return f"{size_bytes / 1e6:,.1f} MB" if size_bytes >= 1_000_000 else f"{size_bytes / 1e3:,.1f} KB"


def render_markdown(report: dict) -> str:
    """El reporte en español llano, para alguien que no programa."""
    config, rows, sides = report["config"], report["filas"], report["puntas"]
    start, end = sides["inicio"], sides["fin"]
    merge, capacity, hours = report["union_de_ids"], report["capacidad"], report["station_hour"]
    routes, gbfs, after = report["routes"], report["entradas"]["gbfs"], report["salidas_segun_su_fin"]
    trips, trip_start, trip_end = report["trips"], report["trips"]["puntas"]["inicio"], report["trips"]["puntas"]["fin"]
    runs = hours["ceros_en_rachas_largas"]
    excluded_ids = config["estaciones_excluidas"]
    lines = [
        "# Reporte de la limpieza de viajes",
        "",
        f"Generado el {report['generado_nueva_york']} (hora de Nueva York) por `python scripts/build_tables.py`.",
        "Este archivo se escribe solo en cada corrida; no lo edites a mano. Los mismos números están en",
        "`limpieza.json`.",
        "",
        "Los datos solo registran viajes que sí ocurrieron (demanda atendida). Aquí no hay ninguna medida",
        "de la demanda que no se pudo atender.",
        "",
        "## 1. Qué se leyó",
        "",
        f"- Periodo de análisis: del {config['periodo']['inicio']} al {config['periodo']['fin']}, los dos días",
        f"  incluidos, en hora local de Nueva York. Sale de `{config['archivo'] or 'la configuración'}`.",
    ]
    lines += [f"- Mes {f['mes']}: {_n(f['filas'])} viajes en `{f['archivo']}`." for f in report["entradas"]["archivos"]]
    lines += [
        f"- Total leído: {_n(rows['leidas'])} viajes.",
        "",
        "## 2. Qué se quitó y por qué",
        "",
        "Viajes completos que se quitaron:",
        "",
        f"- Repetidos: {_n(rows['duplicadas_quitadas'])}. Un mismo viaje puede venir en dos meses seguidos;",
        "  se cuenta una sola vez.",
        f"- Con fechas imposibles: {_n(rows['fechas_invalidas_quitadas'])} ({_n(rows['sin_fecha'])} sin fecha y",
        f"  {_n(rows['terminan_antes_de_empezar'])} que terminan antes de empezar).",
        f"- Quedan {_n(rows['conservadas'])} viajes.",
        "",
        "Cada viaje tiene dos puntas: la salida y la llegada. Cada punta se cuenta por separado, en su",
        "propia estación y con su propia hora. Un viaje puede contar como salida y no como llegada, o al revés.",
        "",
        "| Qué pasó con la punta | Salidas | Llegadas |",
        "|---|---:|---:|",
        f"| Se contó en las tablas | {_n(start['contadas'])} | {_n(end['contadas'])} |",
        f"| No trae estación | {_n(start['sin_estacion'])} | {_n(end['sin_estacion'])} |",
        f"| Estación excluida (ver sección 4) | {_n(start['excluidas'])} | {_n(end['excluidas'])} |",
        f"| Su fecha cae fuera del periodo | {_n(start['fuera_del_periodo'])} | {_n(end['fuera_del_periodo'])} |",
        f"| Total (viajes que quedan) | {_n(rows['conservadas'])} | {_n(rows['conservadas'])} |",
        "",
        "Las salidas se cuentan con la fecha de inicio del viaje y las llegadas con la fecha de fin. Por eso",
        "un viaje que empezó antes del periodo y terminó dentro cuenta solo como llegada. Los archivos crudos",
        "traen días que no son del periodo; la configuración explica por qué el periodo termina donde termina.",
        "",
        "Limpieza de textos (sobre los viajes leídos): "
        f"{_n(report['texto']['ids_con_espacios'])} ids de estación y {_n(report['texto']['nombres_con_espacios'])} "
        "nombres traían espacios en las orillas y se recortaron; "
        f"{_n(report['texto']['ids_vacios'])} ids venían vacíos y se tratan como estación faltante.",
        "",
        "## 3. Estaciones escritas de dos formas",
        "",
        "En el origen una misma estación aparece a veces con dos ids, por ejemplo `2231.1` y `2231.10`, o",
        "`5308.04` y `5308.04_`. Se unen solo si tienen el mismo nombre y sus coordenadas están a",
        f"{config['union_de_ids']['distancia_maxima_m']:g} m o menos. Queda el id con más viajes.",
        "",
        f"- Grupos de ids parecidos: {_n(merge['grupos_encontrados'])}.",
        f"- Se unieron por completo: {_n(merge['grupos_unidos'])} grupos "
        f"({_n(merge['puntas_en_grupos_unidos'])} salidas y llegadas en total).",
        f"  Cambiaron de id {_n(merge['salidas_que_cambiaron_de_id'])} salidas y "
        f"{_n(merge['llegadas_que_cambiaron_de_id'])} llegadas.",
        f"- No se unieron, o solo en parte: {_n(merge['grupos_no_unidos'])} grupos "
        f"({_n(merge['puntas_en_grupos_no_unidos'])} salidas y llegadas).",
        "  Cada escritura, comparada con el id con más viajes de su grupo:",
    ]
    for group in merge["lista_no_unidos"]:
        for v in group["variantes"]:
            if v["es_el_id_que_queda"]:
                continue
            anchor = next(a for a in group["variantes"] if a["es_el_id_que_queda"])
            distance = "sin dato" if v["distancia_m"] is None else f"{v['distancia_m']:.1f} m"
            lines.append(
                f"  - `{anchor['id']}` ({anchor['nombre']}; {_n(anchor['salidas'])} salidas, "
                f"{_n(anchor['llegadas'])} llegadas) y "
                f"`{v['id']}` ({v['nombre']}; {_n(v['salidas'])} salidas, {_n(v['llegadas'])} llegadas). "
                f"Distancia: {distance}. "
                + ("Se unió." if v["se_une"] else f"No se unió: {v['motivo']}.")
            )
    if not merge["lista_no_unidos"]:
        lines.append("  - (ninguno)")
    lines += [
        "",
        "## 4. Estaciones que no entran a las tablas",
        "",
        "La configuración excluye los ids que empiezan con "
        f"{', '.join(excluded_ids['prefijos']) or '(ninguno)'} y los ids "
        f"{', '.join(excluded_ids['ids']) or '(ninguno)'}. El motivo de cada uno está escrito en la configuración.",
        "Un viaje entre una estación normal y una de estas conserva su otra punta.",
        "",
        "| Regla | Estaciones | Salidas que no se cuentan | Llegadas que no se cuentan |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {e['regla']} | {_n(e['estaciones'])} | {_n(e['salidas'])} | {_n(e['llegadas'])} |"
        for e in report["estaciones_excluidas"]
    ]
    if not report["estaciones_excluidas"]:
        lines.append("| (ninguna estación excluida tuvo viajes) | 0 | 0 | 0 |")
    lines += [
        "",
        "## 5. Tablas que se escribieron",
        "",
        f"- `stations`: {_n(report['tablas']['stations']['filas'])} estaciones.",
        f"- `station_hour`: {_n(hours['filas'])} filas (una por estación y hora), con {_n(hours['salidas'])} salidas",
        f"  ({_n(hours['salidas_de_miembros'])} de miembros y {_n(hours['salidas_de_casuales'])} de usuarios "
        f"casuales) y {_n(hours['llegadas'])} llegadas.",
        f"  Va de {hours['primera_hora']} a {hours['ultima_hora']}.",
        f"- `routes`: {_n(routes['pares'])} pares de estaciones con {_n(routes['viajes'])} viajes. De ellos, "
        f"{_n(routes['viajes_misma_estacion'])} viajes",
        f"  salen y llegan a la misma estación ({_n(routes['pares_misma_estacion'])} pares).",
        "",
        "`station_hour` tiene una fila por cada hora entre la primera y la última hora con viajes de cada",
        f"estación, aunque no haya viajes. Por eso {_n(hours['filas_en_cero'])} filas "
        f"({_percent(hours['filas_en_cero'], hours['filas'])}) están en cero.",
        "",
        "### Tabla interna de viajes (`trips`)",
        "",
        "Además se escribió `trips.parquet`. Es una tabla de trabajo para el simulador y **no es uno de los seis",
        "contratos**: el dashboard no la lee y su formato puede cambiar sin avisar.",
        "",
        f"- Una fila por viaje que sale o llega dentro del periodo: {_n(trips['filas'])} viajes, "
        f"{_size(trips['bytes'])}, ordenados por la",
        "  hora de salida.",
        "- Columnas: `ride_id`, `started_at` y `ended_at` (salida y llegada, en hora de Nueva York),",
        "  `start_station_id` y `end_station_id` (la estación de cada punta, con el id que quedó después de la",
        "  limpieza: las escrituras unidas de la sección 3 ya llevan un solo id).",
        "  Además `start_lat`, `start_lng`, `end_lat` y `end_lng`: las coordenadas del viaje tal como vienen en el",
        "  archivo crudo (pueden ir vacías). Una punta vacía conserva las suyas; el simulador las usa para saber",
        "  cerca de qué estación empezó o terminó ese viaje.",
        f"- Ningún `ride_id` se repite. Filas sin `ride_id`: {_n(trips['sin_ride_id'])}.",
        "- Una punta va vacía si el viaje no trae estación, si la estación está excluida (sección 4) o si es",
        "  una estación sin ningún viaje dentro del periodo. No se adivina ni se rellena.",
        "- Una punta cuya fecha cae fuera del periodo conserva su fecha y su estación. No cuenta en",
        "  `station_hour`; el simulador decide qué hacer con ella.",
        "",
        "| Qué trae la punta | Salidas | Llegadas |",
        "|---|---:|---:|",
        f"| Una estación de las tablas | {_n(trip_start['con_estacion'])} | {_n(trip_end['con_estacion'])} |",
        f"| Vacía: el viaje no trae estación | {_n(trip_start['sin_estacion'])} | {_n(trip_end['sin_estacion'])} |",
        f"| Vacía: estación excluida | {_n(trip_start['estacion_excluida'])} | {_n(trip_end['estacion_excluida'])} |",
        "| Vacía: estación sin viajes en el periodo | "
        f"{_n(trip_start['estacion_sin_viajes_en_el_periodo'])} | "
        f"{_n(trip_end['estacion_sin_viajes_en_el_periodo'])} |",
        f"| Total (filas del archivo) | {_n(trips['filas'])} | {_n(trips['filas'])} |",
        f"| De ese total, con la fecha fuera del periodo | {_n(trip_start['fuera_del_periodo'])} | "
        f"{_n(trip_end['fuera_del_periodo'])} |",
        "",
        f"Viajes del archivo sin estación en ninguna de las dos puntas: {_n(trips['sin_estacion_en_ninguna_punta'])}.",
        "",
        "Contar los viajes de este archivo por estación y hora da exactamente `station_hour`, y contarlos por",
        "par de estaciones da exactamente `routes`. Se comprueba en cada corrida leyendo de vuelta el archivo",
        "(sección 7). Son viajes que sí ocurrieron: este archivo tampoco dice nada de la demanda que no se pudo",
        "atender.",
        "",
        "## 6. Capacidad de las estaciones",
        "",
    ]
    if gbfs["foto"] is None:
        lines += [
            "No se encontró el archivo congelado de GBFS (`config/gbfs_congelado.csv`), así que la capacidad quedó "
            "vacía en todas las estaciones.",
        ]
    else:
        lines += [
            f"La capacidad sale de `{gbfs['archivo']}`, el archivo congelado que escribe `scripts/freeze_gbfs.py`: "
            "la capacidad más",
            f"frecuente de cada estación en la captura de GBFS del {gbfs['captura_desde_utc']} al "
            f"{gbfs['captura_hasta_utc']} (UTC),",
            f"con {_n(gbfs['fotos_de_estado'] or 0)} fotos de estado y {_n(gbfs['estados_de_informacion'] or 0)} "
            "estados de información. Como está congelado,",
            "reconstruir las tablas no cambia la capacidad aunque la captura siga creciendo.",
            "",
            f"- Con capacidad: {_n(capacity['con_capacidad'])} estaciones.",
            f"- Capacidad 0 en el archivo: {_n(capacity['capacidad_cero_hoy'])}. Son estaciones que hoy no están instaladas;",
            "  se guarda vacía, no como cero. Al revés no vale: otras estaciones que hoy no están instaladas",
            "  conservan su capacidad en GBFS. Para saber cuáles ya no existen se usa `is_installed` de",
            "  `station_status`, no la capacidad vacía.",
            f"- En GBFS pero sin dato de capacidad: {_n(capacity['en_gbfs_sin_dato_de_capacidad'])}.",
            f"- No están en GBFS: {_n(capacity['no_estan_en_gbfs'])}. Capacidad vacía:",
        ]
        lines += [
            f"  - `{u['id']}` {u['nombre']}: {_n(u['salidas'])} salidas, {_n(u['llegadas'])} llegadas."
            for u in capacity["lista_no_estan_en_gbfs"]
        ]
    lines += [
        "",
        "## 7. Comprobaciones de esta corrida",
        "",
        "Si alguna fallara, el script se detiene y no escribe nada.",
        "",
        "| Comprobación | Esperado | Obtenido | Cumple |",
        "|---|---:|---:|---|",
    ]
    lines += [
        f"| {c['nombre']} | {_n(c['esperado'])} | {_n(c['obtenido'])} | {'sí' if c['cumple'] else 'NO'} |"
        for c in report["comprobaciones"]
    ]
    lines += [
        "",
        "Hora local: las fechas de los viajes vienen en hora de Nueva York sin zona. Se revisó con un viaje",
        "real por mes que la hora de la tabla sea la correcta (cálculo aparte, sin la base de datos):",
        "",
        "| Viaje | Inicio (como viene) | Estación | Hora en Nueva York | Hora en UTC | Coincide |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t['ride_id']} | {t['started_at']} | {t['estacion']} | {t['hora_nueva_york']} | {t['hora_utc']} | "
        f"{'sí' if t['coincide'] else 'NO'} |"
        for t in report["zona_horaria"]
    ]
    lines += [
        "",
        "Además, para la estación de cada uno de esos viajes se contaron aparte sus salidas de ese día, hora",
        "por hora, y coinciden con la tabla: "
        + "; ".join(
            f"`{t['estacion']}` {_n(t['horas_con_salidas_ese_dia_en_station_hour'])} horas con salidas"
            for t in report["zona_horaria"]
        )
        + ".",
        "",
        "En `trips.parquet` se revisó la salida y la llegada de esos mismos viajes, con su minuto y su segundo:",
        "",
        "| Viaje | Salida (como viene) | Llegada (como viene) | Salida en UTC | Llegada en UTC | Coincide |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {t['ride_id']} | {t['started_at']} | {t['ended_at']} | {t['inicio_utc']} | {t['fin_utc']} | "
        f"{'sí' if t['coincide'] else 'NO'} |"
        for t in trips["zona_horaria"]
    ]
    lines += [
        "",
        "## 8. Lo que se sabe que es imperfecto",
        "",
        "- **Solo hay demanda atendida.** Un cero significa que no hubo viajes, no que nadie quisiera uno.",
        f"- **Ceros que no son demanda cero.** {_n(runs['filas'])} filas de `station_hour` "
        f"({_percent(runs['filas'], hours['filas'])}) caen en rachas de {runs['horas_minimas_de_la_racha']} "
        "horas o más",
        f"  sin ningún viaje en la estación ({_n(runs['rachas'])} rachas en {_n(runs['estaciones'])} estaciones). "
        "Puede ser que la",
        "  estación estuviera cerrada o fuera de servicio; el dato no lo dice. Esas filas van en cero igual.",
    ]
    if gbfs["foto"] is None:
        lines.append("- **Capacidad.** No hay capacidad de ninguna estación.")
    else:
        days = gbfs["dias_despues_del_periodo"]
        when = (
            f"que terminó {_n(days)} días después del último día del periodo" if days is not None and days > 0
            else "que no es del periodo completo"
        )
        lines += [
            f"- **La capacidad es la de la captura, no la del periodo.** Viene de una captura de GBFS {when}.",
            f"  {_n(capacity['capacidad_cero_hoy'])} estaciones con viajes en el periodo aparecen en el archivo con "
            f"capacidad 0 y {_n(capacity['no_estan_en_gbfs'])} no están en GBFS:",
            "  todas quedan con capacidad vacía.",
        ]
    lines += [
        f"- **Llegadas sin id de estación.** {_n(report['llegadas_con_nombre_pero_sin_id'])} viajes que terminan "
        "en el periodo traen el nombre de la",
        "  estación de fin pero no su id. No se cuentan como llegadas.",
        f"- **Salidas sin llegada.** De las {_n(start['contadas'])} salidas contadas, "
        f"{_n(after['sin_estacion_de_fin'])} son de viajes sin",
        f"  estación de fin, {_n(after['fin_en_estacion_excluida'])} terminan en una estación excluida y "
        f"{_n(after['fin_en_estacion_sin_viajes_en_el_periodo'])} en una estación sin viajes",
        "  dentro del periodo. Son salidas sin llegada y no aparecen en `routes`.",
        f"- **Salidas y llegadas no suman lo mismo:** {_n(hours['salidas'])} salidas contra "
        f"{_n(hours['llegadas'])} llegadas. Además de lo",
        "  anterior hay llegadas sin salida (viajes sin estación de inicio o que salen de una estación",
        "  excluida) y viajes que cruzan el borde del periodo, que cuentan en una sola punta.",
        f"- **Estaciones que no se unieron.** Los {_n(merge['grupos_no_unidos'])} grupos de la sección 3 quedan "
        "como estaciones",
        "  separadas. Si en realidad son la misma estación, su historia está partida en dos ids.",
        _type_line(report.get("tipo_de_estacion")),
        "",
    ]
    return "\n".join(lines)


def _type_line(kinds: dict | None) -> str:
    """Renglón del reporte sobre `station_type`: cuántas estaciones de cada tipo, o por qué va vacía."""
    if not kinds or not kinds["aplicada"]:
        why = kinds["motivo"] if kinds else "no se aplicó ninguna consulta"
        return f"- **Tipo de estación.** La columna `station_type` va vacía: {why}."
    counts = ", ".join(f"{kind} {_n(count)}" for kind, count in kinds["por_tipo"].items())
    window = kinds["tramo_de_la_consulta"]
    return (
        f"- **Tipo de estación.** `station_type` sale de `{kinds['consulta']}`, con las filas de `station_hour` "
        f"del tramo de entrenamiento ({window['desde']} a {window['antes_de']}, sin incluir el último): {counts}; "
        f"sin tipo (la consulta no las devuelve: no tienen filas en ese tramo), {_n(kinds['sin_tipo'])}."
    )


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def format_summary(report: dict) -> str:
    """Resumen corto para la terminal."""
    rows, merge, hours, capacity = report["filas"], report["union_de_ids"], report["station_hour"], report["capacidad"]
    period = report["config"]["periodo"]
    return "\n".join(
        [
            f"Periodo {period['inicio']} a {period['fin']} (hora de Nueva York).",
            f"Viajes leídos: {rows['leidas']:,}. Repetidos quitados: {rows['duplicadas_quitadas']:,}. "
            f"Fechas inválidas: {rows['fechas_invalidas_quitadas']:,}. Quedan {rows['conservadas']:,}.",
            f"Variantes de id: {merge['grupos_encontrados']} grupos, {merge['grupos_unidos']} unidos y "
            f"{merge['grupos_no_unidos']} sin unir.",
            f"stations: {hours['estaciones']:,} estaciones ({capacity['con_capacidad']:,} con capacidad).",
            f"station_hour: {hours['filas']:,} filas, {hours['salidas']:,} salidas y {hours['llegadas']:,} llegadas. "
            f"En cero: {100 * hours['proporcion_en_cero']:.2f}% de las filas.",
            f"routes: {report['routes']['pares']:,} pares con {report['routes']['viajes']:,} viajes.",
            f"trips (tabla interna, no es contrato): {report['trips']['filas']:,} viajes, "
            f"{_size(report['trips']['bytes'])}.",
            f"Comprobaciones: {len(report['comprobaciones'])} de {len(report['comprobaciones'])} cumplen. "
            f"Zona horaria revisada con {len(report['zona_horaria'])} viajes reales.",
            f"Listo en {report['segundos']:.0f} s.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_tables.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python scripts/build_tables.py [opciones]",
        description=(
            "Limpia los viajes crudos y escribe stations, station_hour y routes, más el reporte de la limpieza.\n"
            "También escribe trips.parquet, una tabla interna de viajes para el simulador (no es un contrato).\n"
            "El periodo, los meses y las reglas salen de config/datos.yaml."
        ),
        epilog="Ejemplo:\n  python scripts/build_tables.py",
    )
    options = parser.add_argument_group("opciones")
    options.add_argument(
        "--salida", metavar="DIR", help="carpeta de las tablas y de reports/ (por defecto data/processed)"
    )
    options.add_argument(
        "--crudos", metavar="DIR", help="carpeta con los Parquet mensuales AAAAMM.parquet (por defecto data/raw/trips)"
    )
    options.add_argument(
        "--gbfs", metavar="ARCHIVO",
        help=(
            "archivo congelado de GBFS con la capacidad, el que escribe scripts/freeze_gbfs.py "
            "(por defecto config/gbfs_congelado.csv)"
        ),
    )
    options.add_argument(
        "--config", metavar="ARCHIVO", help="archivo de periodo y reglas (por defecto config/datos.yaml)"
    )
    options.add_argument(
        "--tipo-sql", metavar="ARCHIVO",
        help=(
            "consulta que da el tipo de cada estación (por defecto notebooks/b/tipo_estacion.sql). Solo se aplica "
            "si sus fechas son las del tramo de entrenamiento"
        ),
    )
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve 0 si todo salió bien, 1 si algo falló, 130 si se interrumpió.

    Un AVISO al final (sin archivo congelado de GBFS, viajes sin ride_id, o falta el mes siguiente al periodo) no cambia el 0.
    """
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else DATOS_YAML
    out_dir = Path(args.salida).expanduser() if args.salida else DATA_PROCESSED
    raw_dir = Path(args.crudos).expanduser() if args.crudos else DATA_RAW / "trips"
    gbfs_file = Path(args.gbfs).expanduser() if args.gbfs else FROZEN_FILE
    try:
        datos = load_datos(config_path)
        tipo_sql = Path(args.tipo_sql).expanduser() if args.tipo_sql else TIPO_ESTACION_SQL
        report = build(
            datos, raw_dir, gbfs_file, out_dir, config_path=config_path, tipo_sql=tipo_sql,
            log=lambda m: print(m, flush=True),
        )
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. No se escribió nada.", file=sys.stderr)
        return 130
    except (BuildError, ContractError, DatosError, OSError) as error:
        print(f"ERROR. {error}", file=sys.stderr)
        return 1
    print()
    print(format_summary(report))
    print(f"\nTablas en: {out_dir}")
    print(f"Reporte:   {out_dir / REPORT_FOLDER / (REPORT_NAME + '.md')}")
    # Las predicciones se calculan sobre station_hour: si la tabla cambió, quedan viejas.
    print("Ahora corre: python scripts/run_baseline.py", flush=True)
    kinds = report["tipo_de_estacion"]
    if kinds["aplicada"]:
        counts = ", ".join(f"{kind} {count:,}" for kind, count in kinds["por_tipo"].items())
        print(f"Tipo de estación ({kinds['consulta']}): {counts}; sin tipo {kinds['sin_tipo']:,}.", flush=True)
    else:
        print(f"Tipo de estación: station_type quedó vacía en todas las estaciones: {kinds['motivo']}.", flush=True)
    # Avisos: la corrida terminó bien, pero hay algo que quien la corrió debe saber.
    if report["entradas"]["gbfs"]["foto"] is None:
        print(
            f"\nAVISO: no existe el archivo congelado de GBFS {gbfs_file}. La capacidad quedó "
            "vacía en TODAS las estaciones.\nCorre: python scripts/freeze_gbfs.py   y vuelve a "
            "correr este script.",
            file=sys.stderr,
        )
    if report["trips"]["sin_ride_id"]:
        print(
            f"\nAVISO: {report['trips']['sin_ride_id']:,} viajes de trips.parquet no traen ride_id. El simulador "
            "usa el ride_id para ordenar los viajes que ocurren en el mismo instante: revisa esos viajes en los "
            "archivos crudos antes de simular.",
            file=sys.stderr,
        )
    siguiente = (datos.periodo.fin + timedelta(days=1)).strftime("%Y%m")
    if siguiente not in datos.meses_crudos:
        print(
            f"\nAVISO: el periodo termina el {datos.periodo.fin} y el mes {siguiente} no está en meses_crudos. "
            "Los archivos traen los viajes que TERMINAN en cada mes: faltan las salidas del último día que "
            "terminaron al día siguiente.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
