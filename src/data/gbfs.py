"""Captura del feed GBFS en vivo de Citi Bike: estado e información de estaciones.

ES URGENTE dejarlo corriendo. El feed no guarda historia: solo publica cuántas
bicis y lugares libres hay en este minuto. Lo que no se captura se pierde.
Sirve para ver cuándo una estación estuvo vacía o llena (posible demanda
insatisfecha), algo que los viajes no muestran porque solo registran la demanda
atendida.

Qué guarda
----------
Un Parquet no se puede ampliar, así que "acumular" es escribir un archivo
pequeño por foto. Cada archivo se escribe con nombre temporal y se renombra al
final: nunca queda uno a medias.

    data/raw/gbfs/
        station_status/date=AAAA-MM-DD/status_AAAAMMDDTHHMMSSZ.parquet
        station_information/info_AAAAMMDDTHHMMSSZ.parquet
        _state.json

- La carpeta `date=` y el sello del nombre son la hora de captura EN UTC, no
  la hora de Nueva York. Las columnas de tiempo también están en UTC.
- station_status: una foto por captura. Si el feed no cambió desde la última
  (mismo `last_updated`), no se escribe otra vez.
- station_information: cambia muy poco. Se escribe en la primera captura y
  después solo cuando cambia su contenido (se compara una huella).
- _state.json recuerda lo último guardado para no duplicar al reiniciar. Si se
  borra no pasa nada grave: se repite una foto.

Cuatro cosas que hay que saber al usar estos datos
--------------------------------------------------
1. Los archivos de viajes identifican la estación con el `short_name` de GBFS
   ("6289.06"), NO con el `station_id` de GBFS, que es otro id (un UUID o un
   número largo). El puente entre los dos es station_information; por eso
   también se captura.
2. `gbfs_station_type` es el campo `station_type` de GBFS ("classic"). No es el
   `station_type` de los contratos del proyecto (oficinas, residencial, ...).
3. Hay dos horas por fila. `snapshot_ts` es cuándo empezó nuestra captura.
   `feed_last_updated` es cuándo generó el feed esos datos: suele ir entre
   unos segundos y poco más de un minuto atrás (se han visto hasta 72 s).
   Para analizar, usa `feed_last_updated`.
4. `last_reported` trae valores raros (1970-01-02) en estaciones que no están
   instaladas. Se guarda tal cual llega; filtra con `is_installed`.

Cómo dejarlo corriendo (desde la raíz del repositorio)
------------------------------------------------------
Primero revisa si ya hay una captura corriendo. Si este comando muestra una
línea con `capture_gbfs.py`, NO inicies otra: dos capturas a la vez escribirían
en la misma carpeta. Si no muestra nada, no hay ninguna.

    ps -o pid=,command= -p "$(cat data/raw/gbfs/captura.pid 2>/dev/null)" 2>/dev/null

Modo continuo, una captura cada 5 minutos, aunque se cierre la terminal:

    mkdir -p data/raw/gbfs
    nohup .venv/bin/python scripts/capture_gbfs.py >> data/raw/gbfs/captura.log 2>&1 &
    echo $! > data/raw/gbfs/captura.pid
    nohup caffeinate -i -w $(cat data/raw/gbfs/captura.pid) > /dev/null 2>&1 &

La última línea es para macOS: pide que la laptop no se duerma por inactividad
mientras la captura siga viva, y termina sola cuando la captura termina. Lleva
`nohup` igual que la captura: sin él se apaga al cerrar la terminal. NO
alcanza con la tapa cerrada: ahí la Mac entra en reposo aunque esté caffeinate.
Deja la laptop abierta y conectada a la corriente.

Si de todos modos se duerme, la captura no se rompe: captura en cuanto la
computadora despierta y sigue. Pero lo que pasó mientras dormía no se recupera.

    tail -f data/raw/gbfs/captura.log      # ver cómo va

Cada captura deja una sola línea como esta (aquí va partida en dos). El
número final son los segundos que tardó; lo normal son 1 o 2. Si tarda mucho
más, la red anda mal o la computadora se está durmiendo a ratos.

    2026-10-10T04:14:43Z status: 2519 estaciones, 2519 filas -> station_status/date=2026-10-10/
    status_20261010T041441Z.parquet (70 KB) | info: 2519 estaciones, sin cambios -> no se escribe [1 s]

Si una captura falla (por ejemplo, sin red), la línea dice ERROR y se vuelve a
intentar al minuto, sin esperar los 5 minutos del turno siguiente.

Cómo detenerlo:

    kill $(cat data/raw/gbfs/captura.pid)  # o Ctrl+C si corre en la terminal

Termina la captura en curso, escribe "captura detenida" y sale.

Alternativa con cron (una captura por llamada; sale con código distinto de 0
si falló). OJO: en macOS cron puede no tener permiso para entrar a carpetas
como Descargas; si falla, usa el modo continuo.

    */5 * * * * cd /ruta/al/repo && .venv/bin/python scripts/capture_gbfs.py --una-vez >> data/raw/gbfs/captura.log 2>&1

Opciones: `--intervalo N` (segundos, mínimo 60 porque el feed se renueva cada
60), `--salida DIR` y `--una-vez`.

Cómo leerlo todo con DuckDB
---------------------------
Estado de cada estación en cada foto, con el `short_name` (el id de los viajes)
y la capacidad tomados de la información más reciente. Si una foto quedó
guardada dos veces, el QUALIFY del final deja una sola. Las rutas son
relativas a la raíz del repositorio:

    import duckdb

    df = duckdb.sql('''
        WITH info AS (
            SELECT *
            FROM read_parquet('data/raw/gbfs/station_information/*.parquet')
            QUALIFY snapshot_ts = max(snapshot_ts) OVER ()
        )
        SELECT s.feed_last_updated, i.short_name, i.name, i.capacity,
               s.num_bikes_available, s.num_docks_available, s.is_renting
        FROM read_parquet('data/raw/gbfs/station_status/*/*.parquet') AS s
        JOIN info AS i USING (station_id)
        QUALIFY row_number() OVER (
            PARTITION BY station_id, s.feed_last_updated ORDER BY s.snapshot_ts
        ) = 1
        ORDER BY s.feed_last_updated, i.short_name
    ''').df()

DuckDB devuelve las horas en la zona de la computadora. Para verlas en hora de
Nueva York, ejecuta antes `duckdb.sql("SET TimeZone = 'America/New_York'")`.
Al leer station_status con ese patrón aparece además la columna `date`, que es
la fecha UTC de la carpeta.

Volumen aproximado
------------------
Medido el 2026-10-10 (UTC) con 2,519 estaciones: cada foto de station_status
pesa unos 70 KB y cada archivo de station_information unos 160 KB.

- Cada 5 minutos (por defecto): 288 fotos al día, unos 20 MB y 725 mil filas.
- Cada 60 segundos: 1,440 fotos al día, unos 100 MB y 3.6 millones de filas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.paths import DATA_RAW

# Documento de descubrimiento: de aquí se leen las URL reales de cada feed.
DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/gbfs.json"
LANGUAGE = "en"

DEFAULT_OUT = DATA_RAW / "gbfs"
STATE_FILE = "_state.json"

MIN_INTERVAL = 60        # segundos. Es el ttl del feed: pedirlo más seguido no trae datos nuevos.
DEFAULT_INTERVAL = 300   # segundos entre capturas en modo continuo
WAIT_SLICE = 5.0         # segundos. La espera entre capturas se revisa en tramos así de cortos.
RETRY_SECONDS = 60       # segundos. Si una captura falla se reintenta así de pronto. No bajarlo de MIN_INTERVAL.

TIMEOUT = (10, 30)       # segundos para conectar y para recibir la respuesta
ATTEMPTS = 3             # intentos por descarga
BACKOFF_SECONDS = 2.0    # espera antes de reintentar: 2 s, luego 4 s
USER_AGENT = "DataRushNY-CitiBike-Rebalanceo/1.0 (captura periodica de GBFS; proyecto estudiantil; python-requests)"

# Todas las horas se guardan como instante en UTC.
UTC_TS = pa.timestamp("us", tz="UTC")

# El esquema es fijo: si el feed omite un campo, la columna queda en nulo.
STATUS_SCHEMA = pa.schema(
    [
        ("snapshot_ts", UTC_TS),                  # cuándo lo bajamos nosotros
        ("feed_last_updated", UTC_TS),            # cuándo lo generó el feed
        ("station_id", pa.string()),              # id de GBFS: UUID o número largo (no es el id de los viajes)
        ("legacy_id", pa.string()),
        ("num_bikes_available", pa.int64()),
        ("num_ebikes_available", pa.int64()),
        ("num_bikes_disabled", pa.int64()),
        ("num_docks_available", pa.int64()),
        ("num_docks_disabled", pa.int64()),
        ("num_scooters_available", pa.int64()),   # no viene en todas las estaciones
        ("num_scooters_unavailable", pa.int64()), # no viene en todas las estaciones
        ("is_installed", pa.bool_()),             # el feed manda 0/1
        ("is_renting", pa.bool_()),
        ("is_returning", pa.bool_()),
        ("last_reported", UTC_TS),                # último reporte de la estación
    ]
)

INFO_SCHEMA = pa.schema(
    [
        ("snapshot_ts", UTC_TS),
        ("feed_last_updated", UTC_TS),
        ("station_id", pa.string()),              # id de GBFS: UUID o número largo
        ("short_name", pa.string()),              # id de los viajes, p. ej. "6289.06"
        ("name", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("capacity", pa.int64()),
        ("region_id", pa.string()),               # no viene en todas las estaciones
        ("gbfs_station_type", pa.string()),       # `station_type` de GBFS ("classic")
        ("has_kiosk", pa.bool_()),
        ("external_id", pa.string()),
    ]
)

# Columnas que en el feed se llaman distinto.
SOURCE_FIELD = {"gbfs_station_type": "station_type"}


class GbfsError(RuntimeError):
    """El feed no respondió o llegó con una estructura inesperada."""


def log_line(message: str) -> None:
    """Una línea con la hora UTC. Sale de inmediato, también con nohup o redirección.

    Nunca lanza: si no se puede escribir, la captura continua debe seguir.
    """
    try:
        print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}", flush=True)
    except (OSError, ValueError):
        pass  # disco lleno o salida cerrada: se pierde la línea, no la captura


# --------------------------------------------------------------------------- #
# Conversión de valores
# --------------------------------------------------------------------------- #
# Un valor ausente o de un tipo inesperado se vuelve nulo. Es preferible un
# nulo a perder la foto completa.

def _to_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        value = int(value) if value.is_integer() else None
    elif isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return None
    if isinstance(value, int) and -2**63 <= value < 2**63:  # lo que cabe en int64
        return value
    return None


def _to_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _to_bool(value):
    """El feed manda 0/1; otras versiones de GBFS mandan true/false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _to_str(value):
    return None if value is None else str(value)


def _to_ts(value):
    """Epoch en segundos -> instante UTC."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _convert(value, arrow_type: pa.DataType):
    if pa.types.is_integer(arrow_type):
        return _to_int(value)
    if pa.types.is_boolean(arrow_type):
        return _to_bool(value)
    if pa.types.is_floating(arrow_type):
        return _to_float(value)
    if pa.types.is_timestamp(arrow_type):
        return _to_ts(value)
    return _to_str(value)


def build_table(payload: dict, schema: pa.Schema, snapshot_ts: datetime) -> pa.Table:
    """Convierte el JSON de un feed en una tabla con el esquema fijo.

    Los campos que el esquema no conoce se ignoran. Una estación sin
    `station_id` se descarta, porque no se podría identificar.
    """
    try:
        stations = payload["data"]["stations"]
    except (KeyError, TypeError):
        stations = None
    if not isinstance(stations, list):
        raise GbfsError("el feed llegó sin lista de estaciones")

    feed_ts = _to_ts(payload.get("last_updated"))
    rows = []
    for station in stations:
        if not isinstance(station, dict) or station.get("station_id") is None:
            continue
        row = {"snapshot_ts": snapshot_ts, "feed_last_updated": feed_ts}
        for column in schema:
            if column.name not in row:
                source = SOURCE_FIELD.get(column.name, column.name)
                row[column.name] = _convert(station.get(source), column.type)
        rows.append(row)
    if not rows:
        raise GbfsError("el feed llegó sin estaciones")
    return pa.Table.from_pylist(rows, schema=schema)


def information_hash(table: pa.Table) -> str:
    """Huella del contenido de station_information.

    No incluye las columnas de tiempo ni depende del orden de las estaciones:
    solo cambia cuando cambia algún dato de alguna estación.
    """
    content = table.drop_columns(["snapshot_ts", "feed_last_updated"]).sort_by("station_id")
    text = json.dumps(content.to_pylist(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Descarga
# --------------------------------------------------------------------------- #

def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def fetch_json(session, url: str) -> dict:
    """Descarga un JSON con reintentos. Lanza GbfsError si no se pudo."""
    error: Exception | None = None
    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as problem:
            error = problem
            continue
        if isinstance(payload, dict):
            return payload
        error = ValueError("la respuesta no es un objeto JSON")
    raise GbfsError(f"no se pudo descargar {url} tras {ATTEMPTS} intentos: {error}")


def discover_feeds(session) -> dict[str, str]:
    """Lee del documento de descubrimiento la URL de cada feed (idioma "en")."""
    document = fetch_json(session, DISCOVERY_URL)
    try:
        urls = {feed["name"]: feed["url"] for feed in document["data"][LANGUAGE]["feeds"]}
    except (KeyError, TypeError):
        raise GbfsError(f"{DISCOVERY_URL}: estructura inesperada, no se encontraron los feeds") from None
    missing = [name for name in ("station_status", "station_information") if name not in urls]
    if missing:
        raise GbfsError(f"{DISCOVERY_URL}: no lista los feeds {missing}")
    return urls


# --------------------------------------------------------------------------- #
# Escritura y estado
# --------------------------------------------------------------------------- #

def _write_parquet(table: pa.Table, path: Path) -> None:
    """Escribe con nombre temporal y renombra: nunca queda un archivo a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_state(out_dir: Path) -> dict:
    """Lo último que se guardó. Si el archivo falta o está dañado, se empieza de cero."""
    try:
        with open(Path(out_dir) / STATE_FILE, encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(out_dir: Path, state: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / (STATE_FILE + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, out_dir / STATE_FILE)


def _describe(path: Path, out_dir: Path) -> str:
    return f"{path.relative_to(out_dir)} ({path.stat().st_size / 1024:.0f} KB)"


def _save_status(payload: dict, out_dir: Path, snapshot_ts: datetime, state: dict) -> str:
    """Guarda la foto de station_status, salvo que el feed no haya cambiado."""
    table = build_table(payload, STATUS_SCHEMA, snapshot_ts)
    summary = f"status: {len(payload['data']['stations'])} estaciones"

    last_updated = payload.get("last_updated")
    feed_ts = _to_ts(last_updated)
    if feed_ts is not None and last_updated == state.get("status_last_updated"):
        return f"{summary}, feed sin cambios desde {feed_ts:%Y-%m-%dT%H:%M:%SZ} -> no se escribe"

    path = out_dir / "station_status" / f"date={snapshot_ts:%Y-%m-%d}" / f"status_{snapshot_ts:%Y%m%dT%H%M%SZ}.parquet"
    _write_parquet(table, path)
    state["status_last_updated"] = last_updated
    state["status_last_file"] = str(path.relative_to(out_dir))
    return f"{summary}, {table.num_rows} filas -> {_describe(path, out_dir)}"


def _save_information(payload: dict, out_dir: Path, snapshot_ts: datetime, state: dict) -> str:
    """Guarda station_information solo la primera vez o cuando cambia su contenido."""
    table = build_table(payload, INFO_SCHEMA, snapshot_ts)
    summary = f"info: {len(payload['data']['stations'])} estaciones"

    digest = information_hash(table)
    last_file = state.get("info_last_file")
    # Si alguien borró el archivo, se escribe de nuevo: sin él no hay puente a los viajes.
    if digest == state.get("info_hash") and last_file and (out_dir / last_file).is_file():
        return f"{summary}, sin cambios -> no se escribe"

    path = out_dir / "station_information" / f"info_{snapshot_ts:%Y%m%dT%H%M%SZ}.parquet"
    _write_parquet(table, path)
    state["info_hash"] = digest
    state["info_last_file"] = str(path.relative_to(out_dir))
    return f"{summary}, {table.num_rows} filas -> {_describe(path, out_dir)}"


# --------------------------------------------------------------------------- #
# Captura
# --------------------------------------------------------------------------- #

def capture_once(out_dir: str | Path = DEFAULT_OUT, session=None, now: datetime | None = None) -> str:
    """Hace una captura y devuelve el resumen de una línea.

    Lanza GbfsError si algo falló. Los dos feeds se guardan por separado: si
    uno falla, el otro se guarda igual y el error lo dice.

    `session` y `now` existen para las pruebas; en uso normal se dejan vacíos.
    """
    out_dir = Path(out_dir)
    snapshot_ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    own_session = session is None
    session = session or new_session()
    state = load_state(out_dir)
    parts: list[str] = []
    failed = False
    try:
        urls = discover_feeds(session)
        for feed, save in (("station_status", _save_status), ("station_information", _save_information)):
            try:
                parts.append(save(fetch_json(session, urls[feed]), out_dir, snapshot_ts, state))
                save_state(out_dir, state)
            except Exception as error:  # un feed con problemas no debe impedir guardar el otro
                failed = True
                parts.append(f"{feed} FALLÓ: {error}")
    finally:
        if own_session:
            session.close()

    summary = " | ".join(parts)
    if failed:
        raise GbfsError(summary)
    return summary


def capture_safely(out_dir: str | Path = DEFAULT_OUT, session=None, now: datetime | None = None) -> bool:
    """Una vuelta del modo continuo: captura, anota el resultado y nunca lanza.

    Devuelve True si la captura salió bien. La línea termina con los segundos
    que tardó: una captura normal tarda pocos; si tarda mucho, la red anda mal
    o la computadora se durmió a medias.
    """
    started = time.time()
    try:
        message, ok = capture_once(out_dir, session, now), True
    except Exception as error:  # se anota y la siguiente vuelta lo vuelve a intentar
        message, ok = f"ERROR {error}", False
    log_line(f"{message} [{time.time() - started:.0f} s]")
    return ok


def check_interval(seconds: int) -> int:
    """Rechaza intervalos menores al ttl del feed."""
    seconds = int(seconds)
    if seconds < MIN_INTERVAL:
        raise ValueError(
            f"el intervalo mínimo es {MIN_INTERVAL} segundos (el feed se renueva cada {MIN_INTERVAL}); "
            f"se pidió {seconds}"
        )
    return seconds


def run_loop(
    interval: int = DEFAULT_INTERVAL,
    out_dir: str | Path = DEFAULT_OUT,
    stop: threading.Event | None = None,
    session=None,
    clock=time.time,
) -> None:
    """Captura cada `interval` segundos hasta que se active `stop`.

    Las capturas caen en inicio + k * intervalo según el reloj del sistema, así
    lo que tarda cada una no se acumula. Si la computadora se durmió o una
    captura tardó más que el intervalo, se captura en cuanto se puede y se
    vuelve a contar el intervalo desde ahí.

    Si una captura falla no se espera el turno entero: se reintenta a los
    RETRY_SECONDS, o en el turno siguiente si llega antes. Un reintento no
    mueve los turnos, salvo que termine después del turno siguiente; entonces
    vale la regla de arriba.

    `session` y `clock` existen para las pruebas.
    """
    interval = check_interval(interval)
    stop = stop or threading.Event()
    target = clock()
    while not stop.is_set():
        ok = capture_safely(out_dir, session, datetime.fromtimestamp(clock(), tz=timezone.utc))
        target += interval
        if target <= clock():  # se perdió al menos un turno
            target = clock() + interval
        wake = target if ok else min(target, clock() + RETRY_SECONDS)
        # Se espera en tramos cortos mirando el reloj del sistema. Una sola
        # espera larga no avanza mientras la computadora duerme, y al
        # despertar la siguiente captura llegaría muy tarde.
        while not stop.is_set():
            remaining = wake - clock()
            if remaining <= 0:
                break
            stop.wait(min(remaining, WAIT_SLICE))
        if wake < target:
            target -= interval  # fue un reintento: no consume el turno


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def _interval_argument(text: str) -> int:
    try:
        seconds = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"debe ser un número entero de segundos, no {text!r}") from None
    try:
        return check_interval(seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_gbfs.py",
        description=(
            "Captura el estado de las estaciones de Citi Bike (feed GBFS) y lo acumula en Parquet. "
            "Sin opciones, captura sin parar hasta que se detenga con Ctrl+C."
        ),
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    parser.add_argument(
        "--intervalo",
        type=_interval_argument,
        default=DEFAULT_INTERVAL,
        metavar="N",
        help=f"segundos entre capturas (mínimo {MIN_INTERVAL}, por defecto {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=DEFAULT_OUT,
        metavar="DIR",
        help="carpeta donde se guardan los Parquet (por defecto data/raw/gbfs del repositorio)",
    )
    parser.add_argument(
        "--una-vez",
        action="store_true",
        help="hace una sola captura y termina; sale con código 1 si falló (útil para cron)",
    )
    args = parser.parse_args(argv)

    if args.una_vez:
        return 0 if capture_safely(args.salida) else 1

    stop = threading.Event()

    def request_stop(signum, frame) -> None:
        # La primera señal pide parar con orden. Una segunda corta de inmediato.
        stop.set()
        signal.signal(signum, signal.SIG_DFL)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log_line(f"captura continua cada {args.intervalo} s en {args.salida} (Ctrl+C para detener)")
    run_loop(args.intervalo, args.salida, stop)
    log_line("captura detenida")
    return 0
