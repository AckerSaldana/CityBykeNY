"""Descarga viajes de Citi Bike por mes y los convierte a Parquet (capa cruda).

Por cada mes pedido (AAAAMM):

1. Busca el archivo en el listado real del bucket público. Nunca arma la URL
   a ciegas, porque los nombres no son uniformes.
   - 2024 en adelante: un zip por mes.
   - 2021 a 2023: un zip por año que trae un zip por mes. Se baja SOLO el mes
     pedido con peticiones de rango HTTP, no el zip anual de más de 1 GB.
2. Descarga a data/raw/zips/. Reanuda si quedó a medias y reintenta si falla
   la red.
3. Extrae todos los CSV a una carpeta temporal y comprueba el encabezado.
4. Escribe UN Parquet por mes: <salida>/<AAAAMM>.parquet. Jersey City va
   siempre a su propia carpeta trips_jc, también con --salida.

Es la capa cruda: no limpia, no filtra y no quita duplicados. Eso es un paso
posterior. Dos cosas que conviene saber al leer el Parquet:

- started_at y ended_at son hora local de Nueva York SIN zona, tal como
  vienen en el CSV.
- Los ids de estación son texto ("5470.10" conserva su cero final; también
  hay ids como "JC013" o "SYS038").

Solo se admite el formato nuevo de columnas: de febrero de 2021 (202102) en
adelante.

Uso:  python scripts/download_trips.py --help
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import struct
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import duckdb
import requests

from src.paths import DATA_RAW

BUCKET_URL = "https://s3.amazonaws.com/tripdata"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Decisión del equipo: solo el formato nuevo de columnas.
FIRST_MONTH = "202102"

DATASETS = ("nyc", "jc")
OUT_FOLDER = {"nyc": "trips", "jc": "trips_jc"}  # dentro de data/raw
ZIPS_FOLDER = "zips"

# Encabezado exacto de los CSV. Si un archivo trae otro, el mes falla.
EXPECTED_COLUMNS = (
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
)

# Tipos de DuckDB. Ids y nombres siempre texto; fechas sin zona.
COLUMN_TYPES = {column: "VARCHAR" for column in EXPECTED_COLUMNS}
COLUMN_TYPES.update(
    started_at="TIMESTAMP",
    ended_at="TIMESTAMP",
    start_lat="DOUBLE",
    start_lng="DOUBLE",
    end_lat="DOUBLE",
    end_lng="DOUBLE",
)

MB = 1_000_000
CHUNK = 1 << 20          # 1 MiB por bloque al descargar y copiar
TIMEOUT = (15, 60)       # segundos: conexión, lectura
MAX_RETRIES = 5          # intentos seguidos sin avanzar antes de rendirse
BACKOFF_SECONDS = 2.0    # espera inicial entre intentos; se duplica cada vez

_MONTH = re.compile(r"^\d{4}(0[1-9]|1[0-2])$")
# Nombres en el bucket. Se aceptan variantes reales como
# "JC-202604-citibike-tripdata.zip", "JC-202609-citibike-tripdata.csv.zip",
# "JC-201708 citibike-tripdata.csv.zip" o "JC-202207-citbike-tripdata.csv.zip".
_MONTHLY_KEY = {
    "nyc": re.compile(r"^(\d{6})\D.*\.zip$"),
    "jc": re.compile(r"^JC-(\d{6})\D.*\.zip$"),
}
_YEARLY_KEY = re.compile(r"^(\d{4})\D.*\.zip$")
_NESTED_ZIP = re.compile(r"^(\d{6})\D.*\.zip$", re.IGNORECASE)


class DownloadError(Exception):
    """Un mes no se pudo descargar o convertir. El mensaje dice por qué."""


class _TransientHTTP(Exception):
    """Respuesta del servidor que vale la pena reintentar (5xx o 429)."""


_NETWORK_ERRORS = (requests.RequestException, ConnectionError, TimeoutError, _TransientHTTP)


@dataclass(frozen=True)
class RemoteObject:
    key: str    # nombre del archivo en el bucket
    size: int   # bytes


@dataclass(frozen=True)
class Member:
    """Una entrada del directorio central de un zip."""

    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int   # 0 = sin compresión, 8 = deflate
    crc: int


@dataclass(frozen=True)
class Source:
    """De dónde sale un mes y cuánto hay que transferir."""

    month: str
    remote: RemoteObject              # zip mensual o zip anual
    nested: Member | None = None      # zip del mes dentro del zip anual
    csvs: tuple[Member, ...] = ()     # CSV del zip mensual (un mes anidado se ve hasta bajarlo)

    @property
    def zip_name(self) -> str:
        return Path(self.nested.name if self.nested else self.remote.key).name

    @property
    def transfer_bytes(self) -> int:
        return self.nested.compress_size if self.nested else self.remote.size

    @property
    def disk_bytes(self) -> int:
        """Espacio aproximado: zip + CSV extraídos + Parquet y margen."""
        if self.nested:
            return self.nested.compress_size + int(2.5 * self.nested.file_size)
        # Con zips sin compresión esto equivale a unas 2.5 veces el zip.
        return self.remote.size + int(1.5 * sum(m.file_size for m in self.csvs))


@dataclass
class MonthResult:
    month: str
    status: str                       # "ok", "ya existía", "error", "interrumpido"
    detail: str = ""
    rows: int | None = None
    min_started: datetime | None = None
    max_started: datetime | None = None
    parquet_bytes: int | None = None
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Meses
# --------------------------------------------------------------------------- #

def parse_month(text: str) -> str:
    if not _MONTH.match(text):
        raise argparse.ArgumentTypeError(f"mes inválido: {text!r}. Usa AAAAMM, por ejemplo 202604.")
    return text


def month_range(first: str, last: str) -> list[str]:
    """Meses de `first` a `last`, ambos incluidos."""
    months = []
    year, month = int(first[:4]), int(first[4:])
    while f"{year}{month:02d}" <= last:
        months.append(f"{year}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


# --------------------------------------------------------------------------- #
# Listado del bucket y ubicación de cada mes
# --------------------------------------------------------------------------- #

def parse_listing(xml_text: str | bytes) -> tuple[list[RemoteObject], bool]:
    """Lee una página del listado S3. Devuelve (archivos, hay_más_páginas)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise DownloadError(f"El listado del bucket no es XML válido ({error}).") from error
    if root.tag != f"{S3_NS}ListBucketResult":
        raise DownloadError(f"El listado del bucket no tiene el formato esperado (raíz {root.tag!r}).")
    objects = [
        RemoteObject(item.findtext(f"{S3_NS}Key"), int(item.findtext(f"{S3_NS}Size")))
        for item in root.iter(f"{S3_NS}Contents")
    ]
    truncated = (root.findtext(f"{S3_NS}IsTruncated") or "").strip().lower() == "true"
    return objects, truncated


def _keys_by_period(objects: list[RemoteObject], pattern: re.Pattern) -> dict[str, list[RemoteObject]]:
    """Agrupa los archivos por el mes (o año) que captura `pattern`."""
    found: dict[str, list[RemoteObject]] = {}
    for obj in objects:
        match = pattern.match(obj.key)
        if match:
            found.setdefault(match[1], []).append(obj)
    return found


def describe_available(objects: list[RemoteObject], dataset: str) -> str:
    """Texto corto con el rango de meses del listado (desde FIRST_MONTH)."""
    parts = []
    if dataset == "nyc":
        years = sorted(y for y in _keys_by_period(objects, _YEARLY_KEY) if y >= FIRST_MONTH[:4])
        if years:
            parts.append(f"{max(years[0] + '01', FIRST_MONTH)} a {years[-1]}12 dentro de zips anuales")
    months = sorted(m for m in _keys_by_period(objects, _MONTHLY_KEY[dataset]) if m >= FIRST_MONTH)
    if months:
        span = months[0] if len(months) == 1 else f"{months[0]} a {months[-1]}"
        text = f"{span} en zips mensuales"
        missing = sorted(set(month_range(months[0], months[-1])) - set(months))
        if missing:
            text += f" (faltan {', '.join(missing)})"
        parts.append(text)
    return "Disponible: " + (" y ".join(parts) if parts else "ningún mes") + "."


def check_supported(month: str) -> None:
    """Rechaza los meses del formato viejo (decisión del equipo)."""
    if month < FIRST_MONTH:
        raise DownloadError(
            f"{month} es anterior a {FIRST_MONTH}. Solo se admite el formato nuevo de columnas, "
            "de febrero de 2021 en adelante."
        )


def find_key(month: str, dataset: str, objects: list[RemoteObject]) -> tuple[str, RemoteObject]:
    """Ubica en el listado el archivo que trae un mes.

    Devuelve ("mensual", archivo) o ("anual", archivo). En el segundo caso el
    mes está en un zip dentro del zip anual.
    """
    check_supported(month)
    candidates = [("mensual", _keys_by_period(objects, _MONTHLY_KEY[dataset]).get(month, []))]
    if dataset == "nyc":
        candidates.append(("anual", _keys_by_period(objects, _YEARLY_KEY).get(month[:4], [])))
    for kind, found in candidates:
        if len(found) > 1:
            names = ", ".join(obj.key for obj in found)
            raise DownloadError(f"Hay más de un archivo para {month} ({names}). No se elige a ciegas.")
        if found:
            return kind, found[0]
    raise DownloadError(f"{month} no está en el índice de '{dataset}'. {describe_available(objects, dataset)}")


def _is_junk(name: str) -> bool:
    """Carpetas, __MACOSX y archivos ocultos (.DS_Store, ._algo.csv)."""
    parts = name.split("/")
    return name.endswith("/") or "__MACOSX" in parts or parts[-1].startswith(".")


def _is_csv(name: str) -> bool:
    return not _is_junk(name) and name.lower().endswith(".csv")


def _nested_month(name: str) -> str | None:
    """Mes de un zip mensual guardado dentro de un zip anual, o None."""
    if _is_junk(name):
        return None
    match = _NESTED_ZIP.match(name.split("/")[-1])
    return match[1] if match else None


# --------------------------------------------------------------------------- #
# Acceso HTTP
# --------------------------------------------------------------------------- #

def _short(error: object) -> str:
    return f"{type(error).__name__}: {str(error)[:160]}" if isinstance(error, Exception) else str(error)


def _wait(attempt: int, what: str, error: object) -> None:
    seconds = min(BACKOFF_SECONDS * 2 ** (attempt - 1), 60)
    print(f"  {what}: fallo de red ({_short(error)}). Reintento en {seconds:.0f} s.", flush=True)
    time.sleep(seconds)


def _retry(action, what: str):
    """Ejecuta `action` y la repite, con espera creciente, si falla la red."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return action()
        except _NETWORK_ERRORS as error:
            if attempt == MAX_RETRIES:
                message = f"{what}: la red falló {MAX_RETRIES} veces seguidas ({_short(error)})."
                raise DownloadError(message) from error
            _wait(attempt, what, error)


def _check_status(response: requests.Response, expected: tuple[int, ...], what: str) -> None:
    code = response.status_code
    if code in expected:
        return
    if code >= 500 or code == 429:
        raise _TransientHTTP(f"el servidor respondió {code}")
    if code == 200 and expected == (206,):
        raise DownloadError(f"{what}: el servidor no admite peticiones de rango.")
    raise DownloadError(f"{what}: el servidor respondió {code}.")


class _Progress:
    """Avance de una descarga sin llenar la pantalla.

    En una terminal reescribe una sola línea. Si la salida va a un archivo,
    escribe una línea cada 10 %.
    """

    def __init__(self, label: str, total: int, done: int = 0):
        self.label = label
        self.total = max(total, 1)
        self.tty = sys.stdout.isatty()
        self.last_time = 0.0
        self.last_step = -1
        if done:
            print(f"{label}: se reanuda desde {done / MB:.1f} MB.", flush=True)

    def update(self, done: int) -> None:
        percent = min(100 * done // self.total, 100)
        text = f"{self.label}: {percent:3d}% ({done / MB:.1f} de {self.total / MB:.1f} MB)"
        if self.tty:
            now = time.monotonic()
            if now - self.last_time >= 0.5 or done >= self.total:
                self.last_time = now
                print("\r" + text, end="", flush=True)
        elif percent // 10 != self.last_step:
            self.last_step = percent // 10
            print(text, flush=True)

    def close(self) -> None:
        if self.tty:
            print(flush=True)


class _RemoteFile(io.RawIOBase):
    """Archivo remoto de solo lectura que lee por rangos HTTP.

    Sirve para que `zipfile` lea el directorio central (está al final del
    zip) sin bajar el archivo entero.
    """

    def __init__(self, bucket: Bucket, remote: RemoteObject):
        self.bucket = bucket
        self.remote = remote
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.remote.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        length = min(len(buffer), self.remote.size - self.position)
        if length <= 0:
            return 0
        data = self.bucket.read_range(self.remote.key, self.position, length)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


class Bucket:
    """Acceso HTTP al bucket público: listado, lecturas por rango y descargas."""

    def __init__(self, base_url: str = BUCKET_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # Bytes tal cual, sin compresión de transporte: los tamaños deben coincidir.
        self.session.headers["Accept-Encoding"] = "identity"
        self._objects: list[RemoteObject] | None = None
        self._listing_error: DownloadError | None = None
        self._members: dict[str, list[Member]] = {}

    def url(self, key: str) -> str:
        return f"{self.base_url}/{quote(key)}"

    def _get_bytes(self, url: str, what: str, *, expected: tuple[int, ...], params=None, headers=None) -> bytes:
        def action() -> bytes:
            # stream=True: el estado se revisa antes de leer el cuerpo.
            with self.session.get(url, params=params, headers=headers, stream=True, timeout=TIMEOUT) as response:
                _check_status(response, expected, what)
                return response.content

        return _retry(action, what)

    def listing(self) -> list[RemoteObject]:
        """Listado completo del bucket. Se pide una sola vez por ejecución."""
        if self._listing_error:
            raise self._listing_error
        if self._objects is None:
            try:
                self._objects = self._fetch_listing()
            except DownloadError as error:
                self._listing_error = error  # no repetir los reintentos en cada mes
                raise
        return self._objects

    def _fetch_listing(self) -> list[RemoteObject]:
        objects: list[RemoteObject] = []
        marker = ""
        while True:
            params = {"marker": marker} if marker else None
            page, truncated = parse_listing(
                self._get_bytes(self.base_url, "listado del bucket", expected=(200,), params=params)
            )
            objects.extend(page)
            if not truncated:
                return objects
            # Listado en varias páginas: la siguiente empieza tras el último nombre.
            if not page or page[-1].key == marker:
                raise DownloadError("El listado del bucket viene truncado y no avanza.")
            marker = page[-1].key

    def read_range(self, key: str, start: int, length: int) -> bytes:
        """Lee `length` bytes de un archivo remoto a partir de `start`."""
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        what = f"lectura parcial de {key}"
        data = self._get_bytes(self.url(key), what, expected=(206,), headers=headers)
        if len(data) != length:
            raise DownloadError(f"{what}: llegaron {len(data)} bytes y se pidieron {length}.")
        return data

    def zip_members(self, remote: RemoteObject) -> list[Member]:
        """Directorio central de un zip remoto (unas 3 peticiones pequeñas)."""
        if remote.key not in self._members:
            reader = io.BufferedReader(_RemoteFile(self, remote), buffer_size=CHUNK)
            try:
                with zipfile.ZipFile(reader) as archive:
                    self._members[remote.key] = [
                        Member(i.filename, i.header_offset, i.compress_size, i.file_size, i.compress_type, i.CRC)
                        for i in archive.infolist()
                    ]
            except zipfile.BadZipFile as error:
                raise DownloadError(f"{remote.key} no es un zip válido ({error}).") from error
        return self._members[remote.key]

    def member_data_start(self, key: str, member: Member) -> int:
        """Posición del primer byte de datos de un miembro dentro del zip remoto."""
        header = self.read_range(key, member.header_offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise DownloadError(f"{key}: el encabezado de {member.name} no es de zip.")
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        return member.header_offset + 30 + name_length + extra_length

    def download_range(self, key: str, start: int, length: int, part: Path, label: str) -> None:
        """Baja `length` bytes de `key` desde `start` al archivo `part`.

        Si `part` ya existe, continúa donde se quedó. Ante un fallo de red
        espera y reanuda desde lo ya descargado.
        """
        what = f"descarga de {key}"
        part.parent.mkdir(parents=True, exist_ok=True)
        done = part.stat().st_size if part.exists() else 0
        if done > length:  # parte de otra versión del archivo: empezar de cero
            part.unlink()
            done = 0
        progress = _Progress(label, length, done)
        failures = 0
        try:
            while done < length:
                before, error = done, None
                try:
                    done = self._stream_into(key, start, length, done, part, progress, what)
                except _NETWORK_ERRORS as caught:
                    error = caught
                    done = part.stat().st_size if part.exists() else 0
                if done > before:
                    failures = 0
                    if error is None:
                        continue
                failures += 1
                problem = error or "respuesta vacía"
                if failures >= MAX_RETRIES:
                    raise DownloadError(
                        f"{what}: la red falló {MAX_RETRIES} veces seguidas ({_short(problem)}). "
                        f"Lo descargado queda en {part} y se reanuda al volver a ejecutar."
                    )
                _wait(failures, what, problem)
        finally:
            progress.close()
        if done != length:
            part.unlink(missing_ok=True)
            raise DownloadError(f"{what}: se esperaban {length} bytes y llegaron {done}. Se borró la descarga.")

    def _stream_into(
        self, key: str, start: int, length: int, done: int, part: Path, progress: _Progress, what: str
    ) -> int:
        """Una petición de descarga. Devuelve cuántos bytes hay ya en `part`."""
        first = start + done
        headers = {"Range": f"bytes={first}-{start + length - 1}"}
        with self.session.get(self.url(key), headers=headers, stream=True, timeout=TIMEOUT) as response:
            mode = "ab"
            if response.status_code == 200 and start == 0:
                done, mode = 0, "wb"  # el servidor ignoró el rango y manda el archivo entero
            else:
                _check_status(response, (206,), what)
                content_range = response.headers.get("Content-Range", "")
                if content_range and not content_range.startswith(f"bytes {first}-"):
                    raise DownloadError(f"{what}: el servidor devolvió otro rango ({content_range}).")
            with open(part, mode) as out:
                for chunk in response.iter_content(CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    progress.update(done)
        return done


def resolve_month(month: str, dataset: str, bucket: Bucket) -> Source:
    """Decide de qué archivo remoto sale un mes, mirando el listado real."""
    kind, remote = find_key(month, dataset, bucket.listing())
    members = bucket.zip_members(remote)
    if kind == "mensual":
        csvs = tuple(m for m in members if _is_csv(m.name))
        if not csvs:
            raise DownloadError(f"{remote.key} no trae ningún CSV.")
        return Source(month, remote, csvs=csvs)

    nested = [m for m in members if _nested_month(m.name) == month]
    if len(nested) != 1:
        inside = sorted({_nested_month(m.name) for m in members} - {None})
        problem = "no trae el mes" if not nested else "trae más de un zip para el mes"
        raise DownloadError(f"{remote.key} {problem} {month}. Meses que trae: {', '.join(inside) or 'ninguno'}.")
    if nested[0].compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise DownloadError(f"{remote.key}: {nested[0].name} usa una compresión no admitida.")
    return Source(month, remote, nested=nested[0])


def describe_source(source: Source) -> str:
    """Dos líneas que dicen de dónde sale el mes y cómo se obtiene."""
    remote = f"{source.remote.key} ({source.remote.size / MB:.1f} MB)"
    indent = " " * (len(source.month) + 2)
    if source.nested:
        return (
            f"{source.month}: dentro del zip anual {remote}, miembro {source.nested.name}.\n"
            f"{indent}Se bajan solo {source.transfer_bytes / MB:.1f} MB por rango HTTP "
            f"({source.nested.file_size / MB:.1f} MB al inflar)."
        )
    packed = "sin compresión" if all(m.compress_type == zipfile.ZIP_STORED for m in source.csvs) else "comprimidos"
    total = sum(m.file_size for m in source.csvs)
    return (
        f"{source.month}: zip mensual {remote}. Se descarga completo.\n"
        f"{indent}{len(source.csvs)} CSV {packed}, {total / MB:.1f} MB al extraer."
    )


# --------------------------------------------------------------------------- #
# Zip local, CSV y Parquet
# --------------------------------------------------------------------------- #

def _free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


def check_disk(folder: Path, needed: int, what: str) -> None:
    free = _free_bytes(folder)
    if free < needed:
        raise DownloadError(
            f"Espacio insuficiente en {folder}: {what} necesita unos {needed / MB:.0f} MB "
            f"y hay {free / MB:.0f} MB libres."
        )


def unpack_member(part: Path, target: Path, member: Member) -> None:
    """Convierte los bytes crudos de un miembro de zip en el archivo final.

    Infla si venía con deflate y comprueba tamaño y CRC contra el directorio
    central. Si algo no cuadra, no deja el archivo final.
    """
    inflater = zlib.decompressobj(-15) if member.compress_type == zipfile.ZIP_DEFLATED else None
    temporary = target.with_name(target.name + ".tmp")
    crc = size = 0
    try:
        with open(part, "rb") as source, open(temporary, "wb") as out:
            while True:
                chunk = source.read(CHUNK)
                data = (inflater.decompress(chunk) if chunk else inflater.flush()) if inflater else chunk
                out.write(data)
                crc = zlib.crc32(data, crc)
                size += len(data)
                if not chunk:
                    break
        if size != member.file_size or crc != member.crc:
            raise DownloadError(
                f"{target.name}: el tamaño o el CRC no coinciden con el zip anual. Descarga dañada."
            )
        os.replace(temporary, target)
    except zlib.error as error:
        raise DownloadError(f"{target.name}: no se pudo inflar ({error}). Descarga dañada.") from error
    finally:
        temporary.unlink(missing_ok=True)


def fetch_zip(source: Source, bucket: Bucket, zip_path: Path) -> None:
    """Deja en `zip_path` el zip mensual. Reutiliza uno ya descargado si el tamaño coincide."""
    expected = source.nested.file_size if source.nested else source.remote.size
    if zip_path.exists() and zip_path.stat().st_size == expected:
        print(f"{source.month}: se reutiliza {zip_path.name}, ya descargado.", flush=True)
        return
    part = zip_path.with_name(zip_path.name + ".part")
    label = f"{source.month} descarga"
    if source.nested is None:
        bucket.download_range(source.remote.key, 0, source.remote.size, part, label)
        os.replace(part, zip_path)
        return
    start = bucket.member_data_start(source.remote.key, source.nested)
    bucket.download_range(source.remote.key, start, source.nested.compress_size, part, label)
    try:
        unpack_member(part, zip_path, source.nested)
    except DownloadError:
        part.unlink(missing_ok=True)  # dañada: la próxima vez se baja de nuevo
        raise
    part.unlink()


def check_header(line: bytes, name: str) -> None:
    """Falla si la primera línea de un CSV no es exactamente el encabezado esperado."""
    text = line.decode("utf-8-sig", errors="replace").rstrip("\r\n")
    header = tuple(next(csv.reader([text]), []))
    if header != EXPECTED_COLUMNS:
        raise DownloadError(
            f"{name}: el encabezado no es el esperado.\n"
            f"  esperado: {','.join(EXPECTED_COLUMNS)}\n"
            f"  recibido: {text[:300]}"
        )


def extract_csvs(zip_path: Path, folder: Path) -> list[Path]:
    """Extrae todos los CSV del zip a `folder`. Ignora __MACOSX, .DS_Store y carpetas.

    El encabezado se comprueba antes de copiar cada archivo. `zipfile`
    verifica el CRC al leer cada miembro completo.
    """
    paths = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted((i for i in archive.infolist() if _is_csv(i.filename)), key=lambda i: i.filename)
        if not members:
            raise DownloadError(f"{zip_path.name} no trae ningún CSV.")
        check_disk(folder, int(1.5 * sum(i.file_size for i in members)), f"extraer {zip_path.name}")
        for number, info in enumerate(members, start=1):
            target = folder / f"{number:02d}_{Path(info.filename).name}"
            with archive.open(info) as source, open(target, "wb") as out:
                first = source.readline()
                check_header(first, info.filename)
                out.write(first)
                shutil.copyfileobj(source, out, CHUNK)
            paths.append(target)
    return paths


def count_data_rows(csv_path: Path) -> int:
    """Filas de datos de un CSV: líneas no vacías menos el encabezado.

    Es un conteo independiente de DuckDB para comprobar que no se pierde
    ninguna fila. Supone que ningún campo trae saltos de línea.
    """
    with open(csv_path, "rb") as file:
        lines = sum(1 for line in file if line.strip())
    return max(lines - 1, 0)


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _was_interrupted(error: Exception) -> bool:
    # DuckDB convierte Ctrl+C en un RuntimeError "Query interrupted".
    if isinstance(error, duckdb.InterruptException):
        return True
    return type(error) is RuntimeError and str(error).strip().lower().startswith("query interrupted")


def csvs_to_parquet(csv_paths: list[Path], parquet: Path, expected_rows: int, work_folder: Path):
    """Une los CSV de un mes en un Parquet. Devuelve (filas, min y max de started_at).

    Los tipos son explícitos (nada se adivina) y no se limpia ni se filtra.
    Se escribe con nombre temporal y se renombra al final: nunca queda un
    Parquet a medias.
    """
    temporary = parquet.with_name(parquet.name + ".tmp")
    files = "[" + ", ".join(_sql_text(path) for path in csv_paths) + "]"
    columns = "{" + ", ".join(f"{_sql_text(name)}: {_sql_text(kind)}" for name, kind in COLUMN_TYPES.items()) + "}"
    connection = duckdb.connect()
    try:
        connection.execute(f"SET temp_directory = {_sql_text(work_folder / 'duckdb')}")
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_csv(
                    {files}, columns = {columns}, header = true, auto_detect = false,
                    delim = ',', quote = '"', escape = '"', nullstr = ''
                )
            ) TO {_sql_text(temporary)} (FORMAT parquet, COMPRESSION zstd)
            """
        )
        rows, first, last = connection.execute(
            "SELECT count(*), min(started_at), max(started_at) FROM read_parquet(?)", [str(temporary)]
        ).fetchone()
        if rows != expected_rows:
            raise DownloadError(
                f"El Parquet tiene {rows} filas y los CSV suman {expected_rows} líneas de datos. No se guarda. "
                "Revisa si algún campo trae saltos de línea."
            )
        os.replace(temporary, parquet)
        return rows, first, last
    except Exception as error:
        if _was_interrupted(error):
            raise KeyboardInterrupt from None
        if isinstance(error, duckdb.Error):
            raise DownloadError(f"DuckDB no pudo convertir los CSV: {str(error)[:600]}") from error
        raise
    finally:
        connection.close()  # sin esto el proceso se queda colgado tras un Ctrl+C
        temporary.unlink(missing_ok=True)


def parquet_stats(parquet: Path):
    """(filas, min y max de started_at) de un Parquet ya escrito."""
    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT count(*), min(started_at), max(started_at) FROM read_parquet(?)", [str(parquet)]
        ).fetchone()
    except duckdb.Error as error:
        raise DownloadError(f"{parquet} existe pero no se pudo leer ({str(error)[:200]}). Usa --forzar.") from error
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Flujo por mes
# --------------------------------------------------------------------------- #

def process_month(
    month: str,
    dataset: str,
    bucket: Bucket,
    out_dir: Path,
    raw_dir: Path,
    *,
    keep_zip: bool = False,
    force: bool = False,
) -> MonthResult:
    """Descarga y convierte un mes. Lanza DownloadError con un mensaje claro si no puede."""
    check_supported(month)  # antes de tocar la red
    parquet = out_dir / f"{month}.parquet"
    if parquet.exists() and not force:
        rows, first, last = parquet_stats(parquet)
        print(f"{month}: ya existe {parquet}. Se omite (usa --forzar para rehacerlo).", flush=True)
        return MonthResult(month, "ya existía", "", rows, first, last, parquet.stat().st_size)

    source = resolve_month(month, dataset, bucket)
    print(describe_source(source), flush=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    check_disk(raw_dir, source.disk_bytes, f"el mes {month}")

    zip_path = raw_dir / ZIPS_FOLDER / source.zip_name
    work_folder = raw_dir / f"_tmp_{dataset}_{month}"
    try:
        fetch_zip(source, bucket, zip_path)
        shutil.rmtree(work_folder, ignore_errors=True)
        work_folder.mkdir(parents=True)
        try:
            csv_paths = extract_csvs(zip_path, work_folder)
        except (zipfile.BadZipFile, zlib.error) as error:
            zip_path.unlink(missing_ok=True)  # ni con --conservar-zip: está dañado
            raise DownloadError(
                f"{zip_path.name} está dañado ({error}). Se borró. Vuelve a ejecutar para bajarlo de nuevo."
            ) from error
        expected_rows = sum(count_data_rows(path) for path in csv_paths)
        print(f"{month}: {len(csv_paths)} CSV, {expected_rows:,} filas. Escribiendo Parquet...", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, first, last = csvs_to_parquet(csv_paths, parquet, expected_rows, work_folder)
    finally:
        # Pase lo que pase (error o Ctrl+C) no quedan CSV temporales.
        shutil.rmtree(work_folder, ignore_errors=True)
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
    return MonthResult(month, "ok", "", rows, first, last, parquet.stat().st_size)


def run(
    months: list[str],
    dataset: str,
    bucket: Bucket,
    out_dir: Path,
    raw_dir: Path,
    *,
    keep_zip: bool = False,
    force: bool = False,
) -> list[MonthResult]:
    """Procesa los meses uno por uno. Un mes que falla no detiene a los demás."""
    results: list[MonthResult] = []
    for month in months:
        started = time.time()  # reloj de pared: es el tiempo que la persona esperó
        print(f"\n== {month} ({dataset}) ==", flush=True)
        try:
            result = process_month(month, dataset, bucket, out_dir, raw_dir, keep_zip=keep_zip, force=force)
        except KeyboardInterrupt:
            print(
                f"\n{month}: interrumpido por el usuario. Si había una descarga a medias, queda como "
                f".part en {raw_dir / ZIPS_FOLDER} y se reanuda al volver a ejecutar.",
                flush=True,
            )
            results.append(MonthResult(month, "interrumpido", seconds=time.time() - started))
            break
        except DownloadError as error:
            result = MonthResult(month, "error", str(error))
        except Exception as error:  # un fallo inesperado tampoco detiene a los demás meses
            result = MonthResult(month, "error", f"{type(error).__name__}: {error}")
        result.seconds = time.time() - started
        if result.status == "error":
            print(f"{month}: ERROR. {result.detail}", flush=True)
        elif result.status == "ok":
            size = result.parquet_bytes / MB
            print(f"{month}: listo en {result.seconds:.0f} s. {result.rows:,} filas, {size:.1f} MB.", flush=True)
        results.append(result)
    return results


def format_summary(results: list[MonthResult]) -> str:
    """Tabla final: mes, filas, rango de started_at, tamaño y estado."""

    def moment(value: datetime | None) -> str:
        return value.isoformat(sep=" ", timespec="milliseconds") if value else "-"

    header = ("Mes", "Filas", "started_at mín", "started_at máx", "Parquet MB", "Estado")
    table = [header]
    for r in results:
        status = r.status if not r.detail else f"{r.status}: {r.detail.splitlines()[0]}"
        table.append(
            (
                r.month,
                f"{r.rows:,}" if r.rows is not None else "-",
                moment(r.min_started),
                moment(r.max_started),
                f"{r.parquet_bytes / MB:.1f}" if r.parquet_bytes is not None else "-",
                status,
            )
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(header) - 1)]
    lines = []
    for row in table:
        cells = [row[0].ljust(widths[0]), row[1].rjust(widths[1]), row[2].ljust(widths[2]),
                 row[3].ljust(widths[3]), row[4].rjust(widths[4]), row[5]]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def list_months(months: list[str], dataset: str, bucket: Bucket) -> int:
    """--listar: qué meses hay en el índice y cómo se obtendría cada mes pedido."""
    failed = False
    try:
        objects = bucket.listing()
        print(f"Índice: {bucket.base_url} ({len(objects)} archivos)")
        print(describe_available(objects, dataset))
        if dataset == "nyc":
            yearly = _keys_by_period(objects, _YEARLY_KEY)
            for year in sorted(y for y in yearly if y >= FIRST_MONTH[:4]):
                for remote in yearly[year]:
                    inside = {_nested_month(m.name) for m in bucket.zip_members(remote)} - {None}
                    usable = sorted(m for m in inside if m >= FIRST_MONTH)
                    print(f"  {remote.key} ({remote.size / MB:.1f} MB): {', '.join(usable) or 'sin meses'}")
        monthly = sorted(m for m in _keys_by_period(objects, _MONTHLY_KEY[dataset]) if m >= FIRST_MONTH)
        if monthly:
            print(f"  zips mensuales ({len(monthly)}):")
            print(textwrap.fill(", ".join(monthly), width=100, initial_indent="    ", subsequent_indent="    "))
    except DownloadError as error:
        print(f"ERROR. {error}")
        return 1
    for month in months:
        print()
        try:
            print(describe_source(resolve_month(month, dataset, bucket)))
        except DownloadError as error:
            failed = True
            print(f"{month}: ERROR. {error}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_trips.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python scripts/download_trips.py [opciones] [AAAAMM ...]",
        description=(
            "Descarga viajes de Citi Bike por mes y deja un Parquet por mes (capa cruda, sin limpiar).\n"
            f"Solo meses desde {FIRST_MONTH}."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python scripts/download_trips.py 202604\n"
            "  python scripts/download_trips.py --desde 202601 --hasta 202609\n"
            "  python scripts/download_trips.py --listar 202609 202102\n"
            "  python scripts/download_trips.py --dataset jc 202609     (prueba rápida, pocos MB)"
        ),
    )
    months = parser.add_argument_group("meses")
    months.add_argument(
        "meses", nargs="*", type=parse_month, metavar="AAAAMM", help="uno o más meses, por ejemplo 202604"
    )
    months.add_argument(
        "--desde", type=parse_month, metavar="AAAAMM", help="primer mes de un rango (se usa con --hasta)"
    )
    months.add_argument("--hasta", type=parse_month, metavar="AAAAMM", help="último mes del rango, incluido")
    options = parser.add_argument_group("opciones")
    options.add_argument(
        "--dataset", choices=DATASETS, default="nyc",
        help="nyc = Nueva York (por defecto); jc = Jersey City, archivos pequeños para pruebas",
    )
    options.add_argument(
        "--listar", action="store_true",
        help="muestra qué meses hay en el índice y cómo se obtendría cada mes pedido; no descarga nada",
    )
    options.add_argument(
        "--salida", metavar="DIR",
        help=(
            "carpeta de los Parquet (por defecto data/raw/trips, o data/raw/trips_jc con --dataset jc). "
            "Con --dataset jc se escribe en DIR/trips_jc, para no mezclar los dos datasets"
        ),
    )
    options.add_argument(
        "--conservar-zip", action="store_true", help="no borra el zip descargado (queda en data/raw/zips)"
    )
    options.add_argument("--forzar", action="store_true", help="rehace el mes aunque su Parquet ya exista")
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    return parser


def main(argv: list[str] | None = None, *, base_url: str = BUCKET_URL, raw_dir: Path = DATA_RAW) -> int:
    """Punto de entrada. Devuelve 0 si todo salió bien, 1 si algún mes falló, 130 si se interrumpió."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.desde) != bool(args.hasta):
        parser.error("--desde y --hasta se usan juntos.")
    months = set(args.meses)
    if args.desde:
        if args.desde > args.hasta:
            parser.error("--desde no puede ser posterior a --hasta.")
        months.update(month_range(args.desde, args.hasta))
    months = sorted(months)
    bucket = Bucket(base_url)

    if args.listar:
        try:
            return list_months(months, args.dataset, bucket)
        except KeyboardInterrupt:
            return 130
    if not months:
        parser.error("indica al menos un mes (AAAAMM) o un rango con --desde y --hasta.")

    raw_dir = Path(raw_dir)
    out_dir = Path(args.salida).expanduser() if args.salida else raw_dir / OUT_FOLDER[args.dataset]
    if args.salida and args.dataset != "nyc":
        out_dir = out_dir / OUT_FOLDER[args.dataset]  # jc nunca comparte carpeta con nyc: el archivo se llama igual
    results = run(months, args.dataset, bucket, out_dir, raw_dir, keep_zip=args.conservar_zip, force=args.forzar)

    print("\n" + format_summary(results))
    print(f"\nParquet en: {out_dir}")
    if any(r.status == "interrumpido" for r in results):
        return 130
    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
