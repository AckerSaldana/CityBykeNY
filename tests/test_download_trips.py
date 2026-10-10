"""Pruebas de la descarga de viajes (src/data/download_trips.py).

No usan internet. Un servidor HTTP local imita al bucket: entrega el listado
XML y sirve zips pequeños de mentira, con soporte de rangos (Range), que es
lo que usa la descarga de un mes guardado dentro de un zip anual.

Los zips y sus filas son INVENTADOS para la prueba. Lo único real es
tests/fixtures/s3_listing.xml, una copia del listado del bucket del
2026-10-09.
"""

from __future__ import annotations

import io
import os
import re
import struct
import subprocess
import sys
import threading
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote
from xml.sax.saxutils import escape

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data import download_trips as dt

REPO = Path(__file__).resolve().parents[1]
REAL_LISTING = Path(__file__).parent / "fixtures" / "s3_listing.xml"
HEADER = ",".join(dt.EXPECTED_COLUMNS)


# --------------------------------------------------------------------------- #
# Bucket de mentira
# --------------------------------------------------------------------------- #

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # sin ruido en la salida de pytest
        pass

    def do_GET(self):
        fake = self.server.fake
        path, _, query = self.path.partition("?")
        path = unquote(path)
        if path == "/tripdata":
            marker = parse_qs(query).get("marker", [""])[0]
            fake.listings.append(marker)
            return self._send(200, fake.listing_xml(marker))
        file = fake.root / path.removeprefix("/tripdata/")
        if not file.is_file():
            return self._send(404, b"NoSuchKey")
        data = file.read_bytes()
        range_header = self.headers.get("Range")
        status, start, body, extra = 200, 0, data, {}
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            start, end = int(match[1]), min(int(match[2]), len(data) - 1)
            status, body = 206, data[start : end + 1]
            extra = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        cut = fake.cut.get(file.name)
        if cut and cut[0] == start:
            # Promete el cuerpo completo y corta la conexión a la mitad (una sola vez).
            del fake.cut[file.name]
            fake.log.append((file.name, range_header, cut[1]))
            return self._send(status, body, extra, only=cut[1])
        fake.log.append((file.name, range_header, len(body)))
        self._send(status, body, extra)

    def _send(self, status, body, headers=None, only=None):
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if only is None:
            self.wfile.write(body)
        else:
            self.wfile.write(body[:only])
            self.wfile.flush()
            self.close_connection = True


class FakeBucket:
    """Servidor local que sirve los archivos de una carpeta como si fuera el bucket."""

    def __init__(self, root: Path):
        self.root = root
        self.log: list[tuple[str, str | None, int]] = []  # (archivo, Range pedido, bytes enviados)
        self.listings: list[str] = []                      # marker de cada petición de listado
        self.cut: dict[str, tuple[int, int]] = {}          # archivo -> (inicio del rango, bytes antes de cortar)
        self.page_size = 1000
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.fake = self
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/tripdata"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def listing_xml(self, marker: str) -> bytes:
        keys = sorted(p.name for p in self.root.iterdir() if p.name > marker)
        page = keys[: self.page_size]
        items = "".join(
            f"<Contents><Key>{escape(key)}</Key><Size>{(self.root / key).stat().st_size}</Size></Contents>"
            for key in page
        )
        truncated = "true" if len(keys) > len(page) else "false"
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>tripdata</Name><IsTruncated>{truncated}</IsTruncated>{items}</ListBucketResult>"
        ).encode()

    def sent(self, key: str) -> int:
        """Bytes enviados de un archivo, sumando todas las peticiones."""
        return sum(n for name, _, n in self.log if name == key)

    def ranges(self, key: str) -> list[str | None]:
        return [range_header for name, range_header, _ in self.log if name == key]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def bucket(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    fake = FakeBucket(root)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def _rapido_y_local(monkeypatch):
    monkeypatch.setattr(dt, "BACKOFF_SECONDS", 0.0)  # reintentos sin esperar
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture()
def real_objects():
    objects, truncated = dt.parse_listing(REAL_LISTING.read_bytes())
    assert not truncated
    return objects


# --------------------------------------------------------------------------- #
# Datos de mentira
# --------------------------------------------------------------------------- #

def row(
    ride_id: str,
    start_id: str = "6289.06",
    end_id: str = "6331.01",
    started: str = "2026-09-16 15:27:34.365",
    ended: str = "2026-09-16 15:30:55.260",
    start_name: str = "Broadway & W 29 St",
) -> str:
    return (
        f"{ride_id},electric_bike,{started},{ended},{start_name},{start_id},"
        f"W 31 St & 7 Ave,{end_id},40.7462009,-73.98855723,40.749156,-73.9916,member"
    )


def csv_bytes(rows: list[str], header: str = HEADER, newline: str = "\n") -> bytes:
    return (newline.join([header, *rows]) + newline).encode()


def many_rows(prefix: str, count: int) -> list[str]:
    return [row(f"{prefix}{i:06d}") for i in range(count)]


# Los zips anuales reales traen 32 bytes de campo extra en cada encabezado local.
LOCAL_EXTRA = struct.pack("<HH", 0x7875, 28) + bytes(28)


def zip_bytes(members: dict[str, bytes], compression: int = zipfile.ZIP_STORED, extra: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            info.extra = extra
            archive.writestr(info, data)
    return buffer.getvalue()


def put(
    bucket: FakeBucket,
    key: str,
    members: dict[str, bytes],
    compression: int = zipfile.ZIP_STORED,
    extra: bytes = b"",
) -> Path:
    path = bucket.root / key
    path.write_bytes(zip_bytes(members, compression, extra))
    return path


def put_yearly(bucket: FakeBucket, inner: bytes, compression: int = zipfile.ZIP_DEFLATED) -> Path:
    """Zip anual como los reales: carpeta, un zip por mes, basura de macOS y campo extra local."""
    filler = os.urandom(300_000)  # otros meses: bytes que no se comprimen y que NO deben bajarse
    return put(
        bucket,
        "2021-citibike-tripdata.zip",
        {
            "2021-citibike-tripdata/": b"",
            "2021-citibike-tripdata/202107-citibike-tripdata.zip": filler,
            "__MACOSX/2021-citibike-tripdata/._202107-citibike-tripdata.zip": b"basura",
            "2021-citibike-tripdata/.DS_Store": b"basura",
            "__MACOSX/2021-citibike-tripdata/._.DS_Store": b"basura",
            "2021-citibike-tripdata/202102-citibike-tripdata.zip": inner,
            "__MACOSX/2021-citibike-tripdata/._202102-citibike-tripdata.zip": b"basura",
            "2021-citibike-tripdata/202103-citibike-tripdata.zip": filler,
        },
        compression,
        LOCAL_EXTRA,
    )


def cli(bucket: FakeBucket, tmp_path: Path, *args: str) -> int:
    return dt.main(list(args), base_url=bucket.url, raw_dir=tmp_path / "raw")


def leftovers(tmp_path: Path) -> list[str]:
    """Todo lo que quedó en data/raw aparte de los Parquet finales."""
    raw = tmp_path / "raw"
    return sorted(
        str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file() and p.suffix != ".parquet"
    ) + sorted(p.name for p in raw.glob("_tmp_*"))


# --------------------------------------------------------------------------- #
# Listado real y ubicación de cada mes
# --------------------------------------------------------------------------- #

def test_el_listado_real_se_lee_completo(real_objects):
    sizes = {obj.key: obj.size for obj in real_objects}
    assert len(real_objects) == 178
    assert sizes["202604-citibike-tripdata.zip"] == 164592563
    assert sizes["2021-citibike-tripdata.zip"] == 1074415437


@pytest.mark.parametrize(
    "dataset, month, kind, key",
    [
        ("nyc", "202609", "mensual", "202609-citibike-tripdata.zip"),
        ("nyc", "202604", "mensual", "202604-citibike-tripdata.zip"),
        ("nyc", "202401", "mensual", "202401-citibike-tripdata.zip"),
        ("nyc", "202102", "anual", "2021-citibike-tripdata.zip"),
        ("nyc", "202312", "anual", "2023-citibike-tripdata.zip"),
        ("jc", "202609", "mensual", "JC-202609-citibike-tripdata.csv.zip"),
        ("jc", "202604", "mensual", "JC-202604-citibike-tripdata.zip"),
        ("jc", "202510", "mensual", "JC-202510-citibike-tripdata.zip"),
        ("jc", "202207", "mensual", "JC-202207-citbike-tripdata.csv.zip"),  # errata real del bucket
    ],
)
def test_cada_mes_se_ubica_en_el_listado_real(real_objects, dataset, month, kind, key):
    found_kind, remote = dt.find_key(month, dataset, real_objects)
    assert (found_kind, remote.key) == (kind, key)


@pytest.mark.parametrize("dataset", ["nyc", "jc"])
@pytest.mark.parametrize("month", ["202101", "201912", "201306"])
def test_los_meses_anteriores_a_202102_se_rechazan(real_objects, dataset, month):
    with pytest.raises(dt.DownloadError, match="anterior a 202102"):
        dt.find_key(month, dataset, real_objects)


def test_un_mes_que_no_existe_dice_que_rango_hay(real_objects):
    with pytest.raises(dt.DownloadError) as nyc:
        dt.find_key("202610", "nyc", real_objects)
    assert "202102 a 202312 dentro de zips anuales" in str(nyc.value)
    assert "202401 a 202609 en zips mensuales" in str(nyc.value)
    with pytest.raises(dt.DownloadError) as jc:
        dt.find_key("202610", "jc", real_objects)
    assert "202102 a 202609 en zips mensuales" in str(jc.value)
    assert "faltan" not in str(jc.value)


def test_un_listado_en_varias_paginas_se_lee_entero(bucket):
    for month in ("202401", "202402", "202403", "202404", "202405"):
        put(bucket, f"{month}-citibike-tripdata.zip", {"a.csv": csv_bytes([])})
    bucket.page_size = 2
    keys = [obj.key for obj in dt.Bucket(bucket.url).listing()]
    assert keys == sorted(p.name for p in bucket.root.iterdir())
    assert bucket.listings == ["", "202402-citibike-tripdata.zip", "202404-citibike-tripdata.zip"]


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def test_rango_de_meses_incluye_ambos_extremos():
    assert dt.month_range("202511", "202602") == ["202511", "202512", "202601", "202602"]
    assert dt.month_range("202604", "202604") == ["202604"]


@pytest.mark.parametrize(
    "args",
    [["2026-09"], ["202613"], ["20269"], ["--desde", "202601"], ["--desde", "202604", "--hasta", "202601"], []],
)
def test_argumentos_invalidos_terminan_con_error(bucket, tmp_path, args):
    with pytest.raises(SystemExit) as exit_info:
        cli(bucket, tmp_path, *args)
    assert exit_info.value.code == 2
    assert bucket.log == [] and bucket.listings == []


def test_el_script_corre_desde_cualquier_carpeta(tmp_path):
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "download_trips.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "AAAAMM" in done.stdout and "--conservar-zip" in done.stdout


def test_listar_no_descarga(bucket, tmp_path, capsys):
    yearly = put_yearly(bucket, zip_bytes({"202102-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 50))}))
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("B", 50))})

    assert cli(bucket, tmp_path, "--listar", "202609", "202102") == 0
    out = capsys.readouterr().out
    assert "202609-citibike-tripdata.zip" in out and "Se descarga completo" in out
    assert "2021-citibike-tripdata/202102-citibike-tripdata.zip" in out and "rango HTTP" in out
    assert "2021-citibike-tripdata.zip" in out and "202102, 202103, 202107" in out  # meses dentro del zip anual
    assert not (tmp_path / "raw").exists()
    assert bucket.sent(yearly.name) < 0.05 * yearly.stat().st_size

    assert cli(bucket, tmp_path, "--listar", "202610") == 1
    assert "202610 no está en el índice" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Descarga y conversión
# --------------------------------------------------------------------------- #

def test_zip_mensual_sin_compresion_conserva_texto_nulos_y_milisegundos(bucket, tmp_path):
    first = [
        row("R1", start_id="5470.10"),
        row("R2", start_id="JC013", end_id=""),
        row("R3", start_id="SYS038", start_name='"Calle, con coma"', started="2026-09-30 23:59:59.999"),
    ]
    second = [row("R4", end_id="HB101"), row("R5", end_id='""', started="2026-09-01 00:00:00.001")]
    put(
        bucket,
        "202609-citibike-tripdata.zip",
        {
            "202609-citibike-tripdata_2.csv": csv_bytes(second, newline="\r\n"),
            "202609-citibike-tripdata_1.csv": csv_bytes(first),
        },
    )

    assert cli(bucket, tmp_path, "202609") == 0

    parquet = tmp_path / "raw" / "trips" / "202609.parquet"
    table = pq.read_table(parquet)
    rows = {r["ride_id"]: r for r in table.to_pylist()}
    assert table.schema.names == list(dt.EXPECTED_COLUMNS)
    assert sorted(rows) == ["R1", "R2", "R3", "R4", "R5"]  # filas de los dos CSV, sin perder ninguna

    # Los ids son texto: no se pierde el cero final ni se rompen los que traen letras.
    for column in ("ride_id", "start_station_id", "end_station_id", "start_station_name", "member_casual"):
        assert table.schema.field(column).type == pa.string()
    assert rows["R1"]["start_station_id"] == "5470.10"
    assert rows["R2"]["start_station_id"] == "JC013"
    assert rows["R3"]["start_station_id"] == "SYS038"
    assert rows["R4"]["end_station_id"] == "HB101"
    assert rows["R3"]["start_station_name"] == "Calle, con coma"

    # Un id vacío queda como nulo, no como texto vacío.
    assert rows["R2"]["end_station_id"] is None
    assert rows["R5"]["end_station_id"] is None
    assert table["end_station_id"].null_count == 2

    # Fechas con milisegundos y SIN zona horaria.
    for column in ("started_at", "ended_at"):
        assert table.schema.field(column).type == pa.timestamp("us")
        assert table.schema.field(column).type.tz is None
    assert rows["R1"]["started_at"] == datetime(2026, 9, 16, 15, 27, 34, 365000)
    assert rows["R1"]["ended_at"] == datetime(2026, 9, 16, 15, 30, 55, 260000)
    assert rows["R3"]["started_at"] == datetime(2026, 9, 30, 23, 59, 59, 999000)
    assert table.schema.field("start_lat").type == pa.float64()
    assert rows["R1"]["start_lng"] == -73.98855723

    assert leftovers(tmp_path) == []  # ni zip, ni .part, ni CSV temporales


def test_zip_mensual_con_deflate_y_nombres_part(bucket, tmp_path, capsys):
    put(
        bucket,
        "202604-citibike-tripdata.zip",
        {
            "202604-citibike-tripdata-part1.csv": csv_bytes(many_rows("A", 700)),
            "202604-citibike-tripdata-part2.csv": b"\xef\xbb\xbf" + csv_bytes(many_rows("B", 300)),  # con BOM
            "202604-citibike-tripdata-part3.csv": csv_bytes(many_rows("C", 1)),
        },
        zipfile.ZIP_DEFLATED,
    )
    assert cli(bucket, tmp_path, "202604") == 0
    table = pq.read_table(tmp_path / "raw" / "trips" / "202604.parquet")
    assert table.num_rows == 1001
    assert len(set(table["ride_id"].to_pylist())) == 1001
    assert "1,001" in capsys.readouterr().out  # la tabla final muestra las filas
    assert leftovers(tmp_path) == []


def test_mes_dentro_del_zip_anual_solo_baja_ese_mes(bucket, tmp_path):
    inner = zip_bytes(
        {
            "202102-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 400) + [row("X1", start_id="5470.10")]),
            "202102-citibike-tripdata_2.csv": csv_bytes(many_rows("B", 200)),
        }
    )
    yearly = put_yearly(bucket, inner)

    assert cli(bucket, tmp_path, "202102") == 0

    table = pq.read_table(tmp_path / "raw" / "trips" / "202102.parquet")
    assert table.num_rows == 601
    assert "5470.10" in table["start_station_id"].to_pylist()
    # Todo se pidió por rangos y se transfirió una fracción pequeña del zip anual.
    assert all(range_header for range_header in bucket.ranges(yearly.name))
    assert bucket.sent(yearly.name) < 0.2 * yearly.stat().st_size
    assert bucket.sent(yearly.name) < len(inner)  # viajó comprimido
    assert leftovers(tmp_path) == []


def test_mes_anidado_danado_se_detecta_por_crc(bucket, tmp_path, capsys):
    inner = zip_bytes({"202102-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 300))})
    yearly = put_yearly(bucket, inner, zipfile.ZIP_STORED)
    # Se cambia un byte del mes guardado dentro del zip anual, ya publicado.
    data = bytearray(yearly.read_bytes())
    position = data.index(inner) + len(inner) // 2
    data[position] ^= 0xFF
    yearly.write_bytes(bytes(data))

    assert cli(bucket, tmp_path, "202102") == 1
    # El mensaje es el de unpack_member; el de zipfile ("Bad CRC-32") no basta.
    assert "no coinciden con el zip anual" in capsys.readouterr().out
    assert not (tmp_path / "raw" / "trips" / "202102.parquet").exists()
    assert leftovers(tmp_path) == []


def test_jersey_city_con_los_dos_sufijos(bucket, tmp_path):
    put(
        bucket,
        "JC-202609-citibike-tripdata.csv.zip",
        {
            "JC-202609-citibike-tripdata.csv": csv_bytes([row("J1", start_id="JC013"), row("J2", start_id="HB101")]),
            "__MACOSX/._JC-202609-citibike-tripdata.csv": b"\x00\x05\x16\x07 no es un CSV",
        },
        zipfile.ZIP_DEFLATED,
    )
    put(bucket, "JC-202604-citibike-tripdata.zip", {"JC-202604-citibike-tripdata.csv": csv_bytes([row("J3")])})

    assert cli(bucket, tmp_path, "--dataset", "jc", "202604", "202609") == 0
    out_dir = tmp_path / "raw" / "trips_jc"
    assert pq.read_table(out_dir / "202609.parquet")["start_station_id"].to_pylist() == ["JC013", "HB101"]
    assert pq.read_table(out_dir / "202604.parquet").num_rows == 1


def test_salida_cambia_la_carpeta_de_los_parquet(bucket, tmp_path):
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    out_dir = tmp_path / "con espacio"
    assert cli(bucket, tmp_path, "202609", "--salida", str(out_dir)) == 0
    assert pq.read_table(out_dir / "202609.parquet").num_rows == 20
    assert not (tmp_path / "raw" / "trips").exists()
    assert leftovers(tmp_path) == []


def test_con_la_misma_salida_jersey_city_no_se_mezcla_con_nueva_york(bucket, tmp_path, capsys):
    # Los dos datasets nombran el archivo igual (<AAAAMM>.parquet): jc va a su subcarpeta.
    put(bucket, "JC-202609-citibike-tripdata.zip", {"JC-202609-citibike-tripdata.csv": csv_bytes(many_rows("J", 3))})
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 50))})
    out_dir = tmp_path / "comun"

    assert cli(bucket, tmp_path, "--dataset", "jc", "202609", "--salida", str(out_dir)) == 0
    jc_out = capsys.readouterr().out
    assert cli(bucket, tmp_path, "202609", "--salida", str(out_dir)) == 0
    assert "ya existía" not in capsys.readouterr().out  # el mes de Nueva York sí se descargó

    nyc = pq.read_table(out_dir / "202609.parquet")["ride_id"].to_pylist()
    jc = pq.read_table(out_dir / "trips_jc" / "202609.parquet")["ride_id"].to_pylist()
    assert sorted(nyc) == [f"A{i:06d}" for i in range(50)]  # las filas de Nueva York, no las de jc
    assert sorted(jc) == ["J000000", "J000001", "J000002"]
    assert len(list(out_dir.rglob("*.parquet"))) == 2
    assert f"Parquet en: {out_dir / 'trips_jc'}" in jc_out  # el mensaje final dice dónde quedó
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "header",
    [
        "tripduration,starttime,stoptime,start station id,start station name,end station id,usertype",  # formato viejo
        HEADER.replace("member_casual", "usertype"),
        HEADER + ",extra",
        HEADER.replace("ride_id,rideable_type", "rideable_type,ride_id"),
        HEADER.replace("started_at", " started_at"),
    ],
)
def test_un_encabezado_distinto_falla_sin_detener_los_demas_meses(bucket, tmp_path, capsys, header):
    put(
        bucket,
        "202605-citibike-tripdata.zip",
        {
            "202605-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 3)),
            "202605-citibike-tripdata_2.csv": csv_bytes(many_rows("B", 3), header=header),
        },
    )
    put(bucket, "202606-citibike-tripdata.zip", {"202606-citibike-tripdata_1.csv": csv_bytes(many_rows("C", 4))})

    assert cli(bucket, tmp_path, "--desde", "202605", "--hasta", "202606") == 1

    out = capsys.readouterr().out
    assert "el encabezado no es el esperado" in out
    out_dir = tmp_path / "raw" / "trips"
    assert not (out_dir / "202605.parquet").exists()
    assert pq.read_table(out_dir / "202606.parquet").num_rows == 4  # el otro mes sí se hizo
    assert leftovers(tmp_path) == []


def test_un_mes_rechazado_o_inexistente_falla_solo_ese_mes(bucket, tmp_path, capsys):
    put(bucket, "202606-citibike-tripdata.zip", {"202606-citibike-tripdata_1.csv": csv_bytes(many_rows("C", 4))})
    assert cli(bucket, tmp_path, "202101", "202606", "202607") == 1
    out = capsys.readouterr().out
    assert "202101 es anterior a 202102" in out
    assert "202607 no está en el índice" in out and "Disponible: 202606 en zips mensuales" in out
    assert [p.name for p in (tmp_path / "raw" / "trips").iterdir()] == ["202606.parquet"]


def test_dos_archivos_para_el_mismo_mes_no_se_elige_a_ciegas(bucket, tmp_path, capsys):
    for key in ("202609-citibike-tripdata.zip", "202609-citibike-tripdata.csv.zip"):
        put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 3))})
    assert cli(bucket, tmp_path, "202609") == 1
    assert "No se elige a ciegas" in capsys.readouterr().out
    assert not list((tmp_path / "raw").rglob("*.parquet"))


def test_si_las_filas_no_cuadran_no_se_guarda(bucket, tmp_path, capsys):
    # Un campo con salto de línea entre comillas: hay más líneas que filas.
    rows = [row("R1"), row("R2", start_name='"Nombre\npartido"')]
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(rows)})
    assert cli(bucket, tmp_path, "202609") == 1
    assert "El Parquet tiene 2 filas y los CSV suman 3" in capsys.readouterr().out
    assert not (tmp_path / "raw" / "trips" / "202609.parquet").exists()
    assert leftovers(tmp_path) == []


def test_la_segunda_corrida_se_salta_el_mes(bucket, tmp_path, capsys):
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    parquet = tmp_path / "raw" / "trips" / "202609.parquet"

    def identity() -> tuple[int, int]:
        info = parquet.stat()
        return info.st_ino, info.st_mtime_ns

    assert cli(bucket, tmp_path, "202609") == 0
    requests_made, written = len(bucket.log) + len(bucket.listings), identity()
    capsys.readouterr()

    assert cli(bucket, tmp_path, "202609") == 0
    out = capsys.readouterr().out
    assert re.search(r"^202609\s+20\s.*ya existía$", out, re.MULTILINE)  # la tabla final trae sus filas
    assert len(bucket.log) + len(bucket.listings) == requests_made  # no tocó la red
    assert identity() == written

    assert cli(bucket, tmp_path, "202609", "--forzar") == 0
    assert len(bucket.log) + len(bucket.listings) > requests_made
    assert identity() != written  # archivo nuevo
    assert pq.read_table(parquet).num_rows == 20


def test_conservar_zip_y_reutilizarlo(bucket, tmp_path):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    whole = f"bytes=0-{remote.stat().st_size - 1}"

    assert cli(bucket, tmp_path, "202609", "--conservar-zip") == 0
    assert leftovers(tmp_path) == [f"zips/{key}"]
    assert (tmp_path / "raw" / "zips" / key).read_bytes() == remote.read_bytes()
    assert bucket.ranges(key).count(whole) == 1

    assert cli(bucket, tmp_path, "202609", "--forzar") == 0  # usa el zip local y luego lo borra
    assert bucket.ranges(key).count(whole) == 1
    assert leftovers(tmp_path) == []


def test_un_zip_guardado_de_otro_tamano_se_baja_de_nuevo(bucket, tmp_path):
    key = "202609-citibike-tripdata.zip"
    put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    stale = tmp_path / "raw" / "zips" / key
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"zip viejo")
    assert cli(bucket, tmp_path, "202609") == 0
    assert pq.read_table(tmp_path / "raw" / "trips" / "202609.parquet").num_rows == 20


def test_un_zip_guardado_y_danado_se_borra_aunque_se_pida_conservarlo(bucket, tmp_path, capsys):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 300))})
    data = bytearray(remote.read_bytes())
    data[len(data) // 2] ^= 0xFF  # mismo tamaño, contenido dañado
    kept = tmp_path / "raw" / "zips" / key
    kept.parent.mkdir(parents=True)
    kept.write_bytes(bytes(data))
    assert cli(bucket, tmp_path, "202609", "--conservar-zip") == 1
    assert "está dañado" in capsys.readouterr().out
    assert leftovers(tmp_path) == []


def test_reanuda_una_descarga_a_medias(bucket, tmp_path, capsys):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 3000))})
    data = remote.read_bytes()
    already = len(data) // 3
    part = tmp_path / "raw" / "zips" / (key + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:already])

    assert cli(bucket, tmp_path, "202609") == 0

    assert f"bytes={already}-{len(data) - 1}" in bucket.ranges(key)  # pidió solo lo que faltaba
    assert f"bytes=0-{len(data) - 1}" not in bucket.ranges(key)
    assert "se reanuda" in capsys.readouterr().out
    assert pq.read_table(tmp_path / "raw" / "trips" / "202609.parquet").num_rows == 3000
    assert leftovers(tmp_path) == []


def test_un_part_mas_grande_que_el_archivo_se_descarta(bucket, tmp_path):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 3000))})
    part = tmp_path / "raw" / "zips" / (key + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * (remote.stat().st_size + 100))
    assert cli(bucket, tmp_path, "202609") == 0
    assert pq.read_table(tmp_path / "raw" / "trips" / "202609.parquet").num_rows == 3000
    assert leftovers(tmp_path) == []


def test_si_se_corta_la_conexion_reintenta_desde_donde_iba(bucket, tmp_path, capsys, monkeypatch):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 3000))})
    size = remote.stat().st_size
    monkeypatch.setattr(dt, "CHUNK", 1024)  # bloques pequeños: el zip de prueba pesa menos de 1 MB
    bucket.cut[key] = (0, 5000)             # la primera descarga se corta a los 5000 bytes

    assert cli(bucket, tmp_path, "202609") == 0

    assert (key, f"bytes=0-{size - 1}", 5000) in bucket.log
    # Se conservan los bloques completos (4 x 1024) y se pide solo el resto.
    assert f"bytes=4096-{size - 1}" in bucket.ranges(key)
    assert bucket.ranges(key).count(f"bytes=0-{size - 1}") == 1
    assert "Reintento" in capsys.readouterr().out
    assert pq.read_table(tmp_path / "raw" / "trips" / "202609.parquet").num_rows == 3000


def test_mes_anidado_reanuda_tras_un_corte(bucket, tmp_path, monkeypatch):
    inner = zip_bytes({"202102-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 5000))})
    yearly = put_yearly(bucket, inner)
    # Dónde empiezan y terminan los bytes comprimidos del mes dentro del zip anual.
    with zipfile.ZipFile(yearly) as archive:
        info = archive.getinfo("2021-citibike-tripdata/202102-citibike-tripdata.zip")
    local_header = yearly.read_bytes()[info.header_offset : info.header_offset + 30]
    name_length, extra_length = struct.unpack("<HH", local_header[26:30])
    assert extra_length == len(LOCAL_EXTRA) == 32  # como en los zips anuales reales
    start = info.header_offset + 30 + name_length + extra_length
    end = start + info.compress_size - 1
    monkeypatch.setattr(dt, "CHUNK", 1024)
    bucket.cut[yearly.name] = (start, 3000)  # la descarga del mes se corta a los 3000 bytes

    assert cli(bucket, tmp_path, "202102") == 0

    assert (yearly.name, f"bytes={start}-{end}", 3000) in bucket.log
    assert f"bytes={start + 2048}-{end}" in bucket.ranges(yearly.name)  # siguió tras los 2 bloques completos
    assert pq.read_table(tmp_path / "raw" / "trips" / "202102.parquet").num_rows == 5000
    assert leftovers(tmp_path) == []


def test_sin_espacio_en_disco_falla_antes_de_descargar(bucket, tmp_path, capsys, monkeypatch):
    key = "202609-citibike-tripdata.zip"
    remote = put(bucket, key, {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    monkeypatch.setattr(dt, "_free_bytes", lambda folder: 10)

    assert cli(bucket, tmp_path, "202609") == 1
    assert "Espacio insuficiente" in capsys.readouterr().out
    assert f"bytes=0-{remote.stat().st_size - 1}" not in bucket.ranges(key)
    assert leftovers(tmp_path) == []


def test_ctrl_c_limpia_y_no_sigue_con_los_demas_meses(bucket, tmp_path, capsys, monkeypatch):
    put(bucket, "202608-citibike-tripdata.zip", {"202608-citibike-tripdata_1.csv": csv_bytes(many_rows("A", 20))})
    put(bucket, "202609-citibike-tripdata.zip", {"202609-citibike-tripdata_1.csv": csv_bytes(many_rows("B", 20))})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(dt, "csvs_to_parquet", interrupted)
    assert cli(bucket, tmp_path, "202608", "202609") == 130
    assert "interrumpido" in capsys.readouterr().out
    assert bucket.ranges("202609-citibike-tripdata.zip") == []  # no siguió con el otro mes
    assert not list((tmp_path / "raw").rglob("*.parquet"))
    assert leftovers(tmp_path) == []


def test_ctrl_c_dentro_de_duckdb_se_reconoce():
    assert dt._was_interrupted(RuntimeError("Query interrupted"))
    assert not dt._was_interrupted(RuntimeError("otro error"))
    assert not dt._was_interrupted(duckdb.ConversionException('línea con el texto "Query interrupted"'))
