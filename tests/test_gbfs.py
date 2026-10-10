"""Pruebas de la captura GBFS. No usan la red.

Los datos están escritos a mano con la misma forma que el feed real, incluidos
sus casos raros: una estación sin num_scooters_*, una sin region_id, una con
last_reported=86400 y una con un campo que el esquema no conoce.
"""

import errno
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from src.data import gbfs

ROOT = Path(__file__).resolve().parents[1]

STATUS_URL = "https://ejemplo.test/en/station_status.json"
INFO_URL = "https://ejemplo.test/en/station_information.json"

FEED_T0 = 1791590400  # 2026-10-10 00:00:00 UTC
FEED_T0_UTC = datetime(2026, 10, 10, tzinfo=timezone.utc)

ID_A = "aaaaaaaa-0000-4000-8000-000000000001"  # estación normal, con todos los campos
ID_B = "bbbbbbbb-0000-4000-8000-000000000002"  # no instalada
ID_C = "cccccccc-0000-4000-8000-000000000003"  # trae un campo desconocido
ID_D = "dddddddd-0000-4000-8000-000000000004"  # sin region_id


def discovery_payload():
    def feeds(language):
        return {
            "feeds": [
                {"name": "system_information", "url": f"https://ejemplo.test/{language}/system_information.json"},
                {"name": "station_information", "url": f"https://ejemplo.test/{language}/station_information.json"},
                {"name": "station_status", "url": f"https://ejemplo.test/{language}/station_status.json"},
            ]
        }

    return {"data": {"en": feeds("en"), "fr": feeds("fr"), "es": feeds("es")}, "last_updated": FEED_T0, "ttl": 60, "version": "1.1"}


def status_payload(last_updated=FEED_T0):
    stations = [
        {
            "station_id": ID_A, "num_bikes_available": 7, "num_bikes_disabled": 1, "num_docks_available": 12,
            "num_docks_disabled": 1, "is_installed": 1, "is_renting": 1, "is_returning": 1,
            "last_reported": FEED_T0 - 12, "legacy_id": "3419", "num_ebikes_available": 3,
            "eightd_has_available_keys": False, "num_scooters_available": 0, "num_scooters_unavailable": 0,
        },
        {   # no instalada: sin num_scooters_* y con last_reported=86400
            "station_id": ID_B, "num_bikes_available": 0, "num_bikes_disabled": 0, "num_docks_available": 0,
            "num_docks_disabled": 0, "is_installed": 0, "is_renting": 0, "is_returning": 0,
            "last_reported": 86400, "legacy_id": "3950", "num_ebikes_available": 0,
            "eightd_has_available_keys": False,
        },
        {   # con un campo que el esquema no conoce
            "station_id": ID_C, "num_bikes_available": 20, "num_bikes_disabled": 0, "num_docks_available": 0,
            "num_docks_disabled": 0, "is_installed": 1, "is_renting": 1, "is_returning": 0,
            "last_reported": FEED_T0 - 40, "legacy_id": "517", "num_ebikes_available": 9,
            "eightd_has_available_keys": False, "num_scooters_available": 2, "num_scooters_unavailable": 1,
            "campo_nuevo": {"detalle": [1, 2, 3]},
        },
        {
            "station_id": ID_D, "num_bikes_available": 0, "num_bikes_disabled": 2, "num_docks_available": 29,
            "num_docks_disabled": 0, "is_installed": 1, "is_renting": 1, "is_returning": 1,
            "last_reported": FEED_T0 - 5, "legacy_id": "4402", "num_ebikes_available": 0,
            "eightd_has_available_keys": True, "num_scooters_available": 0, "num_scooters_unavailable": 0,
        },
    ]
    return {"data": {"stations": stations}, "last_updated": last_updated, "ttl": 60, "version": "1.1"}


def info_payload(last_updated=FEED_T0):
    def station(station_id, short_name, name, lat, lon, capacity, **extra):
        return {
            "station_id": station_id, "name": name, "short_name": short_name, "lon": lon, "lat": lat,
            "region_id": "71", "capacity": capacity,
            "rental_uris": {"ios": f"https://ejemplo.test/app/{short_name}", "android": f"https://ejemplo.test/app/{short_name}"},
            "rental_methods": ["KEY", "CREDITCARD"], "has_kiosk": True, "station_type": "classic",
            "external_id": station_id, "electric_bike_surcharge_waiver": False,
            "eightd_station_services": [], "eightd_has_key_dispenser": False, **extra,
        }

    stations = [
        station(ID_A, "3834.10", "Calle 6 y Avenida 7", 40.6686627, -73.97988067, 21),
        station(ID_B, "6289.06", "Calle 31 y Broadway", 40.74874, -73.98808, 0),
        station(ID_C, "5329.03", "Plaza Central", 40.71911552, -74.00666661, 20, campo_nuevo="x"),
        station(ID_D, "7023.04", "Muelle 40", 40.7285, -74.0118, 31),
    ]
    del stations[3]["region_id"]  # en el feed real falta en algunas estaciones
    return {"data": {"stations": stations}, "last_updated": last_updated, "ttl": 60, "version": "1.1"}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Server Error")

    def json(self):
        return self.payload


class FakeSession:
    """Responde con los datos escritos a mano según la URL. No abre conexiones.

    Cada URL tiene una lista de respuestas que se van consumiendo; la última se
    repite. Una respuesta que es una excepción se lanza en lugar de devolverse.
    """

    def __init__(self, status=None, info=None, discovery=None):
        self.answers = {
            gbfs.DISCOVERY_URL: [discovery_payload() if discovery is None else discovery],
            STATUS_URL: [status_payload() if status is None else status],
            INFO_URL: [info_payload() if info is None else info],
        }
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        answers = self.answers[url]
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, FakeResponse) else FakeResponse(answer)

    def close(self):
        pass


def at(minute):
    """Hora de captura fija, para que los nombres de archivo sean previsibles."""
    return datetime(2026, 10, 10, 0, minute, 30, tzinfo=timezone.utc)


def parquet_files(out, feed):
    return sorted(str(path.relative_to(out)) for path in (out / feed).rglob("*.parquet"))


def read(out, relative):
    return pq.ParquetFile(out / relative).read()


def column(table, name):
    return dict(zip(table["station_id"].to_pylist(), table[name].to_pylist()))


@pytest.fixture(autouse=True)
def sin_red_ni_esperas(monkeypatch):
    """Ninguna prueba abre una sesión real ni espera entre reintentos."""

    def prohibido():
        raise AssertionError("las pruebas no deben usar la red")

    monkeypatch.setattr(gbfs, "new_session", prohibido)
    monkeypatch.setattr(gbfs, "BACKOFF_SECONDS", 0)


@pytest.fixture()
def out(tmp_path):
    return tmp_path / "gbfs"


# --------------------------------------------------------------------------- #
# Rutas, esquema y conversión de valores
# --------------------------------------------------------------------------- #

def test_escribe_un_archivo_por_feed_en_la_ruta_esperada(out):
    session = FakeSession()
    summary = gbfs.capture_once(out, session, now=at(1))

    assert parquet_files(out, "station_status") == ["station_status/date=2026-10-10/status_20261010T000130Z.parquet"]
    assert parquet_files(out, "station_information") == ["station_information/info_20261010T000130Z.parquet"]
    assert "4 estaciones" in summary and "status_20261010T000130Z.parquet" in summary


def test_las_url_salen_del_documento_de_descubrimiento_en_ingles(out):
    session = FakeSession()
    gbfs.capture_once(out, session, now=at(1))
    assert session.calls == [gbfs.DISCOVERY_URL, STATUS_URL, INFO_URL]


def test_esquema_estable_aunque_el_feed_omita_campos(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))

    # Segunda foto: a todas las estaciones les faltan campos opcionales.
    recortado = status_payload(FEED_T0 + 60)
    for station in recortado["data"]["stations"]:
        for campo in ("num_scooters_available", "num_scooters_unavailable", "legacy_id", "last_reported"):
            station.pop(campo, None)
    gbfs.capture_once(out, FakeSession(status=recortado), now=at(2))

    status_files = parquet_files(out, "station_status")
    assert len(status_files) == 2
    for relative in status_files:
        assert pq.read_schema(out / relative).equals(gbfs.STATUS_SCHEMA)
    for relative in parquet_files(out, "station_information"):
        assert pq.read_schema(out / relative).equals(gbfs.INFO_SCHEMA)

    assert gbfs.STATUS_SCHEMA.names == [
        "snapshot_ts", "feed_last_updated", "station_id", "legacy_id", "num_bikes_available",
        "num_ebikes_available", "num_bikes_disabled", "num_docks_available", "num_docks_disabled",
        "num_scooters_available", "num_scooters_unavailable", "is_installed", "is_renting", "is_returning",
        "last_reported",
    ]
    assert gbfs.INFO_SCHEMA.names == [
        "snapshot_ts", "feed_last_updated", "station_id", "short_name", "name", "lat", "lon", "capacity",
        "region_id", "gbfs_station_type", "has_kiosk", "external_id",
    ]


def test_campos_opcionales_ausentes_quedan_en_nulo(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    status = read(out, parquet_files(out, "station_status")[0])
    info = read(out, parquet_files(out, "station_information")[0])

    assert column(status, "num_scooters_available") == {ID_A: 0, ID_B: None, ID_C: 2, ID_D: 0}
    assert column(status, "num_scooters_unavailable") == {ID_A: 0, ID_B: None, ID_C: 1, ID_D: 0}
    assert column(status, "num_bikes_available") == {ID_A: 7, ID_B: 0, ID_C: 20, ID_D: 0}
    assert column(info, "region_id") == {ID_A: "71", ID_B: "71", ID_C: "71", ID_D: None}


def test_is_installed_is_renting_e_is_returning_son_booleanos(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    status = read(out, parquet_files(out, "station_status")[0])

    assert column(status, "is_installed") == {ID_A: True, ID_B: False, ID_C: True, ID_D: True}
    assert column(status, "is_renting") == {ID_A: True, ID_B: False, ID_C: True, ID_D: True}
    assert column(status, "is_returning") == {ID_A: True, ID_B: False, ID_C: False, ID_D: True}


def test_los_epoch_se_guardan_como_instantes_utc(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    status = read(out, parquet_files(out, "station_status")[0])
    info = read(out, parquet_files(out, "station_information")[0])

    for table in (status, info):
        for name in ("snapshot_ts", "feed_last_updated"):
            assert table.schema.field(name).type == pa.timestamp("us", tz="UTC")
            assert set(table[name].to_pylist()) == {at(1) if name == "snapshot_ts" else FEED_T0_UTC}

    last_reported = column(status, "last_reported")
    assert last_reported[ID_A] == datetime(2026, 10, 9, 23, 59, 48, tzinfo=timezone.utc)
    assert last_reported[ID_B] == datetime(1970, 1, 2, tzinfo=timezone.utc)  # 86400: se guarda tal cual


def test_el_station_type_de_gbfs_se_guarda_con_otro_nombre(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    info = read(out, parquet_files(out, "station_information")[0])

    assert "station_type" not in info.schema.names
    assert set(info["gbfs_station_type"].to_pylist()) == {"classic"}
    assert column(info, "short_name")[ID_B] == "6289.06"
    assert column(info, "capacity") == {ID_A: 21, ID_B: 0, ID_C: 20, ID_D: 31}
    assert column(info, "lat")[ID_A] == 40.6686627


def test_valores_de_tipo_inesperado_no_rompen_la_captura(out):
    raro = status_payload()
    raro["data"]["stations"][0].update(
        num_bikes_available="5", num_docks_available=None, is_renting=True, is_returning="si", last_reported="ayer"
    )
    raro["data"]["stations"].append({"sin_id": True})  # no se puede identificar: se descarta
    raro["data"]["stations"].append("no soy una estación")
    gbfs.capture_once(out, FakeSession(status=raro), now=at(1))
    status = read(out, parquet_files(out, "station_status")[0])

    assert status.num_rows == 4
    assert column(status, "num_bikes_available")[ID_A] == 5
    assert column(status, "num_docks_available")[ID_A] is None
    assert column(status, "is_renting")[ID_A] is True
    assert column(status, "is_returning")[ID_A] is None
    assert column(status, "last_reported")[ID_A] is None


def test_un_entero_que_no_cabe_en_int64_queda_en_nulo_y_la_foto_se_guarda(out):
    enorme = status_payload()
    enorme["data"]["stations"][0].update(
        num_bikes_available=2**63, num_docks_available=1e300, num_ebikes_available="9" * 23,
        num_bikes_disabled=-2**63 - 1,
    )
    # Los extremos de int64 sí caben y se guardan.
    enorme["data"]["stations"][3].update(num_bikes_available=2**63 - 1, num_docks_available=-2**63)
    info = info_payload()
    info["data"]["stations"][0]["capacity"] = 2**63
    gbfs.capture_once(out, FakeSession(status=enorme, info=info), now=at(1))
    status = read(out, parquet_files(out, "station_status")[0])

    assert status.num_rows == 4
    for name in ("num_bikes_available", "num_docks_available", "num_ebikes_available", "num_bikes_disabled"):
        assert column(status, name)[ID_A] is None
    assert column(status, "num_bikes_available")[ID_D] == 2**63 - 1
    assert column(status, "num_docks_available")[ID_D] == -2**63
    assert column(status, "num_bikes_available")[ID_C] == 20  # las demás estaciones no cambian
    capacity = column(read(out, parquet_files(out, "station_information")[0]), "capacity")
    assert capacity == {ID_A: None, ID_B: 0, ID_C: 20, ID_D: 31}


# --------------------------------------------------------------------------- #
# No duplicar: mismo feed, información sin cambios, reinicios
# --------------------------------------------------------------------------- #

def test_mismo_last_updated_no_escribe_otro_status(out):
    session = FakeSession()
    gbfs.capture_once(out, session, now=at(1))
    summary = gbfs.capture_once(out, session, now=at(2))

    assert len(parquet_files(out, "station_status")) == 1
    assert "feed sin cambios" in summary and "no se escribe" in summary

    # Cuando el feed avanza, sí se escribe.
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 60)), now=at(3))
    assert len(parquet_files(out, "station_status")) == 2


def test_information_se_escribe_una_vez_y_de_nuevo_solo_si_cambia(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    assert len(parquet_files(out, "station_information")) == 1

    # Mismo contenido, otro last_updated y otro orden: no es un cambio.
    mismo = info_payload(FEED_T0 + 60)
    mismo["data"]["stations"].reverse()
    summary = gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 60), info=mismo), now=at(2))
    assert len(parquet_files(out, "station_information")) == 1
    assert "info: 4 estaciones, sin cambios" in summary

    # Cambia la capacidad de una estación: archivo nuevo.
    distinto = info_payload(FEED_T0 + 120)
    distinto["data"]["stations"][0]["capacity"] = 25
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 120), info=distinto), now=at(3))
    assert parquet_files(out, "station_information") == [
        "station_information/info_20261010T000130Z.parquet",
        "station_information/info_20261010T000330Z.parquet",
    ]
    assert column(read(out, parquet_files(out, "station_information")[1]), "capacity")[ID_A] == 25

    # Y se queda así mientras no vuelva a cambiar.
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 180), info=distinto), now=at(4))
    assert len(parquet_files(out, "station_information")) == 2


def test_el_estado_sobrevive_a_un_reinicio(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    state = json.loads((out / gbfs.STATE_FILE).read_text(encoding="utf-8"))
    assert state["status_last_updated"] == FEED_T0
    assert state["status_last_file"] == "station_status/date=2026-10-10/status_20261010T000130Z.parquet"
    assert state["info_last_file"] == "station_information/info_20261010T000130Z.parquet"

    # "Reinicio": otra llamada, otra sesión, misma carpeta y el feed igual.
    summary = gbfs.capture_once(out, FakeSession(), now=at(9))
    assert len(parquet_files(out, "station_status")) == 1
    assert len(parquet_files(out, "station_information")) == 1
    assert summary.count("no se escribe") == 2


def test_un_estado_danado_o_borrado_no_impide_capturar(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))

    (out / gbfs.STATE_FILE).write_text("{esto no es json", encoding="utf-8")
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 60)), now=at(2))
    assert len(parquet_files(out, "station_status")) == 2

    (out / gbfs.STATE_FILE).unlink()
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 120)), now=at(3))
    assert len(parquet_files(out, "station_status")) == 3
    assert json.loads((out / gbfs.STATE_FILE).read_text(encoding="utf-8"))["status_last_updated"] == FEED_T0 + 120


def test_si_se_borra_el_archivo_de_information_se_vuelve_a_escribir(out):
    gbfs.capture_once(out, FakeSession(), now=at(1))
    (out / "station_information" / "info_20261010T000130Z.parquet").unlink()

    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 60)), now=at(2))
    assert parquet_files(out, "station_information") == ["station_information/info_20261010T000230Z.parquet"]


# --------------------------------------------------------------------------- #
# Archivos completos y lectura con DuckDB
# --------------------------------------------------------------------------- #

def test_no_quedan_archivos_temporales(out):
    for minute in (1, 2, 3):
        gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 60 * minute)), now=at(minute))

    names = sorted(path.name for path in out.rglob("*") if path.is_file())
    assert [name for name in names if not name.endswith(".parquet")] == [gbfs.STATE_FILE]
    for path in out.rglob("*.parquet"):
        assert pq.ParquetFile(path).metadata.num_rows == 4


def test_una_escritura_que_falla_no_deja_archivos_a_medias(out, monkeypatch):
    def write_and_fail(table, where, **options):
        Path(where).write_bytes(b"PAR1 a medias")
        raise OSError("disco lleno")

    with monkeypatch.context() as patch:
        patch.setattr(gbfs.pq, "write_table", write_and_fail)
        with pytest.raises(gbfs.GbfsError, match="disco lleno"):
            gbfs.capture_once(out, FakeSession(), now=at(1))
    assert [path for path in out.rglob("*") if path.is_file()] == []

    # Como no se guardó nada, la siguiente captura sí lo escribe.
    gbfs.capture_once(out, FakeSession(), now=at(2))
    assert len(parquet_files(out, "station_status")) == 1
    assert len(parquet_files(out, "station_information")) == 1


def test_duckdb_lee_toda_la_carpeta_con_un_glob(out):
    # Tres fotos, la última ya en el día siguiente (otra carpeta date=).
    moments = [at(1), at(6), datetime(2026, 10, 11, 0, 0, 30, tzinfo=timezone.utc)]
    for i, moment in enumerate(moments):
        gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 300 * i)), now=moment)
    assert len(parquet_files(out, "station_status")) == 3

    status_glob = (out / "station_status" / "*" / "*.parquet").as_posix()
    info_glob = (out / "station_information" / "*.parquet").as_posix()
    con = duckdb.connect()

    assert con.sql(f"SELECT count(*) FROM read_parquet('{status_glob}')").fetchone() == (12,)
    per_snapshot = con.sql(
        f"SELECT CAST(date AS VARCHAR), count(*) FROM read_parquet('{status_glob}') GROUP BY ALL ORDER BY 1"
    ).fetchall()
    assert per_snapshot == [("2026-10-10", 8), ("2026-10-11", 4)]

    # El cruce de la documentación: status + información más reciente.
    joined = con.sql(
        f"""
        WITH info AS (
            SELECT * FROM read_parquet('{info_glob}')
            QUALIFY snapshot_ts = max(snapshot_ts) OVER ()
        )
        SELECT i.short_name, i.capacity, s.num_bikes_available
        FROM read_parquet('{status_glob}') AS s
        JOIN info AS i USING (station_id)
        ORDER BY s.snapshot_ts, i.short_name
        """
    ).fetchall()
    assert len(joined) == 12
    assert joined[:4] == [("3834.10", 21, 7), ("5329.03", 20, 20), ("6289.06", 0, 0), ("7023.04", 31, 0)]


def documented_query():
    """La consulta de DuckDB tal como está escrita en la documentación del módulo."""
    return gbfs.__doc__.split("duckdb.sql('''")[1].split("''')")[0]


def test_la_consulta_documentada_no_repite_una_foto_guardada_dos_veces(tmp_path, monkeypatch):
    # La consulta usa rutas relativas a la raíz del repositorio; aquí la raíz es tmp_path.
    out = tmp_path / "data" / "raw" / "gbfs"
    gbfs.capture_once(out, FakeSession(), now=at(1))
    (out / gbfs.STATE_FILE).unlink()  # sin estado, la misma foto del feed se guarda otra vez
    gbfs.capture_once(out, FakeSession(), now=at(2))
    gbfs.capture_once(out, FakeSession(status=status_payload(FEED_T0 + 300)), now=at(6))
    assert len(parquet_files(out, "station_status")) == 3
    assert len(parquet_files(out, "station_information")) == 2

    monkeypatch.chdir(tmp_path)
    counts = duckdb.connect().sql(
        f"SELECT count(*), count(DISTINCT (feed_last_updated, short_name)) FROM ({documented_query()})"
    ).fetchone()

    assert counts == (8, 8)  # 2 fotos distintas del feed x 4 estaciones, ninguna repetida


# --------------------------------------------------------------------------- #
# Fallos de descarga
# --------------------------------------------------------------------------- #

def test_fetch_json_reintenta_y_termina_bien():
    session = FakeSession()
    session.answers[STATUS_URL] = [
        requests.ConnectionError("sin conexión"),
        FakeResponse({}, status=503),
        status_payload(),
    ]
    assert gbfs.fetch_json(session, STATUS_URL)["last_updated"] == FEED_T0
    assert session.calls == [STATUS_URL] * 3


def test_fetch_json_se_rinde_tras_los_intentos_configurados():
    session = FakeSession()
    session.answers[STATUS_URL] = [requests.Timeout("tardó demasiado")]
    with pytest.raises(gbfs.GbfsError, match="tras 3 intentos"):
        gbfs.fetch_json(session, STATUS_URL)
    assert len(session.calls) == gbfs.ATTEMPTS == 3


def test_un_feed_con_estructura_inesperada_da_un_error_claro(out):
    with pytest.raises(gbfs.GbfsError, match="station_status"):
        gbfs.capture_once(out, FakeSession(status={"data": {}, "last_updated": FEED_T0}), now=at(1))
    with pytest.raises(gbfs.GbfsError, match="no lista los feeds"):
        gbfs.capture_once(out, FakeSession(discovery={"data": {"en": {"feeds": []}}}), now=at(1))
    with pytest.raises(gbfs.GbfsError, match="estructura inesperada"):
        gbfs.capture_once(out, FakeSession(discovery={"data": {"es": {}}}), now=at(1))


def test_si_falla_information_el_status_se_guarda_igual(out):
    session = FakeSession()
    session.answers[INFO_URL] = [requests.ConnectionError("sin conexión")]
    with pytest.raises(gbfs.GbfsError, match="station_information FALLÓ"):
        gbfs.capture_once(out, session, now=at(1))
    assert len(parquet_files(out, "station_status")) == 1
    assert parquet_files(out, "station_information") == []

    # La siguiente captura completa lo que faltó.
    gbfs.capture_once(out, FakeSession(), now=at(2))
    assert len(parquet_files(out, "station_status")) == 1  # el feed no cambió
    assert len(parquet_files(out, "station_information")) == 1


def test_una_vuelta_que_falla_se_anota_y_no_lanza(out, capsys):
    caido = FakeSession()
    caido.answers[gbfs.DISCOVERY_URL] = [requests.ConnectionError("sin conexión")]

    assert gbfs.capture_safely(out, caido) is False
    line = capsys.readouterr().out
    assert "ERROR" in line and "sin conexión" in line and line.count("\n") == 1
    assert not out.exists()

    # La vuelta siguiente, ya con red, captura con normalidad.
    assert gbfs.capture_safely(out, FakeSession()) is True
    line = capsys.readouterr().out
    assert "ERROR" not in line and "4 estaciones" in line and line.count("\n") == 1
    assert len(parquet_files(out, "station_status")) == 1


LOOP_START = FEED_T0 + 90  # 2026-10-10 00:01:30 UTC


class SimulatedTime(threading.Event):
    """Reloj y señal de paro falsos para probar el modo continuo sin esperar.

    Esperar adelanta el reloj en lugar de dormir. La señal de paro "llega" a
    los `stop_at` segundos. `sleeps` simula que la computadora se durmió: la
    espera número N dura esos segundos de más.
    """

    def __init__(self, stop_at, sleeps=None):
        super().__init__()
        self.now = float(LOOP_START)
        self.stop_at = stop_at
        self.sleeps = sleeps or {}
        self.waits = []

    @property
    def elapsed(self):
        return self.now - LOOP_START

    def clock(self):
        return self.now

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.now += timeout + self.sleeps.get(len(self.waits), 0)
        if self.elapsed >= self.stop_at:
            self.set()
        return self.is_set()


class LoopSession(FakeSession):
    """Cada descarga tarda 1 s del reloj falso y el feed avanza con el tiempo, como el real."""

    def __init__(self, simulated):
        super().__init__()
        self.simulated = simulated

    def get(self, url, timeout=None):
        self.simulated.now += 1
        if url == STATUS_URL:
            return FakeResponse(status_payload(FEED_T0 + int(self.simulated.elapsed)))
        return super().get(url, timeout)


@pytest.fixture()
def capture_starts(monkeypatch):
    """Anota en qué segundo (desde el inicio) arrancó cada captura del modo continuo."""
    starts = []
    capture_once = gbfs.capture_once

    def spy(out_dir, session, now):
        starts.append(now.timestamp() - LOOP_START)
        return capture_once(out_dir, session, now)

    monkeypatch.setattr(gbfs, "capture_once", spy)
    return starts


def test_el_modo_continuo_captura_en_turnos_fijos_sin_acumular_retraso(out, capture_starts):
    simulated = SimulatedTime(stop_at=200)

    gbfs.run_loop(60, out, simulated, LoopSession(simulated), clock=simulated.clock)

    # Cada captura tarda 3 s, pero los turnos siguen cayendo cada 60 s exactos.
    assert capture_starts == [0, 60, 120, 180]
    assert parquet_files(out, "station_status") == [
        "station_status/date=2026-10-10/status_20261010T000130Z.parquet",
        "station_status/date=2026-10-10/status_20261010T000230Z.parquet",
        "station_status/date=2026-10-10/status_20261010T000330Z.parquet",
        "station_status/date=2026-10-10/status_20261010T000430Z.parquet",
    ]
    assert len(parquet_files(out, "station_information")) == 1
    assert max(simulated.waits) <= gbfs.WAIT_SLICE


def test_el_modo_continuo_sigue_despues_de_un_fallo_y_se_detiene_limpio(out, capture_starts, capsys):
    simulated = SimulatedTime(stop_at=130)
    session = LoopSession(simulated)
    session.answers[gbfs.DISCOVERY_URL] = [requests.ConnectionError("sin conexión")] * gbfs.ATTEMPTS + [discovery_payload()]

    gbfs.run_loop(60, out, simulated, session, clock=simulated.clock)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert "ERROR" in lines[0] and "sin conexión" in lines[0]
    assert "ERROR" not in lines[1] and "ERROR" not in lines[2]
    assert capture_starts == [0, 60, 120]
    assert len(parquet_files(out, "station_status")) == 2
    # El paro llegó durante una espera: sale enseguida, sin otra captura y sin temporales.
    assert simulated.is_set() and simulated.elapsed < 180
    assert list(out.rglob("*.tmp")) == []


def test_el_modo_continuo_captura_al_despertar_si_la_computadora_se_durmio(out, capture_starts):
    # La tercera espera dura una hora de más, como si la laptop se hubiera dormido.
    simulated = SimulatedTime(stop_at=4000, sleeps={3: 3600})

    gbfs.run_loop(300, out, simulated, LoopSession(simulated), clock=simulated.clock)

    # Captura en cuanto despierta (no 300 s después) y luego vuelve al ritmo normal.
    assert capture_starts == [0, 3618, 3921]


def test_tras_un_fallo_se_reintenta_al_minuto_sin_mover_los_turnos(out, capture_starts, capsys):
    # Un corte de red corto justo en el segundo turno (a los 300 s).
    simulated = SimulatedTime(stop_at=1000)
    session = LoopSession(simulated)
    session.answers[gbfs.DISCOVERY_URL] = (
        [discovery_payload()] + [requests.ConnectionError("sin red")] * gbfs.ATTEMPTS + [discovery_payload()]
    )

    gbfs.run_loop(300, out, simulated, session, clock=simulated.clock)

    # El turno fallido tardó 3 s. Se reintenta 60 s después (363) y los turnos siguen en 600 y 900.
    assert capture_starts == [0, 300, 363, 600, 900]
    lines = capsys.readouterr().out.splitlines()
    assert ["ERROR" in line for line in lines] == [False, True, False, False, False]
    assert parquet_files(out, "station_status") == [
        "station_status/date=2026-10-10/status_20261010T000130Z.parquet",
        "station_status/date=2026-10-10/status_20261010T000733Z.parquet",
        "station_status/date=2026-10-10/status_20261010T001130Z.parquet",
        "station_status/date=2026-10-10/status_20261010T001630Z.parquet",
    ]


def test_mientras_dura_el_fallo_se_reintenta_cada_minuto_y_nunca_despues_del_turno(out, capture_starts):
    # Fallan el turno de los 300 s y los cuatro reintentos siguientes.
    simulated = SimulatedTime(stop_at=700)
    session = LoopSession(simulated)
    session.answers[gbfs.DISCOVERY_URL] = (
        [discovery_payload()] + [requests.ConnectionError("sin red")] * (5 * gbfs.ATTEMPTS) + [discovery_payload()]
    )

    gbfs.run_loop(300, out, simulated, session, clock=simulated.clock)

    # Tras el fallo de 552 tocaría reintentar en 615, pero el turno de 600 llega antes y manda.
    assert capture_starts == [0, 300, 363, 426, 489, 552, 600]
    assert len(parquet_files(out, "station_status")) == 2


class BrokenOutput:
    """Una salida en la que no se puede escribir, como con el disco lleno."""

    def __init__(self, error):
        self.error = error
        self.attempts = 0

    def write(self, text):
        self.attempts += 1
        raise self.error

    def flush(self):
        raise self.error


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left on device"), ValueError("I/O operation on closed file")],
    ids=["disco_lleno", "salida_cerrada"],
)
def test_si_no_se_puede_escribir_la_linea_del_registro_la_captura_sigue(out, capture_starts, monkeypatch, error):
    simulated = SimulatedTime(stop_at=130)
    broken = BrokenOutput(error)
    monkeypatch.setattr(sys, "stdout", broken)

    gbfs.run_loop(60, out, simulated, LoopSession(simulated), clock=simulated.clock)

    assert broken.attempts == 3  # intentó anotar cada vuelta
    assert capture_starts == [0, 60, 120]
    assert len(parquet_files(out, "station_status")) == 3


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def test_intervalo_menor_a_60_se_rechaza(out, capsys):
    assert gbfs.check_interval(60) == 60
    assert gbfs.check_interval(300) == 300
    with pytest.raises(ValueError, match="mínimo es 60"):
        gbfs.check_interval(59)
    with pytest.raises(ValueError, match="mínimo es 60"):
        gbfs.run_loop(30, out, threading.Event(), FakeSession())

    with pytest.raises(SystemExit) as exit_info:
        gbfs.main(["--intervalo", "59", "--salida", str(out)])
    assert exit_info.value.code == 2
    assert "mínimo es 60" in capsys.readouterr().err
    assert not out.exists()


def test_una_vez_devuelve_0_si_capturo_y_1_si_fallo(out, monkeypatch, capsys):
    monkeypatch.setattr(gbfs, "new_session", FakeSession)
    assert gbfs.main(["--una-vez", "--salida", str(out)]) == 0
    assert len(parquet_files(out, "station_status")) == 1

    def caido():
        session = FakeSession()
        session.answers[STATUS_URL] = [FakeResponse({}, status=500)]
        return session

    monkeypatch.setattr(gbfs, "new_session", caido)
    assert gbfs.main(["--una-vez", "--salida", str(out)]) == 1
    assert "ERROR" in capsys.readouterr().out


# Programa hijo para probar las señales sin arriesgar el proceso de pytest:
# arranca el modo continuo con datos falsos y se manda la señal a sí mismo a
# media captura (después de bajar station_status y antes de station_information).
SIGNAL_CHILD = """
import os, signal, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tests")]
import test_gbfs
from src.data import gbfs

class Session(test_gbfs.FakeSession):
    def get(self, url, timeout=None):
        if url == test_gbfs.STATUS_URL:
            os.kill(os.getpid(), getattr(signal, sys.argv[2]))
        return super().get(url, timeout)

gbfs.new_session = Session
sys.exit(gbfs.main(["--intervalo", "60", "--salida", sys.argv[3]]))
"""


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
def test_una_senal_a_media_captura_termina_la_captura_y_sale_limpio(out, signal_name):
    child = subprocess.run(
        [sys.executable, "-c", SIGNAL_CHILD, str(ROOT), signal_name, str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode == 0, child.stderr

    lines = child.stdout.splitlines()
    assert len(lines) == 3, child.stdout
    assert "captura continua cada 60 s" in lines[0]
    assert "status: 4 estaciones, 4 filas" in lines[1] and "info: 4 estaciones, 4 filas" in lines[1]
    assert lines[2].endswith("captura detenida")
    assert len(parquet_files(out, "station_status")) == 1
    assert len(parquet_files(out, "station_information")) == 1
    assert list(out.rglob("*.tmp")) == []


def test_el_script_corre_desde_cualquier_carpeta(tmp_path):
    script = str(ROOT / "scripts" / "capture_gbfs.py")

    ayuda = subprocess.run([sys.executable, script, "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert ayuda.returncode == 0, ayuda.stderr
    for opcion in ("--intervalo", "--salida", "--una-vez"):
        assert opcion in ayuda.stdout

    # Se rechaza antes de tocar la red.
    rechazo = subprocess.run([sys.executable, script, "--intervalo", "30"], cwd=tmp_path, capture_output=True, text=True)
    assert rechazo.returncode == 2
    assert "mínimo es 60" in rechazo.stderr


def test_la_receta_documentada_revisa_si_ya_corre_y_no_depende_de_la_terminal():
    recipe = gbfs.__doc__.split("Cómo dejarlo corriendo")[1].split("Cómo detenerlo")[0]
    commands = [line.strip() for line in recipe.splitlines() if line.startswith("    ")]

    # Lo primero es ver si ya hay una captura: dos a la vez escribirían en la misma carpeta.
    assert commands[0].startswith("ps ") and "data/raw/gbfs/captura.pid" in commands[0]
    # Todo lo que queda en segundo plano va con nohup; si no, muere al cerrar la terminal.
    background = [command for command in commands if command.endswith("&")]
    assert len(background) == 2 and "caffeinate" in background[1]
    assert all(command.startswith("nohup ") for command in background)
