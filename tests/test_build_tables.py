"""Pruebas de la limpieza y de las tablas reales (src/data/build_tables.py).

No usan internet ni los datos reales: cada prueba escribe en una carpeta
temporal unos pocos viajes INVENTADOS con el formato del Parquet crudo, una
configuración y, si hace falta, una foto de GBFS. Después corre la limpieza
completa y revisa las tablas que salen.

La única excepción es la última prueba: compara dos horas conocidas de
data/processed/station_hour.parquet contra un conteo aparte hecho sobre los
archivos crudos reales. Se SALTA si esos archivos no están en la máquina.

Córrela sola con:  pytest tests/test_build_tables.py
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.contracts import CONTRACTS, read_contract, validate_file
from src.data import build_tables as bt
from src.data.datos_config import Datos, Tramo, load_datos
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE, TZ

REPO = Path(__file__).resolve().parents[1]

# Mismo esquema que deja scripts/download_trips.py: ids y nombres como texto,
# fechas sin zona (hora local de Nueva York), coordenadas decimales.
RAW_SCHEMA = pa.schema(
    [
        ("ride_id", pa.string()),
        ("rideable_type", pa.string()),
        ("started_at", pa.timestamp("us")),
        ("ended_at", pa.timestamp("us")),
        ("start_station_name", pa.string()),
        ("start_station_id", pa.string()),
        ("end_station_name", pa.string()),
        ("end_station_id", pa.string()),
        ("start_lat", pa.float64()),
        ("start_lng", pa.float64()),
        ("end_lat", pa.float64()),
        ("end_lng", pa.float64()),
        ("member_casual", pa.string()),
    ]
)

METRE = 1 / 111_195  # un metro hacia el norte, en grados de latitud

# Estaciones INVENTADAS para las pruebas: id tal como viene en el crudo ->
# (nombre, lat, lon). Los ids imitan las formas reales del origen.
CATALOG = {
    "4327.09": ("Putnam Ave & Nostrand Ave", 40.6840, -73.9500),
    "4327.09 ": ("Putnam Ave & Nostrand Ave", 40.6840, -73.9500),   # con espacio al final
    "6720.07": ("34 Ave & 31 St", 40.7600, -73.9250),
    # Un decimal y dos decimales, mismo nombre y misma coordenada: se unen.
    "5470.10": ("Greenwich St & Hubert St", 40.7210, -74.0100),
    "5470.1": ("Greenwich St & Hubert St", 40.7210, -74.0100),
    # Un decimal SIN gemelo: no se toca.
    "6131.1": ("E 24 St & Park Ave S", 40.7400, -73.9860),
    # Guion bajo, misma coordenada, nombre con espacio al final: se unen.
    "5308.04": ("Metropolitan Ave & Bedford Ave", 40.7150, -73.9600),
    "5308.04_": ("Metropolitan Ave & Bedford Ave ", 40.7150, -73.9600),
    # Mismo nombre pero a 30 metros: NO se unen.
    "5303.06": ("Clinton St & Grand St", 40.7155, -73.9870),
    "5303.06_": ("Clinton St & Grand St", 40.7155 + 30 * METRE, -73.9870),
    # Misma coordenada pero otro nombre: NO se unen.
    "7000.1": ("Calle Uno", 40.7500, -73.9900),
    "7000.10": ("Calle Dos", 40.7500, -73.9900),
    # La escritura con un decimal tiene más viajes: es la que queda.
    "8000.1": ("Avenida Ocho", 40.7700, -73.9700),
    "8000.10": ("avenida   ocho", 40.7700, -73.9700),
    # Empate de viajes: queda la que ordena al final.
    "9000.1": ("Avenida Nueve", 40.7800, -73.9600),
    "9000.10": ("Avenida Nueve", 40.7800, -73.9600),
    # Excluidas por la configuración.
    "HB101": ("Hoboken Terminal", 40.7360, -74.0290),
    "JC013": ("Marin Light Rail", 40.7140, -74.0420),
    "SYS016": ("Morgan Bike Mechanics", 40.7090, -73.9330),
    "Shop Morgan ": ("Shop Morgan", 40.7092, -73.9332),
    "1234.56": ("Morgan HCT Charging", 40.7094, -73.9334),
}

_ride_numbers = itertools.count(1)


def trip(start, end, started_at, *, minutes=10, ended_at=None, member="member", ride_id=None) -> dict:
    """Una fila del Parquet crudo. `start` y `end` son ids de CATALOG, o None si falta la estación."""
    started = datetime.fromisoformat(started_at) if started_at else None
    if ended_at:
        ended = datetime.fromisoformat(ended_at)
    else:
        ended = started + timedelta(minutes=minutes) if started else None
    start_name, start_lat, start_lng = CATALOG[start] if start is not None else (None, None, None)
    end_name, end_lat, end_lng = CATALOG[end] if end is not None else (None, None, None)
    return {
        "ride_id": ride_id or f"R{next(_ride_numbers):015d}",
        "rideable_type": "classic_bike",
        "started_at": started,
        "ended_at": ended,
        "start_station_name": start_name,
        "start_station_id": start,
        "end_station_name": end_name,
        "end_station_id": end,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "member_casual": member,
    }


def some_route(day: str) -> dict:
    """Un viaje completo en otra estación, para las pruebas cuyos viajes no tienen estación de fin.

    El contrato de routes no admite una tabla sin filas.
    """
    return trip("5470.10", "5470.10", f"{day} 12:00:00")


def write_raw(folder: Path, month: str, trips: list[dict], schema: pa.Schema = RAW_SCHEMA) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{month}.parquet"
    pq.write_table(pa.Table.from_pylist(trips, schema=schema), path)
    return path


def write_config(
    folder: Path,
    months,
    inicio: str,
    fin: str,
    *,
    distancia: float = 1.0,
    prefijos=("HB", "JC", "SYS"),
    ids=("1234.56", "Shop Morgan"),
) -> Path:
    """Configuración de prueba. El periodo debe tener al menos tres días (un día por tramo)."""
    first, last = date.fromisoformat(inicio), date.fromisoformat(fin)
    one = timedelta(days=1)
    text = (
        f"meses_crudos: [{', '.join(months)}]\n"
        f"periodo: {{inicio: {first}, fin: {last}}}\n"
        "tramos:\n"
        f"  entrenamiento: {{inicio: {first}, fin: {last - 2 * one}}}\n"
        f"  validacion: {{inicio: {last - one}, fin: {last - one}}}\n"
        f"  prueba: {{inicio: {last}, fin: {last}}}\n"
        f"union_de_ids: {{distancia_maxima_m: {distancia}}}\n"
        "estaciones_excluidas:\n"
        f"  prefijos: [{', '.join(prefijos)}]\n"
        f"  ids: [{', '.join(json.dumps(i) for i in ids)}]\n"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "datos.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_gbfs(folder: Path, stamp: str, capacities: dict) -> Path:
    """Una foto de station_information como las de la captura: short_name -> capacidad."""
    info = folder / "station_information"
    info.mkdir(parents=True, exist_ok=True)
    taken = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    table = pa.table(
        {
            "snapshot_ts": pa.array([taken] * len(capacities), pa.timestamp("us", tz="UTC")),
            "station_id": pa.array([f"gbfs-{i}" for i in range(len(capacities))], pa.string()),
            "short_name": pa.array(list(capacities), pa.string()),
            "capacity": pa.array(list(capacities.values()), pa.int64()),
        }
    )
    path = info / f"info_{stamp}.parquet"
    pq.write_table(table, path)
    return path


@dataclass
class Built:
    out: Path
    report: dict
    stations: pd.DataFrame
    station_hour: pd.DataFrame
    routes: pd.DataFrame

    def hours(self, station: str) -> list[tuple]:
        """Filas de una estación: (hora en UTC, salidas, llegadas), en orden."""
        rows = self.station_hour[self.station_hour["station_id"] == station].sort_values("ts_hour")
        return [
            (ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"), int(d), int(a))
            for ts, d, a in zip(rows["ts_hour"], rows["departures"], rows["arrivals"])
        ]

    def totals(self, station: str) -> tuple[int, int]:
        rows = self.station_hour[self.station_hour["station_id"] == station]
        return int(rows["departures"].sum()), int(rows["arrivals"].sum())

    def route_trips(self) -> dict:
        return {
            (s, e): int(n)
            for s, e, n in zip(self.routes["start_station_id"], self.routes["end_station_id"], self.routes["trips"])
        }


def read_built(out: Path, report: dict | None = None) -> Built:
    if report is None:
        report = json.loads((out / "reports" / "limpieza.json").read_text(encoding="utf-8"))
    return Built(out, report, *(read_contract(name, out / f"{name}.parquet") for name in bt.TABLES))


def run(
    tmp_path: Path, months: dict, inicio: str, fin: str, *, gbfs: Path | None = None, out="salida", **config
) -> Built:
    """Escribe los meses crudos y la configuración en tmp_path y corre la limpieza completa."""
    raw = tmp_path / "crudos"
    for month, trips in months.items():
        write_raw(raw, month, trips)
    config_path = write_config(tmp_path, list(months), inicio, fin, **config)
    out_dir = tmp_path / out
    report = bt.build(load_datos(config_path), raw, gbfs, out_dir, config_path=config_path, log=lambda m: None)
    return read_built(out_dir, report)


# --------------------------------------------------------------------------- #
# Zona horaria
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "local, utc",
    [
        ("2026-07-15 08:05:00", "2026-07-15 12:00"),  # verano: Nueva York va 4 horas detrás de UTC
        ("2026-01-15 08:05:00", "2026-01-15 13:00"),  # invierno: 5 horas
    ],
)
def test_la_hora_local_de_nueva_york_queda_en_el_instante_correcto(tmp_path, local, utc):
    day = date.fromisoformat(local[:10])
    built = run(
        tmp_path,
        {day.strftime("%Y%m"): [trip("4327.09", "6720.07", local, minutes=5)]},
        str(day - timedelta(days=1)),
        str(day + timedelta(days=1)),
    )
    assert built.hours("4327.09") == [(utc, 1, 0)]
    assert built.hours("6720.07") == [(utc, 0, 1)]
    assert str(built.station_hour["ts_hour"].dtype) == f"datetime64[us, {TZ}]"


def test_el_dia_de_23_horas_de_primavera(tmp_path):
    """8 de marzo de 2026: a las 02:00 el reloj salta a las 03:00."""
    trips = [
        trip("4327.09", None, "2026-03-08 01:30:00"),
        trip("4327.09", None, "2026-03-08 03:30:00"),
        # Una hora que no existe ese día: DuckDB la pone en las 03:00 (-04:00).
        # Con el menor ride_id, es el viaje que usa la comprobación de zona horaria.
        trip("6131.1", None, "2026-03-08 02:30:00", ride_id="A-HORA-QUE-NO-EXISTE"),
        trip("6720.07", None, "2026-03-08 00:10:00"),
        trip("6720.07", None, "2026-03-08 23:50:00"),
        some_route("2026-03-08"),
    ]
    built = run(tmp_path, {"202603": trips}, "2026-03-07", "2026-03-09")

    # 01:00 (-05:00) y 03:00 (-04:00) son horas seguidas: no hay fila de las 02:00.
    assert built.hours("4327.09") == [("2026-03-08 06:00", 1, 0), ("2026-03-08 07:00", 1, 0)]
    assert built.hours("6131.1") == [("2026-03-08 07:00", 1, 0)]
    assert len(built.hours("6720.07")) == 23
    assert built.report["zona_horaria"][0]["ride_id"] == "A-HORA-QUE-NO-EXISTE"
    assert built.report["zona_horaria"][0]["hora_utc"] == "2026-03-08T07:00:00+00:00"


def test_el_dia_de_25_horas_de_otono(tmp_path):
    """2 de noviembre de 2025: la 01:00 ocurre dos veces y el dato no dice cuál es."""
    trips = [
        trip("4327.09", None, "2025-11-02 00:30:00"),
        # La 01:xx repetida queda toda en la segunda 01:00 (-05:00), como dice CLAUDE.md.
        trip("4327.09", None, "2025-11-02 01:30:00", ride_id="A-HORA-REPETIDA"),
        trip("4327.09", None, "2025-11-02 02:30:00"),
        trip("6720.07", None, "2025-11-02 00:10:00"),
        trip("6720.07", None, "2025-11-02 23:50:00"),
        some_route("2025-11-02"),
    ]
    built = run(tmp_path, {"202511": trips}, "2025-11-01", "2025-11-03")

    assert built.hours("4327.09") == [
        ("2025-11-02 04:00", 1, 0),  # 00:00 (-04:00)
        ("2025-11-02 05:00", 0, 0),  # primera 01:00 (-04:00): queda en cero
        ("2025-11-02 06:00", 1, 0),  # segunda 01:00 (-05:00)
        ("2025-11-02 07:00", 1, 0),  # 02:00 (-05:00)
    ]
    assert len(built.hours("6720.07")) == 25
    assert built.report["zona_horaria"][0]["hora_utc"] == "2025-11-02T06:00:00+00:00"
    assert validate_file(built.out / "station_hour.parquet", "station_hour") == []


@pytest.mark.parametrize(
    "local, utc",
    [
        ("2026-07-15 08:20:16.088", "2026-07-15 12:00"),
        ("2026-01-15 08:05:00", "2026-01-15 13:00"),
        ("2026-03-08 01:59:59", "2026-03-08 06:00"),
        ("2026-03-08 02:30:00", "2026-03-08 07:00"),  # no existe: cae en las 03:00 (-04:00)
        ("2026-03-08 03:00:00", "2026-03-08 07:00"),
        ("2025-11-02 00:59:59", "2025-11-02 04:00"),
        ("2025-11-02 01:00:00", "2025-11-02 06:00"),  # repetida: la segunda
        ("2025-11-02 01:59:59", "2025-11-02 06:00"),
        ("2025-11-02 02:00:00", "2025-11-02 07:00"),
    ],
)
def test_el_calculo_de_la_hora_en_python_puro(local, utc):
    """Es la referencia contra la que se compara DuckDB en cada corrida."""
    expected = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)
    assert bt.local_hour_instant(datetime.fromisoformat(local)) == expected


def _timezone_trips() -> dict:
    return {
        "202607": [
            trip("4327.09", "6720.07", "2026-07-15 08:05:00", ride_id="ZONA-1"),
            trip("6720.07", "4327.09", "2026-07-15 23:40:00", minutes=30, ride_id="ZONA-2"),
        ]
    }


def test_la_hora_no_depende_de_la_zona_de_la_sesion_de_duckdb(tmp_path, monkeypatch):
    """Aunque la sesión quedara en otra zona, AT TIME ZONE da los mismos instantes."""
    normal = run(tmp_path / "a", _timezone_trips(), "2026-07-14", "2026-07-16")
    original = bt._connect

    def connect_in_tokyo(work):
        connection = original(work)
        connection.execute("SET TimeZone = 'Asia/Tokyo'")
        return connection

    monkeypatch.setattr(bt, "_connect", connect_in_tokyo)
    other = run(tmp_path / "b", _timezone_trips(), "2026-07-14", "2026-07-16")

    assert normal.hours("4327.09")[0] == ("2026-07-15 12:00", 1, 0)
    assert normal.hours("4327.09")[-1] == ("2026-07-16 04:00", 0, 1)  # llegada a las 00:10 del día 16
    for name in bt.TABLES:
        assert pq.read_table(normal.out / f"{name}.parquet").equals(pq.read_table(other.out / f"{name}.parquet")), name


def test_el_script_da_lo_mismo_con_la_computadora_en_otra_zona(tmp_path):
    """Corre scripts/build_tables.py con la zona del sistema en Tokio y en Monterrey."""
    raw = tmp_path / "crudos"
    write_raw(raw, "202607", _timezone_trips()["202607"])
    config_path = write_config(tmp_path, ["202607"], "2026-07-14", "2026-07-16")
    tables = {}
    for zone in ("Asia/Tokyo", "America/Monterrey"):
        out = tmp_path / zone.replace("/", "_")
        done = subprocess.run(
            [
                sys.executable, str(REPO / "scripts" / "build_tables.py"),
                "--salida", str(out), "--crudos", str(raw), "--gbfs", str(tmp_path / "sin_gbfs"),
                "--config", str(config_path),
            ],
            env={**os.environ, "TZ": zone},
            cwd=tmp_path,  # el script debe funcionar desde cualquier carpeta
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert "Comprobaciones:" in done.stdout and "Listo" in done.stdout
        tables[zone] = read_built(out)
        assert tables[zone].hours("4327.09")[0] == ("2026-07-15 12:00", 1, 0)
    for name in bt.TABLES:
        assert pq.read_table(tmp_path / "Asia_Tokyo" / f"{name}.parquet").equals(
            pq.read_table(tmp_path / "America_Monterrey" / f"{name}.parquet")
        ), name


def test_la_zona_del_sistema_si_cambia_la_zona_por_defecto_de_duckdb():
    """Sin esto, la prueba anterior no probaría nada: TZ debe mover la zona de DuckDB."""
    code = (
        "import duckdb; c = duckdb.connect(); "
        "print(c.execute(\"SELECT current_setting('TimeZone')\").fetchone()[0]); c.close()"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "TZ": "Asia/Tokyo"}, capture_output=True, text=True, timeout=60
    )
    assert done.stdout.strip() == "Asia/Tokyo", done.stderr


def test_la_comprobacion_de_zona_horaria_detecta_un_cast_a_secas(tmp_path, monkeypatch):
    """El error clásico: CAST a TIMESTAMPTZ con la sesión en la zona de la computadora."""
    original = bt._connect

    def connect_in_monterrey(work):
        connection = original(work)
        connection.execute("SET TimeZone = 'America/Monterrey'")
        return connection

    monkeypatch.setattr(bt, "_connect", connect_in_monterrey)
    monkeypatch.setattr(bt, "hour_sql", lambda column: f"CAST(date_trunc('hour', {column}) AS TIMESTAMPTZ)")
    with pytest.raises(bt.BuildError, match="zona horaria: el viaje ZONA-1"):
        run(tmp_path, _timezone_trips(), "2026-07-14", "2026-07-16")
    assert not list((tmp_path / "salida").rglob("*.parquet"))


# --------------------------------------------------------------------------- #
# Duplicados, fechas inválidas y texto
# --------------------------------------------------------------------------- #

def test_un_viaje_repetido_en_dos_meses_se_cuenta_una_vez(tmp_path):
    repeated = trip("4327.09", "6720.07", "2026-07-31 23:50:00", minutes=20, ride_id="REPETIDO")
    months = {
        "202607": [repeated, trip("4327.09", "6720.07", "2026-07-31 10:00:00")],
        "202608": [dict(repeated), trip("6720.07", "4327.09", "2026-08-01 09:00:00")],
    }
    built = run(tmp_path, months, "2026-07-30", "2026-08-02")

    assert built.report["filas"]["leidas"] == 4
    assert built.report["filas"]["ride_id_repetidos"] == 1
    assert built.report["filas"]["duplicadas_quitadas"] == 1
    assert built.report["filas"]["conservadas"] == 3
    assert built.totals("4327.09") == (2, 1)
    assert built.totals("6720.07") == (1, 2)
    assert built.route_trips() == {("4327.09", "6720.07"): 2, ("6720.07", "4327.09"): 1}


def test_se_quitan_los_viajes_sin_fecha_o_que_terminan_antes_de_empezar(tmp_path):
    trips = [
        trip("4327.09", "6720.07", "2026-07-02 10:00:00"),
        trip("4327.09", "6720.07", "2026-07-02 11:00:00", ended_at="2026-07-02 10:59:00"),  # termina antes
        trip("4327.09", "6720.07", None, ended_at="2026-07-02 12:00:00"),                   # sin inicio
        {**trip("4327.09", "6720.07", "2026-07-02 13:00:00"), "ended_at": None},            # sin fin
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    rows = built.report["filas"]
    assert (rows["leidas"], rows["sin_fecha"], rows["terminan_antes_de_empezar"], rows["conservadas"]) == (4, 2, 1, 1)
    assert rows["fechas_invalidas_quitadas"] == 3
    assert built.totals("4327.09") == (1, 0)


def test_los_ids_son_texto_y_conservan_el_cero_final(tmp_path):
    trips = [
        trip("5470.10", "4327.09 ", "2026-07-02 10:00:00"),  # el destino trae un espacio al final
        trip("4327.09", "5470.10", "2026-07-02 11:00:00"),
        # Tabulador y salto de línea en las orillas: también se recortan.
        {**trip("5470.10", "4327.09", "2026-07-02 12:00:00"), "end_station_id": "\t4327.09\n"},
        # Espacio duro (U+00A0): Python lo quita con strip() y la limpieza también.
        {**trip("5470.10", "4327.09", "2026-07-02 13:00:00"), "end_station_id": " 4327.09 "},
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert sorted(built.stations["station_id"]) == ["4327.09", "5470.10"]
    assert built.totals("4327.09") == (1, 3)  # las cuatro escrituras de "4327.09" son la misma estación
    assert built.report["texto"]["ids_con_espacios"] == 3
    for name, columns in (
        ("stations", ["station_id"]),
        ("station_hour", ["station_id"]),
        ("routes", ["start_station_id", "end_station_id"]),
    ):
        schema = pq.read_schema(built.out / f"{name}.parquet")
        for column in columns:
            assert schema.field(column).type == pa.string(), (name, column)
    assert set(built.routes["start_station_id"]) | set(built.routes["end_station_id"]) == {"4327.09", "5470.10"}


def test_un_id_vacio_es_una_estacion_que_falta(tmp_path):
    blank = {**trip("4327.09", "6720.07", "2026-07-02 10:00:00"), "end_station_id": "   "}
    trips = [blank, trip("4327.09", "6720.07", "2026-07-02 11:00:00")]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert built.report["texto"]["ids_vacios"] == 1
    assert built.report["puntas"]["fin"]["sin_estacion"] == 1
    assert built.totals("4327.09") == (2, 0)
    assert built.totals("6720.07") == (0, 1)
    # Trae el nombre de la estación de fin pero no su id: se reporta aparte.
    assert built.report["llegadas_con_nombre_pero_sin_id"] == 1


# --------------------------------------------------------------------------- #
# Unión de variantes de un mismo id (decisión 15)
# --------------------------------------------------------------------------- #

def _variant_trips() -> list[dict]:
    day = "2026-07-02"
    return [
        # 5470.10 (2 puntas) y 5470.1 (1 punta): mismo nombre y coordenada.
        trip("5470.10", "4327.09", f"{day} 08:00:00"),
        trip("5470.10", "4327.09", f"{day} 08:20:00"),
        trip("4327.09", "5470.1", f"{day} 09:00:00"),
        # 6131.1 no tiene gemelo.
        trip("6131.1", "4327.09", f"{day} 10:00:00"),
        # 5308.04 (2 puntas) y 5308.04_ (1 punta, nombre con espacio al final).
        trip("5308.04_", "5308.04", f"{day} 11:00:00"),
        trip("5308.04", "4327.09", f"{day} 11:30:00"),
        # 5303.06 (2 puntas) y 5303.06_ (1 punta) están a 30 metros.
        trip("5303.06", "5303.06_", f"{day} 12:00:00"),
        trip("5303.06", "4327.09", f"{day} 12:30:00"),
        # 7000.1 y 7000.10 tienen nombres distintos.
        trip("7000.1", "7000.10", f"{day} 13:00:00"),
        # 8000.1 (3 puntas) le gana a 8000.10 (1 punta).
        trip("8000.1", "8000.1", f"{day} 14:00:00"),
        trip("8000.1", "8000.10", f"{day} 14:30:00"),
        # 9000.1 y 9000.10 empatan con 1 punta cada una.
        trip("9000.1", "9000.10", f"{day} 15:00:00"),
    ]


def test_las_variantes_se_unen_solo_si_coinciden_nombre_y_coordenadas(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")

    assert sorted(built.stations["station_id"]) == [
        "4327.09", "5303.06", "5303.06_", "5308.04", "5470.10", "6131.1", "7000.1", "7000.10", "8000.1", "9000.10",
    ]
    assert built.totals("5470.10") == (2, 1)   # la llegada escrita como 5470.1 cuenta aquí
    assert built.totals("6131.1") == (1, 0)    # un decimal sin gemelo: intacto
    assert built.totals("5308.04") == (2, 1)
    assert built.totals("5303.06") == (2, 0) and built.totals("5303.06_") == (0, 1)
    assert built.totals("7000.1") == (1, 0) and built.totals("7000.10") == (0, 1)
    assert built.totals("8000.1") == (2, 2)    # queda la escritura con más viajes, aunque tenga un decimal
    assert built.totals("9000.10") == (1, 1)   # empate: queda la que ordena al final

    routes = built.route_trips()
    assert routes[("5308.04", "5308.04")] == 1  # tras unir, sale y llega a la misma estación
    assert routes[("5303.06", "5303.06_")] == 1
    assert routes[("8000.1", "8000.1")] == 2
    assert routes[("9000.10", "9000.10")] == 1

    names = dict(zip(built.stations["station_id"], built.stations["name"]))
    assert names["5308.04"] == "Metropolitan Ave & Bedford Ave"  # nombre más frecuente, ya recortado
    assert names["8000.1"] == "Avenida Ocho"


def test_el_reporte_dice_cuantas_variantes_se_unieron_y_cuales_no(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    merge = built.report["union_de_ids"]

    assert (merge["grupos_encontrados"], merge["grupos_unidos"], merge["grupos_no_unidos"]) == (6, 4, 2)
    assert merge["puntas_en_grupos_unidos"] == 3 + 3 + 4 + 2
    assert merge["puntas_en_grupos_no_unidos"] == 3 + 2
    # Cambian de id: 5470.1 (1 llegada), 5308.04_ (1 salida), 8000.10 (1 llegada) y 9000.1 (1 salida).
    assert (merge["salidas_que_cambiaron_de_id"], merge["llegadas_que_cambiaron_de_id"]) == (2, 2)
    assert merge["ids_que_cambiaron"] == 4
    apart = {g["forma_comun"]: g for g in merge["lista_no_unidos"]}
    assert sorted(apart) == ["5303.06", "7000.10"]
    far = next(v for v in apart["5303.06"]["variantes"] if not v["es_el_id_que_queda"])
    assert far["id"] == "5303.06_" and far["distancia_m"] == pytest.approx(30.0, abs=0.1)
    assert "coordenadas a 30.0 m" in far["motivo"]
    other_name = next(v for v in apart["7000.10"]["variantes"] if not v["es_el_id_que_queda"])
    assert other_name["motivo"] == "nombre distinto" and other_name["distancia_m"] == 0.0
    assert {g["id_que_queda"] for g in merge["lista_unidos"]} == {"5470.10", "5308.04", "8000.1", "9000.10"}

    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "`5303.06_`" in text and "coordenadas a 30.0 m (máximo 1 m)" in text
    assert "nombre distinto" in text
    # Un grupo de tres escrituras puede unirse solo en parte: el texto no dice que nada se unió.
    assert "- Se unieron por completo: 4 grupos" in text
    assert "- No se unieron, o solo en parte: 2 grupos" in text
    assert "Cada id sigue como estación aparte" not in text


def test_la_distancia_maxima_sale_de_la_configuracion(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", distancia=50)

    # Con 50 metros de tolerancia el par que está a 30 metros sí se une.
    assert {"5303.06", "5303.06_"} & set(built.stations["station_id"]) == {"5303.06"}
    assert built.totals("5303.06") == (2, 1)
    assert built.report["union_de_ids"]["grupos_no_unidos"] == 1  # solo queda el de nombre distinto


def _datos(**changes) -> Datos:
    periodo = Tramo("periodo", date(2026, 7, 1), date(2026, 7, 3))
    values = dict(
        meses_crudos=("202607",), periodo=periodo, tramos={}, distancia_maxima_m=1.0,
        prefijos_excluidos=("HB", "JC", "SYS"), ids_excluidos=("1234.56", "Shop Morgan"),
    )
    return Datos(**{**values, **changes})


def test_plan_de_ids_con_tres_variantes_y_con_ids_que_no_son_estacion():
    same = ("Calle", 40.7, -74.0)
    plan = bt.plan_ids(
        [
            bt.RawId("100.1", 5, 5, *same),
            bt.RawId("100.10", 1, 0, *same),
            bt.RawId("100.10_", 2, 0, "Calle", 40.7 + 20 * METRE, -74.0),  # a 20 metros de la que queda
            bt.RawId("200.1", 0, 0, None, None, None),     # solo aparece fuera del periodo
            bt.RawId("SYS001", 3, 4, "Taller", 40.7, -74.0),
            bt.RawId("SYS002", 0, 0, None, None, None),    # excluida y sin viajes en el periodo
        ],
        _datos(),
    )
    assert plan.station_of == {
        "100.1": "100.1", "100.10": "100.1", "100.10_": "100.10_", "200.1": None, "SYS001": None, "SYS002": None,
    }
    assert plan.state_of["200.1"] == "inactiva" and plan.state_of["SYS001"] == "excluida"
    assert len(plan.groups) == 1 and plan.groups[0]["unido"] is False
    assert plan.excluded == [
        {
            "regla": "prefijo SYS", "estaciones": 1, "salidas": 3, "llegadas": 4,
            "ids": [{"id": "SYS001", "nombre": "Taller", "salidas": 3, "llegadas": 4}],
        }
    ]


def test_plan_de_ids_se_detiene_si_dos_escrituras_coinciden_entre_si_pero_no_con_la_que_queda():
    """Sin esto, 100.1 y 100.10 quedarían como dos estaciones con el mismo nombre y la misma coordenada."""
    same = ("Calle", 40.7, -74.0)
    with pytest.raises(bt.BuildError, match="100.1 y 100.10 tienen el mismo nombre y la misma coordenada"):
        bt.plan_ids(
            [
                bt.RawId("100.10_", 6, 0, "Calle", 40.7 + 20 * METRE, -74.0),  # el id con más viajes, a 20 metros
                bt.RawId("100.10", 3, 0, *same),
                bt.RawId("100.1", 0, 2, *same),
            ],
            _datos(),
        )


def test_la_forma_comun_solo_sirve_para_encontrar_gemelos():
    assert bt.candidate_form("2231.1") == "2231.10"
    assert bt.candidate_form("2231.10") == "2231.10"
    assert bt.candidate_form("5308.04_") == "5308.04"
    assert bt.candidate_form("5303.1__") == "5303.10"
    assert bt.candidate_form("JC013") == "JC013"
    assert bt.normalise_name("  Avenida   OCHO ") == bt.normalise_name("avenida ocho")
    assert bt.haversine_m(40.7155, -73.9870, 40.7155 + 30 * METRE, -73.9870) == pytest.approx(30.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Estaciones excluidas, puntas sin estación y bordes del periodo
# --------------------------------------------------------------------------- #

def test_las_estaciones_excluidas_no_tienen_filas_y_la_otra_punta_si_cuenta(tmp_path):
    day = "2026-07-02"
    trips = [
        trip("4327.09", "HB101", f"{day} 08:00:00"),          # salida sin llegada
        trip("SYS016", "4327.09", f"{day} 09:00:00"),         # llegada sin salida
        trip("4327.09", "Shop Morgan ", f"{day} 10:00:00"),   # se compara ya sin el espacio
        trip("1234.56", "JC013", f"{day} 11:00:00"),          # no cuenta en ningún lado
        trip("4327.09", "6720.07", f"{day} 12:00:00"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert sorted(built.stations["station_id"]) == ["4327.09", "6720.07"]
    assert set(built.station_hour["station_id"]) == {"4327.09", "6720.07"}
    assert built.totals("4327.09") == (3, 1)
    assert built.totals("6720.07") == (0, 1)
    assert built.route_trips() == {("4327.09", "6720.07"): 1}
    assert built.report["puntas"]["inicio"]["excluidas"] == 2
    assert built.report["puntas"]["fin"]["excluidas"] == 3
    by_rule = {e["regla"]: (e["estaciones"], e["salidas"], e["llegadas"]) for e in built.report["estaciones_excluidas"]}
    assert by_rule == {
        "prefijo HB": (1, 0, 1),
        "prefijo JC": (1, 0, 1),
        "prefijo SYS": (1, 1, 0),
        "id 1234.56": (1, 1, 0),
        "id Shop Morgan": (1, 0, 1),
    }
    assert built.report["salidas_segun_su_fin"]["fin_en_estacion_excluida"] == 2


def test_puntas_sin_estacion_y_bordes_del_periodo(tmp_path):
    a, b = "4327.09", "6720.07"
    trips = [
        trip(a, b, "2026-06-30 23:50:00", minutes=20),   # empieza antes del periodo: solo llegada
        trip(a, b, "2026-07-03 23:50:00", minutes=20),   # termina después: solo salida (y sí es una ruta)
        trip(a, b, "2026-07-04 10:00:00"),               # el día siguiente al fin está en el crudo: no cuenta
        trip(None, b, "2026-07-02 10:00:00"),            # sin estación de inicio: solo llegada
        trip(a, None, "2026-07-02 11:00:00"),            # sin estación de fin: solo salida
        trip(None, None, "2026-07-02 11:30:00"),
        trip(a, b, "2026-07-02 12:00:00"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert built.totals(a) == (3, 0)
    assert built.totals(b) == (0, 3)
    assert built.hours(b)[0] == ("2026-07-01 04:00", 0, 1)    # 00:10 del 1 de julio, hora local
    assert built.hours(a)[-1] == ("2026-07-04 03:00", 1, 0)   # 23:00 del 3 de julio, la última hora del periodo
    assert built.route_trips() == {(a, b): 2}
    assert built.report["puntas"] == {
        "inicio": {"contadas": 3, "sin_estacion": 2, "excluidas": 0, "fuera_del_periodo": 2, "sin_clasificar": 0},
        "fin": {"contadas": 3, "sin_estacion": 2, "excluidas": 0, "fuera_del_periodo": 2, "sin_clasificar": 0},
    }
    assert built.report["salidas_segun_su_fin"] == {
        "con_estacion_de_fin": 2,
        "sin_estacion_de_fin": 1,
        "fin_en_estacion_excluida": 0,
        "fin_en_estacion_sin_viajes_en_el_periodo": 0,
    }
    hours = built.station_hour["ts_hour"]
    assert (hours >= pd.Timestamp("2026-07-01", tz=TZ)).all() and (hours < pd.Timestamp("2026-07-04", tz=TZ)).all()


def test_los_bordes_exactos_del_periodo(tmp_path):
    """El periodo incluye las 00:00:00 del primer día y excluye las 00:00:00 del día siguiente al último."""
    a, b = "4327.09", "6720.07"
    trips = [
        trip(a, b, "2026-07-01 00:00:00", minutes=5),    # empieza en el primer instante: cuenta entero
        trip(a, b, "2026-06-30 23:50:00", minutes=10),   # termina justo a las 00:00:00 del día 1: solo llegada
        trip(a, b, "2026-07-03 23:50:00", minutes=10),   # termina justo a las 00:00:00 del día 4: solo salida
        trip(a, b, "2026-07-04 00:00:00", minutes=5),    # empieza en el primer instante de fuera: no cuenta
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert built.totals(a) == (2, 0)
    assert built.totals(b) == (0, 2)
    assert built.hours(a)[0] == ("2026-07-01 04:00", 1, 0)
    assert built.hours(a)[-1] == ("2026-07-04 03:00", 1, 0)
    assert built.hours(b) == [("2026-07-01 04:00", 0, 2)]
    assert built.route_trips() == {(a, b): 2}
    assert built.report["puntas"]["inicio"]["fuera_del_periodo"] == 2
    assert built.report["puntas"]["fin"]["fuera_del_periodo"] == 2


def test_una_estacion_que_solo_aparece_fuera_del_periodo_no_entra(tmp_path):
    trips = [
        trip("4327.09", "6720.07", "2026-07-02 10:00:00"),
        trip("4327.09", "6131.1", "2026-07-03 23:55:00", minutes=20),  # llega el día 4, fuera del periodo
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert sorted(built.stations["station_id"]) == ["4327.09", "6720.07"]
    assert built.totals("4327.09") == (2, 0)
    assert built.route_trips() == {("4327.09", "6720.07"): 1}
    assert built.report["salidas_segun_su_fin"]["fin_en_estacion_sin_viajes_en_el_periodo"] == 1


# --------------------------------------------------------------------------- #
# Tabla densa, rutas y tipo de usuario
# --------------------------------------------------------------------------- #

def test_station_hour_es_densa_solo_entre_la_primera_y_la_ultima_hora(tmp_path):
    trips = [
        trip("4327.09", None, "2026-07-01 05:10:00"),
        trip("4327.09", None, "2026-07-02 08:20:00"),
        trip(None, "6720.07", "2026-07-02 14:00:00"),  # una sola hora con viajes
        some_route("2026-07-02"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    rows = built.hours("4327.09")
    assert len(rows) == 28  # de las 05:00 del día 1 a las 08:00 del día 2
    assert rows[0] == ("2026-07-01 09:00", 1, 0) and rows[-1] == ("2026-07-02 12:00", 1, 0)
    assert all(row[1:] == (0, 0) for row in rows[1:-1])
    assert built.hours("6720.07") == [("2026-07-02 18:00", 0, 1)]
    zeros = built.station_hour[built.station_hour["departures"] + built.station_hour["arrivals"] == 0]
    assert (zeros[["net_flow", "departures_member", "departures_casual"]] == 0).all().all()
    hours = built.report["station_hour"]  # 28 + 1 filas, más la de some_route
    assert (hours["filas"], hours["filas_con_viajes"], hours["filas_en_cero"]) == (30, 4, 26)
    assert hours["proporcion_en_cero"] == pytest.approx(26 / 30, abs=1e-6)


def test_el_reporte_cuenta_los_ceros_de_rachas_largas(tmp_path):
    trips = [
        trip("4327.09", None, "2026-07-01 00:10:00"),
        trip("4327.09", None, "2026-07-04 01:10:00"),   # 72 horas en cero entre los dos viajes
        trip("6720.07", None, "2026-07-01 00:10:00"),
        trip("6720.07", None, "2026-07-04 00:10:00"),   # 71 horas en cero: no llega
        some_route("2026-07-02"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-05")

    assert built.report["station_hour"]["ceros_en_rachas_largas"] == {
        "horas_minimas_de_la_racha": bt.LONG_ZERO_RUN_HOURS, "rachas": 1, "filas": 72, "estaciones": 1,
    }


def test_rutas_y_salidas_por_tipo_de_usuario(tmp_path):
    a, b = "4327.09", "6720.07"
    hour = "2026-07-02 08"
    trips = [
        trip(a, b, f"{hour}:00:00", member="member"),
        trip(a, b, f"{hour}:10:00", member="member"),
        trip(a, b, f"{hour}:20:00", member="casual"),
        trip(a, a, f"{hour}:30:00", member="casual"),  # sale y llega a la misma estación
        trip(a, a, f"{hour}:40:00", member="member"),
        trip(b, a, f"{hour}:45:00", member="casual"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert built.route_trips() == {(a, b): 3, (a, a): 2, (b, a): 1}
    assert int(built.routes["trips"].sum()) == 6
    row = built.station_hour[built.station_hour["station_id"] == a].iloc[0]
    assert (row["departures"], row["arrivals"], row["net_flow"]) == (5, 3, -2)
    assert (row["departures_member"], row["departures_casual"]) == (3, 2)
    assert (built.station_hour["departures_member"] + built.station_hour["departures_casual"]).equals(
        built.station_hour["departures"]
    )
    assert built.report["routes"] == {
        "pares": 3, "viajes": 6, "pares_misma_estacion": 1, "viajes_misma_estacion": 2,
    }


# --------------------------------------------------------------------------- #
# Estaciones y capacidad de GBFS
# --------------------------------------------------------------------------- #

def _capacity_trips() -> list[dict]:
    return [
        trip("4327.09", "6720.07", "2026-07-02 08:00:00"),
        trip("6720.07", "6131.1", "2026-07-02 09:00:00"),
    ]


def test_la_capacidad_sale_de_la_ultima_foto_de_gbfs(tmp_path):
    gbfs = tmp_path / "gbfs"
    write_gbfs(gbfs, "20261010T040000Z", {"4327.09": 99, "6720.07": 99, "6131.1": 99})
    # El short_name de GBFS también se recorta antes de cruzarlo con el id de los viajes.
    write_gbfs(gbfs, "20261010T050000Z", {"4327.09 ": 30, "6720.07": 0, "9999.99": 12})
    built = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs)

    capacity = dict(zip(built.stations["station_id"], built.stations["capacity"]))
    assert capacity["4327.09"] == 30
    assert pd.isna(capacity["6720.07"])   # 0 en GBFS es "no instalada hoy": se guarda vacía
    assert pd.isna(capacity["6131.1"])    # no está en la foto
    assert built.stations["station_type"].isna().all()
    report = built.report["capacidad"]
    assert (report["con_capacidad"], report["capacidad_cero_hoy"], report["no_estan_en_gbfs"]) == (1, 1, 1)
    assert report["lista_no_estan_en_gbfs"] == [
        {"id": "6131.1", "nombre": "E 24 St & Park Ave S", "salidas": 0, "llegadas": 1}
    ]
    photo = built.report["entradas"]["gbfs"]
    assert photo["foto"] == "info_20261010T050000Z.parquet"
    assert photo["dias_despues_del_periodo"] == 99  # del 3 de julio al 10 de octubre


def test_sin_fotos_de_gbfs_la_capacidad_queda_vacia_y_el_reporte_lo_dice(tmp_path):
    built = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=tmp_path / "no_existe")

    assert built.stations["capacity"].isna().all()
    assert built.report["entradas"]["gbfs"]["foto"] is None
    assert built.report["capacidad"]["con_capacidad"] == 0
    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "No se encontró ninguna foto de GBFS" in text


def test_stations_lleva_nombre_y_coordenada_mas_frecuentes(tmp_path):
    odd = {**trip("4327.09", "6720.07", "2026-07-02 10:00:00"), "start_station_name": "Otro nombre", "start_lat": 40.9}
    trips = [
        trip("4327.09", "6720.07", "2026-07-02 08:00:00"),
        trip("4327.09", "6720.07", "2026-07-02 09:00:00"),
        odd,
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    row = built.stations.set_index("station_id").loc["4327.09"]
    assert (row["name"], row["lat"], row["lon"]) == ("Putnam Ave & Nostrand Ave", 40.6840, -73.9500)
    assert set(built.stations["station_id"]) == set(built.station_hour["station_id"])


# --------------------------------------------------------------------------- #
# Contratos, reporte y fallos
# --------------------------------------------------------------------------- #

def test_las_tres_salidas_cumplen_su_contrato_y_el_reporte_existe(tmp_path):
    gbfs = tmp_path / "gbfs"
    write_gbfs(gbfs, "20261010T050000Z", {"4327.09": 30})
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs)

    for name in bt.TABLES:
        assert validate_file(built.out / f"{name}.parquet", name) == [], name
        assert pq.read_schema(built.out / f"{name}.parquet").equals(CONTRACTS[name].schema), name
    written = sorted(p.name for p in built.out.iterdir())
    assert written == ["reports", "routes.parquet", "station_hour.parquet", "stations.parquet"]

    on_disk = json.loads((built.out / "reports" / "limpieza.json").read_text(encoding="utf-8"))
    assert on_disk == built.report
    for key in (
        "generado_utc", "config", "entradas", "filas", "texto", "puntas", "union_de_ids", "estaciones_excluidas",
        "capacidad", "tablas", "station_hour", "routes", "zona_horaria", "comprobaciones",
    ):
        assert key in on_disk, key
    assert on_disk["config"]["periodo"] == {"inicio": "2026-07-01", "fin": "2026-07-03"}
    assert on_disk["tablas"]["station_hour"]["filas"] == len(built.station_hour)
    assert on_disk["comprobaciones"] and all(c["cumple"] for c in on_disk["comprobaciones"])
    assert on_disk["zona_horaria"] and all(t["coincide"] for t in on_disk["zona_horaria"])

    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    for expected in ("demanda atendida", "La capacidad es la de hoy", "rachas de 72 horas", "Salidas sin llegada"):
        assert expected in text, expected
    assert "recuperad" not in text.lower()  # decisión 9: nunca "viajes recuperados"
    # Capacidad 0 es "no instalada hoy", pero no toda estación no instalada trae capacidad 0.
    assert "Al revés no vale" in text and "`is_installed`" in text
    assert "GBFS pone 0 a las estaciones" not in text


def test_dos_corridas_dan_tablas_identicas(tmp_path):
    first = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", out="uno")
    second = run(tmp_path, {"202607": list(reversed(_variant_trips()))}, "2026-07-01", "2026-07-03", out="dos")
    for name in ("station_hour", "routes"):
        assert pq.read_table(first.out / f"{name}.parquet").equals(pq.read_table(second.out / f"{name}.parquet")), name
    assert first.stations.equals(second.stations)


def test_si_una_tabla_no_cuadra_con_los_viajes_no_se_escribe_nada(tmp_path, monkeypatch):
    """Sabotaje: routes pierde una fila. La comprobación de sumas debe detenerlo todo."""
    good = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    before = {p.name: p.read_bytes() for p in good.out.rglob("*") if p.is_file()}

    original = bt._routes_table
    monkeypatch.setattr(bt, "_routes_table", lambda con: original(con).slice(1))
    with pytest.raises(bt.BuildError, match="suma de routes.trips"):
        run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    with pytest.raises(bt.BuildError, match="No se escribió nada"):
        run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", out="otra")

    # Lo que ya había sigue igual, y en una carpeta nueva no queda nada.
    assert {p.name: p.read_bytes() for p in good.out.rglob("*") if p.is_file()} == before
    assert not [p for p in (tmp_path / "otra").rglob("*") if p.is_file()]


def test_si_station_hour_pierde_una_hora_tambien_se_detecta(tmp_path, monkeypatch):
    original = bt._station_hour_table

    def without_last_row(con):
        table = original(con)
        return table.slice(0, table.num_rows - 1)

    monkeypatch.setattr(bt, "_station_hour_table", without_last_row)
    with pytest.raises(bt.BuildError, match="suma de"):
        run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    assert not list((tmp_path / "salida").rglob("*.parquet"))


def test_si_station_hour_queda_corrida_una_hora_la_comprobacion_de_zona_lo_detecta(tmp_path, monkeypatch):
    """Sabotaje: todas las horas una hora tarde. Las sumas cuadran y el contrato se cumple;
    solo lo nota la comparación contra el conteo por hora hecho en Python."""
    original = bt._station_hour_table

    def one_hour_late(con):
        table = original(con)
        late = pc.add(table["ts_hour"], pa.scalar(timedelta(hours=1), pa.duration("us")))
        return table.set_column(table.schema.get_field_index("ts_hour"), "ts_hour", late)

    monkeypatch.setattr(bt, "_station_hour_table", one_hour_late)
    with pytest.raises(bt.BuildError, match="zona horaria: las salidas por hora de la estación"):
        run(tmp_path, _timezone_trips(), "2026-07-14", "2026-07-16")
    assert not list((tmp_path / "salida").rglob("*.parquet"))


def test_un_tipo_de_usuario_desconocido_detiene_la_corrida(tmp_path):
    trips = [trip("4327.09", "6720.07", "2026-07-02 08:00:00", member="otro")]
    with pytest.raises(bt.BuildError, match="miembros \\+ casuales"):
        run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")


def test_falta_un_mes_crudo(tmp_path, capsys):
    raw = tmp_path / "crudos"
    write_raw(raw, "202607", [trip("4327.09", "6720.07", "2026-07-02 08:00:00")])
    config_path = write_config(tmp_path, ["202607", "202608"], "2026-07-01", "2026-07-03")

    code = bt.main(["--salida", str(tmp_path / "salida"), "--crudos", str(raw), "--config", str(config_path)])

    assert code == 1
    error = capsys.readouterr().err
    assert "Faltan meses crudos" in error and "python scripts/download_trips.py 202608" in error
    assert not (tmp_path / "salida").exists()


def _run_main(tmp_path, capsys, fin: str, gbfs: Path) -> tuple[int, str, str]:
    """Corre el script sobre julio inventado, del día 1 a `fin`. Devuelve (código, salida, errores)."""
    raw = tmp_path / "crudos"
    write_raw(raw, "202607", [trip("4327.09", "6720.07", "2026-07-02 08:00:00")])
    config_path = write_config(tmp_path, ["202607"], "2026-07-01", fin)
    code = bt.main(
        ["--salida", str(tmp_path / "salida"), "--crudos", str(raw), "--gbfs", str(gbfs), "--config", str(config_path)]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_el_script_avisa_si_no_hay_ninguna_foto_de_gbfs(tmp_path, capsys):
    """La corrida termina bien, pero stations sale sin capacidad: no puede pasar en silencio."""
    code, out, err = _run_main(tmp_path, capsys, "2026-07-03", tmp_path / "no_existe")
    assert code == 0 and "Comprobaciones:" in out and "(0 con capacidad)" in out
    assert "AVISO: no hay ninguna foto de GBFS en " + str(tmp_path / "no_existe" / "station_information") in err
    assert "TODAS las estaciones" in err and "python scripts/capture_gbfs.py --una-vez" in err

    gbfs = tmp_path / "gbfs"
    write_gbfs(gbfs, "20261010T050000Z", {"4327.09": 30})
    code, out, err = _run_main(tmp_path, capsys, "2026-07-03", gbfs)
    assert code == 0 and "(1 con capacidad)" in out
    assert "AVISO" not in err


def test_el_script_avisa_si_el_periodo_llega_al_ultimo_dia_del_ultimo_mes(tmp_path, capsys):
    """Los archivos traen los viajes que TERMINAN en cada mes: sin el mes siguiente faltan salidas del último día."""
    gbfs = tmp_path / "gbfs"
    write_gbfs(gbfs, "20261010T050000Z", {"4327.09": 30})
    code, _, err = _run_main(tmp_path, capsys, "2026-07-31", gbfs)
    assert code == 0
    assert "AVISO: el periodo termina el 2026-07-31 y el mes 202608 no está en meses_crudos" in err
    assert "TERMINAN" in err

    code, _, err = _run_main(tmp_path, capsys, "2026-07-30", gbfs)  # el día 31 también está en el archivo de julio
    assert code == 0
    assert "AVISO" not in err


def test_un_id_guardado_como_numero_se_rechaza(tmp_path):
    """Un id leído como número ya perdió el cero final: no se convierte a texto a escondidas."""
    fields = [pa.field(f.name, pa.float64()) if f.name == "start_station_id" else f for f in RAW_SCHEMA]
    row = {**trip("4327.09", "6720.07", "2026-07-02 08:00:00"), "start_station_id": 5470.1}
    raw = tmp_path / "crudos"
    write_raw(raw, "202607", [row], schema=pa.schema(fields))
    config_path = write_config(tmp_path, ["202607"], "2026-07-01", "2026-07-03")

    with pytest.raises(bt.BuildError, match="start_station_id es double y debe ser texto"):
        bt.build(load_datos(config_path), raw, None, tmp_path / "salida", log=lambda m: None)


@pytest.mark.parametrize("folder", [DATA_RAW, DATA_RAW / "trips", DATA_SAMPLE])
def test_la_salida_no_puede_ser_la_carpeta_cruda_ni_la_de_muestra(tmp_path, folder):
    raw = tmp_path / "crudos"
    write_raw(raw, "202607", [trip("4327.09", "6720.07", "2026-07-02 08:00:00")])
    config_path = write_config(tmp_path, ["202607"], "2026-07-01", "2026-07-03")
    before = sorted(folder.iterdir()) if folder.exists() else None

    with pytest.raises(bt.BuildError, match="La carpeta de salida no puede ser"):
        bt.build(load_datos(config_path), raw, None, folder, log=lambda m: None)
    assert (sorted(folder.iterdir()) if folder.exists() else None) == before


def test_la_ayuda_esta_en_espanol(capsys):
    with pytest.raises(SystemExit) as stop:
        bt.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for option in ("--salida", "--crudos", "--gbfs", "--config"):
        assert option in text


# --------------------------------------------------------------------------- #
# Con los datos reales (se salta si no están en la máquina)
# --------------------------------------------------------------------------- #

# Viajes reales conocidos (vistos el 10 de octubre de 2026 en los archivos crudos):
# (ride_id, mes del archivo, started_at tal como viene, estación, hora local, segundos UTC desde 1970).
KNOWN_TRIPS = [
    ("0005122465C6A4A8", "202607", "2026-07-15 08:20:16.088", "4327.09", "2026-07-15 08:00:00", 1784116800),
    ("0002A41A4CDD2718", "202608", "2026-08-12 17:20:32.632", "6720.07", "2026-08-12 17:00:00", 1786568400),
]


@pytest.mark.parametrize("ride_id, month, started_at, station, local_hour, epoch", KNOWN_TRIPS)
def test_horas_conocidas_de_los_datos_reales(ride_id, month, started_at, station, local_hour, epoch):
    """La fila de station_hour de un viaje real conocido existe y sus salidas cuadran.

    El conteo de referencia se hace aquí con pyarrow sobre los archivos crudos,
    sin DuckDB y sin el SQL de la limpieza.
    """
    processed = DATA_PROCESSED / "station_hour.parquet"
    datos = load_datos()
    raw = {m: DATA_RAW / "trips" / f"{m}.parquet" for m in datos.meses_crudos}
    missing = [str(p.relative_to(REPO)) for p in (processed, *raw.values()) if not p.exists()]
    if month not in raw:
        pytest.skip(f"config/datos.yaml ya no usa el mes {month}: este viaje conocido no aplica.")
    if missing:
        pytest.skip(
            f"Faltan archivos reales ({', '.join(missing)}). Los meses crudos se bajan con "
            "scripts/download_trips.py; después corre scripts/build_tables.py."
        )

    # 1) El viaje conocido está en el crudo tal como se anotó.
    hour_start = datetime.fromisoformat(local_hour)
    in_hour = [("started_at", ">=", hour_start), ("started_at", "<", hour_start + timedelta(hours=1))]
    columns = ["ride_id", "started_at", "start_station_id"]
    found = pa.concat_tables(pq.read_table(path, columns=columns, filters=in_hour) for path in raw.values()).to_pandas()
    known = found[found["ride_id"] == ride_id]
    assert len(known) == 1, f"el viaje {ride_id} no está en los archivos crudos"
    assert known["started_at"].iloc[0] == pd.Timestamp(started_at)
    assert known["start_station_id"].iloc[0] == station

    # 2) Conteo aparte: salidas de esa estación (y de las escrituras que se le unieron) en esa hora local.
    variants = {station}
    report_path = DATA_PROCESSED / "reports" / "limpieza.json"
    if report_path.exists():
        for group in json.loads(report_path.read_text(encoding="utf-8"))["union_de_ids"]["lista_unidos"]:
            if group["id_que_queda"] == station:
                variants |= {v["id"] for v in group["variantes"]}
    found = found.drop_duplicates("ride_id")
    expected = int(found["start_station_id"].str.strip().isin(variants).sum())
    assert expected >= 1

    # 3) La fila existe en la tabla, en el instante correcto, con esas salidas.
    table = pq.read_table(processed, filters=[("station_id", "==", station)]).to_pandas()
    instant = pd.Timestamp(local_hour, tz=TZ)
    assert int(instant.timestamp()) == epoch  # p. ej. 08:00 de Nueva York en julio son las 12:00 UTC
    row = table[table["ts_hour"] == instant]
    assert len(row) == 1, f"no hay fila para {station} a las {local_hour} de Nueva York"
    assert int(row["departures"].iloc[0]) == expected

    # 4) La tabla no se sale del periodo de la configuración.
    edges = pc.min_max(pq.read_table(processed, columns=["ts_hour"])["ts_hour"]).as_py()
    assert edges["min"] >= datos.periodo.start and edges["max"] < datos.periodo.end
