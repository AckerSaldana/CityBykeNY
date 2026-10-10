"""Pruebas de la limpieza y de las tablas reales (src/data/build_tables.py).

No usan internet ni los datos reales: cada prueba escribe en una carpeta
temporal unos pocos viajes INVENTADOS con el formato del Parquet crudo, una
configuración y, si hace falta, un archivo congelado de GBFS. Después corre la limpieza
completa y revisa las tablas que salen.

Cada corrida de prueba compara además trips.parquet (la tabla interna de
viajes) contra station_hour y routes, con un conteo aparte hecho en pandas.

La única excepción son las dos últimas pruebas: comparan horas conocidas de
data/processed/station_hour.parquet y de trips.parquet contra un conteo aparte
hecho sobre los archivos crudos reales. Se SALTAN si esos archivos no están en
la máquina.

Córrela sola con:  pytest tests/test_build_tables.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
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


def write_gbfs(path: Path, capacities: dict, until: str = "2026-10-10T17:00:00Z") -> Path:
    """Un archivo congelado de GBFS como el que escribe scripts/freeze_gbfs.py: short_name -> capacidad."""
    lines = [
        "# Archivo congelado de prueba.",
        "# captura_desde_utc: 2026-10-10T04:00:00Z",
        f"# captura_hasta_utc: {until}",
        "# fotos_de_estado: 12",
        "# estados_de_informacion: 2",
        "short_name,capacity,ocupacion_inicial,fotos",
    ]
    lines += [f"{name},{capacity},{'0.500000' if capacity else ''},{3 if capacity else 0}"
              for name, capacity in capacities.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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

    def trip_rows(self) -> list[tuple]:
        """Filas de trips.parquet, en el orden del archivo: (salida en UTC, llegada en UTC, inicio, fin).

        Una punta sin estación sale como None.
        """
        return [row[1:] for row in self.trip_rows_with_id()]

    def trip_rows_with_id(self) -> list[tuple]:
        """Como `trip_rows`, con el ride_id por delante."""
        def utc(stamp: datetime) -> str:
            text = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
            return text.removesuffix("000").removesuffix(".000")  # sin milisegundos si son cero

        return [
            (r["ride_id"], utc(r["started_at"]), utc(r["ended_at"]), r["start_station_id"], r["end_station_id"])
            for r in pq.read_table(self.out / "trips.parquet").to_pylist()
        ]

    def trips_check_names(self) -> list[str]:
        return [c["nombre"] for c in self.report["comprobaciones"] if c["nombre"].startswith("trips: ")]


# La limpieza hace 17 comprobaciones sobre las tres tablas y 13 más sobre trips.parquet.
TABLE_CHECKS = 17
TRIPS_CHECKS = 13


def read_built(out: Path, report: dict | None = None) -> Built:
    if report is None:
        report = json.loads((out / "reports" / "limpieza.json").read_text(encoding="utf-8"))
    return Built(out, report, *(read_contract(name, out / f"{name}.parquet") for name in bt.TABLES))


def assert_trips_reproduce_tables(built: Built, inicio: str, fin: str) -> None:
    """trips.parquet da las mismas salidas, llegadas y rutas que station_hour y routes.

    El conteo se hace aquí con pandas, sin DuckDB y sin el SQL de la limpieza:
    una salida cuenta en la hora de started_at si cae en el periodo y trae
    estación; una llegada, en la hora de ended_at; una ruta, si el viaje sale
    en el periodo y trae las dos estaciones.
    """
    trips = pq.read_table(built.out / "trips.parquet").to_pandas()
    periodo = Tramo("periodo", date.fromisoformat(inicio), date.fromisoformat(fin))

    def from_trips(station: str, stamp: str) -> dict:
        rows = trips[periodo.contains(trips[stamp]) & trips[station].notna()]
        hours = rows[stamp].dt.tz_convert("UTC").dt.floor("h")  # en UTC: las horas de Nueva York son enteras
        return {key: int(n) for key, n in rows.groupby([rows[station], hours]).size().items()}

    def from_table(column: str) -> dict:
        rows = built.station_hour[built.station_hour[column] > 0]
        return {
            (station, hour.tz_convert("UTC")): int(n)
            for station, hour, n in zip(rows["station_id"], rows["ts_hour"], rows[column])
        }

    assert from_trips("start_station_id", "started_at") == from_table("departures")
    assert from_trips("end_station_id", "ended_at") == from_table("arrivals")
    both = trips[
        periodo.contains(trips["started_at"]) & trips["start_station_id"].notna() & trips["end_station_id"].notna()
    ]
    pairs = {key: int(n) for key, n in both.groupby(["start_station_id", "end_station_id"]).size().items()}
    assert pairs == built.route_trips()
    in_file = set(trips["start_station_id"].dropna()) | set(trips["end_station_id"].dropna())
    assert in_file <= set(built.stations["station_id"])


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
    built = read_built(out_dir, report)
    assert_trips_reproduce_tables(built, inicio, fin)  # en TODAS las pruebas que corren la limpieza
    return built


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


@pytest.mark.parametrize(
    "local, utc",
    [
        ("2026-07-15 08:20:16.088", "2026-07-15 12:20:16.088"),
        ("2026-01-15 08:05:00", "2026-01-15 13:05:00"),
        ("2026-03-08 01:59:59.999", "2026-03-08 06:59:59.999"),
        ("2026-03-08 02:30:00", "2026-03-08 07:30:00"),          # no existe: se lee como 03:30 (-04:00)
        ("2026-03-08 03:10:00", "2026-03-08 07:10:00"),          # y queda ANTES que la 02:30 de arriba
        ("2025-11-02 00:59:59.999", "2025-11-02 04:59:59.999"),
        ("2025-11-02 01:30:00", "2025-11-02 06:30:00"),          # repetida: la segunda
        ("2025-11-02 02:00:00", "2025-11-02 07:00:00"),
    ],
)
def test_el_calculo_del_instante_en_python_puro(local, utc):
    """Lo mismo sin recortar a la hora: es la referencia de started_at y ended_at en trips.parquet."""
    expected = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)
    assert bt.local_instant(datetime.fromisoformat(local)) == expected


def test_los_microsegundos_desde_1970_son_enteros_exactos():
    assert bt.epoch_us(datetime(1970, 1, 1, 0, 0, 1, 500, tzinfo=timezone.utc)) == 1_000_500
    assert bt.epoch_us(datetime(2026, 7, 15, 12, tzinfo=timezone.utc)) == 1784116800 * 1_000_000
    assert bt.epoch_us(bt.local_instant(datetime(2026, 7, 15, 8))) == 1784116800 * 1_000_000


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
    for name in (*bt.TABLES, bt.TRIPS_NAME):
        assert pq.read_table(normal.out / f"{name}.parquet").equals(pq.read_table(other.out / f"{name}.parquet")), name
    # Con la sesión en Tokio DuckDB entrega las fechas etiquetadas con esa zona: el archivo no la hereda.
    assert pq.read_schema(other.out / "trips.parquet").equals(bt.TRIPS_SCHEMA)
    assert other.trip_rows()[0] == ("2026-07-15 12:05:00", "2026-07-15 12:15:00", "4327.09", "6720.07")


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
        assert tables[zone].trip_rows()[0] == ("2026-07-15 12:05:00", "2026-07-15 12:15:00", "4327.09", "6720.07")
        assert "trips (tabla interna, no es contrato): 2 viajes" in done.stdout
    for name in (*bt.TABLES, bt.TRIPS_NAME):
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


def test_la_capacidad_sale_del_archivo_congelado_de_gbfs(tmp_path):
    gbfs = write_gbfs(tmp_path / "gbfs_congelado.csv", {"4327.09": 30, "6720.07": 0, "9999.99": 12})
    built = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs)

    capacity = dict(zip(built.stations["station_id"], built.stations["capacity"]))
    assert capacity["4327.09"] == 30
    assert pd.isna(capacity["6720.07"])   # 0 en GBFS es "no instalada hoy": se guarda vacía
    assert pd.isna(capacity["6131.1"])    # no está en el archivo
    assert built.stations["station_type"].isna().all()
    report = built.report["capacidad"]
    assert (report["con_capacidad"], report["capacidad_cero_hoy"], report["no_estan_en_gbfs"]) == (1, 1, 1)
    assert report["lista_no_estan_en_gbfs"] == [
        {"id": "6131.1", "nombre": "E 24 St & Park Ave S", "salidas": 0, "llegadas": 1}
    ]
    photo = built.report["entradas"]["gbfs"]
    assert photo["foto"] == "gbfs_congelado.csv" and photo["estaciones_en_el_archivo"] == 3
    assert (photo["captura_desde_utc"], photo["captura_hasta_utc"]) == ("2026-10-10T04:00:00Z", "2026-10-10T17:00:00Z")
    assert photo["dias_despues_del_periodo"] == 99  # del 3 de julio al 10 de octubre
    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "archivo congelado" in text and "2026-10-10T04:00:00Z" in text


def test_la_capacidad_no_cambia_al_reconstruir_con_el_mismo_archivo_congelado(tmp_path):
    """El archivo está congelado: dos construcciones dan la misma capacidad, y los ids se leen como texto."""
    gbfs = write_gbfs(tmp_path / "gbfs_congelado.csv", {"4327.09": 30, "6720.07": 21, "6131.1": 19, "6131.10": 55})
    first = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs, out="uno")
    second = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs, out="dos")
    assert first.stations.equals(second.stations)
    capacity = dict(zip(first.stations["station_id"], first.stations["capacity"]))
    assert capacity == {"4327.09": 30, "6131.1": 19, "6720.07": 21}  # 6131.1 no toma la capacidad de 6131.10


def test_un_archivo_congelado_ilegible_detiene_la_limpieza(tmp_path):
    bad = tmp_path / "gbfs_congelado.csv"
    bad.write_text("short_name,capacity\n4327.09,30\n", encoding="utf-8")
    with pytest.raises(bt.BuildError, match="le faltan columnas"):
        run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=bad)


def test_sin_archivo_de_gbfs_la_capacidad_queda_vacia_y_el_reporte_lo_dice(tmp_path):
    built = run(tmp_path, {"202607": _capacity_trips()}, "2026-07-01", "2026-07-03", gbfs=tmp_path / "no_existe.csv")

    assert built.stations["capacity"].isna().all()
    assert built.report["entradas"]["gbfs"]["foto"] is None
    assert built.report["capacidad"]["con_capacidad"] == 0
    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "No se encontró el archivo congelado de GBFS" in text


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
# Tabla interna de viajes (trips.parquet)
# --------------------------------------------------------------------------- #

def test_trips_tiene_las_columnas_y_los_tipos_acordados(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    path = built.out / "trips.parquet"

    schema = pq.read_schema(path)
    assert schema.names == [
        "ride_id", "started_at", "ended_at", "start_station_id", "end_station_id",
        "start_lat", "start_lng", "end_lat", "end_lng",
    ]
    assert [str(kind) for kind in schema.types] == [
        "string", f"timestamp[us, tz={TZ}]", f"timestamp[us, tz={TZ}]", "string", "string",
        "double", "double", "double", "double",
    ]
    assert schema.equals(bt.TRIPS_SCHEMA)
    frame = pq.read_table(path).to_pandas()
    assert str(frame["started_at"].dtype) == str(frame["ended_at"].dtype) == f"datetime64[us, {TZ}]"
    assert len(frame) == len(_variant_trips()) and frame["ride_id"].is_unique
    # Es una tabla interna: no es uno de los seis contratos ni entra en la lista de tablas de contrato.
    assert bt.TRIPS_NAME == "trips" and bt.TRIPS_NAME not in CONTRACTS and bt.TRIPS_NAME not in bt.TABLES


def test_trips_lleva_el_id_que_queda_despues_de_unir_variantes(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")

    # Salida en UTC (en julio, cuatro horas más que la hora local) -> (inicio, fin).
    by_departure = {row[0]: row[2:] for row in built.trip_rows()}
    day = "2026-07-02"
    assert by_departure[f"{day} 13:00:00"] == ("4327.09", "5470.10")   # la llegada venía como 5470.1
    assert by_departure[f"{day} 14:00:00"] == ("6131.1", "4327.09")    # un decimal sin gemelo: intacto
    assert by_departure[f"{day} 15:00:00"] == ("5308.04", "5308.04")   # la salida venía como 5308.04_
    assert by_departure[f"{day} 16:00:00"] == ("5303.06", "5303.06_")  # a 30 metros: siguen separadas
    assert by_departure[f"{day} 17:00:00"] == ("7000.1", "7000.10")    # nombres distintos: siguen separadas
    assert by_departure[f"{day} 18:30:00"] == ("8000.1", "8000.1")     # la llegada venía como 8000.10
    assert by_departure[f"{day} 19:00:00"] == ("9000.10", "9000.10")   # la salida venía como 9000.1
    in_file = {station for row in built.trip_rows() for station in row[2:]}
    assert in_file == set(built.stations["station_id"])
    assert in_file.isdisjoint({"5470.1", "5308.04_", "8000.10", "9000.1"})  # las escrituras que desaparecieron


def test_trips_lleva_los_ids_ya_recortados(tmp_path):
    trips = [
        trip("5470.10", "4327.09 ", "2026-07-02 10:00:00"),
        {**trip("5470.10", "4327.09", "2026-07-02 12:00:00"), "end_station_id": "\t4327.09\n"},
        {**trip("4327.09", "5470.10", "2026-07-02 13:00:00"), "start_station_id": "\u00a04327.09\u00a0"},
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")
    assert [row[2:] for row in built.trip_rows()] == [
        ("5470.10", "4327.09"), ("5470.10", "4327.09"), ("4327.09", "5470.10"),
    ]


def test_trips_deja_vacia_la_punta_sin_estacion_excluida_o_sin_viajes_en_el_periodo(tmp_path):
    a, b = "4327.09", "6720.07"
    day = "2026-07-02"
    trips = [
        trip(a, "HB101", f"{day} 08:00:00", ride_id="A-NUEVA-JERSEY"),
        trip("SYS016", a, f"{day} 09:00:00", ride_id="DESDE-UN-TALLER"),
        trip("1234.56", "JC013", f"{day} 10:00:00", ride_id="DOS-EXCLUIDAS"),
        trip(None, b, f"{day} 11:00:00", ride_id="SIN-INICIO"),
        trip(a, None, f"{day} 12:00:00", ride_id="SIN-FIN"),
        trip(None, None, f"{day} 13:00:00", ride_id="SIN-NINGUNA"),
        {**trip(a, b, f"{day} 14:00:00", ride_id="ID-EN-BLANCO"), "end_station_id": "   "},
        # 6131.1 solo aparece con una llegada del día 4, fuera del periodo: no es estación de las tablas.
        trip(a, "6131.1", "2026-07-03 23:55:00", minutes=20, ride_id="A-ESTACION-SIN-VIAJES"),
        trip(a, b, f"{day} 15:00:00", ride_id="COMPLETO"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert sorted(built.stations["station_id"]) == [a, b]
    assert {row[0]: row[3:] for row in built.trip_rows_with_id()} == {
        "A-NUEVA-JERSEY": (a, None),
        "DESDE-UN-TALLER": (None, a),
        "DOS-EXCLUIDAS": (None, None),     # sigue en el archivo: es un viaje limpio que sale en el periodo
        "SIN-INICIO": (None, b),
        "SIN-FIN": (a, None),
        "SIN-NINGUNA": (None, None),
        "ID-EN-BLANCO": (a, None),
        "A-ESTACION-SIN-VIAJES": (a, None),
        "COMPLETO": (a, b),
    }
    info = built.report["trips"]
    assert info["filas"] == 9 and info["sin_ride_id"] == 0
    assert info["puntas"] == {
        "inicio": {
            "con_estacion": 5, "sin_estacion": 2, "estacion_excluida": 2, "estacion_sin_viajes_en_el_periodo": 0,
            "sin_clasificar": 0, "fuera_del_periodo": 0,
        },
        "fin": {
            "con_estacion": 3, "sin_estacion": 3, "estacion_excluida": 2, "estacion_sin_viajes_en_el_periodo": 1,
            "sin_clasificar": 0, "fuera_del_periodo": 1,
        },
    }
    assert info["sin_estacion_en_ninguna_punta"] == 2
    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "| Una estación de las tablas | 5 | 3 |" in text
    assert "| Vacía: el viaje no trae estación | 2 | 3 |" in text
    assert "| Vacía: estación excluida | 2 | 2 |" in text
    assert "| Vacía: estación sin viajes en el periodo | 0 | 1 |" in text
    assert "| Total (filas del archivo) | 9 | 9 |" in text
    assert "| De ese total, con la fecha fuera del periodo | 0 | 1 |" in text
    assert "Viajes del archivo sin estación en ninguna de las dos puntas: 2." in text


def test_trips_conserva_la_punta_que_cae_fuera_del_periodo(tmp_path):
    """Un viaje que empieza antes del periodo y termina dentro está, con sus dos fechas y sus dos estaciones."""
    a, b = "4327.09", "6720.07"
    trips = [
        trip(a, b, "2026-06-30 23:50:00", minutes=20, ride_id="EMPIEZA-ANTES"),
        trip(a, b, "2026-06-30 23:50:00", minutes=10, ride_id="LLEGA-EN-EL-PRIMER-INSTANTE"),   # 00:00:00 del día 1
        trip(a, b, "2026-06-30 23:00:00", minutes=10, ride_id="TODO-ANTES"),
        trip(a, b, "2026-07-02 12:00:00", ride_id="DENTRO"),
        trip(a, b, "2026-07-03 23:50:00", minutes=10, ride_id="LLEGA-EN-EL-PRIMER-INSTANTE-DE-FUERA"),
        trip(a, b, "2026-07-03 23:50:00", minutes=20, ride_id="TERMINA-DESPUES"),
        trip(a, b, "2026-07-04 00:00:00", minutes=5, ride_id="TODO-DESPUES"),
    ]
    built = run(tmp_path, {"202607": trips}, "2026-07-01", "2026-07-03")

    assert built.trip_rows_with_id() == [
        ("LLEGA-EN-EL-PRIMER-INSTANTE", "2026-07-01 03:50:00", "2026-07-01 04:00:00", a, b),
        ("EMPIEZA-ANTES", "2026-07-01 03:50:00", "2026-07-01 04:10:00", a, b),
        ("DENTRO", "2026-07-02 16:00:00", "2026-07-02 16:10:00", a, b),
        ("LLEGA-EN-EL-PRIMER-INSTANTE-DE-FUERA", "2026-07-04 03:50:00", "2026-07-04 04:00:00", a, b),
        ("TERMINA-DESPUES", "2026-07-04 03:50:00", "2026-07-04 04:10:00", a, b),
    ]
    # En las tablas cada punta cuenta solo si su fecha cae en el periodo: 3 salidas y 3 llegadas.
    assert built.totals(a) == (3, 0) and built.totals(b) == (0, 3)
    assert built.route_trips() == {(a, b): 3}
    info = built.report["trips"]
    assert info["filas"] == 5
    assert info["puntas"]["inicio"]["fuera_del_periodo"] == 2 and info["puntas"]["fin"]["fuera_del_periodo"] == 2
    assert info["puntas"]["inicio"]["con_estacion"] == info["puntas"]["fin"]["con_estacion"] == 5
    assert info["primera_salida"] == "2026-06-30T23:50:00-04:00"
    assert info["ultima_salida"] == "2026-07-03T23:50:00-04:00"


@pytest.mark.parametrize(
    "local_start, local_end, utc_start, utc_end, offset",
    [
        # Verano: Nueva York va 4 horas detrás de UTC. Invierno: 5 horas.
        ("2026-07-15 08:05:30.250", "2026-07-15 08:47:10", "2026-07-15 12:05:30.250", "2026-07-15 12:47:10", "-04:00"),
        ("2026-01-15 08:05:30.250", "2026-01-15 08:47:10", "2026-01-15 13:05:30.250", "2026-01-15 13:47:10", "-05:00"),
    ],
)
def test_trips_guarda_la_salida_y_la_llegada_en_hora_de_nueva_york(
    tmp_path, local_start, local_end, utc_start, utc_end, offset
):
    day = date.fromisoformat(local_start[:10])
    one_trip = trip("4327.09", "6720.07", local_start, ended_at=local_end, ride_id="UNICO")
    built = run(
        tmp_path, {day.strftime("%Y%m"): [one_trip]}, str(day - timedelta(days=1)), str(day + timedelta(days=1))
    )

    assert built.trip_rows() == [(utc_start, utc_end, "4327.09", "6720.07")]
    table = pq.read_table(built.out / "trips.parquet")
    for column, local in (("started_at", local_start), ("ended_at", local_end)):
        assert table.schema.field(column).type == pa.timestamp("us", tz=TZ)
        stamp = table[column][0].as_py()
        assert stamp.replace(tzinfo=None) == datetime.fromisoformat(local)  # el mismo reloj que el dato crudo
        assert stamp.isoformat()[-6:] == offset
    assert built.report["trips"]["zona_horaria"] == [
        {
            "ride_id": "UNICO",
            "started_at": local_start,
            "ended_at": local_end + ".000",
            "inicio_utc": utc_start.replace(" ", "T") + "000+00:00",
            "fin_utc": utc_end.replace(" ", "T") + "+00:00",
            "inicio_en_el_archivo_utc": utc_start.replace(" ", "T") + "000+00:00",
            "fin_en_el_archivo_utc": utc_end.replace(" ", "T") + "+00:00",
            "coincide": True,
        }
    ]


def test_trips_en_el_dia_de_23_horas_se_ordena_por_instante(tmp_path):
    """8 de marzo de 2026: la 02:30 no existe y se lee como 03:30, así que queda DESPUÉS de la 03:10."""
    trips = [
        trip("4327.09", None, "2026-03-08 01:30:00", ride_id="UNA-Y-MEDIA"),
        trip("4327.09", None, "2026-03-08 02:30:00", ride_id="HORA-QUE-NO-EXISTE"),
        trip("4327.09", None, "2026-03-08 03:10:00", ride_id="TRES-Y-DIEZ"),
        some_route("2026-03-08"),
    ]
    built = run(tmp_path, {"202603": trips}, "2026-03-07", "2026-03-09")

    assert [row[:3] for row in built.trip_rows_with_id()][:3] == [
        ("UNA-Y-MEDIA", "2026-03-08 06:30:00", "2026-03-08 06:40:00"),
        ("TRES-Y-DIEZ", "2026-03-08 07:10:00", "2026-03-08 07:20:00"),
        ("HORA-QUE-NO-EXISTE", "2026-03-08 07:30:00", "2026-03-08 07:40:00"),
    ]
    assert built.hours("4327.09") == [("2026-03-08 06:00", 1, 0), ("2026-03-08 07:00", 2, 0)]


def test_trips_en_el_dia_de_25_horas_la_hora_repetida_es_la_segunda(tmp_path):
    """2 de noviembre de 2025: toda la 01:xx se lee como la segunda (-05:00), igual que en station_hour.

    Un viaje que sale a las 00:50 y llega a la 01:10 queda con 80 minutos y no con 20: el dato no dice
    cuál de las dos 01:10 es.
    """
    trips = [
        trip("4327.09", "6720.07", "2025-11-02 00:50:00", minutes=20, ride_id="CRUZA-A-LA-HORA-REPETIDA"),
        trip("4327.09", "6720.07", "2025-11-02 01:30:00", ride_id="EN-LA-HORA-REPETIDA"),
        trip("4327.09", "6720.07", "2025-11-02 02:30:00", ride_id="DESPUES"),
    ]
    built = run(tmp_path, {"202511": trips}, "2025-11-01", "2025-11-03")

    assert [row[:3] for row in built.trip_rows_with_id()] == [
        ("CRUZA-A-LA-HORA-REPETIDA", "2025-11-02 04:50:00", "2025-11-02 06:10:00"),
        ("EN-LA-HORA-REPETIDA", "2025-11-02 06:30:00", "2025-11-02 06:40:00"),
        ("DESPUES", "2025-11-02 07:30:00", "2025-11-02 07:40:00"),
    ]


def _unordered_months() -> dict:
    a, b = "4327.09", "6720.07"
    return {
        "202607": [
            trip(a, b, "2026-07-31 23:55:00", minutes=10, ride_id="CRUZA-EL-MES"),
            trip(b, a, "2026-07-02 08:00:00", ride_id="EMPATE-2"),
            trip(a, b, "2026-07-02 08:00:00", ride_id="EMPATE-1"),   # misma salida y llegada: decide el ride_id
            trip(a, b, "2026-07-02 08:00:00", minutes=5, ride_id="MISMA-SALIDA-LLEGA-ANTES"),
            trip(b, a, "2026-07-01 00:00:00", ride_id="PRIMERO"),
        ],
        "202608": [
            trip(a, b, "2026-08-01 07:00:00", ride_id="AGOSTO"),
            trip(a, b, "2026-07-15 07:00:00", ride_id="DE-JULIO-EN-EL-ARCHIVO-DE-AGOSTO"),
        ],
    }


def test_trips_va_ordenado_por_la_salida_sin_importar_el_orden_de_entrada(tmp_path):
    built = run(tmp_path, _unordered_months(), "2026-07-01", "2026-08-02", out="uno")

    assert [row[0] for row in built.trip_rows_with_id()] == [
        "PRIMERO", "MISMA-SALIDA-LLEGA-ANTES", "EMPATE-1", "EMPATE-2", "DE-JULIO-EN-EL-ARCHIVO-DE-AGOSTO",
        "CRUZA-EL-MES", "AGOSTO",
    ]
    frame = pq.read_table(built.out / "trips.parquet").to_pandas()
    assert frame["started_at"].is_monotonic_increasing

    # Con las filas de entrada al revés sale exactamente el mismo archivo.
    reversed_months = {month: list(reversed(trips)) for month, trips in _unordered_months().items()}
    other = run(tmp_path, reversed_months, "2026-07-01", "2026-08-02", out="dos")
    assert pq.read_table(built.out / "trips.parquet").equals(pq.read_table(other.out / "trips.parquet"))

    # Como va ordenado, se pueden leer unos días con un filtro sobre started_at (así lo lee el simulador).
    august = pq.read_table(
        built.out / "trips.parquet", filters=[("started_at", ">=", pd.Timestamp("2026-08-01", tz=TZ))]
    )
    assert august["ride_id"].to_pylist() == ["AGOSTO"]


def test_trips_en_varios_grupos_del_parquet_sigue_ordenado_y_completo(tmp_path, monkeypatch):
    """Con los datos reales el archivo se escribe por partes: aquí se fuerzan partes de dos filas."""
    monkeypatch.setattr(bt, "TRIPS_ROWS_PER_GROUP", 2)
    built = run(tmp_path, _unordered_months(), "2026-07-01", "2026-08-02")

    file = pq.ParquetFile(built.out / "trips.parquet")
    assert file.metadata.num_rows == 7 and file.metadata.num_row_groups >= 4
    assert [row[0] for row in built.trip_rows_with_id()][0] == "PRIMERO"
    assert pq.read_table(built.out / "trips.parquet").to_pandas()["started_at"].is_monotonic_increasing
    assert all(c["cumple"] for c in built.report["comprobaciones"])


def test_un_viaje_repetido_en_dos_meses_aparece_una_vez_en_trips(tmp_path):
    repeated = trip("4327.09", "6720.07", "2026-07-31 23:50:00", minutes=20, ride_id="REPETIDO")
    months = {"202607": [repeated], "202608": [dict(repeated), trip("6720.07", "4327.09", "2026-08-01 09:00:00")]}
    built = run(tmp_path, months, "2026-07-30", "2026-08-02")
    assert [row[0] for row in built.trip_rows_with_id()].count("REPETIDO") == 1
    assert built.report["trips"]["filas"] == 2


def test_un_viaje_sin_ride_id_sigue_en_trips_y_el_script_lo_avisa(tmp_path, capsys):
    """Sin ride_id el viaje no se puede comparar con otros: se conserva, se cuenta y se avisa al terminar."""
    raw = tmp_path / "crudos"
    trips = [
        {**trip("4327.09", "6720.07", "2026-07-02 08:00:00"), "ride_id": None},
        {**trip("4327.09", "6720.07", "2026-07-02 08:00:00"), "ride_id": None},  # igual al anterior: son dos viajes
        trip("6720.07", "4327.09", "2026-07-02 09:00:00", ride_id="CON-ID"),
    ]
    write_raw(raw, "202607", trips)
    config_path = write_config(tmp_path, ["202607"], "2026-07-01", "2026-07-03")
    out = tmp_path / "salida"

    code = bt.main(
        ["--salida", str(out), "--crudos", str(raw), "--gbfs", str(tmp_path / "sin_gbfs"), "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    assert code == 0 and "Comprobaciones: 30 de 30 cumplen." in captured.out
    assert "AVISO: 2 viajes de trips.parquet no traen ride_id" in captured.err
    built = read_built(out)
    assert [row[0] for row in built.trip_rows_with_id()] == [None, None, "CON-ID"]
    assert built.report["trips"]["sin_ride_id"] == 2
    assert built.totals("4327.09") == (2, 1)
    assert_trips_reproduce_tables(built, "2026-07-01", "2026-07-03")


def test_la_limpieza_hace_sus_17_comprobaciones_y_las_de_trips(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")

    checks = built.report["comprobaciones"]
    names = [c["nombre"] for c in checks]
    assert len(checks) == TABLE_CHECKS + TRIPS_CHECKS == 30
    assert all(c["cumple"] for c in checks)
    assert len(set(names)) == len(names)
    assert not any(name.startswith("trips: ") for name in names[:TABLE_CHECKS])
    assert names[TABLE_CHECKS:] == built.trips_check_names() == [
        "trips: columnas con nombre o tipo distinto al acordado",
        "trips: filas del archivo = viajes conservados con la salida o la llegada dentro del periodo",
        "trips: filas con la salida dentro del periodo = viajes conservados que empiezan en el periodo",
        "trips: filas con la llegada dentro del periodo = viajes conservados que terminan en el periodo",
        "trips: filas con estación de inicio = conteo aparte sobre los viajes",
        "trips: filas con estación de fin = conteo aparte sobre los viajes",
        "trips: filas con un ride_id que ya salió en otra fila",
        "trips: ids de estación que no están en stations",
        "trips: horas de estación donde las salidas del archivo no dan station_hour.departures",
        "trips: horas de estación donde las llegadas del archivo no dan station_hour.arrivals",
        "trips: pares de estaciones donde el archivo no da routes.trips",
        "trips: filas fuera de orden por started_at",
        "trips: viajes de la revisión de zona horaria cuya salida o llegada no cae en el instante calculado aparte",
    ]
    found = {c["nombre"]: (c["esperado"], c["obtenido"]) for c in checks[TABLE_CHECKS:]}
    assert found[names[TABLE_CHECKS + 1]] == (12, 12)   # los 12 viajes de _variant_trips
    assert found[names[TABLE_CHECKS + 4]] == (12, 12) and found[names[TABLE_CHECKS + 5]] == (12, 12)
    # Cada comprobación sale como una fila de la tabla del reporte para personas.
    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    for check in checks:
        assert f"| {check['nombre']} | {check['esperado']:,} | {check['obtenido']:,} | sí |" in text, check["nombre"]


def test_el_reporte_describe_trips_y_dice_que_no_es_un_contrato(tmp_path):
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")

    info = built.report["trips"]
    assert info["es_contrato"] is False
    assert Path(info["archivo"]).name == "trips.parquet"
    assert info["bytes"] == (built.out / "trips.parquet").stat().st_size > 0
    assert info["columnas"] == {
        "ride_id": "string",
        "started_at": f"timestamp[us, tz={TZ}]",
        "ended_at": f"timestamp[us, tz={TZ}]",
        "start_station_id": "string",
        "end_station_id": "string",
        "start_lat": "double",
        "start_lng": "double",
        "end_lat": "double",
        "end_lng": "double",
    }
    assert (info["primera_salida"], info["ultima_salida"]) == ("2026-07-02T08:00:00-04:00", "2026-07-02T15:00:00-04:00")
    assert "trips" not in built.report["tablas"]

    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    assert "### Tabla interna de viajes (`trips`)" in text
    assert "**no es uno de los seis\ncontratos**" in text
    assert "12 viajes" in text and "ordenados por la" in text
    assert "da exactamente `station_hour`" in text and "da exactamente `routes`" in text
    assert "tampoco dice nada de la demanda que no se pudo" in text  # decisión 9
    assert "recuperad" not in text.lower()
    summary = bt.format_summary(built.report)
    assert "trips (tabla interna, no es contrato): 12 viajes" in summary
    assert "Comprobaciones: 30 de 30 cumplen." in summary


def _alter_trips(monkeypatch, change) -> None:
    """Sabotaje: la limpieza escribe trips.parquet y después `change` (tabla -> tabla) lo altera."""
    original = bt._write_trips

    def write_then_alter(con, path):
        original(con, path)
        pq.write_table(change(pq.read_table(path)), path)
        return path.stat().st_size

    monkeypatch.setattr(bt, "_write_trips", write_then_alter)


def _with_column(table: pa.Table, column: str, change) -> pa.Table:
    """La tabla con una columna cambiada: `change` recibe la lista de valores y devuelve la nueva."""
    kind = table.schema.field(column).type
    values = pa.array(change(table[column].to_pylist()), kind)
    return table.set_column(table.schema.get_field_index(column), column, values)


def _one_hour_late(values: list) -> list:
    return [value + timedelta(hours=1) for value in values]


def _in_utc(table: pa.Table) -> pa.Table:
    """La misma tabla con las fechas etiquetadas en UTC: mismos instantes, otro tipo de columna."""
    in_utc = pa.timestamp("us", tz="UTC")
    fields = [pa.field(f.name, in_utc) if pa.types.is_timestamp(f.type) else f for f in table.schema]
    return table.cast(pa.schema(fields))


SABOTAGES = [
    # (qué se le hace al archivo ya escrito, comprobación que debe detenerlo)
    (lambda t: t.slice(0, t.num_rows - 1),
     "trips: filas del archivo = viajes conservados con la salida o la llegada dentro del periodo: "
     "se esperaba 12 y se obtuvo 11"),
    (lambda t: t.take(pa.array(range(t.num_rows - 1, -1, -1))),
     "trips: filas fuera de orden por started_at: se esperaba 0 y se obtuvo 11"),
    (lambda t: _with_column(t, "started_at", _one_hour_late),
     "trips: horas de estación donde las salidas del archivo no dan station_hour.departures"),
    (lambda t: _with_column(t, "ended_at", _one_hour_late),
     "trips: horas de estación donde las llegadas del archivo no dan station_hour.arrivals"),
    # La escritura 5470.1 vuelve a aparecer: es un id que no existe en stations.
    (lambda t: _with_column(t, "end_station_id", lambda ids: ["5470.1" if i == "5470.10" else i for i in ids]),
     "trips: ids de estación que no están en stations: se esperaba 0 y se obtuvo 1"),
    (lambda t: _with_column(t, "end_station_id", lambda ids: [None, *ids[1:]]),
     "trips: pares de estaciones donde el archivo no da routes.trips"),
    (lambda t: _with_column(t, "start_station_id", lambda ids: [None, *ids[1:]]),
     "trips: filas con estación de inicio = conteo aparte sobre los viajes: se esperaba 12 y se obtuvo 11"),
    # Un ride_id copiado de otra fila: el simulador lo usa para desempatar y no puede venir repetido.
    (lambda t: _with_column(t, "ride_id", lambda ids: [*ids[:-1], ids[-2]]),
     "trips: filas con un ride_id que ya salió en otra fila: se esperaba 0 y se obtuvo 1"),
    (_in_utc, "trips.parquet no quedó con las columnas acordadas"),
    (lambda t: t.rename_columns(["ride_id", "started_at", "ended_at", "start_station", "end_station", *t.column_names[5:]]),
     "trips.parquet no quedó con las columnas acordadas"),
    (lambda t: t.drop_columns(["ride_id"]), "trips.parquet no quedó con las columnas acordadas"),
]


@pytest.mark.parametrize("change, message", SABOTAGES, ids=[message[:60] for _, message in SABOTAGES])
def test_si_trips_no_cuadra_con_las_tablas_no_se_escribe_nada(tmp_path, monkeypatch, change, message):
    """Cada comprobación de trips.parquet detecta su propio daño, hecho sobre el archivo ya escrito."""
    good = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    before = {p.name: p.read_bytes() for p in good.out.rglob("*") if p.is_file()}
    assert "trips.parquet" in before

    _alter_trips(monkeypatch, change)
    with pytest.raises(bt.BuildError, match=re.escape(message)):
        run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03")
    with pytest.raises(bt.BuildError, match="No se escribió nada"):
        run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", out="otra")

    # Lo que ya había sigue igual (tablas, trips y reporte), y en una carpeta nueva no queda nada.
    assert {p.name: p.read_bytes() for p in good.out.rglob("*") if p.is_file()} == before
    assert not [p for p in (tmp_path / "otra").rglob("*") if p.is_file()]


def test_las_comprobaciones_de_trips_detectan_un_cast_a_secas(tmp_path, monkeypatch):
    """El mismo error clásico de zona horaria, ahora solo en started_at y ended_at de trips.parquet."""
    original = bt._connect

    def connect_in_monterrey(work):
        connection = original(work)
        connection.execute("SET TimeZone = 'America/Monterrey'")
        return connection

    monkeypatch.setattr(bt, "_connect", connect_in_monterrey)
    monkeypatch.setattr(bt, "instant_sql", lambda column: f"CAST({column} AS TIMESTAMPTZ)")
    with pytest.raises(bt.BuildError) as error:
        run(tmp_path, _timezone_trips(), "2026-07-14", "2026-07-16")

    message = str(error.value)
    assert "trips: horas de estación donde las salidas del archivo no dan station_hour.departures" in message
    assert "trips: horas de estación donde las llegadas del archivo no dan station_hour.arrivals" in message
    assert (
        "trips: viajes de la revisión de zona horaria cuya salida o llegada no cae en el instante calculado "
        "aparte: se esperaba 0 y se obtuvo 1"
    ) in message
    assert "zona horaria: el viaje" not in message  # station_hour quedó bien: lo dañado es solo el archivo de viajes
    assert not list((tmp_path / "salida").rglob("*.parquet"))


def test_trips_conserva_las_coordenadas_crudas_tambien_en_una_punta_sin_estacion(tmp_path):
    """El simulador necesita saber cerca de qué estación terminó un viaje sin estación de fin o excluida."""
    loose = {**trip("4327.09", None, "2026-07-02 08:00:00", ride_id="SUELTO"), "end_lat": 40.7005, "end_lng": -73.9905}
    excluded = {**trip("4327.09", "6720.07", "2026-07-02 09:00:00", ride_id="A_NJ"),
                "end_station_id": "JC001", "end_station_name": "Grove St", "end_lat": 40.7195, "end_lng": -74.0431}
    no_point = trip("4327.09", None, "2026-07-02 10:00:00", ride_id="SIN_PUNTO")
    normal = trip("4327.09", "6720.07", "2026-07-02 11:00:00", ride_id="NORMAL")
    built = run(tmp_path, {"202607": [loose, excluded, no_point, normal]}, "2026-07-01", "2026-07-03")

    frame = pq.read_table(built.out / "trips.parquet").to_pandas().set_index("ride_id")
    assert pd.isna(frame.loc["SUELTO", "end_station_id"])
    assert (frame.loc["SUELTO", "end_lat"], frame.loc["SUELTO", "end_lng"]) == (40.7005, -73.9905)
    assert pd.isna(frame.loc["A_NJ", "end_station_id"])   # excluida: sin id, pero con su coordenada
    assert (frame.loc["A_NJ", "end_lat"], frame.loc["A_NJ", "end_lng"]) == (40.7195, -74.0431)
    assert pd.isna(frame.loc["SIN_PUNTO", "end_lat"]) and pd.isna(frame.loc["SIN_PUNTO", "end_lng"])
    start = CATALOG["4327.09"]
    assert (frame.loc["NORMAL", "start_lat"], frame.loc["NORMAL", "start_lng"]) == (start[1], start[2])
    end = CATALOG["6720.07"]
    assert (frame.loc["NORMAL", "end_lat"], frame.loc["NORMAL", "end_lng"]) == (end[1], end[2])


# --------------------------------------------------------------------------- #
# Contratos, reporte y fallos
# --------------------------------------------------------------------------- #

def test_las_tres_salidas_cumplen_su_contrato_y_el_reporte_existe(tmp_path):
    gbfs = write_gbfs(tmp_path / "gbfs_congelado.csv", {"4327.09": 30})
    built = run(tmp_path, {"202607": _variant_trips()}, "2026-07-01", "2026-07-03", gbfs=gbfs)

    for name in bt.TABLES:
        assert validate_file(built.out / f"{name}.parquet", name) == [], name
        assert pq.read_schema(built.out / f"{name}.parquet").equals(CONTRACTS[name].schema), name
    written = sorted(p.name for p in built.out.iterdir())
    assert written == ["reports", "routes.parquet", "station_hour.parquet", "stations.parquet", "trips.parquet"]

    on_disk = json.loads((built.out / "reports" / "limpieza.json").read_text(encoding="utf-8"))
    assert on_disk == built.report
    for key in (
        "generado_utc", "config", "entradas", "filas", "texto", "puntas", "union_de_ids", "estaciones_excluidas",
        "capacidad", "tablas", "station_hour", "routes", "trips", "zona_horaria", "comprobaciones",
    ):
        assert key in on_disk, key
    assert sorted(on_disk["tablas"]) == sorted(bt.TABLES)  # trips no es contrato: va aparte
    assert on_disk["config"]["periodo"] == {"inicio": "2026-07-01", "fin": "2026-07-03"}
    assert on_disk["tablas"]["station_hour"]["filas"] == len(built.station_hour)
    assert on_disk["comprobaciones"] and all(c["cumple"] for c in on_disk["comprobaciones"])
    assert on_disk["zona_horaria"] and all(t["coincide"] for t in on_disk["zona_horaria"])

    text = (built.out / "reports" / "limpieza.md").read_text(encoding="utf-8")
    for expected in ("demanda atendida", "La capacidad es la de la captura", "rachas de 72 horas", "Salidas sin llegada"):
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
    # trips.parquet: los ride_id salen de un contador, así que se compara todo lo demás, en orden.
    assert first.trip_rows() == second.trip_rows()


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


def test_el_script_avisa_si_no_existe_el_archivo_congelado_de_gbfs(tmp_path, capsys):
    """La corrida termina bien, pero stations sale sin capacidad: no puede pasar en silencio."""
    code, out, err = _run_main(tmp_path, capsys, "2026-07-03", tmp_path / "no_existe.csv")
    assert code == 0 and "Comprobaciones:" in out and "(0 con capacidad)" in out
    assert "trips (tabla interna, no es contrato): 1 viajes" in out
    assert "AVISO: no existe el archivo congelado de GBFS " + str(tmp_path / "no_existe.csv") in err
    assert "TODAS las estaciones" in err and "python scripts/freeze_gbfs.py" in err

    gbfs = write_gbfs(tmp_path / "gbfs_congelado.csv", {"4327.09": 30})
    code, out, err = _run_main(tmp_path, capsys, "2026-07-03", gbfs)
    assert code == 0 and "(1 con capacidad)" in out
    assert "AVISO" not in err


def test_el_script_avisa_si_el_periodo_llega_al_ultimo_dia_del_ultimo_mes(tmp_path, capsys):
    """Los archivos traen los viajes que TERMINAN en cada mes: sin el mes siguiente faltan salidas del último día."""
    gbfs = write_gbfs(tmp_path / "gbfs_congelado.csv", {"4327.09": 30})
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
    assert "trips.parquet" in text and "no es un contrato" in text


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


@pytest.mark.parametrize("ride_id, month, started_at, station, local_hour, epoch", KNOWN_TRIPS)
def test_trips_real_trae_el_viaje_conocido_y_cuadra_con_station_hour(
    ride_id, month, started_at, station, local_hour, epoch
):
    """El viaje real conocido está en trips.parquet con su instante exacto, y su hora da las salidas de station_hour.

    Solo mira una hora de julio y una de agosto (tramo de entrenamiento).
    """
    path = DATA_PROCESSED / "trips.parquet"
    processed = DATA_PROCESSED / "station_hour.parquet"
    report_path = DATA_PROCESSED / "reports" / "limpieza.json"
    missing = [str(p.relative_to(REPO)) for p in (path, processed, report_path) if not p.exists()]
    if missing:
        pytest.skip(f"Faltan archivos reales ({', '.join(missing)}). Se generan con: python scripts/build_tables.py")
    if month not in load_datos().meses_crudos:
        pytest.skip(f"config/datos.yaml ya no usa el mes {month}: este viaje conocido no aplica.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "trips" not in report:
        pytest.skip(
            "reports/limpieza.json es de una corrida anterior a trips.parquet. Corre: python scripts/build_tables.py"
        )

    assert pq.read_schema(path).equals(bt.TRIPS_SCHEMA)
    assert pq.ParquetFile(path).metadata.num_rows == report["trips"]["filas"]

    # Solo se leen los viajes que salen en esa hora: el archivo va ordenado por started_at.
    hour = pd.Timestamp(local_hour, tz=TZ)
    assert int(hour.timestamp()) == epoch
    in_hour = [("started_at", ">=", hour), ("started_at", "<", hour + pd.Timedelta(hours=1))]
    trips = pq.read_table(path, filters=in_hour).to_pandas()
    known = trips[trips["ride_id"] == ride_id]
    assert len(known) == 1, f"el viaje {ride_id} no está en trips.parquet"
    assert known["started_at"].iloc[0] == pd.Timestamp(started_at, tz=TZ)
    assert known["start_station_id"].iloc[0] == station
    assert (known["ended_at"] >= known["started_at"]).all()

    table = pq.read_table(processed, filters=[("station_id", "==", station)]).to_pandas()
    row = table[table["ts_hour"] == hour]
    assert len(row) == 1
    assert int((trips["start_station_id"] == station).sum()) == int(row["departures"].iloc[0])
