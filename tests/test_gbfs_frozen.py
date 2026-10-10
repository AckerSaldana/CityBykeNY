"""Pruebas del archivo congelado de GBFS (src/data/gbfs_frozen.py y scripts/freeze_gbfs.py).

Con una captura inventada, escrita aquí con el mismo formato que deja
scripts/capture_gbfs.py:

a) `capacity` es la capacidad más frecuente, pesando cada estado de
   station_information por las fotos de station_status tomadas mientras estuvo
   vigente; en un empate queda la más grande.
b) `ocupacion_inicial` es el promedio de bicis disponibles entre esa capacidad
   en las fotos de la hora local en que empieza la simulación (00:00 a 00:59
   de Nueva York), entre 0 y 1, solo con la estación instalada.
c) El archivo se lee igual que se escribió, con los ids como texto, y su
   encabezado dice de qué fechas es la captura.
d) El script no pisa un archivo que ya existe, salvo con --reemplazar: está
   congelado para que reconstruir no cambie los resultados.
e) El archivo del repositorio existe, es válido y es de octubre de 2026.

Todo corre sin red y sin tocar data/raw.
"""

from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.gbfs import INFO_SCHEMA, STATUS_SCHEMA
from src.data.gbfs_frozen import COLUMNS, FROZEN_FILE, START_HOUR, GbfsFrozenError, freeze, load_frozen, main, render


def utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def write_info(folder, when: str, stations: dict) -> None:
    """Un estado de station_information: {id de GBFS: (short_name, capacidad)}."""
    taken = utc(when)
    columns = {field.name: [None] * len(stations) for field in INFO_SCHEMA}
    columns["snapshot_ts"] = columns["feed_last_updated"] = [taken] * len(stations)
    columns["station_id"] = list(stations)
    columns["short_name"] = [short_name for short_name, _ in stations.values()]
    columns["capacity"] = [capacity for _, capacity in stations.values()]
    path = folder / "station_information" / f"info_{taken:%Y%m%dT%H%M%SZ}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=INFO_SCHEMA), path)


def write_status(folder, when: str, bikes: dict, not_installed: tuple = ()) -> None:
    """Una foto de station_status: {id de GBFS: bicis disponibles}. El feed es de un minuto antes."""
    taken = utc(when)
    columns = {field.name: [None] * len(bikes) for field in STATUS_SCHEMA}
    columns["snapshot_ts"] = [taken] * len(bikes)
    columns["feed_last_updated"] = [taken - pd.Timedelta(minutes=1)] * len(bikes)
    columns["station_id"] = list(bikes)
    columns["num_bikes_available"] = list(bikes.values())
    columns["is_installed"] = [station not in not_installed for station in bikes]
    path = folder / "station_status" / f"date={taken:%Y-%m-%d}" / f"status_{taken:%Y%m%dT%H%M%SZ}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=STATUS_SCHEMA), path)


@pytest.fixture()
def capture(tmp_path):
    """Captura del 10 de octubre de 2026. Nueva York va 4 horas detrás de UTC: las 04:xx UTC son las 00:xx.

    Estados de información (id de GBFS -> short_name, capacidad):
      04:00  g1 -> "100.1" 20   g2 -> "100.10" 10   g3 -> "200.01" 30   g4 -> "300.01" 0   g5 -> " 400.01 " 12
      05:30  igual, pero "100.1" pasa a 24 y "200.01" a 18
      09:00  igual que el de 05:30, pero "200.01" pasa a 40
    Fotos de estado: 04:05, 04:20, 04:50 (estado 1); 05:40, 06:00, 07:00 (estado 2); 09:10 (estado 3).
      "100.1":   20 x3, 24 x4          -> 24
      "200.01":  30 x3, 18 x3, 40 x1   -> empate 3 a 3: la más grande, 30
    Bicis disponibles en las fotos de las 00:xx locales (04:05, 04:20, 04:50):
      g1: 12, 12, 18 -> media 14 entre 24 lugares = 0.583333
      g2: 10, 12, 14 -> media 12 entre 10 = 1.2 -> se recorta a 1
      g3: 0, 0, 0    -> 0
      g4: capacidad 0 -> sin ocupación
      g5: instalada solo en la primera foto (3 bicis de 12 = 0.25)
    """
    folder = tmp_path / "gbfs"
    base = {"g1": ("100.1", 20), "g2": ("100.10", 10), "g3": ("200.01", 30), "g4": ("300.01", 0), "g5": (" 400.01 ", 12)}
    write_info(folder, "2026-10-10 04:00", base)
    write_info(folder, "2026-10-10 05:30", {**base, "g1": ("100.1", 24), "g3": ("200.01", 18)})
    write_info(folder, "2026-10-10 09:00", {**base, "g1": ("100.1", 24), "g3": ("200.01", 40)})
    write_status(folder, "2026-10-10 04:05", {"g1": 12, "g2": 10, "g3": 0, "g4": 0, "g5": 3})
    write_status(folder, "2026-10-10 04:20", {"g1": 12, "g2": 12, "g3": 0, "g4": 0, "g5": 9}, not_installed=("g5",))
    write_status(folder, "2026-10-10 04:50", {"g1": 18, "g2": 14, "g3": 0, "g4": 0, "g5": 9}, not_installed=("g5",))
    for when in ("2026-10-10 05:40", "2026-10-10 06:00", "2026-10-10 07:00", "2026-10-10 09:10"):
        write_status(folder, when, {"g1": 1, "g2": 1, "g3": 29, "g4": 0, "g5": 12})   # otras horas: no cuentan
    return folder


def test_la_capacidad_es_la_mas_frecuente_pesada_por_fotos_y_el_empate_va_a_la_mas_grande(capture):
    table = freeze(capture).table.set_index("short_name")
    assert list(freeze(capture).table.columns) == list(COLUMNS)
    assert table.index.tolist() == ["100.1", "100.10", "200.01", "300.01", "400.01"]   # ordenado y sin espacios
    assert table["capacity"].tolist() == [24, 10, 30, 0, 12]
    # Un estado de información sin ninguna foto de estado mientras estuvo vigente no pesa nada.
    write_info(capture, "2026-10-10 09:20", {"g1": ("100.1", 99), "g3": ("200.01", 99)})
    write_info(capture, "2026-10-10 09:30", {"g1": ("100.1", 24), "g3": ("200.01", 40)})
    again = freeze(capture).table.set_index("short_name")
    assert again.loc["100.1", "capacity"] == 24 and again.loc["200.01", "capacity"] == 30


def test_la_ocupacion_inicial_es_la_media_de_las_fotos_de_la_primera_hora_local(capture):
    assert START_HOUR == 0   # la simulación empieza a la medianoche de Nueva York
    frozen = freeze(capture)
    table = frozen.table.set_index("short_name")
    assert table["ocupacion_inicial"].round(6).tolist()[:3] == [round(14 / 24, 6), 1.0, 0.0]
    assert pd.isna(table.loc["300.01", "ocupacion_inicial"])           # capacidad 0
    assert table.loc["400.01", "ocupacion_inicial"] == 0.25           # solo la foto en que estaba instalada
    assert table["fotos"].tolist() == [3, 3, 3, 0, 1]
    assert frozen.meta == {
        "captura_desde_utc": "2026-10-10T04:05:00Z",
        "captura_hasta_utc": "2026-10-10T09:10:00Z",
        "fotos_de_estado": 7,
        "estados_de_informacion": 3,
        "hora_local_del_inventario": 0,
        "fotos_en_esa_hora": 3,
    }
    # Con otra hora local cambian las fotos que entran. La hora es la del feed (un minuto antes de la captura):
    # a la 01:xx caen las fotos de las 05:40 y las 06:00 UTC, con 1 bici cada una.
    other = freeze(capture, hour=1)
    assert other.meta["fotos_en_esa_hora"] == 2
    assert other.table.set_index("short_name").loc["100.1", "ocupacion_inicial"] == round(1 / 24, 6)
    # Sin fotos en esa hora no hay ocupación de nadie, pero la capacidad sí sale.
    none = freeze(capture, hour=15).table
    assert none["ocupacion_inicial"].isna().all() and none["capacity"].tolist() == [24, 10, 30, 0, 12]


def test_el_archivo_se_lee_igual_que_se_escribio_con_los_ids_como_texto(capture, tmp_path):
    frozen = freeze(capture)
    path = tmp_path / "gbfs_congelado.csv"
    text = render(frozen)
    path.write_text(text, encoding="utf-8")
    assert text.startswith("# Capacidad e inventario inicial CONGELADOS")
    assert "Captura del 2026-10-10T04:05:00Z al 2026-10-10T09:10:00Z (UTC)" in text
    assert "la captura NO es de los días que se simulan" in text
    loaded = load_frozen(path)
    assert loaded.meta == frozen.meta
    assert loaded.table["short_name"].tolist() == ["100.1", "100.10", "200.01", "300.01", "400.01"]   # 100.1 no es 100.10
    assert loaded.table["capacity"].tolist() == [24, 10, 30, 0, 12]
    pd.testing.assert_series_equal(loaded.table["ocupacion_inicial"], frozen.table["ocupacion_inicial"])
    assert loaded.table["fotos"].tolist() == [3, 3, 3, 0, 1]
    assert render(loaded) == text   # escribir lo leído da el mismo archivo


@pytest.mark.parametrize(
    "body, message",
    [
        ("short_name,capacity\nA,10\n", "le faltan columnas"),
        ("short_name,capacity,ocupacion_inicial,fotos\nA,10,0.5,3\nA,12,0.5,3\n", "sin repetidos"),
        ("short_name,capacity,ocupacion_inicial,fotos\nA,-1,0.5,3\n", "capacity debe ser un entero"),
        ("short_name,capacity,ocupacion_inicial,fotos\nA,10.5,0.5,3\n", "capacity debe ser un entero"),
        ("short_name,capacity,ocupacion_inicial,fotos\nA,10,1.5,3\n", "ocupacion_inicial debe estar entre 0 y 1"),
        ("short_name,capacity,ocupacion_inicial,fotos\nA,10,mucha,3\n", "ocupacion_inicial debe estar entre 0 y 1"),
    ],
)
def test_un_archivo_congelado_mal_escrito_se_rechaza(tmp_path, body, message):
    path = tmp_path / "malo.csv"
    path.write_text("# captura_desde_utc: 2026-10-10T04:05:00Z\n" + body, encoding="utf-8")
    with pytest.raises(GbfsFrozenError, match=message):
        load_frozen(path)
    with pytest.raises(GbfsFrozenError, match="python scripts/freeze_gbfs.py"):
        load_frozen(tmp_path / "no_existe.csv")


def test_sin_fotos_no_se_puede_congelar(tmp_path):
    with pytest.raises(GbfsFrozenError, match="station_information"):
        freeze(tmp_path / "vacia")
    write_info(tmp_path / "solo_info", "2026-10-10 04:00", {"g1": ("100.1", 20)})
    with pytest.raises(GbfsFrozenError, match="station_status"):
        freeze(tmp_path / "solo_info")


def test_el_script_no_pisa_un_archivo_que_ya_existe_salvo_que_se_pida(capture, tmp_path, capsys):
    out = tmp_path / "config" / "gbfs_congelado.csv"
    assert main(["--gbfs", str(capture), "--salida", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "7 fotos de estado" in printed and "Estaciones: 5." in printed and "python scripts/build_tables.py" in printed
    assert "La captura no es de los días que se simulan" in printed
    first = out.read_text(encoding="utf-8")
    assert load_frozen(out).table["capacity"].tolist() == [24, 10, 30, 0, 12]

    # La captura sigue creciendo: con dos fotos más, "200.01" empata a 3 entre 30, 18 y 40 y ganaría 40.
    write_status(capture, "2026-10-10 09:15", {"g1": 1, "g2": 1, "g3": 29, "g4": 0, "g5": 12})
    write_status(capture, "2026-10-10 09:20", {"g1": 1, "g2": 1, "g3": 29, "g4": 0, "g5": 12})
    assert main(["--gbfs", str(capture), "--salida", str(out)]) == 1
    refused = capsys.readouterr().out
    assert "ya existe" in refused and "--reemplazar" in refused
    assert out.read_text(encoding="utf-8") == first   # congelado: no cambió
    assert main(["--gbfs", str(capture), "--salida", str(out), "--reemplazar"]) == 0
    capsys.readouterr()
    assert load_frozen(out).table.set_index("short_name").loc["200.01", "capacity"] == 40

    assert main(["--gbfs", str(tmp_path / "no_hay"), "--salida", str(tmp_path / "otro.csv")]) == 1
    assert "ERROR." in capsys.readouterr().out and not (tmp_path / "otro.csv").exists()


def test_el_archivo_congelado_del_repositorio_es_valido_y_dice_de_cuando_es_la_captura():
    assert FROZEN_FILE.exists(), "Falta config/gbfs_congelado.csv. Se crea con: python scripts/freeze_gbfs.py"
    frozen = load_frozen(FROZEN_FILE)
    table, meta = frozen.table, frozen.meta
    assert len(table) > 2000 and table["short_name"].is_unique
    assert (table["capacity"] >= 0).all() and (table["capacity"] > 0).sum() > 2000
    occupancy = table["ocupacion_inicial"].dropna()
    assert len(occupancy) > 2000 and occupancy.between(0, 1).all()
    assert table.loc[table["capacity"] == 0, "ocupacion_inicial"].isna().all()
    # La captura es de octubre de 2026: no se traslapa con los días simulados (decisión 16).
    assert meta["captura_desde_utc"].startswith("2026-10-") and meta["captura_hasta_utc"] >= meta["captura_desde_utc"]
    assert meta["hora_local_del_inventario"] == START_HOUR and meta["fotos_en_esa_hora"] >= 1
    assert int(table["fotos"].max()) == meta["fotos_en_esa_hora"]
    assert FROZEN_FILE.read_text(encoding="utf-8").startswith("# Capacidad e inventario inicial CONGELADOS")
