"""Pruebas de los contratos de datos (los cinco archivos que conectan al equipo).

Qué revisan:

1. Los archivos de data/sample cumplen su contrato. Deben existir siempre: si
   falta uno, la prueba FALLA.
2. Los archivos de data/processed cumplen su contrato. Mientras un archivo no
   exista, su prueba se SALTA (sale como "skipped" con el motivo). En cuanto
   alguien lo genera, `pytest` lo valida solo.
3. Los archivos de una misma carpeta son coherentes entre sí, y en
   data/processed no hay datos sintéticos (ni estaciones "SIM-..." ni copias
   de data/sample).
4. El validador sí detecta cada error: tablas pequeñas hechas a mano a las que
   se les rompe una sola cosa.
5. Nadie cambió por accidente las definiciones de src/contracts.py.

Cómo leer un fallo de los puntos 1 a 3: el mensaje trae la lista de problemas
en español, uno por renglón. Córrela sola con:  pytest tests/test_contracts.py
"""

import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.contracts import (
    CONTRACTS,
    ContractError,
    read_contract,
    to_contract_table,
    validate_file,
    validate_table,
    write_contract,
)
from src.paths import DATA_PROCESSED, DATA_SAMPLE, ROOT, TZ

FOLDERS = {"sample": DATA_SAMPLE, "processed": DATA_PROCESSED}


# --------------------------------------------------------------------------- #
# 1 y 2) Archivos en disco
# --------------------------------------------------------------------------- #

def _short(path: Path) -> str:
    """Ruta relativa a la raíz del repositorio, para mensajes más cortos."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _existing_file(folder: str, name: str) -> Path:
    """Ruta del archivo. Si no existe: en sample es un fallo, en processed la prueba se salta."""
    path = FOLDERS[folder] / f"{name}.parquet"
    if path.exists():
        return path
    if folder == "sample":
        pytest.fail(
            f"Falta {_short(path)}. Los datos de muestra van en Git y deben existir siempre.\n"
            "Para regenerarlos: python scripts/make_sample_data.py",
            pytrace=False,
        )
    pytest.skip(f"{_short(path)} todavía no existe. Se validará solo en cuanto alguien lo genere.")


def _read(path: Path, columns: list[str]) -> pa.Table:
    """Lee algunas columnas de un Parquet. Si no se puede, falla con un mensaje legible."""
    try:
        present = pq.read_schema(path).names
        absent = [column for column in columns if column not in present]
        table = None if absent else pq.read_table(path, columns=columns)
    except Exception as error:  # archivo dañado o que no es Parquet
        pytest.fail(f"{_short(path)} no se pudo leer como Parquet: {error}", pytrace=False)
    if absent:
        pytest.fail(
            f"{_short(path)} no tiene las columnas {absent}. Mira la prueba de contrato de ese archivo.",
            pytrace=False,
        )
    return table


@pytest.mark.parametrize(
    "folder, name",
    [pytest.param(folder, name, id=f"{folder}/{name}") for folder in FOLDERS for name in CONTRACTS],
)
def test_el_archivo_cumple_su_contrato(folder, name):
    path = _existing_file(folder, name)
    try:
        problems = validate_file(path, name)
    except Exception as error:  # archivo dañado o que no es Parquet
        pytest.fail(f"{_short(path)} no se pudo leer como Parquet: {error}", pytrace=False)
    if problems:
        pytest.fail(
            f"{_short(path)} NO cumple el contrato '{name}'. Problemas encontrados:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + f"\nColumnas acordadas, en orden: {CONTRACTS[name].columns}\n"
            f"Cómo arreglarlo: guarda la tabla con src.contracts.write_contract(tabla, '{name}', ruta). "
            "Convierte los tipos y valida antes de escribir.",
            pytrace=False,
        )


# --------------------------------------------------------------------------- #
# 3) Coherencia entre archivos de la misma carpeta
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "folder, name",
    [
        pytest.param(folder, name, id=f"{folder}/{name}")
        for folder in FOLDERS
        for name in ("station_hour", "predictions", "recommendations")
    ],
)
def test_toda_estacion_usada_existe_en_stations(folder, name):
    stations_path = _existing_file(folder, "stations")
    path = _existing_file(folder, name)
    known = set(_read(stations_path, ["station_id"])["station_id"].unique().to_pylist())
    used = set(_read(path, ["station_id"])["station_id"].unique().to_pylist()) - {None}
    unknown = sorted(used - known)
    if unknown:
        pytest.fail(
            f"{_short(path)} usa {len(unknown)} station_id que no están en {_short(stations_path)}.\n"
            f"Primeros casos: {unknown[:10]}",
            pytrace=False,
        )


def test_en_la_muestra_validacion_va_antes_que_prueba_final():
    """Tramos ordenados en el tiempo (decisión 4): toda fila 'val' es anterior a toda fila 'test'."""
    path = _existing_file("sample", "predictions")
    rows = _read(path, ["ts_hour", "split"]).to_pandas()
    val = rows.loc[rows["split"] == "val", "ts_hour"]
    test = rows.loc[rows["split"] == "test", "ts_hour"]
    assert len(val) and len(test), f"{_short(path)} debe traer filas 'val' y filas 'test'"
    assert val.max() < test.min(), (
        f"{_short(path)}: hay filas 'val' ({val.max()}) que no son anteriores a la primera fila 'test' ({test.min()})"
    )


def _is_copy_of_sample(path: Path) -> bool:
    """¿El archivo trae exactamente la misma tabla que su tocayo de data/sample?"""
    sample = FOLDERS["sample"] / path.name
    if not sample.exists():
        return False
    try:
        if pq.read_metadata(path).num_rows != pq.read_metadata(sample).num_rows:
            return False  # así no se lee completo un archivo real, que puede ser grande
        return pq.read_table(path).equals(pq.read_table(sample))
    except Exception as error:  # archivo dañado o que no es Parquet
        pytest.fail(f"{_short(path)} no se pudo leer como Parquet: {error}", pytrace=False)


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_data_processed_no_trae_datos_sinteticos(name):
    """data/processed es solo para salidas reales: la muestra sintética no puede acabar ahí.

    El dashboard decide si avisa "datos sintéticos" por la carpeta de donde
    lee. Con la muestra en data/processed mostraría números inventados como si
    fueran resultados (decisión 9 de CLAUDE.md), y las demás pruebas pasarían.
    """
    path = _existing_file("processed", name)
    fix = (
        "Borra ese archivo: en data/processed solo van salidas reales del flujo. "
        "Mientras falte, el dashboard usa data/sample y lo avisa."
    )
    if "station_id" in CONTRACTS[name].columns:
        ids = _read(path, ["station_id"])["station_id"]
        fake = pc.sum(pc.starts_with(ids.cast(pa.string()), "SIM-")).as_py() or 0
        if fake:
            pytest.fail(
                f"{_short(path)} trae {fake} filas de estaciones sintéticas (station_id que empieza con 'SIM-'). {fix}",
                pytrace=False,
            )
    if _is_copy_of_sample(path):
        pytest.fail(f"{_short(path)} es una copia de data/sample (datos sintéticos). {fix}", pytrace=False)


# --- Las pruebas de arriba sí muerden (se apuntan a una carpeta temporal) ---

def test_un_archivo_real_que_no_cumple_hace_fallar_la_prueba(tmp_path, monkeypatch):
    """Simula que alguien deja en data/processed un archivo con net_flow mal calculado."""
    broken = _with(_valid("station_hour"), "net_flow", [0, 1, 1])
    pq.write_table(broken, tmp_path / "station_hour.parquet")
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    with pytest.raises(pytest.fail.Exception) as failure:
        test_el_archivo_cumple_su_contrato("processed", "station_hour")
    message = str(failure.value)
    assert "NO cumple el contrato 'station_hour'" in message
    assert "  - net_flow: 1 filas donde no es llegadas menos salidas" in message


def test_un_archivo_que_no_es_parquet_falla_con_mensaje_legible(tmp_path, monkeypatch):
    (tmp_path / "stations.parquet").write_text("esto no es un Parquet", encoding="utf-8")
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    with pytest.raises(pytest.fail.Exception, match="no se pudo leer como Parquet"):
        test_el_archivo_cumple_su_contrato("processed", "stations")


def test_si_falta_un_archivo_real_se_salta_y_si_falta_uno_de_muestra_falla(tmp_path, monkeypatch):
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    monkeypatch.setitem(FOLDERS, "sample", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="todavía no existe"):
        test_el_archivo_cumple_su_contrato("processed", "station_hour")
    with pytest.raises(pytest.fail.Exception, match="deben existir siempre"):
        test_el_archivo_cumple_su_contrato("sample", "station_hour")


def test_una_estacion_que_no_esta_en_stations_hace_fallar_la_prueba(tmp_path, monkeypatch):
    pq.write_table(_valid("stations"), tmp_path / "stations.parquet")
    pq.write_table(_with(_valid("station_hour"), "station_id", ["A", "A", "Z"]), tmp_path / "station_hour.parquet")
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    with pytest.raises(pytest.fail.Exception) as failure:
        test_toda_estacion_usada_existe_en_stations("processed", "station_hour")
    message = str(failure.value)
    assert "usa 1 station_id que no están en" in message
    assert "['Z']" in message


def test_la_muestra_copiada_a_data_processed_hace_fallar_la_prueba(tmp_path, monkeypatch):
    """Simula `cp data/sample/*.parquet data/processed/`: cumple los contratos, pero es sintética."""
    for name in CONTRACTS:
        shutil.copy(_existing_file("sample", name), tmp_path)
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    for name in CONTRACTS:
        test_el_archivo_cumple_su_contrato("processed", name)  # la prueba de contrato no lo nota
        # policy_curve no tiene station_id: se reconoce por ser idéntica a la de la muestra.
        expected = "estaciones sintéticas" if "station_id" in CONTRACTS[name].columns else "es una copia de data/sample"
        with pytest.raises(pytest.fail.Exception, match=expected) as failure:
            test_data_processed_no_trae_datos_sinteticos(name)
        assert f"{name}.parquet" in str(failure.value) and "Borra ese archivo" in str(failure.value)


def test_un_archivo_real_no_se_confunde_con_la_muestra(tmp_path, monkeypatch):
    for name in CONTRACTS:
        pq.write_table(_valid(name), tmp_path / f"{name}.parquet")
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    for name in CONTRACTS:
        test_data_processed_no_trae_datos_sinteticos(name)  # ids A, B, C: no falla

    # Una curva con las mismas filas que la de la muestra pero otros números tampoco es una copia.
    sample_curve = pq.read_table(_existing_file("sample", "policy_curve"))
    shifted = _with(sample_curve, "ci_high", [value + 1.0 for value in sample_curve["ci_high"].to_pylist()])
    pq.write_table(shifted, tmp_path / "policy_curve.parquet")
    test_data_processed_no_trae_datos_sinteticos("policy_curve")


def test_sin_archivo_real_la_prueba_de_sinteticos_se_salta(tmp_path, monkeypatch):
    monkeypatch.setitem(FOLDERS, "processed", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="todavía no existe"):
        test_data_processed_no_trae_datos_sinteticos("stations")


# --------------------------------------------------------------------------- #
# 4) El validador detecta cada error
# --------------------------------------------------------------------------- #

def _ts(text: str) -> pd.Timestamp:
    """Instante a partir de la hora del reloj de Nueva York."""
    return pd.Timestamp(text, tz=TZ)


H6, H8, H9, H12 = (_ts(f"2025-10-13 {hour:02d}:00") for hour in (6, 8, 9, 12))


def _valid_frame(name: str) -> pd.DataFrame:
    """Tabla pequeña, hecha a mano, que cumple el contrato `name`."""
    if name == "stations":
        return pd.DataFrame(
            {
                "station_id": ["A", "B", "C"],
                "name": ["Uno", "Dos", "Tres"],
                "lat": [40.70, 40.71, 40.72],
                "lon": [-74.00, -73.99, -73.98],
                "capacity": [20, 30, 25],
                "station_type": ["oficinas", "residencial", "mixta"],
            }
        )
    if name == "station_hour":
        return pd.DataFrame(
            {
                "station_id": ["A", "A", "B"],
                "ts_hour": [H8, H9, H8],
                "departures": [5, 3, 0],
                "arrivals": [2, 4, 1],
                "net_flow": [-3, 1, 1],
                "departures_member": [4, 1, 0],
                "departures_casual": [1, 2, 0],
            }
        )
    if name == "predictions":
        return pd.DataFrame(
            {
                "station_id": ["A", "A"],
                "ts_hour": [H8, H8],
                "model": ["baseline", "lgbm"],
                "split": ["val", "val"],
                "dep_q10": [1.0, 1.0],
                "dep_q50": [2.0, 2.5],
                "dep_q90": [4.0, 4.0],
                "arr_q10": [0.0, 0.0],
                "arr_q50": [1.5, 1.0],
                "arr_q90": [3.0, 3.0],
            }
        )
    if name == "recommendations":
        return pd.DataFrame(
            {
                "station_id": ["A", "A", "A"],
                "window_start": [H6, H6, H6],
                "window_end": [H12, H12, H12],
                "policy": ["none", "historical", "model"],
                "start_inventory_target": [10, 12, 7],
                "bikes_to_move": [0, 2, -3],  # negativo es válido: retirar bicis
                "risk_level": ["alto", "medio", "bajo"],
                "feasible": [True, True, False],
            }
        )
    if name == "policy_curve":
        return pd.DataFrame(
            {
                "policy": ["none", "historical", "model"],
                "budget": [0, 100, 100],
                "risk_station_hours": [420.0, 300.0, 250.0],
                "ci_low": [400.0, 280.0, 230.0],
                "ci_high": [440.0, 320.0, 270.0],
            }
        )
    raise KeyError(name)


def _valid(name: str) -> pa.Table:
    """La misma tabla válida, ya con los tipos exactos del contrato."""
    frame = _valid_frame(name)
    return pa.table({column: frame[column].tolist() for column in frame.columns}, schema=CONTRACTS[name].schema)


def _with(table: pa.Table, column: str, values: list) -> pa.Table:
    """Copia de la tabla con los valores de UNA columna cambiados (mismo tipo)."""
    position = table.schema.get_field_index(column)
    return table.set_column(position, column, pa.array(values, type=table.schema.field(column).type))


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_la_tabla_valida_de_ejemplo_cumple(name):
    assert validate_table(_valid(name), name) == []


def test_capacity_y_station_type_admiten_nulos():
    table = _with(_with(_valid("stations"), "capacity", [20, None, 25]), "station_type", ["oficinas", None, None])
    assert validate_table(table, "stations") == []


NAN, INF = float("nan"), float("inf")


def _cases(rows: list[tuple]) -> list:
    """Convierte renglones (qué se rompe, ...) en casos de pytest con ese texto como nombre."""
    return [pytest.param(*rest, id=label) for label, *rest in rows]


# Cada renglón: qué se rompe, contrato, cambio a la tabla, texto que debe salir en el problema.
WRONG_COLUMNS = _cases(
    [
        ("falta una columna", "station_hour", lambda t: t.drop_columns(["net_flow"]), "faltan columnas: ['net_flow']"),
        ("sobra una columna", "stations", lambda t: t.append_column("zona", pa.array(["x", "y", "z"])),
         "sobran columnas: ['zona']"),
        ("columnas en otro orden", "policy_curve", lambda t: t.select(list(reversed(t.schema.names))),
         "orden de columnas distinto"),
    ]
)

# Cada renglón: qué se rompe, contrato, columna, tipo equivocado, texto que debe salir en el problema.
WRONG_TYPES = _cases(
    [
        ("entero de 32 bits", "station_hour", "departures", pa.int32(), "departures: tipo int32 (esperado int64)"),
        ("capacity como decimal", "stations", "capacity", pa.float64(), "capacity: tipo double (esperado int64)"),
        ("budget como decimal", "policy_curve", "budget", pa.float64(), "budget: tipo double (esperado int64)"),
        ("decimal de 32 bits", "predictions", "dep_q50", pa.float32(), "dep_q50: tipo float (esperado double)"),
        ("feasible como entero", "recommendations", "feasible", pa.int64(), "feasible: tipo int64 (esperado bool)"),
        # Lo que deja pandas si se guarda sin write_contract.
        ("texto large_string", "stations", "station_id", pa.large_string(),
         "station_id: tipo large_string (esperado string)"),
        ("fecha en nanosegundos", "predictions", "ts_hour", pa.timestamp("ns", tz=TZ),
         "ts_hour: tipo timestamp[ns, tz=America/New_York]"),
        # Lo que deja DuckDB si se guarda sin write_contract.
        ("fecha en UTC", "station_hour", "ts_hour", pa.timestamp("us", tz="UTC"),
         "ts_hour: tipo timestamp[us, tz=UTC]"),
        ("window_start en UTC", "recommendations", "window_start", pa.timestamp("us", tz="UTC"),
         "window_start: tipo timestamp[us, tz=UTC]"),
        ("fecha sin zona", "station_hour", "ts_hour", pa.timestamp("us"), "ts_hour: tipo timestamp[us] (esperado"),
    ]
)

# Cada renglón: qué se rompe, contrato, {columna: valores nuevos}, texto que debe salir en el problema.
# Cuando hay dos columnas, la segunda solo acompaña a la primera para que no salte otra regla.
WRONG_VALUES = _cases(
    [
        # --- Nulos donde el contrato no los permite ---
        ("nulo en departures", "station_hour", {"departures": [None, 3, 0]}, "departures: 1 nulos"),
        ("nulo en station_id", "stations", {"station_id": [None, "B", "C"]}, "station_id: 1 nulos"),
        ("nulo en un cuantil", "predictions", {"dep_q50": [None, 2.5]}, "dep_q50: 1 nulos"),
        ("nulo en feasible", "recommendations", {"feasible": [True, None, False]}, "feasible: 1 nulos"),
        # --- Llave repetida, aunque el resto de la fila sea distinto ---
        ("misma estación dos veces", "stations", {"station_id": ["A", "A", "C"]},
         "1 filas duplicadas por llave ['station_id']"),
        # --- stations ---
        ("capacity negativa", "stations", {"capacity": [20, -1, 25]}, "capacity: 1 valores negativos"),
        ("latitud imposible", "stations", {"lat": [95.0, 40.71, 40.72]}, "lat/lon fuera de rango"),
        ("longitud imposible", "stations", {"lon": [-74.0, -190.0, -73.98]}, "lat/lon fuera de rango"),
        ("latitud imposible al sur", "stations", {"lat": [-95.0, 40.71, 40.72]}, "lat/lon fuera de rango"),
        ("longitud imposible al este", "stations", {"lon": [-74.0, 190.0, -73.98]}, "lat/lon fuera de rango"),
        ("NaN en lat", "stations", {"lat": [NAN, 40.71, 40.72]}, "lat: 1 valores NaN o infinitos"),
        ("station_type de GBFS", "stations", {"station_type": ["classic", "residencial", "mixta"]},
         "station_type: valores no permitidos ['classic']"),
        # --- station_hour ---
        ("net_flow mal calculado", "station_hour", {"net_flow": [0, 1, 1]},
         "net_flow: 1 filas donde no es llegadas menos salidas"),
        ("arrivals negativo", "station_hour", {"arrivals": [-1, 4, 1], "net_flow": [-6, 1, 1]},
         "arrivals: 1 valores negativos"),
        ("miembros negativo", "station_hour", {"departures_member": [-1, 1, 0], "departures_casual": [6, 2, 0]},
         "departures_member: 1 valores negativos"),
        ("casuales negativo", "station_hour", {"departures_member": [6, 1, 0], "departures_casual": [-1, 2, 0]},
         "departures_casual: 1 valores negativos"),
        ("miembros + casuales > total", "station_hour", {"departures_member": [6, 1, 0]},
         "miembros + casuales supera el total"),
        ("hora con minutos", "station_hour", {"ts_hour": [_ts("2025-10-13 08:30"), H9, H8]},
         "ts_hour: 1 valores que no caen en hora exacta"),
        ("hora con segundos", "station_hour", {"ts_hour": [_ts("2025-10-13 08:00:30"), H9, H8]},
         "ts_hour: 1 valores que no caen en hora exacta"),
        # --- predictions ---
        ("cuantil negativo", "predictions", {"dep_q10": [-1.0, 1.0]}, "dep_q10: 1 valores negativos"),
        ("cuantil de llegadas negativo", "predictions", {"arr_q10": [-1.0, 0.0]}, "arr_q10: 1 valores negativos"),
        ("q10 > q50 en salidas", "predictions", {"dep_q10": [3.0, 1.0]}, "dep: 1 filas con cuantiles cruzados"),
        ("q50 > q90 en llegadas", "predictions", {"arr_q90": [1.0, 3.0]}, "arr: 1 filas con cuantiles cruzados"),
        ("NaN en un cuantil", "predictions", {"dep_q50": [NAN, 2.5]}, "dep_q50: 1 valores NaN o infinitos"),
        ("infinito en un cuantil", "predictions", {"arr_q90": [INF, 3.0]}, "arr_q90: 1 valores NaN o infinitos"),
        ("model desconocido", "predictions", {"model": ["xgboost", "lgbm"]},
         "model: valores no permitidos ['xgboost']"),
        ("split desconocido", "predictions", {"split": ["train", "val"]}, "split: valores no permitidos ['train']"),
        ("pronóstico fuera de hora exacta", "predictions", {"ts_hour": [_ts("2025-10-13 08:15"), H8]},
         "ts_hour: 1 valores que no caen en hora exacta"),
        # --- recommendations ---
        ("policy desconocida", "recommendations", {"policy": ["none", "historical", "modelo"]},
         "policy: valores no permitidos ['modelo']"),
        ("risk_level desconocido", "recommendations", {"risk_level": ["alto", "medio", "high"]},
         "risk_level: valores no permitidos ['high']"),
        ("policy none que mueve bicis", "recommendations", {"bikes_to_move": [3, 2, -3]},
         "bikes_to_move: 1 filas con policy 'none' que mueven bicis"),
        ("ventana de duración cero", "recommendations", {"window_end": [H6, H12, H12]},
         "window_end: 1 filas donde no es posterior a window_start"),
        ("ventana que termina antes de empezar", "recommendations", {"window_end": [H12, _ts("2025-10-13 05:00"), H12]},
         "window_end: 1 filas donde no es posterior a window_start"),
        ("inventario objetivo negativo", "recommendations", {"start_inventory_target": [10, -1, 7]},
         "start_inventory_target: 1 valores negativos"),
        # --- policy_curve ---
        ("policy desconocida en la curva", "policy_curve", {"policy": ["none", "historical", "lgbm"]},
         "policy: valores no permitidos ['lgbm']"),
        ("budget negativo", "policy_curve", {"budget": [0, -100, 100]}, "budget: 1 valores negativos"),
        ("ci_low negativo", "policy_curve", {"ci_low": [-5.0, 280.0, 230.0]}, "ci_low: 1 valores negativos"),
        ("ci_low arriba del valor", "policy_curve", {"ci_low": [430.0, 280.0, 230.0]},
         "ci_low/ci_high: 1 filas donde el intervalo no contiene"),
        ("ci_high abajo del valor", "policy_curve", {"ci_high": [440.0, 320.0, 240.0]},
         "ci_low/ci_high: 1 filas donde el intervalo no contiene"),
        ("NaN en la curva", "policy_curve", {"risk_station_hours": [420.0, NAN, 250.0]},
         "risk_station_hours: 1 valores NaN o infinitos"),
    ]
)


def _assert_single_problem(problems: list[str], expected_text: str) -> None:
    """Se rompió una sola cosa: debe salir un solo problema, y debe ser el esperado."""
    assert len(problems) == 1 and expected_text in problems[0], problems


@pytest.mark.parametrize("name, break_it, expected_text", WRONG_COLUMNS)
def test_el_validador_detecta_columnas_mal(name, break_it, expected_text):
    _assert_single_problem(validate_table(break_it(_valid(name)), name), expected_text)


@pytest.mark.parametrize("name, column, wrong_type, expected_text", WRONG_TYPES)
def test_el_validador_detecta_tipos_mal(name, column, wrong_type, expected_text):
    table = _valid(name)
    broken = table.set_column(table.schema.get_field_index(column), column, table[column].cast(wrong_type))
    _assert_single_problem(validate_table(broken, name), expected_text)


@pytest.mark.parametrize("name, new_values, expected_text", WRONG_VALUES)
def test_el_validador_detecta_valores_mal(name, new_values, expected_text):
    table = _valid(name)
    for column, values in new_values.items():
        table = _with(table, column, values)
    _assert_single_problem(validate_table(table, name), expected_text)


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_el_validador_detecta_filas_repetidas(name):
    table = _valid(name)
    problems = validate_table(pa.concat_tables([table, table.slice(0, 1)]), name)
    assert len(problems) == 1 and f"1 filas duplicadas por llave {list(CONTRACTS[name].key)}" in problems[0], problems


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_el_validador_rechaza_una_tabla_sin_filas(name):
    # Un archivo vacío casi siempre es un paso anterior que falló en silencio.
    _assert_single_problem(validate_table(_valid(name).slice(0, 0), name), "no tiene filas")


def test_el_validador_detecta_fracciones_de_segundo_en_ts_hour():
    table = _valid("station_hour")
    shifted = pc.add(table["ts_hour"], pa.scalar(500_000, pa.duration("us")))  # 08:00:00.5
    _assert_single_problem(validate_table(_with(table, "ts_hour", shifted.to_pylist()), "station_hour"), "hora exacta")


# --- to_contract_table ---

def test_to_contract_table_rechaza_fechas_sin_zona():
    frame = _valid_frame("station_hour")
    frame["ts_hour"] = frame["ts_hour"].dt.tz_localize(None)
    with pytest.raises(ContractError, match="zona horaria"):
        to_contract_table(frame, "station_hour")


def test_to_contract_table_convierte_utc_al_mismo_instante_en_nueva_york():
    frame = _valid_frame("station_hour")
    frame["ts_hour"] = frame["ts_hour"].dt.tz_convert("UTC")
    assert frame["ts_hour"].iloc[0] == pd.Timestamp("2025-10-13 12:00", tz="UTC")  # las 08:00 de Nueva York

    table = to_contract_table(frame, "station_hour")
    assert table.schema.field("ts_hour").type == pa.timestamp("us", tz="America/New_York")
    # Mismo instante (se compara en microsegundos desde 1970, que no depende de la zona)...
    expected = _valid("station_hour")["ts_hour"]
    assert table["ts_hour"].cast(pa.int64()).to_pylist() == expected.cast(pa.int64()).to_pylist()
    # ...y al leerlo, el reloj marca la hora de Nueva York.
    local = table.to_pandas()["ts_hour"]
    assert str(local.dt.tz) == "America/New_York"
    assert local.dt.hour.tolist() == [8, 9, 8]
    assert validate_table(table, "station_hour") == []


def test_to_contract_table_corrige_tipos_orden_y_columnas_de_mas():
    """Una tabla con los tipos que pandas pone por omisión queda con el esquema exacto."""
    frame = _valid_frame("station_hour")
    frame["departures"] = frame["departures"].astype("int32")
    frame["ts_hour"] = frame["ts_hour"].dt.as_unit("ns")
    frame["comentario"] = "sobra"
    frame = frame[list(reversed(frame.columns))]

    table = to_contract_table(frame, "station_hour")
    assert table.schema.names == CONTRACTS["station_hour"].columns
    assert table.schema.types == CONTRACTS["station_hour"].schema.types
    assert validate_table(table, "station_hour") == []


def test_to_contract_table_avisa_si_falta_una_columna():
    with pytest.raises(ContractError, match="faltan columnas.*net_flow"):
        to_contract_table(_valid_frame("station_hour").drop(columns="net_flow"), "station_hour")


def test_to_contract_table_guarda_capacity_vacia_como_nulo_entero():
    """pandas vuelve decimal una columna entera con vacíos. En el contrato sigue siendo entera."""
    frame = _valid_frame("stations")
    frame["capacity"] = [20, float("nan"), 25]
    table = to_contract_table(frame, "stations")
    assert table.schema.field("capacity").type == pa.int64()
    assert table["capacity"].to_pylist() == [20, None, 25]


def test_to_contract_table_no_recorta_decimales_en_una_columna_entera():
    """5.5 salidas no se guarda como 5: la conversión falla en lugar de recortar."""
    frame = _valid_frame("station_hour")
    frame["departures"] = [5.5, 3.0, 0.0]
    with pytest.raises(ContractError, match="no se pudo convertir"):
        to_contract_table(frame, "station_hour")


def _files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


# --- to_contract_table no cambia de familia de tipo (texto, número, sí/no) ---
# Un station_id leído como número ya perdió los ceros finales ("5470.10" llega
# como 5470.1). Volverlo texto sin avisar guardaría "5470.1", que es otra estación.

# Cada renglón: qué llega mal, contrato, columna, valores, texto del error (expresión regular).
WRONG_KINDS = _cases(
    [
        ("station_id decimal", "stations", "station_id", [6289.06, 5470.10, 72.0],
         "station_id llegó como double y el contrato pide string.*pierde los ceros finales"),
        ("station_id entero", "stations", "station_id", [6289, 5470, 72],
         "station_id llegó como int64 y el contrato pide string"),
        ("nombre sí/no", "stations", "name", [True, False, True], "name llegó como bool y el contrato pide string"),
        ("conteo como texto", "station_hour", "departures", ["5", "3", "0"],
         "departures llegó como (large_)?string y el contrato pide int64"),
        ("conteo como sí/no", "station_hour", "departures", [True, True, False],
         "departures llegó como bool y el contrato pide int64"),
        ("cuantil como texto", "predictions", "dep_q50", ["2.0", "2.5"],
         "dep_q50 llegó como (large_)?string y el contrato pide double"),
        ("feasible como entero", "recommendations", "feasible", [2, 0, -1],
         "feasible llegó como int64 y el contrato pide bool"),
        ("feasible como texto", "recommendations", "feasible", ["true", "false", "true"],
         "feasible llegó como (large_)?string y el contrato pide bool"),
    ]
)


@pytest.mark.parametrize("name, column, values, expected_text", WRONG_KINDS)
def test_to_contract_table_no_cambia_de_familia_de_tipo(name, column, values, expected_text, tmp_path):
    frame = _valid_frame(name)
    frame[column] = values
    with pytest.raises(ContractError, match=expected_text):
        to_contract_table(frame, name)
    with pytest.raises(ContractError, match=expected_text):
        write_contract(frame, name, tmp_path / f"{name}.parquet")
    assert _files(tmp_path) == []  # no se escribió nada


def test_un_csv_leido_con_los_tipos_por_omision_no_pasa_como_station_id(tmp_path):
    """El caso real: pandas lee la columna de ids como número y el cero final se pierde ahí mismo."""
    csv = tmp_path / "stations.csv"
    csv.write_text(
        "station_id,name,lat,lon,capacity,station_type\n"
        "5303.10,Uno,40.70,-74.00,20,oficinas\n"
        "2231.10,Dos,40.71,-73.99,30,residencial\n"
        "6289.06,Tres,40.72,-73.98,25,mixta\n",
        encoding="utf-8",
    )
    target = tmp_path / "stations.parquet"

    as_number = pd.read_csv(csv)
    assert as_number["station_id"].tolist() == [5303.1, 2231.1, 6289.06]
    with pytest.raises(ContractError, match="station_id llegó como double"):
        write_contract(as_number, "stations", target)
    assert not target.exists()

    # Leído como texto, el id llega completo y se guarda completo.
    as_text = pd.read_csv(csv, dtype={"station_id": str})
    write_contract(as_text, "stations", target)
    assert pq.read_table(target)["station_id"].to_pylist() == ["5303.10", "2231.10", "6289.06"]


# Lo que SÍ se convierte: misma familia de tipo, o una columna toda vacía (no hay valor que dañar).
# Cada renglón: qué llega, contrato, columna, valores, lo que debe quedar en la tabla.
SAME_KIND_FROM_PANDAS = _cases(
    [
        ("station_type toda vacía (None)", "stations", "station_type", [None, None, None], [None, None, None]),
        ("station_type toda vacía (NaN)", "stations", "station_type", [NAN, NAN, NAN], [None, None, None]),
        ("capacity toda vacía (None)", "stations", "capacity", [None, None, None], [None, None, None]),
        ("station_id como categoría de pandas", "stations", "station_id", pd.Categorical(["A", "B", "C"]),
         ["A", "B", "C"]),
        ("capacity decimal sin fracción", "stations", "capacity", [20.0, 30.0, 25.0], [20, 30, 25]),
        ("lat entera", "stations", "lat", [40, 41, 42], [40.0, 41.0, 42.0]),
        ("feasible con el tipo boolean de pandas", "recommendations", "feasible",
         pd.array([True, True, False], dtype="boolean"), [True, True, False]),
    ]
)

SAME_KIND_FROM_ARROW = _cases(
    [
        # Lo que entrega DuckDB: sum() es decimal128(38, 0) y un NULL sin tipo es int32.
        ("suma de DuckDB (decimal128)", "station_hour", "departures", pa.array([5, 3, 0], type=pa.decimal128(38, 0)),
         [5, 3, 0]),
        ("NULL de DuckDB (int32 todo vacío)", "stations", "station_type", pa.array([None, None, None], type=pa.int32()),
         [None, None, None]),
        ("texto string_view", "stations", "station_id", pa.array(["A", "B", "C"], type=pa.string_view()),
         ["A", "B", "C"]),
        ("texto en diccionario", "stations", "station_id", pa.array(["A", "B", "C"]).dictionary_encode(),
         ["A", "B", "C"]),
    ]
)


def _assert_converted(table: pa.Table, name: str, column: str, stored: list) -> None:
    assert table.schema.equals(CONTRACTS[name].schema)
    assert table[column].to_pylist() == stored
    assert validate_table(table, name) == []


@pytest.mark.parametrize("name, column, values, stored", SAME_KIND_FROM_PANDAS)
def test_to_contract_table_convierte_dentro_de_la_misma_familia(name, column, values, stored):
    frame = _valid_frame(name)
    frame[column] = values
    _assert_converted(to_contract_table(frame, name), name, column, stored)


@pytest.mark.parametrize("name, column, values, stored", SAME_KIND_FROM_ARROW)
def test_to_contract_table_convierte_tablas_de_arrow_de_la_misma_familia(name, column, values, stored):
    table = _valid(name)
    table = table.set_column(table.schema.get_field_index(column), column, values)
    _assert_converted(to_contract_table(table, name), name, column, stored)


# --- to_contract_table: qué recibe ---

STATION_HOUR_SQL = """
    SELECT station_id, ts_hour,
           sum(dep) AS departures, sum(arr) AS arrivals, sum(arr) - sum(dep) AS net_flow,
           sum(mem) AS departures_member, sum(dep) - sum(mem) AS departures_casual
    FROM (VALUES ('A', TIMESTAMPTZ '2025-10-13 08:00:00-04', 5, 2, 4),
                 ('A', TIMESTAMPTZ '2025-10-13 09:00:00-04', 3, 4, 1),
                 ('B', TIMESTAMPTZ '2025-10-13 08:00:00-04', 0, 1, 0)) AS t(station_id, ts_hour, dep, arr, mem)
    GROUP BY ALL
    ORDER BY ALL
"""


@pytest.mark.parametrize("how", ["to_arrow_table", "arrow", "df"])
def test_to_contract_table_acepta_lo_que_devuelve_duckdb(how):
    """El flujo real: una consulta con sum() en DuckDB, entregada de las tres formas habituales."""
    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone='America/New_York'")
        result = getattr(connection.sql(STATION_HOUR_SQL), how)()  # .arrow() entrega un lector por lotes
        table = to_contract_table(result, "station_hour")
    finally:
        connection.close()
    assert table.equals(_valid("station_hour"))


def test_to_contract_table_acepta_un_lector_por_lotes_de_arrow():
    reader = _valid("station_hour").to_reader()
    assert isinstance(reader, pa.RecordBatchReader)
    assert to_contract_table(reader, "station_hour").equals(_valid("station_hour"))


@pytest.mark.parametrize(
    "data, kind",
    [({"station_id": ["A"]}, "dict"), ([1, 2, 3], "list"), (None, "NoneType")],
    ids=["un diccionario", "una lista", "None"],
)
def test_to_contract_table_rechaza_lo_que_no_es_una_tabla(data, kind):
    with pytest.raises(ContractError, match=f"se esperaba un DataFrame de pandas o una tabla de Arrow y llegó {kind}"):
        to_contract_table(data, "station_hour")


# --- write_contract y read_contract ---


def test_write_contract_no_escribe_nada_si_la_tabla_no_cumple(tmp_path):
    frame = _valid_frame("station_hour")
    frame["net_flow"] = 0
    with pytest.raises(ContractError) as error:
        write_contract(frame, "station_hour", tmp_path / "station_hour.parquet")
    assert "net_flow: 3 filas" in str(error.value)  # el error dice qué está mal
    assert _files(tmp_path) == []                    # ni el archivo ni un temporal


def test_write_contract_no_escribe_nada_si_la_fecha_no_trae_zona(tmp_path):
    frame = _valid_frame("station_hour")
    frame["ts_hour"] = frame["ts_hour"].dt.tz_localize(None)
    with pytest.raises(ContractError, match="zona horaria"):
        write_contract(frame, "station_hour", tmp_path / "station_hour.parquet")
    assert _files(tmp_path) == []


def test_write_contract_que_falla_no_toca_el_archivo_anterior(tmp_path):
    path = write_contract(_valid_frame("station_hour"), "station_hour", tmp_path / "station_hour.parquet")
    before = path.read_bytes()
    frame = _valid_frame("station_hour")
    frame["arrivals"] = -1
    with pytest.raises(ContractError):
        write_contract(frame, "station_hour", path)
    assert _files(tmp_path) == [path]
    assert path.read_bytes() == before


def test_write_contract_interrumpido_al_guardar_no_dana_el_archivo_anterior(tmp_path, monkeypatch):
    """Se escribe primero en un temporal: si el guardado se corta a la mitad, el archivo anterior sigue entero."""
    path = write_contract(_valid_frame("station_hour"), "station_hour", tmp_path / "station_hour.parquet")
    before = path.read_bytes()

    def interrupted(table, where, **options):
        Path(where).write_bytes(b"archivo a medias")
        raise OSError("disco lleno")

    monkeypatch.setattr("src.contracts.pq.write_table", interrupted)
    with pytest.raises(OSError, match="disco lleno"):
        write_contract(_valid_frame("station_hour"), "station_hour", path)
    assert path.read_bytes() == before
    assert validate_file(path, "station_hour") == []


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_write_contract_guarda_un_archivo_que_cumple(name, tmp_path):
    path = write_contract(_valid_frame(name), name, tmp_path / "carpeta_nueva" / f"{name}.parquet")
    assert _files(tmp_path) == [path]  # no queda el temporal
    assert validate_file(path, name) == []
    assert pq.read_table(path).equals(_valid(name))


def test_read_contract_conserva_enteros_con_nulos_y_la_zona(tmp_path):
    stations = _valid_frame("stations")
    stations["capacity"] = pd.array([20, None, 25], dtype="Int64")
    path = write_contract(stations, "stations", tmp_path / "stations.parquet")
    back = read_contract("stations", path)
    assert str(back["capacity"].dtype) == "Int64"
    assert back["capacity"].isna().tolist() == [False, True, False]
    assert back["capacity"].dropna().tolist() == [20, 25]

    path = write_contract(_valid_frame("station_hour"), "station_hour", tmp_path / "station_hour.parquet")
    back = read_contract("station_hour", path)
    assert str(back["ts_hour"].dt.tz) == "America/New_York"
    assert back["ts_hour"].tolist() == _valid_frame("station_hour")["ts_hour"].tolist()


# --------------------------------------------------------------------------- #
# 5) Las definiciones de src/contracts.py son las acordadas
# --------------------------------------------------------------------------- #
# Esta es la especificación escrita a mano, tal como la acordó el equipo (ver
# CLAUDE.md). Si esta prueba falla, alguien cambió src/contracts.py. Un cambio
# de contrato se hace en los tres lugares (src/contracts.py, CLAUDE.md y aquí)
# y avisando al otro integrante.

STR, INT, FLOAT, BOOL = pa.string(), pa.int64(), pa.float64(), pa.bool_()
NY_TIME = pa.timestamp("us", tz="America/New_York")

SPEC = {
    "stations": {
        "columns": [
            ("station_id", STR),
            ("name", STR),
            ("lat", FLOAT),
            ("lon", FLOAT),
            ("capacity", INT),
            ("station_type", STR),
        ],
        "key": ["station_id"],
        "nullable": {"capacity", "station_type"},
        # Categorías de config/operacion.yaml. Pendiente de confirmar por logística.
        "allowed": {"station_type": {"oficinas", "residencial", "recreativa", "mixta"}},
    },
    "station_hour": {
        "columns": [
            ("station_id", STR),
            ("ts_hour", NY_TIME),
            ("departures", INT),
            ("arrivals", INT),
            ("net_flow", INT),
            ("departures_member", INT),
            ("departures_casual", INT),
        ],
        "key": ["station_id", "ts_hour"],
        "nullable": set(),
        "allowed": {},
    },
    "predictions": {
        "columns": [
            ("station_id", STR),
            ("ts_hour", NY_TIME),
            ("model", STR),
            ("split", STR),
            ("dep_q10", FLOAT),
            ("dep_q50", FLOAT),
            ("dep_q90", FLOAT),
            ("arr_q10", FLOAT),
            ("arr_q50", FLOAT),
            ("arr_q90", FLOAT),
        ],
        "key": ["station_id", "ts_hour", "model", "split"],
        "nullable": set(),
        "allowed": {"model": {"baseline", "lgbm"}, "split": {"val", "test"}},
    },
    "recommendations": {
        "columns": [
            ("station_id", STR),
            ("window_start", NY_TIME),
            ("window_end", NY_TIME),
            ("policy", STR),
            ("start_inventory_target", INT),
            ("bikes_to_move", INT),
            ("risk_level", STR),
            ("feasible", BOOL),
        ],
        "key": ["station_id", "window_start", "policy"],
        "nullable": set(),
        "allowed": {"policy": {"none", "historical", "model"}, "risk_level": {"alto", "medio", "bajo"}},
    },
    "policy_curve": {
        "columns": [
            ("policy", STR),
            ("budget", INT),
            ("risk_station_hours", FLOAT),
            ("ci_low", FLOAT),
            ("ci_high", FLOAT),
        ],
        "key": ["policy", "budget"],
        "nullable": set(),
        "allowed": {"policy": {"none", "historical", "model"}},
    },
}


def test_son_exactamente_los_cinco_contratos_acordados():
    assert list(CONTRACTS) == list(SPEC)


@pytest.mark.parametrize("name", list(SPEC))
def test_la_definicion_del_contrato_es_la_acordada(name):
    assert name in CONTRACTS, f"src/contracts.py ya no define el contrato '{name}'"
    contract, spec = CONTRACTS[name], SPEC[name]
    assert contract.name == name
    assert contract.filename == f"{name}.parquet"
    # Nombres en orden y tipos. Se comparan como texto para que el fallo se lea fácil...
    assert [(f.name, str(f.type)) for f in contract.schema] == [(column, str(kind)) for column, kind in spec["columns"]]
    # ...y también como tipos de Arrow, que es la comparación exacta.
    assert contract.schema.types == [kind for _, kind in spec["columns"]]
    assert list(contract.key) == spec["key"]
    assert set(contract.nullable) == spec["nullable"]
    assert {column: set(options) for column, options in contract.allowed.items()} == spec["allowed"]
