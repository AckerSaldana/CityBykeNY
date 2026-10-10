"""Pruebas del generador de datos SINTÉTICOS de muestra (src/data/sample_data.py).

Qué revisan:

1. El generador da siempre las mismas tablas (semilla fija).
2. data/sample es lo que el generador produce HOY. Si alguien cambia el
   generador, o las franjas, los presupuestos o la ocupacion_respaldo de
   config/operacion.yaml, y no vuelve a correr el script, la prueba FALLA y
   dice qué correr.
3. Los datos se distinguen como sintéticos: ids "SIM-..." y nombres
   "Sintética ...".
4. `feasible` es de la estación y la franja: vale lo mismo en las tres
   políticas.
5. `ocupacion_respaldo` se usa como quedó decidido: es el inventario con el
   que empiezan las tres políticas y el que queda cuando no hay solución
   factible. La regla histórica no lo usa.
6. station_hour es densa (una fila por estación y hora, con ceros, solo entre
   el primer y el último viaje de cada estación) y hay estaciones que no
   cubren todo el periodo.
7. Las rutas cuadran con station_hour: lo que sale de cada estación suma sus
   salidas y lo que llega suma sus llegadas.
8. El generador se niega a escribir en data/processed y en data/raw, que son
   solo para datos reales.

Si falla el punto 2, corre:  python scripts/make_sample_data.py
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.contracts import CONTRACTS, POLICIES, to_contract_table, validate_table
from src.data import sample_data
from src.data.sample_data import (
    DEFAULT_CAPACITY,
    DEFAULT_OCCUPANCY,
    DST_DAY,
    END,
    START,
    build_sample,
    check_output_folder,
    main,
    write_sample,
)
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE, TZ
from src.sim.operacion import load_operacion

COMMAND = "python scripts/make_sample_data.py"
REFUSAL = "No se escriben datos SINTÉTICOS"


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return build_sample()


# --------------------------------------------------------------------------- #
# 1) Determinista
# --------------------------------------------------------------------------- #

def test_el_generador_da_siempre_las_mismas_tablas(tables):
    again = build_sample()
    assert list(tables) == list(again) == list(CONTRACTS)
    for name in CONTRACTS:
        pd.testing.assert_frame_equal(tables[name], again[name])


# --------------------------------------------------------------------------- #
# 2) data/sample está al día
# --------------------------------------------------------------------------- #

def _outdated(folder: Path, fresh: Path) -> list[str]:
    """Archivos de `folder` que no coinciden con los que el generador acaba de escribir en `fresh`."""
    names = []
    for path in sorted(fresh.iterdir()):
        current = folder / path.name
        if not current.exists():
            names.append(path.name)
        elif path.suffix == ".parquet":
            # Se comparan las tablas (columnas, tipos y valores), no los bytes del archivo.
            if not pq.read_table(path).equals(pq.read_table(current)):
                names.append(path.name)
        elif path.read_text(encoding="utf-8") != current.read_text(encoding="utf-8"):
            names.append(path.name)
    return names


def _fail_if_outdated(folder: Path, tmp_path: Path) -> None:
    """Genera la muestra en una carpeta temporal y la compara con `folder`."""
    write_sample(tmp_path)
    outdated = _outdated(folder, tmp_path)
    if outdated:
        pytest.fail(
            f"data/sample está desactualizado. Ya no coinciden con lo que produce el generador: {outdated}.\n"
            "Cambió src/data/sample_data.py, o cambiaron las franjas o los presupuestos de config/operacion.yaml.\n"
            f"Para regenerarlos: {COMMAND}",
            pytrace=False,
        )


def test_data_sample_es_lo_que_genera_el_script_hoy(tmp_path):
    _fail_if_outdated(DATA_SAMPLE, tmp_path)


# --- La prueba de arriba sí muerde ---

def _changed_config(change) -> dict:
    """Copia de config/operacion.yaml ya leído, con un cambio hecho por `change`."""
    cfg = copy.deepcopy(load_operacion())
    change(cfg)
    return cfg


def _one_more_budget(cfg: dict) -> None:
    cfg["presupuestos_diarios"] = [*cfg["presupuestos_diarios"], max(cfg["presupuestos_diarios"]) + 123]


def _one_window_less(cfg: dict) -> None:
    assert len(cfg["franjas"]) >= 2, "la prueba necesita al menos dos franjas en config/operacion.yaml"
    cfg["franjas"] = cfg["franjas"][:-1]


def _another_fallback(cfg: dict) -> None:
    """Mueve la ocupacion_respaldo de todos los tipos y franjas a un valor que hoy no tiene ninguno."""
    for rule in cfg["estrategia_por_tipo"].values():
        for window, value in rule["ocupacion_respaldo"].items():
            rule["ocupacion_respaldo"][window] = 0.9 if value < 0.5 else 0.1


@pytest.mark.parametrize(
    "change, stale_file",
    [
        pytest.param(_one_more_budget, "policy_curve.parquet", id="un presupuesto más"),
        pytest.param(_one_window_less, "recommendations.parquet", id="una franja menos"),
        pytest.param(_another_fallback, "recommendations.parquet", id="otra ocupación de respaldo"),
    ],
)
def test_si_cambia_operacion_yaml_y_no_se_regenera_la_prueba_falla(change, stale_file, tmp_path, monkeypatch):
    """Simula que alguien edita config/operacion.yaml y no vuelve a correr el script."""
    cfg = _changed_config(change)
    monkeypatch.setattr(sample_data, "load_operacion", lambda: cfg)
    with pytest.raises(pytest.fail.Exception) as failure:
        _fail_if_outdated(DATA_SAMPLE, tmp_path)
    message = str(failure.value)
    assert stale_file in message
    assert f"Para regenerarlos: {COMMAND}" in message


def test_si_cambia_el_generador_y_no_se_regenera_la_prueba_falla(tmp_path, monkeypatch):
    monkeypatch.setattr(sample_data, "SEED", sample_data.SEED + 1)  # cualquier cambio que mueva los números
    with pytest.raises(pytest.fail.Exception) as failure:
        _fail_if_outdated(DATA_SAMPLE, tmp_path)
    message = str(failure.value)
    for name in ("stations", "station_hour", "predictions", "recommendations", "routes"):
        assert f"{name}.parquet" in message
    assert f"Para regenerarlos: {COMMAND}" in message


def test_si_falta_un_archivo_de_la_muestra_la_prueba_falla(tmp_path):
    empty = tmp_path / "muestra_vacia"
    empty.mkdir()
    with pytest.raises(pytest.fail.Exception, match="LEEME.md"):
        _fail_if_outdated(empty, tmp_path / "recien_generada")


# --------------------------------------------------------------------------- #
# 3) Se distinguen como sintéticos
# --------------------------------------------------------------------------- #

def test_las_estaciones_se_distinguen_como_sinteticas(tables):
    assert tables["stations"]["name"].str.startswith("Sintética").all()
    for name in ("stations", "station_hour", "predictions", "recommendations"):
        assert tables[name]["station_id"].str.startswith("SIM-").all(), name
    for column in ("start_station_id", "end_station_id"):
        assert tables["routes"][column].str.startswith("SIM-").all(), column


# --------------------------------------------------------------------------- #
# 4) feasible es de la estación y la franja
# --------------------------------------------------------------------------- #

def test_feasible_vale_lo_mismo_en_las_tres_politicas(tables):
    """El contrato lo define así: falso cuando ningún inventario inicial evita a la vez vaciarse y llenarse.

    Eso depende de la estación y de la franja, no de la política ni de si se movieron bicis.
    """
    recommendations = tables["recommendations"]
    per_window = recommendations.groupby(["station_id", "window_start"])["feasible"]
    assert (per_window.size() == len(POLICIES)).all()
    assert (per_window.nunique() == 1).all(), "feasible cambia entre políticas en la misma estación y franja"

    infeasible = recommendations[~recommendations["feasible"]]
    assert 0 < len(infeasible) < 0.2 * len(recommendations)  # "unas pocas", como dice el LEEME
    assert set(infeasible["policy"]) == set(POLICIES)        # también con "none", que no mueve nada
    assert (infeasible["bikes_to_move"] == 0).any()


# --------------------------------------------------------------------------- #
# 5) ocupacion_respaldo: inventario inicial común y valor sin solución factible
# --------------------------------------------------------------------------- #

def _fallback_inventory(tables: dict[str, pd.DataFrame], cfg: dict) -> pd.Series:
    """Inventario de respaldo de cada fila de recommendations, calculado aquí aparte.

    Es la ocupacion_respaldo de la franja (según el tipo de la estación) por la
    capacidad, redondeado. Sin capacidad o sin tipo se usan los valores que el
    generador supone y que el LEEME de la muestra declara.
    """
    stations = tables["stations"].set_index("station_id")
    recommendations = tables["recommendations"]
    window_by_start = {window["inicio"]: window["nombre"] for window in cfg["franjas"]}
    window = recommendations["window_start"].dt.strftime("%H:%M").map(window_by_start)
    assert window.notna().all(), "hay recomendaciones cuya franja no está en config/operacion.yaml"
    kind = recommendations["station_id"].map(stations["station_type"])
    capacity = recommendations["station_id"].map(stations["capacity"]).fillna(DEFAULT_CAPACITY).astype(int)
    rules = cfg["estrategia_por_tipo"]
    occupancy = [
        rules[k]["ocupacion_respaldo"][w] if isinstance(k, str) else DEFAULT_OCCUPANCY for k, w in zip(kind, window)
    ]
    return pd.Series(np.rint(capacity * occupancy).astype(int), index=recommendations.index)


def _assert_fallback_is_used_as_decided(tables: dict[str, pd.DataFrame], cfg: dict) -> None:
    recommendations = tables["recommendations"]
    fallback = _fallback_inventory(tables, cfg)
    assert fallback.nunique() > 5  # no es un solo número: cambia con la capacidad, el tipo y la franja

    # Inventario inicial común: antes de mover bicis, las tres políticas tienen el de respaldo.
    before_moving = recommendations["start_inventory_target"] - recommendations["bikes_to_move"]
    assert (before_moving == fallback).all()
    untouched = recommendations["policy"] == "none"
    assert (recommendations.loc[untouched, "start_inventory_target"] == fallback[untouched]).all()

    # Sin solución factible: las tres políticas se quedan con el de respaldo y no mueven nada.
    infeasible = ~recommendations["feasible"]
    assert set(recommendations.loc[infeasible, "policy"]) == set(POLICIES)
    assert (recommendations.loc[infeasible, "start_inventory_target"] == fallback[infeasible]).all()
    assert (recommendations.loc[infeasible, "bikes_to_move"] == 0).all()


def test_ocupacion_respaldo_es_el_inventario_inicial_comun_y_el_valor_sin_solucion(tables):
    _assert_fallback_is_used_as_decided(tables, load_operacion())


def test_el_inventario_de_respaldo_sale_del_yaml_y_no_del_codigo(tables):
    """Con otra ocupacion_respaldo en la configuración, el inventario inicial cambia con ella."""
    cfg = _changed_config(_another_fallback)
    changed = build_sample(cfg)
    _assert_fallback_is_used_as_decided(changed, cfg)
    original, moved = _fallback_inventory(tables, load_operacion()), _fallback_inventory(changed, cfg)
    assert len(original) == len(moved) and (original != moved).mean() > 0.8


def test_la_regla_historica_no_busca_el_inventario_de_respaldo(tables):
    """`historical` no usa ocupacion_respaldo: donde hay solución factible casi nunca coincide con él.

    Antes de la decisión, la muestra ponía a la regla histórica a buscar justo
    ocupacion_respaldo por capacidad. Si eso volviera, coincidirían casi todas.
    """
    recommendations = tables["recommendations"]
    fallback = _fallback_inventory(tables, load_operacion())
    rows = (recommendations["policy"] == "historical") & recommendations["feasible"]
    same = recommendations.loc[rows, "start_inventory_target"] == fallback[rows]
    assert rows.sum() > 100 and same.mean() < 0.25


# --------------------------------------------------------------------------- #
# 6) station_hour es densa y hay estaciones que no cubren todo el periodo
# --------------------------------------------------------------------------- #

def _local_day(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_localize(None).dt.normalize()


def test_station_hour_es_densa_entre_el_primer_y_el_ultimo_viaje(tables):
    """La regla de tabla densa, revisada aquí a mano con pandas (sin pasar por el validador del contrato)."""
    station_hour = tables["station_hour"]
    trips = station_hour["departures"] + station_hour["arrivals"]
    for station_id, rows in station_hour.groupby("station_id"):
        every_hour = pd.date_range(rows["ts_hour"].min(), rows["ts_hour"].max(), freq="h")
        assert rows["ts_hour"].tolist() == every_hour.tolist(), f"{station_id}: faltan horas o están desordenadas"
        assert trips[rows.index[0]] > 0, f"{station_id}: empieza con una fila sin viajes"
        assert trips[rows.index[-1]] > 0, f"{station_id}: termina con una fila sin viajes"
    assert (trips == 0).sum() > 1000  # las horas en cero de en medio sí están

    # Ya no es una malla completa: cada estación empieza y termina cuando tuvo viajes.
    per_station = station_hour.groupby("station_id").size()
    hours = pd.date_range(START, END, freq="h", tz=TZ, inclusive="left")
    assert per_station.nunique() > 1 and per_station.max() <= len(hours)
    assert len(station_hour) < len(per_station) * len(hours)
    # Y el validador del contrato dice lo mismo.
    assert validate_table(to_contract_table(station_hour, "station_hour"), "station_hour") == []


def test_la_malla_completa_ya_no_cumple_el_contrato(tables):
    """Si el generador dejara de recortar, la validación lo diría: se rellena la malla y se valida."""
    station_hour = tables["station_hour"]
    hours = pd.date_range(START, END, freq="h", tz=TZ, inclusive="left")
    grid = pd.MultiIndex.from_product([tables["stations"]["station_id"], hours], names=["station_id", "ts_hour"])
    full = station_hour.set_index(["station_id", "ts_hour"]).reindex(grid, fill_value=0).reset_index()
    assert len(full) == len(grid) > len(station_hour)
    assert full["departures"].sum() == station_hour["departures"].sum()  # mismos viajes, más filas en cero

    problems = validate_table(to_contract_table(full, "station_hour"), "station_hour")
    assert len(problems) == 1, problems
    assert "con filas en cero antes de su primer viaje o después del último" in problems[0]


def test_hay_estaciones_que_no_cubren_todo_el_periodo(tables):
    station_hour = tables["station_hour"]
    span = station_hour.groupby("station_id")["ts_hour"].agg(["min", "max"])
    first_day, last_day = _local_day(span["min"]), _local_day(span["max"])
    period_start, period_end = pd.Timestamp(START), pd.Timestamp(END) - pd.Timedelta(days=1)

    several_days = pd.Timedelta(days=3)
    starts_late = first_day >= period_start + several_days
    ends_early = last_day <= period_end - several_days
    assert starts_late.any(), "ninguna estación empieza a medio periodo"
    assert ends_early.any(), "ninguna estación termina antes del final"
    assert 2 <= (starts_late | ends_early).sum() <= 3
    # Las demás van del primer al último día.
    rest = ~(starts_late | ends_early)
    assert (first_day[rest] == period_start).all() and (last_day[rest] == period_end).all()

    # Aun así, todas pasan por el día de 25 horas y lo traen completo.
    long_day = station_hour[_local_day(station_hour["ts_hour"]) == DST_DAY.tz_localize(None)]
    per_station = long_day.groupby("station_id").size()
    assert len(per_station) == len(tables["stations"]) and (per_station == 25).all()


def test_fuera_de_su_tramo_una_estacion_no_tiene_pronostico_ni_recomendaciones(tables):
    station_hour, predictions, recommendations = (
        tables[name] for name in ("station_hour", "predictions", "recommendations")
    )
    known = pd.MultiIndex.from_frame(station_hour[["station_id", "ts_hour"]])

    forecast = pd.MultiIndex.from_frame(predictions[["station_id", "ts_hour"]])
    assert forecast.isin(known).all(), "hay pronóstico para horas sin fila en station_hour"
    # Y al revés: toda fila de station_hour de validación y prueba tiene su pronóstico, en cada modelo.
    evaluated = station_hour[station_hour["ts_hour"] >= predictions["ts_hour"].min()]
    evaluated = pd.MultiIndex.from_frame(evaluated[["station_id", "ts_hour"]])
    assert set(predictions["model"]) == {"baseline", "lgbm"}
    for model, rows in predictions.groupby("model"):
        with_forecast = pd.MultiIndex.from_frame(rows[["station_id", "ts_hour"]])
        assert evaluated.isin(with_forecast).all(), f"faltan pronósticos del modelo {model}"

    windows = pd.MultiIndex.from_frame(
        recommendations[["station_id", "window_start"]].rename(columns={"window_start": "ts_hour"})
    )
    assert windows.isin(known).all(), "hay recomendaciones para franjas en que la estación no tiene filas"
    # Las estaciones que terminan antes tienen menos recomendaciones que las demás.
    per_station = recommendations.groupby("station_id").size()
    assert per_station.min() < per_station.max()
    assert set(per_station.index) == set(tables["stations"]["station_id"])


# --------------------------------------------------------------------------- #
# 7) Las rutas cuadran con station_hour
# --------------------------------------------------------------------------- #

def test_las_rutas_suman_las_salidas_y_las_llegadas_de_station_hour(tables):
    """Cuadra exacto por los dos lados, estación por estación."""
    routes = tables["routes"]
    totals = tables["station_hour"].groupby("station_id")[["departures", "arrivals"]].sum()
    leaving = routes.groupby("start_station_id")["trips"].sum().reindex(totals.index, fill_value=0)
    arriving = routes.groupby("end_station_id")["trips"].sum().reindex(totals.index, fill_value=0)

    assert (leaving == totals["departures"]).all(), "lo que sale por rutas no suma las salidas de station_hour"
    assert (arriving == totals["arrivals"]).all(), "lo que llega por rutas no suma las llegadas de station_hour"
    assert routes["trips"].sum() == totals["departures"].sum() == totals["arrivals"].sum()
    # Ninguna ruta usa una estación que no esté en station_hour.
    assert set(routes["start_station_id"]) | set(routes["end_station_id"]) <= set(totals.index)


def test_cada_viaje_inventado_llega_en_la_hora_en_que_sale(tables):
    """Lo dice el LEEME de la muestra: en cada hora, todas las llegadas suman lo mismo que todas las salidas."""
    per_hour = tables["station_hour"].groupby("ts_hour")[["departures", "arrivals"]].sum()
    assert (per_hour["departures"] == per_hour["arrivals"]).all()
    assert (per_hour["departures"] > 0).mean() > 0.9  # no es que todo valga cero


def test_las_rutas_son_pocas_y_tienen_forma_creible(tables):
    stations, routes = tables["stations"].set_index("station_id"), tables["routes"]
    assert (routes["trips"] >= 1).all()
    assert not routes.duplicated(["start_station_id", "end_station_id"]).any()

    same_station = routes["start_station_id"] == routes["end_station_id"]
    assert same_station.any(), "no hay viajes redondos (misma estación de inicio y de fin)"
    possible = len(stations) ** 2
    assert len(routes) < 0.9 * possible, "existen casi todos los pares: deben faltar los de estaciones lejanas"

    # Más viajes entre estaciones cercanas: distancia aproximada en kilómetros, en línea recta.
    start = stations.loc[routes["start_station_id"], ["lat", "lon"]].to_numpy()
    end = stations.loc[routes["end_station_id"], ["lat", "lon"]].to_numpy()
    km = np.hypot((start[:, 0] - end[:, 0]) * 111.0, (start[:, 1] - end[:, 1]) * 84.0)
    near = routes.loc[~same_station & (km < 1.0), "trips"]
    far = routes.loc[km > 3.0, "trips"]
    assert len(near) > 20 and len(far) > 20 and near.mean() > 2 * far.mean()

    # Más viajes entre una estación residencial y una de oficinas que entre otros pares igual de lejanos.
    kinds = stations["station_type"]
    pair = set(zip(kinds[routes["start_station_id"]], kinds[routes["end_station_id"]]))
    assert {("residencial", "oficinas"), ("oficinas", "residencial")} <= pair
    start_kind = kinds[routes["start_station_id"]].to_numpy()
    end_kind = kinds[routes["end_station_id"]].to_numpy()
    commute = ((start_kind == "residencial") & (end_kind == "oficinas")) | (
        (start_kind == "oficinas") & (end_kind == "residencial")
    )
    for low, high in ((2.0, 3.0), (3.0, 4.5), (4.5, 6.0)):
        band = (km > low) & (km <= high)
        assert (band & commute).sum() > 20 and (band & ~commute).sum() > 20
        assert routes.loc[band & commute, "trips"].mean() > routes.loc[band & ~commute, "trips"].mean(), (low, high)


# --------------------------------------------------------------------------- #
# 8) No escribe donde van los datos reales
# --------------------------------------------------------------------------- #

@pytest.fixture
def no_writing(monkeypatch):
    """Por si la negativa dejara de funcionar: que la prueba no pueda escribir nada."""

    def forbidden(*args, **kwargs):
        raise AssertionError("el generador siguió adelante en vez de negarse")

    for step in ("build_sample", "write_contract", "write_leeme"):
        monkeypatch.setattr(sample_data, step, forbidden)


def _files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


@pytest.mark.parametrize("folder", [DATA_PROCESSED, DATA_RAW], ids=["data/processed", "data/raw"])
def test_el_generador_se_niega_a_escribir_en_las_carpetas_de_datos_reales(folder, no_writing):
    assert folder.is_dir()  # ya existe: aunque la negativa fallara, la prueba no crearía nada
    with pytest.raises(ValueError, match=REFUSAL):
        write_sample(folder)
    # Desde la línea de comandos: termina con el mensaje y código de salida distinto de cero.
    with pytest.raises(SystemExit) as exit_info:
        main(["--salida", str(folder)])
    assert isinstance(exit_info.value.code, str) and REFUSAL in exit_info.value.code


def test_tampoco_escribe_dentro_de_esas_carpetas_ni_llegando_por_otra_ruta(tmp_path, monkeypatch, no_writing):
    """Se prueba con carpetas de mentira, para no acercarse a las reales."""
    processed, raw = tmp_path / "data" / "processed", tmp_path / "data" / "raw"
    processed.mkdir(parents=True)
    raw.mkdir()
    monkeypatch.setattr(sample_data, "DATA_PROCESSED", processed)
    monkeypatch.setattr(sample_data, "DATA_RAW", raw)
    monkeypatch.chdir(tmp_path)  # para probar rutas relativas, como las de --salida

    refused = [
        processed / "sub" / "carpeta",
        raw / "gbfs",
        Path("data/processed"),
        Path("data/raw/x"),
        tmp_path / "app" / ".." / "data" / "processed",
    ]
    shouting = tmp_path / "data" / "PROCESSED"
    if shouting.exists():  # macOS y Windows: es la misma carpeta con otra capitalización
        refused.append(shouting)
    for out_dir in refused:
        with pytest.raises(ValueError, match=REFUSAL):
            check_output_folder(out_dir)
    with pytest.raises(ValueError, match=REFUSAL):
        write_sample(processed / "sub" / "carpeta")
    with pytest.raises(SystemExit, match=REFUSAL):
        main(["--salida", "data/raw/x"])
    assert not (processed / "sub").exists() and not (raw / "x").exists()  # se niega antes de crear nada
    assert _files(tmp_path) == []

    # Cualquier otra carpeta se acepta, también una vecina de nombre parecido.
    for out_dir in (tmp_path / "salida", tmp_path / "data" / "sample", tmp_path / "data" / "processed_copia"):
        assert check_output_folder(out_dir) == out_dir
