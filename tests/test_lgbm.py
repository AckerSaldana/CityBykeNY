"""Pruebas del modelo LightGBM (src/models/lgbm.py) y de sus variables.

Qué revisan, con tablas pequeñas hechas a mano y un config con tramos cortos:

a) Las variables: salen de constructores registrados (los revisa
   tests/test_no_leakage.py), sus valores son los del cálculo a mano y las
   variables fijas de la estación usan solo el tramo de ajuste.
b) Ajustar y predecir corre, y la salida cumple el contrato predictions: una
   fila por cada fila de station_hour del tramo (las mismas llaves que la
   línea base), sin vacíos, sin negativos y con q10 <= q50 <= q90.
c) La prueba final: sus filas se escriben sin leer ningún objetivo de ese
   tramo; el modelo ajustado no cambia si cambian sus valores; el pronóstico
   de un día no depende de ese día ni de los siguientes; validación no ve la
   prueba final; y nada la califica.
d) Al guardar se conservan las filas de la línea base.
e) Dos corridas dan lo mismo.
f) El script de punta a punta y la revisión de que las salidas siguen al día.

Todo corre sin red y sin tocar data/processed. La última prueba mira los
archivos reales y se salta si todavía no existen.
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.contracts import PREDICTIONS, QUANTILE_COLUMNS, read_contract, to_contract_table, validate_file, validate_table, write_contract
from src.data.datos_config import load_datos
from src.features.estaticas import STATIC_COLUMNS, station_statics
from src.features.historia import CALENDAR_COLUMNS, LAG_COLUMNS, TOTAL_COLUMNS, calendario, rezagos_misma_hora, totales_dias_previos
from src.features.registry import load_builders
from src.models import baseline, lgbm
from src.models.baseline import input_fingerprint
from src.models.lgbm import (
    BOOSTER_NAMES,
    FEATURES,
    FINGERPRINT_NAME,
    MODEL,
    LgbmError,
    feature_matrix,
    main,
    pinball_loss,
    postprocess,
    predict_all,
    predict_split,
    quantile_scores,
    save_predictions,
)
from src.models.metrics import METRIC_COLUMNS, FinalTestError, evaluate
from src.paths import DATA_PROCESSED, DATOS_YAML, TZ

KEYS = ["station_id", "ts_hour"]

# Tres semanas de entrenamiento y una de validación y de prueba. El 1 de junio de 2026 es lunes.
SHORT = (("2026-06-01", "2026-06-21"), ("2026-06-22", "2026-06-28"), ("2026-06-29", "2026-07-05"))
# Hiperparámetros para tablas de unos miles de filas: árboles chicos y pocas rondas.
TINY = {"num_leaves": 7, "min_data_in_leaf": 5, "learning_rate": 0.2, "max_bin": 31}
THREADS = 2


# --------------------------------------------------------------------------- #
# Ayudas
# --------------------------------------------------------------------------- #

def write_config(folder, tramos=SHORT) -> str:
    train, val, test = tramos
    path = folder / "datos.yaml"
    meses = ", ".join(pd.period_range(train[0], test[1], freq="M").strftime("%Y%m"))
    path.write_text(
        f"meses_crudos: [{meses}]\n"
        f"periodo: {{inicio: {train[0]}, fin: {test[1]}}}\n"
        "tramos:\n"
        f"  entrenamiento: {{inicio: {train[0]}, fin: {train[1]}}}\n"
        f"  validacion: {{inicio: {val[0]}, fin: {val[1]}}}\n"
        f"  prueba: {{inicio: {test[0]}, fin: {test[1]}}}\n"
        "union_de_ids: {distancia_maxima_m: 1.0}\n"
        "estaciones_excluidas: {prefijos: [], ids: []}\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def datos(tmp_path):
    return load_datos(write_config(tmp_path))


def hourly(station: str, first: str, last: str, departures=1, arrivals=2) -> pd.DataFrame:
    """Una fila por hora de `first` al último instante de `last`, hora de Nueva York."""
    first = first if " " in first else first + " 00:00"
    last = last if " " in last else last + " 23:00"
    ts = pd.date_range(first, last, freq="h", tz=TZ).as_unit("us")
    return pd.DataFrame({"station_id": station, "ts_hour": ts, "departures": departures, "arrivals": arrivals})


def world(seed: int = 7) -> pd.DataFrame:
    """Cinco estaciones con un patrón por hora y ruido. Una empieza en validación (sin historia de ajuste)."""
    rng = np.random.default_rng(seed)
    first, last = SHORT[0][0], SHORT[2][1]
    parts = []
    for i, station in enumerate(("5470.10", "5470.1", "6289.06", "4327.09")):
        part = hourly(station, first, last)
        hour = part["ts_hour"].dt.hour.to_numpy()
        level = (i + 1) * (1 + np.sin(hour / 24 * 2 * np.pi) ** 2 * 3)
        part["departures"] = rng.poisson(level)
        part["arrivals"] = rng.poisson(level[::-1])
        parts.append(part)
    late = hourly("NUEVA", "2026-06-24", last)
    late["departures"] = rng.poisson(1.0, len(late))
    late["arrivals"] = rng.poisson(1.0, len(late))
    return pd.concat([*parts, late], ignore_index=True)


def stations_of(table: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(table["station_id"].unique())
    return pd.DataFrame(
        {
            "station_id": ids,
            "name": [f"Estación {i}" for i in ids],
            "lat": 40.7 + 0.01 * np.arange(len(ids)),
            "lon": -74.0 + 0.01 * np.arange(len(ids)),
            "capacity": pd.array([20, 30, None, 40, 25][: len(ids)], dtype="Int64"),
            "station_type": pd.array([None] * len(ids), dtype="str"),
        }
    )


def as_station_hour(table: pd.DataFrame) -> pd.DataFrame:
    """Completa las columnas del contrato station_hour para poder guardar la tabla."""
    table = table.copy()
    edges = ~table.duplicated("station_id", keep="first") | ~table.duplicated("station_id", keep="last")
    table.loc[edges, "departures"] = table.loc[edges, "departures"].clip(lower=1)
    table["net_flow"] = table["arrivals"] - table["departures"]
    table["departures_member"] = table["departures"]
    table["departures_casual"] = 0
    return table


def changed(table: pd.DataFrame, rows) -> pd.DataFrame:
    """Copia con salidas y llegadas muy distintas en las filas marcadas."""
    out = table.copy()
    rows = np.asarray(rows)
    out.loc[rows, "departures"] = out.loc[rows, "departures"] * 7 + 50
    out.loc[rows, "arrivals"] = out.loc[rows, "arrivals"] * 5 + 80
    return out


def at(table: pd.DataFrame, station: str, when: str) -> pd.Series:
    row = table[(table["station_id"] == station) & (table["ts_hour"] == pd.Timestamp(when, tz=TZ))]
    assert len(row) == 1, f"se esperaba una fila de {station} en {when}"
    return row.iloc[0]


def fit_all(table, datos):
    return predict_all(table, stations_of(table), datos, params=TINY, num_threads=THREADS)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Una corrida completa (validación y prueba final) sobre el mundo pequeño. Varias pruebas la comparten."""
    datos = load_datos(write_config(tmp_path_factory.mktemp("config")))
    table = world()
    predictions, summaries = fit_all(table, datos)
    return table, datos, predictions, summaries


# --------------------------------------------------------------------------- #
# a) Variables
# --------------------------------------------------------------------------- #

def test_las_variables_de_station_hour_salen_de_constructores_registrados():
    builders = load_builders()
    for name, function in (
        ("rezagos_misma_hora", rezagos_misma_hora),
        ("totales_dias_previos", totales_dias_previos),
        ("calendario", calendario),
    ):
        assert builders[name] is function, f"{name} no está registrado: tests/test_no_leakage.py no lo revisaría"
    assert FEATURES == LAG_COLUMNS + TOTAL_COLUMNS + CALENDAR_COLUMNS + STATIC_COLUMNS
    assert len(set(FEATURES)) == len(FEATURES)


def test_los_rezagos_son_los_del_calculo_a_mano():
    # departures = horas desde el inicio, así cada valor dice de qué hora salió.
    table = hourly("A", "2026-06-01", "2026-07-05")
    table["departures"] = np.arange(len(table))
    table["arrivals"] = 3 * table["departures"] + 1
    other = hourly("B", "2026-06-01", "2026-07-05", departures=1000, arrivals=1000)
    features = rezagos_misma_hora(pd.concat([other, table], ignore_index=True).sample(frac=1.0, random_state=1))

    row = at(features, "A", "2026-06-30 08:00")  # día 29 de la tabla (cuenta desde 0), hora 8
    now = 29 * 24 + 8
    previous = np.array([now - 24 * days for days in range(1, 29)], dtype=float)
    assert row["dep_rez_1d"] == now - 24 and row["dep_rez_2d"] == now - 48
    assert row["dep_rez_7d"] == now - 168 and row["dep_rez_14d"] == now - 336
    assert row["arr_rez_1d"] == 3 * (now - 24) + 1
    assert row["dep_media_7d"] == previous[:7].mean() and row["dep_media_28d"] == previous.mean()
    assert row["dep_q10_28d"] == pytest.approx(np.quantile(previous, 0.10))
    assert row["dep_q90_28d"] == pytest.approx(np.quantile(previous, 0.90))
    same_weekday = previous[[6, 13, 20, 27]]
    assert row["dep_media_dia_4s"] == same_weekday.mean()
    assert row["dep_min_dia_4s"] == same_weekday.min() and row["dep_max_dia_4s"] == same_weekday.max()
    assert row["dias_con_historia_28d"] == 28

    # Tercer día: solo hay dos días previos. Lo que no existe queda vacío y los resúmenes usan lo que hay.
    early = at(features, "A", "2026-06-03 05:00")
    assert early["dep_rez_1d"] == 29 and early["dep_rez_2d"] == 5
    assert np.isnan(early["dep_rez_7d"]) and np.isnan(early["dep_media_dia_4s"]) and np.isnan(early["dep_min_dia_4s"])
    assert early["dep_media_28d"] == 17 and early["dias_con_historia_28d"] == 2
    assert early["dep_q10_28d"] == pytest.approx(5 + 0.1 * 24)
    first = at(features, "A", "2026-06-01 05:00")
    assert first[LAG_COLUMNS[:-1]].isna().all() and first["dias_con_historia_28d"] == 0


def test_los_totales_son_los_de_los_dias_de_calendario_previos():
    table = hourly("A", "2026-06-01", "2026-06-12", departures=1, arrivals=2)
    table.loc[table["ts_hour"].dt.day == 9, "departures"] = 5   # el día 9 tiene 120 salidas
    table = table[~((table["ts_hour"].dt.day == 10) & (table["ts_hour"].dt.hour >= 12))]  # el día 10, solo 12 filas
    features = totales_dias_previos(table.reset_index(drop=True))
    assert at(features, "A", "2026-06-10 03:00")["dep_total_1d"] == 120
    assert at(features, "A", "2026-06-10 03:00")["arr_total_1d"] == 48
    assert at(features, "A", "2026-06-11 20:00")["dep_total_1d"] == 12      # suma de las filas que hay
    assert at(features, "A", "2026-06-11 20:00")["dep_total_7d"] == 24
    assert at(features, "A", "2026-06-11 20:00")["dep_total_media_7d"] == pytest.approx((12 + 120 + 24 * 5) / 7)
    assert np.isnan(at(features, "A", "2026-06-01 00:00")["dep_total_1d"])
    assert np.isnan(at(features, "A", "2026-06-05 00:00")["dep_total_7d"])


def test_el_dia_de_25_horas_usa_el_promedio_de_la_hora_repetida_y_su_total_completo():
    hours = pd.date_range("2025-10-30", "2025-11-05", freq="h", tz=TZ, inclusive="left").as_unit("us")
    table = pd.DataFrame({"station_id": "A", "ts_hour": hours, "departures": np.arange(len(hours)), "arrivals": 0})
    first = table.loc[table["ts_hour"] == pd.Timestamp("2025-11-02 01:00-04:00"), "departures"].iloc[0]
    lags, totals = rezagos_misma_hora(table), totales_dias_previos(table)
    monday = lags[lags["ts_hour"] == pd.Timestamp("2025-11-03 01:00", tz=TZ)].iloc[0]
    assert monday["dep_rez_1d"] == first + 0.5                       # promedio de las dos 01:00
    sunday = table[table["ts_hour"].dt.day == 2]
    assert len(sunday) == 25
    assert totals[totals["ts_hour"] == pd.Timestamp("2025-11-03 10:00", tz=TZ)].iloc[0]["dep_total_1d"] == sunday["departures"].sum()


def test_el_calendario_marca_fin_de_semana_y_festivo_federal():
    table = hourly("A", "2026-07-02", "2026-07-06")
    features = calendario(table)
    friday = at(features, "A", "2026-07-03 09:00")     # Independence Day observado (el 4 cae en sábado)
    assert (friday["hora"], friday["dia_semana"], friday["fin_de_semana"], friday["festivo"]) == (9, 4, 0, 1)
    saturday = at(features, "A", "2026-07-04 23:00")
    assert (saturday["hora"], saturday["dia_semana"], saturday["fin_de_semana"]) == (23, 5, 1)
    assert at(features, "A", "2026-07-06 00:00")[["fin_de_semana", "festivo"]].sum() == 0


def test_las_variables_fijas_usan_solo_el_tramo_de_ajuste(datos):
    table = world()
    stations = stations_of(table)
    in_train = datos.tramos["entrenamiento"].contains(table["ts_hour"]).to_numpy()
    statics = station_statics(table, table[in_train], stations)
    assert list(statics.columns) == STATIC_COLUMNS and len(statics) == len(table)

    # Cálculo a mano: estación 6289.06, hora 08, de lunes a viernes, solo entrenamiento.
    local = table["ts_hour"].dt.tz_localize(None)
    cell = in_train & (table["station_id"] == "6289.06") & (local.dt.hour == 8)
    row = statics[(table["station_id"] == "6289.06") & (table["ts_hour"] == pd.Timestamp("2026-07-01 08:00", tz=TZ))].iloc[0]
    assert row["dep_media_estacion_hora"] == pytest.approx(table.loc[cell, "departures"].mean())
    assert row["arr_media_estacion_tipo_hora"] == pytest.approx(table.loc[cell & (local.dt.dayofweek < 5), "arrivals"].mean())
    assert row["dep_media_estacion"] == pytest.approx(table.loc[in_train & (table["station_id"] == "6289.06"), "departures"].mean())
    assert row["capacidad"] == 40 and row["lat"] == pytest.approx(40.73)

    # Cambiar todo lo que no es entrenamiento no mueve ninguna variable fija.
    again = station_statics(changed(table, ~in_train), changed(table, ~in_train)[in_train], stations)
    pd.testing.assert_frame_equal(statics, again)

    # Una estación sin filas de ajuste queda con promedios vacíos; la capacidad nula queda vacía.
    new = statics[(table["station_id"] == "NUEVA").to_numpy()]
    assert new[[c for c in STATIC_COLUMNS if "media" in c]].isna().all().all()
    assert statics.loc[(table["station_id"] == "5470.10").to_numpy(), "capacidad"].isna().all()


def test_la_matriz_de_variables_respeta_el_orden_de_las_filas(datos):
    table = world().sample(frac=1.0, random_state=3)  # índice desordenado
    history = datos.tramos["entrenamiento"].contains(table["ts_hour"]).to_numpy()
    x = feature_matrix(table, history, stations_of(table))
    assert list(x.columns) == FEATURES and len(x) == len(table) and (x.dtypes == np.float32).all()
    np.testing.assert_array_equal(x["hora"].to_numpy(), table["ts_hour"].dt.hour.to_numpy())
    np.testing.assert_array_equal(x["dep_rez_1d"].to_numpy(), rezagos_misma_hora(table)["dep_rez_1d"].to_numpy(dtype=np.float32))


# --------------------------------------------------------------------------- #
# b) Ajustar y predecir: contrato
# --------------------------------------------------------------------------- #

def test_las_predicciones_cumplen_el_contrato_y_cubren_las_filas_de_cada_tramo(fitted):
    table, datos, predictions, summaries = fitted
    assert list(predictions.columns) == PREDICTIONS.columns
    assert validate_table(to_contract_table(predictions, "predictions"), "predictions") == []
    assert set(predictions["model"]) == {MODEL} and set(predictions["split"]) == {"val", "test"}

    values = predictions[list(QUANTILE_COLUMNS)].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    for prefix in ("dep", "arr"):
        assert (predictions[f"{prefix}_q10"] <= predictions[f"{prefix}_q50"]).all()
        assert (predictions[f"{prefix}_q50"] <= predictions[f"{prefix}_q90"]).all()

    # Las mismas llaves que la línea base: una fila por cada fila de station_hour del tramo.
    base, _ = baseline.predict_all(table, datos)
    for split, name in (("val", "validacion"), ("test", "prueba")):
        ours = predictions.loc[predictions["split"] == split, KEYS].reset_index(drop=True)
        theirs = base.loc[base["split"] == split, KEYS].reset_index(drop=True)
        pd.testing.assert_frame_equal(ours, theirs)
        assert len(ours) == datos.tramos[name].contains(table["ts_hour"]).sum() == summaries[split].rows
    assert "NUEVA" in set(predictions["station_id"])  # la estación sin historia de ajuste también se predice

    assert set(summaries["val"].rounds) == set(BOOSTER_NAMES)
    assert all(1 <= rounds <= lgbm.MAX_ROUNDS for rounds in summaries["val"].rounds.values())
    assert summaries["test"].rounds == summaries["val"].rounds  # no se vuelven a elegir
    assert summaries["test"].history_rows > summaries["val"].history_rows  # reajuste con entrenamiento más validación


def test_el_modelo_aprende_el_patron_por_hora(fitted):
    """No es una prueba de calidad: solo que el pronóstico no es una constante y sigue a los datos."""
    table, datos, predictions, _ = fitted
    actual = table[datos.tramos["validacion"].contains(table["ts_hour"])]
    metrics = evaluate(predictions, actual, model=MODEL, split="val", prueba_final=datos.tramos["prueba"])
    general = metrics[metrics["group_by"] == "general"].set_index("target")
    constant = (actual["departures"] - actual["departures"].median()).abs().mean()
    assert general.loc["dep", "mae"] < constant
    assert 0.5 < general.loc["dep", "coverage"] <= 1.0


def test_postprocess_recorta_en_cero_y_ordena_los_cuantiles():
    raw = {name: np.array([1.0, 2.0, 3.0]) for name in BOOSTER_NAMES}
    raw["dep_q10"] = np.array([-0.5, 5.0, 1.0])   # negativo; cruzado con q50 y q90; bien
    raw["dep_q50"] = np.array([-0.2, 2.0, 2.0])
    raw["dep_q90"] = np.array([0.3, 4.0, 3.0])
    fixed, negatives, crossed = postprocess(raw)
    np.testing.assert_array_equal(fixed["dep_q10"], [0.0, 2.0, 1.0])
    np.testing.assert_array_equal(fixed["dep_q50"], [0.0, 4.0, 2.0])
    np.testing.assert_array_equal(fixed["dep_q90"], [0.3, 5.0, 3.0])
    assert negatives["dep_q10"] == 1 and negatives["dep_q50"] == 1 and negatives["dep_q90"] == 0
    assert crossed == {"dep": 1, "arr": 0}   # la fila negativa queda ordenada con solo recortar


def test_la_perdida_de_cuantil_es_la_del_calculo_a_mano():
    assert pinball_loss([10, 0], [6, 2], 0.9) == pytest.approx((0.9 * 4 + 0.1 * 2) / 2)
    assert pinball_loss([10, 0], [6, 2], 0.5) == pytest.approx(0.5 * 3)


# --------------------------------------------------------------------------- #
# c) La prueba final no se lee como objetivo ni se califica
# --------------------------------------------------------------------------- #

def test_los_objetivos_nunca_salen_de_la_prueba_final(monkeypatch, datos):
    """Espía: _labels es el único lugar donde el modelo lee objetivos. Se anota de qué fechas los pidió."""
    table = world()
    original, seen = lgbm._labels, []

    def spy(station_hour, rows):
        seen.append(station_hour.loc[rows, "ts_hour"].max())
        return original(station_hour, rows)

    monkeypatch.setattr(lgbm, "_labels", spy)
    fit_all(table, datos)
    test_start, val_start = datos.tramos["prueba"].start, datos.tramos["validacion"].start
    # Validación: objetivos de entrenamiento (ajuste) y de validación (parada temprana). Prueba: solo la historia.
    assert len(seen) == 3
    assert seen[0] < val_start and val_start <= seen[1] < test_start
    assert val_start <= seen[2] < test_start
    assert all(latest < test_start for latest in seen)


def test_el_modelo_de_la_prueba_final_no_cambia_si_cambian_los_valores_de_la_prueba(fitted):
    table, datos, _, summaries = fitted
    stations, rounds = stations_of(table), summaries["val"].rounds
    in_test = datos.tramos["prueba"].contains(table["ts_hour"]).to_numpy()
    _, _, boosters = predict_split(table, stations, datos, "test", rounds=rounds, params=TINY, num_threads=THREADS)
    _, _, others = predict_split(changed(table, in_test), stations, datos, "test", rounds=rounds, params=TINY, num_threads=THREADS)
    for name in BOOSTER_NAMES:
        assert boosters[name].model_to_string() == others[name].model_to_string(), f"{name} leyó valores de la prueba final"


def test_el_pronostico_de_un_dia_de_la_prueba_no_depende_de_ese_dia_ni_de_los_siguientes(fitted):
    """Los conteos de días de la prueba ya transcurridos entran como rezagos; los del día y los posteriores, no."""
    table, datos, predictions, summaries = fitted
    stations, rounds = stations_of(table), summaries["val"].rounds
    original = predictions[predictions["split"] == "test"].reset_index(drop=True)
    for day in ("2026-06-29", "2026-07-02", "2026-07-05"):   # primer día, uno intermedio y el último
        cut = pd.Timestamp(day, tz=TZ)
        other, _, _ = predict_split(
            changed(table, table["ts_hour"] >= cut), stations, datos, "test", rounds=rounds, params=TINY, num_threads=THREADS
        )
        until = (original["ts_hour"] < cut + pd.Timedelta(days=1)).to_numpy()
        pd.testing.assert_frame_equal(original[until], other[until])
    # Con el primer día alterado, ningún pronóstico de la prueba cambió hasta ese día: el caso "2026-06-29"
    # de arriba compara todas las filas del primer día, que solo miran entrenamiento y validación.


def test_validacion_no_ve_la_prueba_final(fitted):
    table, datos, predictions, _ = fitted
    stations = stations_of(table)
    val = predictions[predictions["split"] == "val"].reset_index(drop=True)
    in_test = datos.tramos["prueba"].contains(table["ts_hour"]).to_numpy()
    for other_table in (changed(table, in_test), table[~in_test]):
        other, summary, _ = predict_split(other_table, stations, datos, "val", params=TINY, num_threads=THREADS)
        pd.testing.assert_frame_equal(val, other)


def test_la_prueba_final_exige_rondas_ya_fijadas_y_nada_la_califica(fitted):
    table, datos, predictions, _ = fitted
    with pytest.raises(LgbmError, match="rondas deben venir ya fijadas"):
        predict_split(table, stations_of(table), datos, "test", params=TINY)
    final = datos.tramos["prueba"]
    actual = table[final.contains(table["ts_hour"])]
    with pytest.raises(FinalTestError):
        evaluate(predictions, actual, model=MODEL, split="test", prueba_final=final)
    with pytest.raises(FinalTestError):
        quantile_scores(predictions, actual, model=MODEL, split="test", prueba_final=final)
    relabelled = predictions[predictions["split"] == "test"].assign(split="val")  # filas de prueba marcadas "val"
    with pytest.raises(FinalTestError):
        quantile_scores(relabelled, actual, model=MODEL, split="val", prueba_final=final)


# --------------------------------------------------------------------------- #
# d) Guardar conserva la línea base
# --------------------------------------------------------------------------- #

def test_guardar_conserva_las_filas_de_la_linea_base_y_reemplaza_las_de_lgbm(fitted, tmp_path):
    table, datos, predictions, _ = fitted
    path = tmp_path / "predictions.parquet"
    assert save_predictions(predictions, path) == 0          # sin archivo previo
    base, _ = baseline.predict_all(table, datos)
    assert baseline.save_predictions(base, path) == len(predictions)   # la línea base conserva las de lgbm

    moved = predictions.assign(dep_q90=predictions["dep_q90"] + 1)
    assert save_predictions(moved, path) == len(base)        # lgbm conserva las de la línea base
    saved = read_contract("predictions", path)
    assert validate_file(path, "predictions") == []
    assert (saved["model"] == "baseline").sum() == len(base) and (saved["model"] == MODEL).sum() == len(predictions)
    kept = saved[saved["model"] == "baseline"].sort_values(KEYS + ["split"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, base.sort_values(KEYS + ["split"]).reset_index(drop=True), check_dtype=False)
    ours = saved[saved["model"] == MODEL].sort_values(["split"] + KEYS)
    np.testing.assert_allclose(ours["dep_q90"].to_numpy(), moved.sort_values(["split"] + KEYS)["dep_q90"].to_numpy())


# --------------------------------------------------------------------------- #
# e) Dos corridas dan lo mismo
# --------------------------------------------------------------------------- #

def test_dos_corridas_dan_las_mismas_predicciones(fitted):
    table, datos, predictions, summaries = fitted
    again, summaries_again = fit_all(table, datos)
    pd.testing.assert_frame_equal(predictions, again)
    assert summaries_again["val"].rounds == summaries["val"].rounds


# --------------------------------------------------------------------------- #
# f) El script de punta a punta
# --------------------------------------------------------------------------- #

@pytest.fixture
def workspace(tmp_path):
    """Carpeta con station_hour, stations y la línea base ya corrida, y su config."""
    data = tmp_path / "datos"
    table = as_station_hour(world())
    write_contract(table, "station_hour", data / "station_hour.parquet")
    write_contract(stations_of(table), "stations", data / "stations.parquet")
    config_path = write_config(tmp_path)
    assert baseline.main(["--datos", str(data), "--config", config_path]) == 0
    return data, config_path, table


def run(data, config_path) -> int:
    return main(["--datos", str(data), "--config", config_path, "--hilos", str(THREADS)], params=TINY)


def test_el_script_escribe_predicciones_metricas_y_reporte(workspace, capsys):
    data, config_path, table = workspace
    capsys.readouterr()
    before = read_contract("predictions", data / "predictions.parquet")
    assert run(data, config_path) == 0
    printed = capsys.readouterr().out

    assert validate_file(data / "predictions.parquet", "predictions") == []
    predictions = read_contract("predictions", data / "predictions.parquet")
    assert len(predictions) == 2 * len(before)
    kept = predictions[predictions["model"] == "baseline"].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept.sort_values(KEYS + ["split"], ignore_index=True), before.sort_values(KEYS + ["split"], ignore_index=True))
    ours = predictions[predictions["model"] == MODEL]
    for split in ("val", "test"):
        merged = ours.loc[ours["split"] == split, KEYS].merge(before.loc[before["split"] == split, KEYS], on=KEYS, how="outer", indicator=True)
        assert (merged["_merge"] == "both").all()

    # Métricas: las dos en validación y nada de la prueba final.
    metrics = pq.read_table(data / "reports" / "metricas_validacion.parquet").to_pandas()
    assert list(metrics.columns) == METRIC_COLUMNS
    assert set(metrics["model"]) == {"baseline", MODEL} and set(metrics["split"]) == {"val"}
    for model in ("baseline", MODEL):
        part = metrics[metrics["model"] == model]
        assert set(part["group_by"]) == {"general", "hora", "tipo_de_dia", "estacion"}
        assert (part.loc[part["group_by"] == "general", "n"] == (ours["split"] == "val").sum()).all()

    report = (data / "reports" / "metricas_lgbm.md").read_text(encoding="utf-8")
    for text in (
        "## General", "## Por tipo de día", "## Por hora del día", "## Pérdida de cuantil", "## Dónde LightGBM sale peor",
        "## Variables", "## Hiperparámetros y rondas", "## Corrección antes de escribir", "## Cómo leer esto",
        "MAE base", "MAE lgbm", "RMSE base", "RMSE lgbm", "pinball q10", "pinball q90",
        "q50 es la media de la celda, no la mediana", "Comparar también RMSE", "demanda atendida", "NO se calificó",
        "La prueba final no se ha calificado", "dep_rez_14d", "arr_total_1d", "festivo",
    ):
        assert text in report, f"al reporte le falta: {text}"
    for forbidden in ("viajes recuperados", "MAPE"):
        assert forbidden not in report and forbidden not in printed
    assert "VALIDACIÓN" in printed and "no se calificó" in printed and "se conservaron" in printed
    assert "General, línea base" in printed and "General, LightGBM" in printed and "Tiempo" in printed

    stored = json.loads((data / "reports" / FINGERPRINT_NAME).read_text(encoding="utf-8"))
    assert stored["entrada"] == input_fingerprint(read_contract("station_hour", data / "station_hour.parquet"), load_datos(config_path))
    assert stored["modelo"]["variables"] == FEATURES and set(stored["modelo"]["rondas"]) == set(BOOSTER_NAMES)


def test_el_script_solo_entrega_filas_de_validacion_a_las_metricas(workspace, monkeypatch, capsys):
    data, config_path, _ = workspace
    datos = load_datos(config_path)
    seen = []

    def spy(function):
        def wrapper(predictions, station_hour, **kwargs):
            seen.append((kwargs["split"], station_hour["ts_hour"].max()))
            return function(predictions, station_hour, **kwargs)
        return wrapper

    monkeypatch.setattr(lgbm, "evaluate", spy(lgbm.evaluate))
    monkeypatch.setattr(lgbm, "quantile_scores", spy(lgbm.quantile_scores))
    assert run(data, config_path) == 0
    assert len(seen) == 4
    assert all(split == "val" and latest < datos.tramos["prueba"].start for split, latest in seen)


def test_dos_corridas_del_script_dan_archivos_identicos(workspace, capsys):
    data, config_path, _ = workspace
    names = ("predictions.parquet", "reports/metricas_validacion.parquet", "reports/metricas_lgbm.md", f"reports/{FINGERPRINT_NAME}")
    assert run(data, config_path) == 0
    first = {name: (data / name).read_bytes() for name in names}
    assert run(data, config_path) == 0
    for name in names:
        assert (data / name).read_bytes() == first[name], f"{name} cambió entre dos corridas iguales"


def test_el_script_pide_la_linea_base_antes(tmp_path, capsys):
    data = tmp_path / "datos"
    table = as_station_hour(world())
    write_contract(table, "station_hour", data / "station_hour.parquet")
    write_contract(stations_of(table), "stations", data / "stations.parquet")
    assert run(data, write_config(tmp_path)) == 1
    assert "run_baseline.py" in capsys.readouterr().out
    assert not (data / "predictions.parquet").exists()


def check_outputs_are_current(data_dir, config_path) -> None:
    """Falla si las filas de LightGBM en `data_dir` no salieron del station_hour y del config de hoy.

    Ningún valor real de la prueba final sale del disco: de ese tramo solo se leen estación y hora.
    """
    source = data_dir / "station_hour.parquet"
    datos = load_datos(config_path)
    rows = pq.read_table(source, columns=KEYS).to_pandas()
    predictions = pq.read_table(data_dir / "predictions.parquet", columns=KEYS + ["model", "split"]).to_pandas()
    predictions = predictions[predictions["model"] == MODEL]
    if predictions.empty:
        pytest.skip("predictions.parquet todavía no trae filas de LightGBM. Corre: python scripts/run_lgbm.py")

    rerun = "Vuelve a correr: python scripts/run_lgbm.py"
    fix = f"station_hour o config/datos.yaml cambiaron después de LightGBM. {rerun}"
    for split, name in (("val", "validacion"), ("test", "prueba")):
        expected = rows[datos.tramos[name].contains(rows["ts_hour"])]
        merged = expected.merge(predictions.loc[predictions["split"] == split, KEYS], on=KEYS, how="outer", indicator=True)
        assert (merged["_merge"] == "both").all(), f"split {split}: las filas no coinciden con station_hour. {fix}"

    fingerprint_path = data_dir / "reports" / FINGERPRINT_NAME
    assert fingerprint_path.exists(), f"Falta reports/{FINGERPRINT_NAME}: no se sabe si las predicciones siguen al día. {rerun}"
    history = pq.read_table(
        source, columns=KEYS + ["departures", "arrivals"], filters=[("ts_hour", "<", datos.tramos["prueba"].start)]
    ).to_pandas()
    stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert stored["entrada"] == input_fingerprint(history, datos), f"La historia o las fechas no son las de LightGBM. {fix}"

    metrics = pq.read_table(data_dir / "reports" / "metricas_validacion.parquet", columns=["model", "split"]).to_pandas()
    scored = set(metrics.loc[metrics["model"] == MODEL, "split"])
    assert scored == {"val"}, f"LightGBM solo se califica en validación (decisión 4) y el Parquet trae {scored}"


def test_la_revision_nota_una_historia_distinta(workspace, capsys):
    data, config_path, table = workspace
    assert run(data, config_path) == 0
    check_outputs_are_current(data, config_path)
    in_train = load_datos(config_path).tramos["entrenamiento"].contains(table["ts_hour"]).to_numpy()
    write_contract(as_station_hour(changed(table, in_train)), "station_hour", data / "station_hour.parquet")
    with pytest.raises(AssertionError, match="Vuelve a correr: python scripts/run_lgbm.py"):
        check_outputs_are_current(data, config_path)


# --------------------------------------------------------------------------- #
# Archivos reales (se salta si no existen)
# --------------------------------------------------------------------------- #

def test_las_predicciones_reales_de_lgbm_siguen_al_dia():
    """Las filas reales de LightGBM corresponden al station_hour y al config de hoy. No se califica nada."""
    for needed in (DATA_PROCESSED / "station_hour.parquet", DATA_PROCESSED / "predictions.parquet"):
        if not needed.exists():
            pytest.skip(f"{needed.relative_to(DATA_PROCESSED.parents[1])} todavía no existe. Se revisará en cuanto se genere.")
    check_outputs_are_current(DATA_PROCESSED, DATOS_YAML)
