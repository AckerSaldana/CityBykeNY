"""Pruebas de la línea base (src/models/baseline.py).

Qué revisan, con tablas pequeñas hechas a mano y un config con tramos cortos:

a) La media y los cuantiles de una celda son los del cálculo a mano.
b) Los tres niveles de respaldo cuando una celda no tiene historia.
c) El recorte que ordena q10 <= q50 <= q90, y cuántas filas lo necesitaron
   (sin contar diferencias de redondeo).
d) Sin fuga de información (decisión 5): las predicciones de un tramo no
   cambian si cambia algo de ese tramo o de después. La prueba final sí usa
   validación como historia.
e) Cambios de horario: días de 25 y de 23 horas, con las horas y el día de
   la semana por reloj local.
f) Al guardar se reemplazan solo las filas de la línea base.
g) El script de punta a punta: archivos, contrato y dos corridas idénticas.
   No corre si station_hour no cumple su contrato o si se construyó con otro
   periodo, y a las métricas solo les entrega filas de validación.
h) La huella de la entrada (reports/linea_base_entrada.json): con ella se
   nota si station_hour o el config cambiaron después de correr la línea base.

Todo corre sin red y sin tocar data/processed. La última prueba mira los
archivos reales y se salta si todavía no existen.
"""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.contracts import (
    PREDICTIONS,
    QUANTILE_COLUMNS,
    read_contract,
    to_contract_table,
    validate_file,
    validate_table,
    write_contract,
)
from src.data.datos_config import load_datos
from src.models import baseline
from src.models.baseline import (
    FINGERPRINT_NAME,
    MODEL,
    BaselineError,
    input_fingerprint,
    main,
    predict_all,
    predict_split,
    save_predictions,
)
from src.models.metrics import METRIC_COLUMNS
from src.paths import DATA_PROCESSED, DATOS_YAML, TZ

KEYS = ["station_id", "ts_hour"]

# Tres semanas de entrenamiento y una de validación y de prueba. El 1 de junio de 2026 es lunes.
SHORT = (("2026-06-01", "2026-06-21"), ("2026-06-22", "2026-06-28"), ("2026-06-29", "2026-07-05"))
# Doce semanas de entrenamiento: hacen falta 11 valores o más en una celda para que el recorte ocurra.
LONG = (("2026-06-01", "2026-08-23"), ("2026-08-24", "2026-08-30"), ("2026-08-31", "2026-09-06"))
# Diez semanas de entrenamiento: una celda con exactamente 10 valores, donde el recorte no puede ocurrir.
TEN = (("2026-06-01", "2026-08-09"), ("2026-08-10", "2026-08-16"), ("2026-08-17", "2026-08-23"))


# --------------------------------------------------------------------------- #
# Ayudas
# --------------------------------------------------------------------------- #

def write_config(folder, tramos) -> str:
    """Escribe un datos.yaml con los tres tramos dados y devuelve su ruta."""
    train, val, test = tramos
    path = folder / "datos.yaml"
    # La lista debe traer todos los meses que toca el periodo: load_datos lo exige.
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


def config(tmp_path, tramos):
    return load_datos(write_config(tmp_path, tramos))


def hourly(station: str, first: str, last: str, departures=1, arrivals=2) -> pd.DataFrame:
    """Una fila por hora (por instante) de `first` al último instante de `last`, hora de Nueva York."""
    first = first if " " in first else first + " 00:00"
    last = last if " " in last else last + " 23:00"
    ts = pd.date_range(first, last, freq="h", tz=TZ).as_unit("us")
    return pd.DataFrame({"station_id": station, "ts_hour": ts, "departures": departures, "arrivals": arrivals})


def put(table: pd.DataFrame, station: str, when: str, **values) -> None:
    """Cambia salidas o llegadas de una estación en una hora local."""
    row = (table["station_id"] == station) & (table["ts_hour"] == pd.Timestamp(when, tz=TZ))
    assert row.sum() == 1, f"se esperaba una fila de {station} en {when}"
    for column, value in values.items():
        table.loc[row, column] = value


def at(predictions: pd.DataFrame, station: str, when: str) -> pd.Series:
    """La fila de predicción de una estación en una hora local."""
    row = predictions[(predictions["station_id"] == station) & (predictions["ts_hour"] == pd.Timestamp(when, tz=TZ))]
    assert len(row) == 1, f"se esperaba una predicción de {station} en {when}"
    return row.iloc[0]


def quantiles(row: pd.Series, prefix: str) -> tuple:
    return tuple(row[f"{prefix}_q{q}"] for q in (10, 50, 90))


def random_world(tramos, seed: int = 7) -> pd.DataFrame:
    """Tres estaciones con conteos al azar. Una empieza tarde y otra tiene muchos ceros."""
    first, last = tramos[0][0], tramos[2][1]
    late = str(pd.Timestamp(tramos[0][1]) - pd.Timedelta(days=2))[:10]
    table = pd.concat(
        [hourly("5470.10", first, last), hourly("5470.1", first, last), hourly("HB101", late, last)],
        ignore_index=True,
    )
    rng = np.random.default_rng(seed)
    table["departures"] = rng.poisson(3, len(table))
    table["arrivals"] = rng.poisson(3, len(table))
    quiet = table["station_id"] == "5470.1"
    table.loc[quiet, "departures"] = rng.poisson(0.1, int(quiet.sum()))
    return table


def as_station_hour(table: pd.DataFrame) -> pd.DataFrame:
    """Completa las columnas del contrato station_hour para poder guardar la tabla."""
    table = table.copy()
    # La primera y la última hora de cada estación llevan al menos un viaje (decisión 12).
    edges = ~table.duplicated("station_id", keep="first") | ~table.duplicated("station_id", keep="last")
    table.loc[edges, "departures"] = table.loc[edges, "departures"].clip(lower=1)
    table["net_flow"] = table["arrivals"] - table["departures"]
    table["departures_member"] = table["departures"]
    table["departures_casual"] = 0
    return table


# --------------------------------------------------------------------------- #
# a) Media y cuantiles de una celda
# --------------------------------------------------------------------------- #

def test_la_media_y_los_cuantiles_de_la_celda_son_los_del_calculo_a_mano(tmp_path):
    datos = config(tmp_path, SHORT)
    table = hourly("A", "2026-06-01", "2026-07-05")
    # Martes 08:00 de las tres semanas de entrenamiento.
    for day, departures, arrivals in (("2026-06-02", 2, 1), ("2026-06-09", 4, 1), ("2026-06-16", 9, 7)):
        put(table, "A", f"{day} 08:00", departures=departures, arrivals=arrivals)
    put(table, "A", "2026-06-23 08:00", departures=25)  # martes de validación

    val, summary = predict_split(table, datos, "val")
    row = at(val, "A", "2026-06-23 08:00")
    # Salidas 2, 4, 9: media 5; q10 = 2 + 0.2 * (4 - 2); q90 = 4 + 0.8 * (9 - 4).
    assert quantiles(row, "dep") == pytest.approx((2.4, 5.0, 8.0))
    # Llegadas 1, 1, 7: media 3; q10 = 1; q90 = 1 + 0.8 * (7 - 1).
    assert quantiles(row, "arr") == pytest.approx((1.0, 3.0, 5.8))
    # Una celda con el mismo valor todas las semanas.
    assert quantiles(at(val, "A", "2026-06-23 09:00"), "dep") == (1.0, 1.0, 1.0)
    assert row["model"] == "baseline" and row["split"] == "val"
    assert summary.rows == 168 and summary.history_rows == 3 * 168 and summary.stations == 1
    assert summary.levels == {1: 168, 2: 0, 3: 0}

    # Para la prueba final la historia incluye validación: salidas 2, 4, 9 y 25.
    test, summary = predict_split(table, datos, "test")
    row = at(test, "A", "2026-06-30 08:00")
    assert quantiles(row, "dep") == pytest.approx((2.6, 10.0, 20.2))
    assert row["split"] == "test"
    assert summary.history_rows == 4 * 168


def test_q50_es_la_media_y_no_la_mediana(tmp_path):
    datos = config(tmp_path, SHORT)
    table = hourly("A", "2026-06-01", "2026-07-05")
    for day, departures in (("2026-06-02", 0), ("2026-06-09", 0), ("2026-06-16", 9)):
        put(table, "A", f"{day} 08:00", departures=departures)
    val, _ = predict_split(table, datos, "val")
    assert at(val, "A", "2026-06-23 08:00")["dep_q50"] == pytest.approx(3.0)  # la mediana sería 0


# --------------------------------------------------------------------------- #
# b) Niveles de respaldo
# --------------------------------------------------------------------------- #

@pytest.fixture
def fallback_world() -> pd.DataFrame:
    """A cubre todo. B tiene tres días de historia. C nace en validación. D tiene unas horas de historia."""
    table = pd.concat(
        [
            hourly("A", "2026-06-01", "2026-07-05"),
            hourly("B", "2026-06-19", "2026-07-05"),         # viernes, sábado y domingo de historia
            hourly("C", "2026-06-24", "2026-07-05"),         # miércoles de validación
            hourly("D", "2026-06-21 10:00", "2026-07-05"),   # historia: domingo de 10:00 a 23:00
        ],
        ignore_index=True,
    )
    for day, departures in (("2026-06-19", 3), ("2026-06-20", 6), ("2026-06-21", 9)):
        put(table, "B", f"{day} 08:00", departures=departures)
    put(table, "D", "2026-06-21 10:00", departures=40)
    return table


def test_los_tres_niveles_de_respaldo(tmp_path, fallback_world):
    val, summary = predict_split(fallback_world, config(tmp_path, SHORT), "val")

    # Nivel 1: B tiene un viernes en su historia.
    assert quantiles(at(val, "B", "2026-06-26 08:00"), "dep") == (3.0, 3.0, 3.0)
    # Nivel 2: B no tiene ningún lunes. Se usan sus 08:00 de todos los días: 3, 6 y 9.
    assert quantiles(at(val, "B", "2026-06-22 08:00"), "dep") == pytest.approx((3.6, 6.0, 8.4))
    # Nivel 3: C no tiene historia. Viernes 08:00 de todas las estaciones: A con 1, 1, 1 y B con 3.
    assert quantiles(at(val, "C", "2026-06-26 08:00"), "dep") == pytest.approx((1.0, 1.5, 2.4))
    assert quantiles(at(val, "C", "2026-06-26 08:00"), "arr") == (2.0, 2.0, 2.0)
    # Nivel 2: D tiene las 10:00 de un domingo. Nivel 3: D no tiene ninguna 08:00.
    assert quantiles(at(val, "D", "2026-06-22 10:00"), "dep") == (40.0, 40.0, 40.0)
    assert quantiles(at(val, "D", "2026-06-22 08:00"), "dep") == (1.0, 1.0, 1.0)

    # A: 168 en nivel 1. B: 72 en nivel 1 y 96 en nivel 2. C: 120 en nivel 3.
    # D: 14 en nivel 1, 84 en nivel 2 y 70 en nivel 3.
    assert summary.levels == {1: 168 + 72 + 14, 2: 96 + 84, 3: 120 + 70}
    assert sum(summary.levels.values()) == summary.rows == len(val)
    assert summary.stations == 4


def test_la_prueba_final_usa_menos_respaldo_porque_su_historia_incluye_validacion(tmp_path, fallback_world):
    _, summary = predict_split(fallback_world, config(tmp_path, SHORT), "test")
    # C tiene de miércoles a domingo en validación: lunes y martes van por nivel 2.
    assert summary.levels == {1: 3 * 168 + 120, 2: 48, 3: 0}


def test_una_historia_de_menos_de_una_semana_falla_con_mensaje_claro(tmp_path):
    """Sin un nivel 3 no se inventa un valor: una estación nueva en un día que la historia no trae."""
    tramos = (("2026-06-01", "2026-06-03"), ("2026-06-04", "2026-06-10"), ("2026-06-11", "2026-06-17"))
    datos = config(tmp_path, tramos)
    old = hourly("A", "2026-06-01", "2026-06-17")   # historia: lunes, martes y miércoles
    new = hourly("B", "2026-06-04", "2026-06-17")   # nace en validación, un jueves
    # A sola sí se puede predecir: usa el nivel 2 (su misma hora en otros días).
    _, summary = predict_split(old, datos, "val")
    assert summary.levels == {1: 72, 2: 96, 3: 0}
    with pytest.raises(BaselineError, match="al menos una semana completa"):
        predict_split(pd.concat([old, new], ignore_index=True), datos, "val")


def test_sin_filas_en_la_historia_o_en_el_tramo_falla_con_mensaje_claro(tmp_path):
    datos = config(tmp_path, SHORT)
    with pytest.raises(BaselineError, match="en la historia"):
        predict_split(hourly("A", "2026-06-22", "2026-07-05"), datos, "val")
    with pytest.raises(BaselineError, match="en el tramo que se predice"):
        predict_split(hourly("A", "2026-06-01", "2026-06-28"), datos, "test")


def test_una_fecha_sin_zona_o_una_columna_faltante_se_rechazan(tmp_path):
    datos = config(tmp_path, SHORT)
    table = hourly("A", "2026-06-01", "2026-07-05")
    naive = table.assign(ts_hour=table["ts_hour"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="zona horaria"):
        predict_split(naive, datos, "val")
    with pytest.raises(BaselineError, match="arrivals"):
        predict_split(table.drop(columns="arrivals"), datos, "val")
    with pytest.raises(BaselineError, match="split desconocido"):
        predict_split(table, datos, "train")


# --------------------------------------------------------------------------- #
# c) Cuantiles ordenados y recorte
# --------------------------------------------------------------------------- #

def test_el_recorte_ordena_los_cuantiles_y_se_cuenta(tmp_path):
    datos = config(tmp_path, LONG)
    table = hourly("A", "2026-06-01", "2026-09-06")
    mondays = pd.date_range("2026-06-01", "2026-08-17", freq="7D")  # los 12 lunes de entrenamiento
    assert len(mondays) == 12
    for i, monday in enumerate(mondays):
        # Salidas: once ceros y un 120. Llegadas: once 12 y un cero.
        put(table, "A", f"{monday.date()} 08:00", departures=120 if i == 5 else 0, arrivals=0 if i == 5 else 12)

    val, summary = predict_split(table, datos, "val")
    row = at(val, "A", "2026-08-24 08:00")
    # Salidas: media 10, q10 = 0 y q90 empírico = 0, menor que la media: sube a 10.
    assert quantiles(row, "dep") == pytest.approx((0.0, 10.0, 10.0))
    # Llegadas: media 11, q10 empírico = 12, mayor que la media: baja a 11. q90 = 12.
    assert quantiles(row, "arr") == pytest.approx((11.0, 11.0, 12.0))
    assert summary.clipped == {"dep_q10": 0, "dep_q90": 1, "arr_q10": 1, "arr_q90": 0}


def test_una_diferencia_de_redondeo_no_se_cuenta_como_recorte(tmp_path):
    """Nueve valores iguales y uno distinto: el cuantil es igual a la media, salvo el redondeo de la interpolación."""
    datos = config(tmp_path, TEN)
    table = hourly("A", "2026-06-01", "2026-08-23", departures=0, arrivals=1)
    mondays = pd.date_range("2026-06-01", "2026-08-03", freq="7D")  # los 10 lunes de entrenamiento
    assert len(mondays) == 10
    # Salidas: nueve ceros y un 1; media 0.1 y q90 = 0.1. Llegadas: nueve 1 y un cero; media 0.9 y q10 = 0.9.
    put(table, "A", f"{mondays[3].date()} 08:00", departures=1, arrivals=0)

    val, summary = predict_split(table, datos, "val")
    row = at(val, "A", "2026-08-10 08:00")
    assert quantiles(row, "dep") == pytest.approx((0.0, 0.1, 0.1))
    assert quantiles(row, "arr") == pytest.approx((0.9, 0.9, 1.0))
    assert row["dep_q50"] <= row["dep_q90"] and row["arr_q10"] <= row["arr_q50"]  # quedan ordenados
    assert summary.clipped == {"dep_q10": 0, "dep_q90": 0, "arr_q10": 0, "arr_q90": 0}


def test_las_predicciones_cumplen_el_contrato(tmp_path):
    table = random_world(LONG)
    predictions, summaries = predict_all(table, config(tmp_path, LONG))

    assert list(predictions.columns) == PREDICTIONS.columns
    assert validate_table(to_contract_table(predictions, "predictions"), "predictions") == []
    values = predictions[list(QUANTILE_COLUMNS)].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all()
    for prefix in ("dep", "arr"):
        assert (predictions[f"{prefix}_q10"] <= predictions[f"{prefix}_q50"]).all()
        assert (predictions[f"{prefix}_q50"] <= predictions[f"{prefix}_q90"]).all()
    assert set(predictions["model"]) == {"baseline"}
    assert predictions["split"].value_counts().to_dict() == {s: summaries[s].rows for s in ("val", "test")}
    # La estación con muchos ceros necesita recorte: en varias celdas la media queda arriba del q90 empírico.
    assert summaries["val"].clipped["dep_q90"] > 0
    assert (predictions["dep_q90"] == predictions["dep_q50"]).sum() >= summaries["val"].clipped["dep_q90"]


def test_el_orden_de_las_filas_de_entrada_no_cambia_el_resultado(tmp_path):
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    base, summaries = predict_all(table, datos)
    shuffled, again = predict_all(table.sample(frac=1.0, random_state=3), datos)
    pd.testing.assert_frame_equal(shuffled, base)
    assert again == summaries
    # Salen ordenadas por estación y hora dentro de cada tramo.
    for split in ("val", "test"):
        rows = base[base["split"] == split]
        pd.testing.assert_frame_equal(rows, rows.sort_values(KEYS, kind="stable"))


def test_el_id_de_estacion_se_conserva_como_texto(tmp_path):
    predictions, _ = predict_all(random_world(SHORT), config(tmp_path, SHORT))
    assert set(predictions["station_id"]) == {"5470.10", "5470.1", "HB101"}


# --------------------------------------------------------------------------- #
# d) Sin fuga de información
# --------------------------------------------------------------------------- #

def changed(table: pd.DataFrame, rows: pd.Series) -> pd.DataFrame:
    """Copia de la tabla con otras salidas y llegadas en las filas indicadas."""
    out = table.copy()
    out.loc[rows, "departures"] = out.loc[rows, "departures"] * 3 + 50
    out.loc[rows, "arrivals"] = out.loc[rows, "arrivals"] * 2 + 70
    return out


def test_las_predicciones_de_un_tramo_solo_tienen_filas_de_ese_tramo(tmp_path):
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    for split, name in (("val", "validacion"), ("test", "prueba")):
        predictions, _ = predict_split(table, datos, split)
        tramo = datos.tramos[name]
        assert tramo.contains(predictions["ts_hour"]).all()
        expected = table.loc[tramo.contains(table["ts_hour"]), KEYS]
        merged = expected.merge(predictions[KEYS], on=KEYS, how="outer", indicator=True)
        assert (merged["_merge"] == "both").all() and len(merged) == len(predictions)


def test_validacion_no_cambia_si_cambian_valores_de_validacion_o_de_despues(tmp_path):
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    base, _ = predict_split(table, datos, "val")

    from_val = table["ts_hour"] >= datos.tramos["validacion"].start
    again, _ = predict_split(changed(table, from_val), datos, "val")
    pd.testing.assert_frame_equal(again, base)

    # Tampoco si desaparecen filas: toda la prueba final, o la mitad de validación.
    without_test, _ = predict_split(table[table["ts_hour"] < datos.tramos["prueba"].start], datos, "val")
    pd.testing.assert_frame_equal(without_test, base)
    keep = ~from_val | (np.arange(len(table)) % 2 == 0)
    half, _ = predict_split(table[keep], datos, "val")
    merged = half.merge(base, on=KEYS + ["model", "split"], suffixes=("", "_base"))
    assert len(merged) == len(half) < len(base)
    for column in QUANTILE_COLUMNS:
        assert (merged[column] == merged[f"{column}_base"]).all()


def test_validacion_si_cambia_si_cambia_el_entrenamiento(tmp_path):
    """Para saber que la prueba anterior muerde: la historia sí mueve las predicciones."""
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    base, _ = predict_split(table, datos, "val")
    in_train = datos.tramos["entrenamiento"].contains(table["ts_hour"])
    again, _ = predict_split(changed(table, in_train), datos, "val")
    assert not again[list(QUANTILE_COLUMNS)].equals(base[list(QUANTILE_COLUMNS)])


def test_la_prueba_final_no_cambia_si_cambian_sus_valores_y_si_cambia_con_validacion(tmp_path):
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    base, _ = predict_split(table, datos, "test")

    in_test = datos.tramos["prueba"].contains(table["ts_hour"])
    again, _ = predict_split(changed(table, in_test), datos, "test")
    pd.testing.assert_frame_equal(again, base)

    # La historia de la prueba final incluye validación.
    in_val = datos.tramos["validacion"].contains(table["ts_hour"])
    moved, _ = predict_split(changed(table, in_val), datos, "test")
    assert (moved["dep_q50"] != base["dep_q50"]).all()
    assert (moved["arr_q50"] != base["arr_q50"]).all()


# --------------------------------------------------------------------------- #
# e) Cambios de horario
# --------------------------------------------------------------------------- #

def by_local_hour(table: pd.DataFrame) -> pd.DataFrame:
    """Salidas = hora del reloj local; llegadas = 23 menos esa hora."""
    hour = table["ts_hour"].dt.hour.astype("int64")
    return table.assign(departures=hour, arrivals=23 - hour)


def test_una_historia_que_cruza_los_dos_cambios_de_horario_usa_la_hora_local(tmp_path):
    # Entrenamiento del 20 de octubre de 2025 al 15 de marzo de 2026: cruza el 2 de noviembre y el 8 de marzo.
    tramos = (("2025-10-20", "2026-03-15"), ("2026-03-16", "2026-03-22"), ("2026-03-23", "2026-03-29"))
    table = by_local_hour(hourly("A", "2025-10-20", "2026-03-29"))
    day = table["ts_hour"].dt.strftime("%Y-%m-%d")
    assert (day == "2025-11-02").sum() == 25 and (day == "2026-03-08").sum() == 23
    # Las dos 01:00 del día de 25 horas, con valores distintos.
    repeated = table.index[(day == "2025-11-02") & (table["ts_hour"].dt.hour == 1)]
    assert len(repeated) == 2
    table.loc[repeated, "departures"] = [10, 30]

    val, summary = predict_split(table, config(tmp_path, tramos), "val")
    hour = val["ts_hour"].dt.hour
    sunday_1am = (val["ts_hour"].dt.dayofweek == 6) & (hour == 1)
    # Con la hora local cada celda tiene un solo valor. Con UTC o un desfase fijo se mezclarían dos horas.
    assert (val.loc[~sunday_1am, "dep_q50"] == hour[~sunday_1am]).all()
    assert (val["arr_q50"] == 23 - hour).all()
    # Domingo 01:00: 21 domingos, uno con dos filas. 20 valores de 1, más 10 y 30, entre 22 filas.
    assert val.loc[sunday_1am, "dep_q50"].item() == pytest.approx(60 / 22)
    assert summary.levels == {1: 168, 2: 0, 3: 0}


def test_el_dia_de_la_semana_sale_del_reloj_local_aunque_la_historia_cruce_un_cambio_de_horario(tmp_path):
    """Con el día de la semana tomado de UTC, las últimas horas de cada día caerían en la celda del día siguiente."""
    tramos = (("2025-10-06", "2025-11-09"), ("2025-11-10", "2025-11-16"), ("2025-11-17", "2025-11-23"))
    table = hourly("A", "2025-10-06", "2025-11-23")
    local = table["ts_hour"]
    table["departures"] = (local.dt.dayofweek * 24 + local.dt.hour).astype("int64")
    table["arrivals"] = (10 * local.dt.dayofweek).astype("int64")

    predictions, _ = predict_all(table, config(tmp_path, tramos))
    ts = predictions["ts_hour"]
    assert (predictions["dep_q50"] == ts.dt.dayofweek * 24 + ts.dt.hour).all()
    assert (predictions["arr_q50"] == 10 * ts.dt.dayofweek).all()


@pytest.mark.parametrize(
    "tramos, special_day, hours",
    [
        pytest.param(
            (("2025-10-06", "2025-10-26"), ("2025-10-27", "2025-11-02"), ("2025-11-03", "2025-11-09")),
            "2025-11-02", 25, id="dia-de-25-horas",
        ),
        pytest.param(
            (("2026-02-16", "2026-03-01"), ("2026-03-02", "2026-03-08"), ("2026-03-09", "2026-03-15")),
            "2026-03-08", 23, id="dia-de-23-horas",
        ),
    ],
)
def test_predecir_un_tramo_con_cambio_de_horario(tmp_path, tramos, special_day, hours):
    table = by_local_hour(hourly("A", tramos[0][0], tramos[2][1]))
    predictions, summaries = predict_all(table, config(tmp_path, tramos))

    assert validate_table(to_contract_table(predictions, "predictions"), "predictions") == []
    val = predictions[predictions["split"] == "val"]
    assert len(val) == 6 * 24 + hours
    assert (val["ts_hour"].dt.strftime("%Y-%m-%d") == special_day).sum() == hours
    # Validación y prueba final quedan en distinto horario que parte de su historia.
    assert (predictions["dep_q50"] == predictions["ts_hour"].dt.hour).all()
    assert (predictions["arr_q50"] == 23 - predictions["ts_hour"].dt.hour).all()
    assert summaries["test"].levels[1] == summaries["test"].rows == 168


# --------------------------------------------------------------------------- #
# f) Guardar sin borrar otros modelos
# --------------------------------------------------------------------------- #

def test_guardar_reemplaza_solo_las_filas_de_la_linea_base(tmp_path):
    predictions, _ = predict_all(random_world(SHORT), config(tmp_path, SHORT))
    path = tmp_path / "predictions.parquet"

    lgbm = predictions.assign(model="lgbm")
    lgbm[list(QUANTILE_COLUMNS)] = lgbm[list(QUANTILE_COLUMNS)] + 100.0
    stale = predictions.head(50).assign(dep_q10=0.0, dep_q50=777.0, dep_q90=777.0)  # línea base de una corrida vieja
    write_contract(pd.concat([lgbm, stale], ignore_index=True), "predictions", path)

    kept = save_predictions(predictions, path)
    assert kept == len(lgbm)
    assert validate_file(path, "predictions") == []
    saved = read_contract("predictions", path)
    order = KEYS + ["split"]
    pd.testing.assert_frame_equal(
        saved[saved["model"] == "lgbm"].sort_values(order, ignore_index=True),
        lgbm.sort_values(order, ignore_index=True),
    )
    pd.testing.assert_frame_equal(
        saved[saved["model"] == MODEL].sort_values(order, ignore_index=True),
        predictions.sort_values(order, ignore_index=True),
    )
    assert not (saved["dep_q50"] == 777.0).any()

    # Guardar otra vez no duplica nada.
    assert save_predictions(predictions, path) == len(lgbm)
    assert len(read_contract("predictions", path)) == len(saved)


def test_guardar_sin_archivo_previo_escribe_solo_la_linea_base(tmp_path):
    predictions, _ = predict_all(random_world(SHORT), config(tmp_path, SHORT))
    path = tmp_path / "nueva" / "predictions.parquet"
    assert save_predictions(predictions, path) == 0
    assert validate_file(path, "predictions") == []
    assert pq.read_metadata(path).num_rows == len(predictions)


# --------------------------------------------------------------------------- #
# g) El script de punta a punta
# --------------------------------------------------------------------------- #

@pytest.fixture
def workspace(tmp_path):
    """Carpeta con un station_hour.parquet pequeño que cumple el contrato, y su config."""
    data = tmp_path / "datos"
    table = as_station_hour(random_world(SHORT))
    write_contract(table, "station_hour", data / "station_hour.parquet")
    return data, write_config(tmp_path, SHORT), table


def test_el_script_escribe_predicciones_y_metricas_de_validacion(workspace, tmp_path, capsys):
    data, config_path, table = workspace
    out = tmp_path / "salida"
    assert main(["--datos", str(data), "--salida", str(out), "--config", config_path]) == 0
    printed = capsys.readouterr().out

    predictions_path = out / "predictions.parquet"
    assert validate_file(predictions_path, "predictions") == []
    predictions = read_contract("predictions", predictions_path)
    datos = load_datos(config_path)
    for split, name in (("val", "validacion"), ("test", "prueba")):
        in_tramo = datos.tramos[name].contains(table["ts_hour"])
        assert (predictions["split"] == split).sum() == in_tramo.sum()

    # Las métricas son solo de validación: ni una fila de la prueba final.
    metrics = pq.read_table(out / "reports" / "metricas_validacion.parquet").to_pandas()
    assert list(metrics.columns) == METRIC_COLUMNS
    assert set(metrics["split"]) == {"val"} and set(metrics["model"]) == {"baseline"}
    assert set(metrics["group_by"]) == {"general", "hora", "franja", "tipo_de_dia", "estacion"}
    # Por franja se califican las mismas filas: las franjas suman el tramo, en salidas y en llegadas.
    by_band = metrics[metrics["group_by"] == "franja"]
    assert (by_band.groupby("target")["n"].sum() == (predictions["split"] == "val").sum()).all()
    assert set(metrics["target"]) == {"dep", "arr"}
    n_val = int((predictions["split"] == "val").sum())
    assert (metrics.loc[metrics["group_by"] == "general", "n"] == n_val).all()

    # La huella de la entrada queda junto a las métricas y es la de la tabla que se leyó.
    fingerprint = json.loads((out / "reports" / FINGERPRINT_NAME).read_text(encoding="utf-8"))
    assert fingerprint == input_fingerprint(read_contract("station_hour", data / "station_hour.parquet"), datos)
    assert FINGERPRINT_NAME in printed

    report = (out / "reports" / "metricas_validacion.md").read_text(encoding="utf-8")
    for text in (
        "## General", "## Por tipo de día", "## Por hora del día", "## Por franja del día",
        "Las franjas son las del reporte de la simulación", "## Respaldo", "## Recortes", "## Cómo leer esto",
        "demanda atendida", "no la mediana", "NO se calificó", "La prueba final no se ha calificado",
        # Avisos para no leer de más: nivel 3, media contra mediana, cobertura y WAPE mayor que 100 %.
        "Las filas de nivel 3 no son un pronóstico de esa estación",
        "ese es la mediana", "Comparar también RMSE",
        "no quiere decir que el intervalo esté calibrado",
        # Tres semanas de historia: 3 valores por celda y 80 x (3 - 1) / (3 + 1) = 40.
        "Con 3 valores por celda, el intervalo empírico de 10 % a 90 % cubre en teoría cerca de 40 % de",
        "predecir cero en todas las filas de ese grupo tendría menor MAE",
        "redondeo del cálculo no cuenta como recorte",
    ):
        assert text in report, f"al reporte le falta: {text}"
    for forbidden in ("viajes recuperados", "MAPE"):
        assert forbidden not in report and forbidden not in printed
    assert "VALIDACIÓN" in printed and "no se calificó" in printed
    assert "General" in printed and "Por tipo de día" in printed and "Por hora local" in printed and "Por franja del día" in printed


def test_dos_corridas_del_script_dan_archivos_identicos(workspace, tmp_path, capsys):
    data, config_path, _ = workspace
    out = tmp_path / "salida"
    names = (
        "predictions.parquet", "reports/metricas_validacion.parquet", "reports/metricas_validacion.md",
        f"reports/{FINGERPRINT_NAME}",
    )
    args = ["--datos", str(data), "--salida", str(out), "--config", config_path]
    assert main(args) == 0
    first = {name: (out / name).read_bytes() for name in names}
    assert main(args) == 0
    assert {name: (out / name).read_bytes() for name in names} == first


def test_el_script_sin_salida_escribe_junto_a_los_datos(workspace, capsys):
    data, config_path, _ = workspace
    assert main(["--datos", str(data), "--config", config_path]) == 0
    assert (data / "predictions.parquet").exists()
    assert (data / "reports" / "metricas_validacion.md").exists()
    assert (data / "reports" / FINGERPRINT_NAME).exists()


def test_el_script_conserva_las_predicciones_de_otro_modelo(workspace, capsys):
    data, config_path, table = workspace
    predictions, _ = predict_all(table, load_datos(config_path))
    lgbm = predictions.assign(model="lgbm")
    write_contract(lgbm, "predictions", data / "predictions.parquet")

    assert main(["--datos", str(data), "--config", config_path]) == 0
    saved = read_contract("predictions", data / "predictions.parquet")
    assert saved["model"].value_counts().to_dict() == {"lgbm": len(lgbm), "baseline": len(predictions)}
    assert f"se conservaron {len(lgbm):,} filas de otros modelos" in capsys.readouterr().out


def test_el_script_sin_station_hour_avisa_y_no_escribe_nada(tmp_path, capsys):
    config_path = write_config(tmp_path, SHORT)
    empty = tmp_path / "vacia"
    empty.mkdir()
    assert main(["--datos", str(empty), "--config", config_path]) == 1
    assert "No existe" in capsys.readouterr().out
    assert list(empty.iterdir()) == []


def test_el_script_con_un_config_invalido_avisa_y_no_escribe_nada(workspace, tmp_path, capsys):
    data, _, _ = workspace
    bad = tmp_path / "malo.yaml"
    bad.write_text("periodo: {inicio: 2026-06-01, fin: 2026-07-05}\n", encoding="utf-8")
    assert main(["--datos", str(data), "--config", str(bad)]) == 1
    assert "ERROR" in capsys.readouterr().out
    assert not (data / "predictions.parquet").exists()


def test_el_script_solo_entrega_filas_de_validacion_a_las_metricas(workspace, tmp_path, capsys, monkeypatch):
    """Decisión 4: a `evaluate` no le llega ni una fila real de la prueba final, y sí las fechas que debe proteger."""
    data, config_path, _ = workspace
    seen = {}
    real_evaluate = baseline.evaluate

    def spy(predictions, station_hour, **kwargs):
        seen["station_hour"], seen["kwargs"] = station_hour, kwargs
        return real_evaluate(predictions, station_hour, **kwargs)

    monkeypatch.setattr(baseline, "evaluate", spy)
    assert main(["--datos", str(data), "--salida", str(tmp_path / "salida"), "--config", config_path]) == 0
    datos = load_datos(config_path)
    assert len(seen["station_hour"]) > 0
    assert datos.tramos["validacion"].contains(seen["station_hour"]["ts_hour"]).all()
    assert seen["kwargs"]["split"] == "val" and seen["kwargs"]["prueba_final"] == datos.tramos["prueba"]
    assert not seen["kwargs"].get("usar_prueba_final", False)


def test_el_script_no_corre_si_station_hour_no_cumple_su_contrato(tmp_path, capsys):
    """Una fila repetida en la historia contaría doble en la media sin que nada avise."""
    config_path = write_config(tmp_path, SHORT)
    data = tmp_path / "datos"
    data.mkdir()
    table = to_contract_table(as_station_hour(random_world(SHORT)), "station_hour")
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), data / "station_hour.parquet")  # sin write_contract

    out = tmp_path / "salida"
    assert main(["--datos", str(data), "--salida", str(out), "--config", config_path]) == 1
    printed = capsys.readouterr().out
    assert "ERROR" in printed and "no cumple el contrato station_hour" in printed
    assert "1 filas duplicadas por llave" in printed
    assert not out.exists()


def test_el_script_no_corre_si_las_tablas_se_construyeron_con_otro_periodo(workspace, tmp_path, capsys):
    """build_tables deja en reports/limpieza.json el periodo que usó. Si no es el del config, la línea base se detiene."""
    data, config_path, _ = workspace
    cleaning = data / "reports" / "limpieza.json"
    cleaning.parent.mkdir()

    def built_with(inicio: str, fin: str) -> None:
        cleaning.write_text(json.dumps({"config": {"periodo": {"inicio": inicio, "fin": fin}}}), encoding="utf-8")

    # El mismo periodo del config: corre.
    built_with("2026-06-01", "2026-07-05")
    assert main(["--datos", str(data), "--salida", str(tmp_path / "igual"), "--config", config_path]) == 0
    capsys.readouterr()

    # Las tablas terminan un día antes de lo que dice el config: se detiene y no escribe nada.
    built_with("2026-06-01", "2026-07-04")
    out = tmp_path / "distinto"
    assert main(["--datos", str(data), "--salida", str(out), "--config", config_path]) == 1
    printed = capsys.readouterr().out
    assert "se construyó con el periodo 2026-06-01 a 2026-07-04" in printed
    assert "dice 2026-06-01 a 2026-07-05" in printed and "python scripts/build_tables.py" in printed
    assert not out.exists()

    # Un limpieza.json que no se puede leer tampoco se pasa por alto.
    cleaning.write_text("{", encoding="utf-8")
    assert main(["--datos", str(data), "--salida", str(out), "--config", config_path]) == 1
    assert "No se pudo leer el periodo" in capsys.readouterr().out
    assert not out.exists()


# --------------------------------------------------------------------------- #
# h) La huella de la entrada y la revisión de que las salidas siguen al día
# --------------------------------------------------------------------------- #

def test_la_huella_cambia_con_la_historia_y_no_depende_de_la_prueba_final(tmp_path):
    datos = config(tmp_path, SHORT)
    table = random_world(SHORT)
    base = input_fingerprint(table, datos)

    # Fechas del config, y de la historia: filas, sumas y un número de control.
    assert base["periodo"] == {"inicio": "2026-06-01", "fin": "2026-07-05"}
    in_train = datos.tramos["entrenamiento"].contains(table["ts_hour"])
    train = base["tramos"]["entrenamiento"]
    assert (train["inicio"], train["fin"], train["filas"]) == ("2026-06-01", "2026-06-21", int(in_train.sum()))
    assert train["salidas"] == int(table.loc[in_train, "departures"].sum())
    assert train["llegadas"] == int(table.loc[in_train, "arrivals"].sum())
    assert set(base["tramos"]["validacion"]) == set(train) == {"inicio", "fin", "filas", "salidas", "llegadas", "control"}
    # De la prueba final solo van las fechas.
    assert base["tramos"]["prueba"] == {"inicio": "2026-06-29", "fin": "2026-07-05"}
    assert json.loads(json.dumps(base)) == base  # se guarda en JSON sin perder nada

    # No depende del orden de las filas.
    assert input_fingerprint(table.sample(frac=1.0, random_state=3), datos) == base
    # Tampoco de la prueba final: ni de sus valores ni de que sus filas estén.
    in_test = datos.tramos["prueba"].contains(table["ts_hour"])
    assert input_fingerprint(changed(table, in_test), datos) == base
    assert input_fingerprint(table[~in_test], datos) == base

    # Sí cambia con la historia, también cuando las filas y las sumas son las mismas:
    # una salida pasa de una hora de entrenamiento a otra.
    one, other = table.index[in_train & (table["station_id"] == "5470.10")][:2]
    swapped = table.copy()
    swapped.loc[[one, other], "departures"] = [table.at[one, "departures"] + 1, table.at[other, "departures"] - 1]
    moved = input_fingerprint(swapped, datos)["tramos"]["entrenamiento"]
    assert (moved["filas"], moved["salidas"], moved["llegadas"]) == (train["filas"], train["salidas"], train["llegadas"])
    assert moved["control"] != train["control"]
    in_val = datos.tramos["validacion"].contains(table["ts_hour"])
    assert input_fingerprint(changed(table, in_val), datos)["tramos"]["validacion"] != base["tramos"]["validacion"]

    # Y con las fechas del config: la prueba final termina un día antes.
    shorter = (SHORT[0], SHORT[1], ("2026-06-29", "2026-07-04"))
    assert input_fingerprint(table, load_datos(write_config(tmp_path, shorter))) != base


def check_outputs_are_current(data_dir, config_path) -> None:
    """Falla si las salidas de la línea base en `data_dir` no salieron del station_hour y del config de hoy.

    De station_hour se leen la estación y la hora de todas las filas, y los
    valores solo de la historia (entrenamiento y validación). Ningún valor
    real de la prueba final sale del disco.
    """
    source = data_dir / "station_hour.parquet"
    datos = load_datos(config_path)
    rows = pq.read_table(source, columns=KEYS).to_pandas()
    predictions = pq.read_table(data_dir / "predictions.parquet", columns=KEYS + ["model", "split"]).to_pandas()
    predictions = predictions[predictions["model"] == MODEL]
    if predictions.empty:
        pytest.skip("predictions.parquet todavía no trae filas de la línea base. Corre: python scripts/run_baseline.py")

    rerun = "Vuelve a correr: python scripts/run_baseline.py"
    fix = f"station_hour o config/datos.yaml cambiaron después de la línea base. {rerun}"
    for split, name in (("val", "validacion"), ("test", "prueba")):
        expected = rows[datos.tramos[name].contains(rows["ts_hour"])]
        got = predictions.loc[predictions["split"] == split, KEYS]
        merged = expected.merge(got, on=KEYS, how="outer", indicator=True)
        assert (merged["_merge"] == "both").all(), f"split {split}: las filas no coinciden con station_hour. {fix}"

    # Las mismas filas no bastan: con otros valores en la historia las predicciones serían otras.
    fingerprint_path = data_dir / "reports" / FINGERPRINT_NAME
    assert fingerprint_path.exists(), (
        f"Falta reports/{FINGERPRINT_NAME}: sin la huella no se sabe si las predicciones siguen al día. {rerun}"
    )
    history = pq.read_table(
        source,
        columns=KEYS + ["departures", "arrivals"],
        filters=[("ts_hour", "<", datos.tramos["prueba"].start)],  # nada de la prueba final
    ).to_pandas()
    stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    assert stored == input_fingerprint(history, datos), f"La historia o las fechas no son las de la línea base. {fix}"

    metrics_path = data_dir / "reports" / "metricas_validacion.parquet"
    if metrics_path.exists():
        metrics = pq.read_table(metrics_path, columns=["model", "split"]).to_pandas()
        scored = set(metrics.loc[metrics["model"] == MODEL, "split"])
        assert scored == {"val"}, f"La línea base solo se califica en validación (decisión 4) y el reporte trae {scored}"


def test_la_revision_nota_una_historia_distinta_aunque_las_filas_sean_las_mismas(workspace, tmp_path, capsys):
    """Antes solo se comparaban estación y hora: reconstruir station_hour con otros valores pasaba sin aviso."""
    data, config_path, table = workspace
    args = ["--datos", str(data), "--config", config_path]
    assert main(args) == 0
    check_outputs_are_current(data, config_path)

    # Cambiar valores de la prueba final no deja nada viejo: la línea base no los lee.
    datos = load_datos(config_path)
    in_test = datos.tramos["prueba"].contains(table["ts_hour"])
    write_contract(as_station_hour(changed(table, in_test)), "station_hour", data / "station_hour.parquet")
    check_outputs_are_current(data, config_path)

    # Las mismas filas con otros valores en el entrenamiento: las predicciones guardadas ya son viejas.
    in_train = datos.tramos["entrenamiento"].contains(table["ts_hour"])
    write_contract(as_station_hour(changed(table, in_train)), "station_hour", data / "station_hour.parquet")
    with pytest.raises(AssertionError, match="Vuelve a correr: python scripts/run_baseline.py"):
        check_outputs_are_current(data, config_path)

    # Al volver a correr la línea base, todo queda al día otra vez.
    assert main(args) == 0
    check_outputs_are_current(data, config_path)

    # Sin la huella no se puede saber: también pide volver a correr.
    (data / "reports" / FINGERPRINT_NAME).unlink()
    with pytest.raises(AssertionError, match=f"Falta reports/{FINGERPRINT_NAME}"):
        check_outputs_are_current(data, config_path)


# --------------------------------------------------------------------------- #
# Archivos reales (se salta si no existen)
# --------------------------------------------------------------------------- #

def test_las_predicciones_reales_cubren_las_filas_de_cada_tramo():
    """Las salidas reales de la línea base corresponden al station_hour y al config de hoy.

    No se mira ningún valor real de la prueba final: de ese tramo solo se leen estación y hora.
    """
    for needed in (DATA_PROCESSED / "station_hour.parquet", DATA_PROCESSED / "predictions.parquet"):
        if not needed.exists():
            pytest.skip(f"{needed.relative_to(DATA_PROCESSED.parents[1])} todavía no existe. Se revisará en cuanto se genere.")
    check_outputs_are_current(DATA_PROCESSED, DATOS_YAML)
