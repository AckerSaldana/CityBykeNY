"""Pruebas de las métricas de pronóstico (src/models/metrics.py).

Qué revisan:

a) MAE, RMSE, WAPE, cobertura y ancho medio contra un cálculo a mano.
b) WAPE queda vacío (NaN) cuando la suma de los valores reales es 0.
c) El tipo de día: un festivo entre semana es "festivo", un sábado es "fin de
   semana", y la fecha es la del reloj de Nueva York.
d) `evaluate` arma la tabla larga con sus cinco desgloses. No califica si
   hay valores vacíos, cuantiles cruzados o filas reales sin predicción.
   El desglose por franja usa las franjas del reporte de la simulación
   (las de operación, la noche y "resto").
e) El candado de la prueba final (decisión 4): calificar "test" falla salvo
   que se pida de forma explícita, y también una fila con fecha de la prueba
   final aunque su etiqueta diga "val".
f) Guardar las métricas no borra las de otros modelos ni las de otro tramo.

Las fechas con las que se prueba el candado son inventadas: aquí no se lee
ningún dato real.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data.datos_config import Tramo, load_datos
from src.models.metrics import (
    DAY_TYPES,
    GROUP_BYS,
    METRIC_COLUMNS,
    METRICS_SCHEMA,
    FinalTestError,
    day_type,
    evaluate,
    federal_holidays,
    format_table,
    metrics_by_group,
    report_bands,
    write_metrics,
)
from src.paths import DATOS_YAML, OPERACION_YAML, TZ
from src.sim.engine import NIGHT, OUTSIDE, Franja, Reglas, hour_labels
from src.sim.operacion import load_operacion

# La prueba final de estas pruebas empieza después de las horas del ejemplo (4 a 7 de septiembre de 2026).
# Se le pasa a `evaluate` para no depender de las fechas de config/datos.yaml.
FINAL = Tramo("prueba", date(2026, 9, 15), date(2026, 9, 29))

# Ejemplo a mano: cuatro filas.
ACTUAL = [0, 2, 5, 10]
Q10 = [0, 1, 4, 2]
Q50 = [1, 2, 3, 6]
Q90 = [2, 2, 8, 9]


def ts(*texts: str) -> pd.Series:
    """Horas locales de Nueva York, con zona."""
    return pd.Series(pd.DatetimeIndex(list(texts)).tz_localize(TZ).as_unit("us"))


# --------------------------------------------------------------------------- #
# a) y b) Valores a mano
# --------------------------------------------------------------------------- #

def test_las_metricas_son_las_del_calculo_a_mano():
    result = metrics_by_group(ACTUAL, Q10, Q50, Q90)
    assert list(result.index) == ["todas"]
    row = result.loc["todas"]
    assert row["n"] == 4
    assert row["mae"] == pytest.approx((1 + 0 + 2 + 4) / 4)
    assert row["rmse"] == pytest.approx(math.sqrt((1 + 0 + 4 + 16) / 4))
    assert row["wape"] == pytest.approx(7 / 17)               # suma de errores entre suma de reales
    assert row["coverage"] == pytest.approx(3 / 4)            # el 10 queda fuera de [2, 9]
    assert row["mean_width"] == pytest.approx((2 + 1 + 4 + 7) / 4)


def test_las_metricas_por_grupo_son_las_del_calculo_a_mano():
    result = metrics_by_group(ACTUAL, Q10, Q50, Q90, group=["a", "b", "a", "b"])
    assert list(result.index) == ["a", "b"]
    a, b = result.loc["a"], result.loc["b"]
    # Grupo a: reales 0 y 5 contra 1 y 3. Grupo b: reales 2 y 10 contra 2 y 6.
    assert (a["n"], b["n"]) == (2, 2)
    assert a["mae"] == pytest.approx(1.5) and b["mae"] == pytest.approx(2.0)
    assert a["rmse"] == pytest.approx(math.sqrt(2.5)) and b["rmse"] == pytest.approx(math.sqrt(8.0))
    assert a["wape"] == pytest.approx(3 / 5) and b["wape"] == pytest.approx(4 / 12)
    assert a["coverage"] == pytest.approx(1.0) and b["coverage"] == pytest.approx(0.5)
    assert a["mean_width"] == pytest.approx(3.0) and b["mean_width"] == pytest.approx(4.0)


def test_la_cobertura_incluye_los_bordes_del_intervalo():
    result = metrics_by_group([1, 3, 0], [1, 2, 0], [2, 2, 0], [3, 3, 0]).loc["todas"]
    assert result["coverage"] == pytest.approx(1.0)  # real igual a q10, igual a q90 e intervalo [0, 0]


def test_wape_queda_vacio_cuando_los_reales_suman_cero():
    result = metrics_by_group([0, 0, 0, 4], [0, 0, 0, 0], [1, 0, 2, 4], [1, 0, 2, 4], group=["x", "x", "x", "y"])
    assert math.isnan(result.loc["x", "wape"])
    assert result.loc["x", "mae"] == pytest.approx(1.0)       # lo demás sí se calcula
    assert result.loc["x", "coverage"] == pytest.approx(1.0)  # un cero real cae dentro de [0, q90]
    assert result.loc["y", "wape"] == pytest.approx(0.0)


def test_no_se_calcula_mape():
    assert METRIC_COLUMNS == [
        "model", "split", "target", "group_by", "group", "n", "mae", "rmse", "wape", "coverage", "mean_width",
    ]
    assert not any("mape" in column for column in metrics_by_group(ACTUAL, Q10, Q50, Q90).columns)


# --------------------------------------------------------------------------- #
# c) Tipo de día
# --------------------------------------------------------------------------- #

def test_tipo_de_dia():
    hours = ts(
        "2026-09-04 08:00",   # viernes
        "2026-09-05 08:00",   # sábado
        "2026-09-06 23:00",   # domingo
        "2026-09-07 00:00",   # lunes, Labor Day
        "2026-09-07 23:00",   # sigue siendo Labor Day en Nueva York (en UTC ya es martes)
        "2026-09-08 00:00",   # martes
        "2026-07-03 12:00",   # viernes en que se observa el 4 de julio
        "2026-07-04 12:00",   # sábado 4 de julio: fin de semana
    )
    expected = [
        "laborable", "fin de semana", "fin de semana", "festivo", "festivo", "laborable", "festivo", "fin de semana",
    ]
    assert day_type(hours).tolist() == expected
    # El mismo instante guardado en UTC da el mismo tipo de día.
    assert day_type(hours.dt.tz_convert("UTC")).tolist() == expected
    assert set(expected) == set(DAY_TYPES)


def test_tipo_de_dia_conserva_el_indice_y_rechaza_fechas_sin_zona():
    hours = ts("2026-09-05 08:00", "2026-09-07 08:00")
    hours.index = [10, 20]
    assert day_type(hours).to_dict() == {10: "fin de semana", 20: "festivo"}
    with pytest.raises(ValueError, match="zona horaria"):
        day_type(hours.dt.tz_localize(None))
    assert day_type(hours.iloc[:0]).empty


def test_los_festivos_son_los_federales():
    found = federal_holidays(2026, 2026)
    assert found[pd.Timestamp("2026-09-07").date()] == "Labor Day"
    assert pd.Timestamp("2026-11-03").date() not in found  # día de elecciones: festivo del estado, no federal


# --------------------------------------------------------------------------- #
# d) Tabla larga
# --------------------------------------------------------------------------- #

@pytest.fixture
def scored():
    """Dos estaciones y cuatro horas: viernes, sábado y dos horas del lunes festivo."""
    hours = ts("2026-09-04 08:00", "2026-09-05 08:00", "2026-09-07 08:00", "2026-09-07 09:00")
    station_hour = pd.DataFrame(
        {
            "station_id": ["5470.10"] * 4 + ["5470.1"] * 4,
            "ts_hour": pd.concat([hours, hours], ignore_index=True),
            "departures": ACTUAL + [0, 0, 0, 0],
            "arrivals": [3, 3, 3, 3] + ACTUAL,
        }
    )
    predictions = station_hour[["station_id", "ts_hour"]].copy()
    predictions["model"] = "baseline"
    predictions["split"] = "val"
    predictions["dep_q10"] = [float(v) for v in Q10 + [0, 0, 0, 0]]
    predictions["dep_q50"] = [float(v) for v in Q50 + [1, 1, 1, 1]]
    predictions["dep_q90"] = [float(v) for v in Q90 + [1, 1, 1, 1]]
    predictions["arr_q10"] = [float(v) for v in [3, 3, 3, 3] + Q10]
    predictions["arr_q50"] = [float(v) for v in [3, 3, 3, 3] + Q50]
    predictions["arr_q90"] = [float(v) for v in [3, 3, 3, 3] + Q90]
    return predictions, station_hour


def pick(metrics: pd.DataFrame, target: str, group_by: str, group: str) -> pd.Series:
    rows = metrics[(metrics["target"] == target) & (metrics["group_by"] == group_by) & (metrics["group"] == group)]
    assert len(rows) == 1
    return rows.iloc[0]


# Franjas de estas pruebas, escritas aquí: mañana de 06:00 a 12:00, tarde de 15:00 a 21:00 y noche de 22:00
# a 06:00. Lo demás (12:00 a 15:00 y 21:00 a 22:00) es "resto".
FRANJAS = (Franja("manana", "05:00", "06:00", "12:00", 0.5), Franja("tarde", "14:00", "15:00", "21:00", 0.5))
BANDS = hour_labels(FRANJAS, ("22:00", "06:00"))


def test_evaluate_devuelve_la_tabla_larga_con_los_cinco_desgloses(scored):
    predictions, station_hour = scored
    metrics = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL, franjas=BANDS)
    assert set(metrics["group_by"]) == set(GROUP_BYS)

    assert list(metrics.columns) == METRIC_COLUMNS
    assert set(metrics["model"]) == {"baseline"} and set(metrics["split"]) == {"val"}
    for target in ("dep", "arr"):
        of_target = metrics[metrics["target"] == target]
        groups = {group_by: of_target.loc[of_target["group_by"] == group_by, "group"].tolist() for group_by in set(of_target["group_by"])}
        assert groups == {
            "general": ["todas"],
            "hora": ["08", "09"],
            "franja": ["manana"],   # las cuatro horas del ejemplo son de la mañana
            "tipo_de_dia": ["laborable", "fin de semana", "festivo"],
            "estacion": ["5470.1", "5470.10"],
        }
    assert metrics["n"].dtype == "int64"

    # General de salidas: las cuatro filas del ejemplo a mano más cuatro ceros predichos como 1.
    general = pick(metrics, "dep", "general", "todas")
    assert general["n"] == 8
    assert general["mae"] == pytest.approx((7 + 4) / 8)
    assert general["rmse"] == pytest.approx(math.sqrt((21 + 4) / 8))
    assert general["wape"] == pytest.approx(11 / 17)
    assert general["coverage"] == pytest.approx((3 + 4) / 8)
    assert general["mean_width"] == pytest.approx((14 + 4) / 8)

    # Por estación: una es el ejemplo a mano; la otra no tiene salidas reales y su WAPE queda vacío.
    station = pick(metrics, "dep", "estacion", "5470.10")
    assert station["mae"] == pytest.approx(1.75) and station["wape"] == pytest.approx(7 / 17)
    assert math.isnan(pick(metrics, "dep", "estacion", "5470.1")["wape"])
    assert pick(metrics, "arr", "estacion", "5470.1")["mae"] == pytest.approx(1.75)
    assert pick(metrics, "arr", "estacion", "5470.10")["mae"] == pytest.approx(0.0)

    # Por tipo de día y por hora (hora local).
    assert pick(metrics, "dep", "tipo_de_dia", "festivo")["n"] == 4
    assert pick(metrics, "dep", "tipo_de_dia", "festivo")["mae"] == pytest.approx((2 + 4 + 1 + 1) / 4)
    assert pick(metrics, "dep", "tipo_de_dia", "fin de semana")["mae"] == pytest.approx((0 + 1) / 2)
    assert pick(metrics, "dep", "hora", "09")["n"] == 2
    assert pick(metrics, "dep", "hora", "09")["mae"] == pytest.approx((4 + 1) / 2)


def test_la_franja_de_cada_hora_es_la_del_reporte_de_la_simulacion():
    assert list(BANDS) == [NIGHT] * 6 + ["manana"] * 6 + [OUTSIDE] * 3 + ["tarde"] * 6 + [OUTSIDE] + [NIGHT] * 2
    # Sin franja de noche, lo que no es franja de operación es "resto".
    assert list(hour_labels(FRANJAS)) == [OUTSIDE] * 6 + ["manana"] * 6 + [OUTSIDE] * 3 + ["tarde"] * 6 + [OUTSIDE] * 3
    # Donde la noche se encima con una franja de operación, la hora cuenta en la franja de operación.
    assert list(hour_labels(FRANJAS, ("20:00", "07:00")))[18:] == ["tarde"] * 3 + [NIGHT] * 3
    assert list(hour_labels(FRANJAS, ("20:00", "07:00")))[:8] == [NIGHT] * 6 + ["manana"] * 2
    # Una noche que no cruza la medianoche.
    assert list(hour_labels(FRANJAS, ("00:00", "05:00")))[:7] == [NIGHT] * 5 + [OUTSIDE] + ["manana"]
    # Un borde que no cae en hora exacta: la hora cuenta donde empieza (las 12:00 empiezan en la mañana).
    late = (Franja("manana", "05:00", "06:00", "12:30", 1.0),)
    assert hour_labels(late)[12] == "manana" and hour_labels(late)[13] == OUTSIDE


def test_las_franjas_por_defecto_salen_de_los_dos_archivos_de_config():
    franjas = Reglas.from_operacion(load_operacion(OPERACION_YAML)).franjas
    noche = load_datos(DATOS_YAML).simulacion.franja_noche
    bands = report_bands()
    assert bands == hour_labels(franjas, tuple(noche)) and len(bands) == 24
    assert {franja.nombre for franja in franjas} <= set(bands) <= {*(f.nombre for f in franjas), NIGHT, OUTSIDE}


def test_las_metricas_por_franja_son_las_del_calculo_a_mano():
    # Una estación y seis horas: dos de la mañana, una de "resto", una de la tarde y dos de la noche (una
    # antes y otra después de la medianoche).
    hours = ts("2026-09-04 08:00", "2026-09-04 11:00", "2026-09-04 13:00", "2026-09-04 16:00",
               "2026-09-04 23:00", "2026-09-05 02:00")
    station_hour = pd.DataFrame(
        {"station_id": ["A"] * 6, "ts_hour": hours, "departures": [4, 6, 2, 9, 1, 0], "arrivals": [0] * 6}
    )
    predictions = station_hour[["station_id", "ts_hour"]].copy()
    predictions["model"], predictions["split"] = "baseline", "val"
    predictions["dep_q10"] = [2.0, 2.0, 0.0, 5.0, 0.0, 0.0]
    predictions["dep_q50"] = [3.0, 8.0, 2.0, 6.0, 3.0, 1.0]
    predictions["dep_q90"] = [5.0, 9.0, 4.0, 8.0, 4.0, 2.0]
    for q in ("q10", "q50", "q90"):
        predictions[f"arr_{q}"] = 0.0
    metrics = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL, franjas=BANDS)

    rows = metrics[(metrics["group_by"] == "franja") & (metrics["target"] == "dep")]
    assert rows["group"].tolist() == ["manana", "tarde", NIGHT, OUTSIDE]   # el orden del reporte de la simulación
    assert rows["n"].tolist() == [2, 1, 2, 1]
    assert rows["n"].sum() == pick(metrics, "dep", "general", "todas")["n"]   # las franjas suman el tramo
    # Mañana: errores 1 y 2; reales 4 y 6; las dos dentro del intervalo; anchos 3 y 7.
    manana = pick(metrics, "dep", "franja", "manana")
    assert manana["mae"] == pytest.approx(1.5) and manana["rmse"] == pytest.approx(math.sqrt(2.5))
    assert manana["wape"] == pytest.approx(3 / 10) and manana["coverage"] == 1.0 and manana["mean_width"] == 5.0
    # Tarde: real 9 contra 6, fuera del intervalo (5 a 8).
    tarde = pick(metrics, "dep", "franja", "tarde")
    assert (tarde["mae"], tarde["coverage"], tarde["wape"]) == (3.0, 0.0, pytest.approx(3 / 9))
    # Noche: las 23:00 y las 02:00 del día siguiente van juntas. Errores 2 y 1; reales 1 y 0.
    noche = pick(metrics, "dep", "franja", NIGHT)
    assert noche["mae"] == pytest.approx(1.5) and noche["wape"] == pytest.approx(3 / 1) and noche["coverage"] == 1.0
    # Resto: las 13:00, sin error.
    assert pick(metrics, "dep", "franja", OUTSIDE)["mae"] == 0.0
    # Sin llegadas reales, el WAPE de llegadas queda vacío en todas las franjas.
    assert metrics.loc[(metrics["group_by"] == "franja") & (metrics["target"] == "arr"), "wape"].isna().all()

    # Otras franjas, otro reparto de las mismas filas; y hacen falta las 24 horas.
    todo_resto = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL,
                          franjas=[OUTSIDE] * 24)
    assert todo_resto.loc[todo_resto["group_by"] == "franja", "group"].tolist() == [OUTSIDE, OUTSIDE]
    assert pick(todo_resto, "dep", "franja", OUTSIDE)["mae"] == pick(metrics, "dep", "general", "todas")["mae"]
    with pytest.raises(ValueError, match="24 etiquetas"):
        evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL, franjas=BANDS[:23])
    # Sin `franjas`, las de config/operacion.yaml y config/datos.yaml.
    default = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL)
    assert set(default.loc[default["group_by"] == "franja", "group"]) <= set(report_bands())


def test_evaluate_solo_califica_el_modelo_y_el_tramo_pedidos(scored):
    predictions, station_hour = scored
    base = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL)
    # Filas de otro modelo y de la prueba final, con valores absurdos y sin valor real: no cuentan.
    other = predictions.assign(model="lgbm", dep_q50=999.0, dep_q90=999.0)
    final = predictions.assign(split="test", ts_hour=predictions["ts_hour"] + pd.Timedelta(days=14), dep_q50=999.0, dep_q90=999.0)
    mixed = pd.concat([final, other, predictions], ignore_index=True)
    again = evaluate(mixed, station_hour, model="baseline", split="val", prueba_final=FINAL)
    pd.testing.assert_frame_equal(again, base)


def test_evaluate_falla_si_no_hay_predicciones_o_si_falta_el_valor_real(scored):
    predictions, station_hour = scored
    with pytest.raises(ValueError, match="No hay predicciones"):
        evaluate(predictions, station_hour, model="lgbm", split="val", prueba_final=FINAL)
    with pytest.raises(ValueError, match="no tienen fila en station_hour"):
        evaluate(predictions, station_hour.iloc[1:], model="baseline", split="val", prueba_final=FINAL)


def test_evaluate_no_califica_valores_vacios(scored):
    """Un vacío contaría como error cero y la fila seguiría contando en n: las métricas saldrían mejor de lo que son."""
    predictions, station_hour = scored
    holes = predictions.copy()
    holes.loc[[0, 1, 2], "dep_q50"] = np.nan
    with pytest.raises(ValueError, match="valores vacíos.*dep_q50: 3"):
        evaluate(holes, station_hour, model="baseline", split="val", prueba_final=FINAL)
    # Un valor real vacío en llegadas, con las salidas completas.
    no_arrivals = station_hour.astype({"arrivals": "float64"})
    no_arrivals.loc[[4, 5], "arrivals"] = np.nan
    with pytest.raises(ValueError, match="valores vacíos.*arrivals: 2"):
        evaluate(predictions, no_arrivals, model="baseline", split="val", prueba_final=FINAL)


def test_evaluate_no_califica_cuantiles_cruzados(scored):
    predictions, station_hour = scored
    crossed = predictions.copy()
    crossed.loc[0, ["arr_q10", "arr_q90"]] = [9.0, 0.0]  # el ancho del intervalo saldría negativo
    with pytest.raises(ValueError, match="1 filas con arr_q10 mayor que arr_q90"):
        evaluate(crossed, station_hour, model="baseline", split="val", prueba_final=FINAL)


def test_evaluate_exige_una_prediccion_por_cada_fila_real(scored):
    """Para comparar dos modelos sobre las mismas filas: un modelo que predice menos filas no pasa sin avisar."""
    predictions, station_hour = scored
    half = predictions.iloc[:4]  # solo la primera estación
    with pytest.raises(ValueError, match="4 filas de station_hour no tienen predicción") as error:
        evaluate(half, station_hour, model="baseline", split="val", prueba_final=FINAL)
    assert "exigir_todas=False" in str(error.value)
    # Con el permiso explícito se califican solo las filas que sí se predijeron.
    partial = evaluate(half, station_hour, model="baseline", split="val", prueba_final=FINAL, exigir_todas=False)
    assert pick(partial, "dep", "general", "todas")["n"] == 4
    # Pasando solo las filas reales de esas predicciones no hace falta el permiso.
    same = evaluate(half, station_hour.iloc[:4], model="baseline", split="val", prueba_final=FINAL)
    pd.testing.assert_frame_equal(same, partial)


# --------------------------------------------------------------------------- #
# e) Candado de la prueba final
# --------------------------------------------------------------------------- #

def test_calificar_la_prueba_final_falla_si_no_se_pide_de_forma_explicita(scored):
    predictions, station_hour = scored
    final = predictions.assign(split="test")
    with pytest.raises(FinalTestError, match="Decisión 4") as error:
        evaluate(final, station_hour, model="baseline", split="test")
    assert "una sola vez" in str(error.value) and "usar_prueba_final=True" in str(error.value)
    # El candado salta antes de mirar los datos: tampoco importa si no hay filas de ese tramo.
    with pytest.raises(FinalTestError):
        evaluate(predictions, station_hour, model="baseline", split="test")
    with pytest.raises(FinalTestError):
        evaluate(final, station_hour, model="baseline", split="test", usar_prueba_final=False)


def test_el_candado_mira_las_fechas_y_no_solo_la_etiqueta(scored):
    predictions, station_hour = scored
    # Si la prueba final empezara el lunes 7, cuatro filas marcadas "val" caerían dentro.
    early = Tramo("prueba", date(2026, 9, 7), date(2026, 9, 13))
    with pytest.raises(FinalTestError, match="4 predicciones.*desde 2026-09-07") as error:
        evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=early)
    assert "Decisión 4" in str(error.value) and "usar_prueba_final=True" in str(error.value)
    # Con el permiso explícito sí se califican.
    allowed = evaluate(
        predictions, station_hour, model="baseline", split="val", prueba_final=early, usar_prueba_final=True
    )
    assert pick(allowed, "dep", "general", "todas")["n"] == 8
    # Si empieza el martes 8, ninguna fila cae dentro.
    later = Tramo("prueba", date(2026, 9, 8), date(2026, 9, 14))
    assert pick(
        evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=later), "dep", "general", "todas"
    )["n"] == 8


def test_el_candado_cuenta_el_dia_por_el_reloj_de_nueva_york():
    """Las 23:00 del domingo 6 ya son lunes 7 en UTC, pero todavía no son prueba final. Las 00:00 del 7 sí."""
    hours = ts("2026-09-06 23:00", "2026-09-07 00:00")
    station_hour = pd.DataFrame({"station_id": "A", "ts_hour": hours, "departures": [1, 2], "arrivals": [0, 1]})
    predictions = station_hour[["station_id", "ts_hour"]].assign(model="lgbm", split="val")
    for column in ("dep_q10", "dep_q50", "dep_q90", "arr_q10", "arr_q50", "arr_q90"):
        predictions[column] = 1.0
    final = Tramo("prueba", date(2026, 9, 7), date(2026, 9, 13))
    with pytest.raises(FinalTestError, match="1 predicciones"):
        evaluate(predictions, station_hour, model="lgbm", split="val", prueba_final=final)
    before = evaluate(predictions.iloc[:1], station_hour.iloc[:1], model="lgbm", split="val", prueba_final=final)
    assert pick(before, "dep", "general", "todas")["n"] == 1
    # Lo mismo con las fechas guardadas en UTC.
    in_utc = predictions.assign(ts_hour=predictions["ts_hour"].dt.tz_convert("UTC"))
    with pytest.raises(FinalTestError, match="1 predicciones"):
        evaluate(in_utc, station_hour, model="lgbm", split="val", prueba_final=final)
    # Una fecha sin zona se rechaza con el mensaje de siempre.
    naive = predictions.assign(ts_hour=predictions["ts_hour"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="zona horaria"):
        evaluate(naive, station_hour, model="lgbm", split="val", prueba_final=final)


def test_sin_decir_cual_es_la_prueba_final_el_candado_usa_config_datos_yaml(scored):
    """Las mismas filas inventadas, movidas al primer día de la prueba final del proyecto y marcadas "val"."""
    predictions, station_hour = scored
    final = load_datos(DATOS_YAML).tramos["prueba"]
    hours = pd.Series(pd.date_range(final.start, periods=4, freq="h").as_unit("us"))
    moved_actual = station_hour.assign(ts_hour=pd.concat([hours, hours], ignore_index=True))
    moved = predictions.assign(ts_hour=moved_actual["ts_hour"])
    with pytest.raises(FinalTestError, match=f"8 predicciones.*desde {final.inicio}"):
        evaluate(moved, moved_actual, model="baseline", split="val")


def test_la_prueba_final_se_califica_solo_con_el_argumento_explicito(scored):
    predictions, station_hour = scored
    final = predictions.assign(split="test")
    metrics = evaluate(final, station_hour, model="baseline", split="test", usar_prueba_final=True)
    assert set(metrics["split"]) == {"test"}
    assert pick(metrics, "dep", "general", "todas")["n"] == 8


# --------------------------------------------------------------------------- #
# f) Guardar y mostrar
# --------------------------------------------------------------------------- #

def test_write_metrics_guarda_el_esquema_y_conserva_otros_modelos(scored, tmp_path):
    predictions, station_hour = scored
    path = tmp_path / "reports" / "metricas_validacion.parquet"
    baseline = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL)
    lgbm = evaluate(predictions.assign(model="lgbm"), station_hour, model="lgbm", split="val", prueba_final=FINAL)

    write_metrics(lgbm, path)
    write_metrics(baseline, path)
    assert pq.read_schema(path).equals(METRICS_SCHEMA, check_metadata=False)
    saved = pq.read_table(path).to_pandas()
    assert saved["model"].value_counts().to_dict() == {"lgbm": len(lgbm), "baseline": len(baseline)}

    # Volver a guardar la línea base reemplaza sus filas, no las duplica.
    changed = baseline.assign(mae=baseline["mae"] + 1)
    write_metrics(changed, path)
    saved = pq.read_table(path).to_pandas()
    assert saved["model"].value_counts().to_dict() == {"lgbm": len(lgbm), "baseline": len(baseline)}
    again = saved[saved["model"] == "baseline"].reset_index(drop=True)
    assert np.allclose(again["mae"], changed["mae"])
    assert again["wape"].isna().sum() == changed["wape"].isna().sum() > 0  # los vacíos siguen vacíos


def test_write_metrics_conserva_el_otro_tramo_del_mismo_modelo(scored, tmp_path):
    """Guardar la prueba final de un modelo no borra su validación, ni al revés."""
    predictions, station_hour = scored
    path = tmp_path / "metricas.parquet"
    lgbm = predictions.assign(model="lgbm")
    val = evaluate(lgbm, station_hour, model="lgbm", split="val", prueba_final=FINAL)
    final = evaluate(lgbm.assign(split="test"), station_hour, model="lgbm", split="test", usar_prueba_final=True)

    write_metrics(val, path)
    write_metrics(final, path)
    counts = {("lgbm", "val"): len(val), ("lgbm", "test"): len(final)}
    saved = pq.read_table(path).to_pandas()
    assert saved.groupby(["model", "split"]).size().to_dict() == counts

    # Volver a guardar validación reemplaza solo las filas de validación.
    write_metrics(val.assign(mae=val["mae"] + 1), path)
    saved = pq.read_table(path).to_pandas()
    assert saved.groupby(["model", "split"]).size().to_dict() == counts
    assert np.allclose(saved.loc[saved["split"] == "val", "mae"], val["mae"] + 1)
    assert np.allclose(saved.loc[saved["split"] == "test", "mae"], final["mae"])


def test_write_metrics_no_pisa_un_archivo_con_otras_columnas(tmp_path):
    path = tmp_path / "metricas_validacion.parquet"
    pd.DataFrame({"otra": [1]}).to_parquet(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="otras columnas"):
        write_metrics(pd.DataFrame(columns=METRIC_COLUMNS), path)
    assert path.read_bytes() == before


def test_format_table_lista_primero_salidas_y_despues_llegadas(scored):
    predictions, station_hour = scored
    metrics = evaluate(predictions, station_hour, model="baseline", split="val", prueba_final=FINAL)

    lines = format_table(metrics, "tipo_de_dia", "tipo de día").splitlines()
    assert len(lines) == 2 + 6 and len({len(line) for line in lines}) == 1  # encabezado, raya y seis filas alineadas
    cells = [[cell.strip() for cell in line.split("|")[1:-1]] for line in lines]
    assert cells[0] == ["objetivo", "tipo de día", "filas", "MAE", "RMSE", "WAPE", "cobertura", "ancho medio"]
    assert [row[:2] for row in cells[2:]] == [
        ["salidas", "laborable"], ["salidas", "fin de semana"], ["salidas", "festivo"],
        ["llegadas", "laborable"], ["llegadas", "fin de semana"], ["llegadas", "festivo"],
    ]
    # Festivo, salidas: reales 5, 10, 0 y 0 contra 3, 6, 1 y 1. Anchos 4, 7, 1 y 1.
    assert cells[4] == ["salidas", "festivo", "4", "2.000", "2.345", "53.3 %", "75.0 %", "3.250"]

    # La estación sin salidas reales muestra "n/d" en su WAPE.
    by_station = [[cell.strip() for cell in line.split("|")[1:-1]] for line in format_table(metrics, "estacion", "estación").splitlines()]
    assert by_station[2][:6] == ["salidas", "5470.1", "4", "1.000", "1.000", "n/d"]
