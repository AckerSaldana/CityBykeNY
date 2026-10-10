"""Pruebas de la simulación completa (src/sim/simulate.py y scripts/run_simulation.py).

Con una red chica y datos hechos aquí mismo:

a) Las tablas cumplen sus contratos y traen las filas esperadas.
b) El día de calentamiento no cuenta en las métricas ni en las
   recomendaciones, pero su inventario sí pasa al día siguiente.
c) Con presupuesto 0 las tres políticas dan lo mismo que no intervenir, y
   "none" da lo mismo con cualquier presupuesto.
d) Dos corridas con las mismas entradas dan salidas idénticas.
e) El candado de la prueba final (decisión 4).
f) El intervalo por remuestreo de días se repite con la semilla y contiene a
   la media.
g) Qué estaciones se simulan y cuáles quedan fuera, con sus conteos.
h) Nivel de riesgo y `feasible` de las recomendaciones, a mano.
i) El script de punta a punta, y lo que lee de cada archivo.
j) El reporte usa el lenguaje honesto y declara las limitaciones.
k) Lo que el reporte dice en palabras sale de los números de la corrida: la
   lectura de la primera tabla, las diferencias entre las dos políticas, las
   bicis que entran y salen, y las estaciones con historia y sin pronóstico.
l) La huella de las entradas: con ella una prueba dice, con el comando que hay
   que correr, cuándo las salidas ya no corresponden a las tablas, las reglas o
   el config de hoy.

Todo corre sin red y sin tocar data/processed. Las dos últimas pruebas miran
los archivos reales y se saltan si todavía no existen.
"""

import copy
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.contracts import (
    POLICIES,
    QUANTILE_COLUMNS,
    STATION_TYPES,
    read_contract,
    to_contract_table,
    validate_file,
    validate_table,
    write_contract,
)
from src.data.datos_config import Tramo, load_datos
from src.models.metrics import FinalTestError
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML, TZ
from src.sim.engine import OUTSIDE, Reglas, SimulacionError, build_network, initial_inventory
from src.sim.policies import ForecastWindow
from src.sim.simulate import (
    FINAL_FOLDER,
    RERUN,
    build_report,
    check_final_test,
    current_fingerprint,
    interval,
    limitations,
    load_forecast,
    load_history,
    load_trips,
    main,
    resample_days,
    risk_levels,
    run_simulation,
    select_stations,
    stale_inputs,
    stale_message,
    summarize_recommendations,
    summary_lines,
)

TRAINING = Tramo("entrenamiento", date(2026, 8, 24), date(2026, 8, 31))
VALIDATION = Tramo("validacion", date(2026, 9, 1), date(2026, 9, 4))
FINAL = Tramo("prueba", date(2026, 9, 5), date(2026, 9, 6))
SETTINGS = SimpleNamespace(dias_de_calentamiento=1, remuestreos=300, nivel_del_intervalo=0.9, semilla=7)
SIMULATED = ["A", "B", "C", "D", "E", "F"]   # G no tiene capacidad y H no tiene viajes en validación

DATOS_TEXT = """\
meses_crudos: [202608, 202609]
periodo: {inicio: 2026-08-24, fin: 2026-09-06}
tramos:
  entrenamiento: {inicio: 2026-08-24, fin: 2026-08-31}
  validacion:    {inicio: 2026-09-01, fin: 2026-09-04}
  prueba:        {inicio: 2026-09-05, fin: 2026-09-06}
union_de_ids: {distancia_maxima_m: 1.0}
estaciones_excluidas: {prefijos: [HB], ids: []}
simulacion: {dias_de_calentamiento: 1, remuestreos: 300, nivel_del_intervalo: 0.90, semilla: 7}
"""


def make_cfg(**changes) -> dict:
    cfg = {
        "estado": "provisional",
        "confirmado_por": None,
        "fecha_confirmacion": None,
        "franjas": [
            {"nombre": "manana", "momento_intervencion": "05:00", "inicio": "06:00", "fin": "12:00",
             "parte_del_presupuesto": 0.5},
            {"nombre": "tarde", "momento_intervencion": "14:00", "inicio": "15:00", "fin": "21:00",
             "parte_del_presupuesto": 0.5},
        ],
        "umbral_riesgo": {"bicis_minimas": 2, "lugares_libres_minimos": 2},
        "presupuestos_por_estacion": [0, 0.5, 1.0, 2.0],
        "presupuesto_de_referencia": 1.0,
        "distancia_maxima_km": 3.0,
        "tipo_si_falta": "mixta",
        "estrategia_por_tipo": {
            kind: {"prioridad": 2, "ocupacion_respaldo": {"manana": 0.5, "tarde": 0.5}} for kind in STATION_TYPES
        },
    }
    cfg.update(copy.deepcopy(changes))
    return cfg


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ)


def stamps(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(values).as_unit("us")


def make_trips(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"])
    for column in ("started_at", "ended_at"):
        frame[column] = stamps([ts(value) for value in frame[column]])
    return frame


def in_tramo(trips: pd.DataFrame, tramo: Tramo) -> pd.DataFrame:
    """Lo que entrega el lector: viajes con salida o llegada en el tramo."""
    keep = tramo.contains(trips["started_at"]) | tramo.contains(trips["ended_at"])
    return trips[keep].reset_index(drop=True)


def make_case(seed: int = 0) -> dict:
    """Ocho estaciones y dos semanas de datos al azar: entrenamiento, validación y prueba.

    A a F se simulan. G no tiene capacidad (y sí viajes). H tiene capacidad y
    ningún viaje en validación. F no tiene historia en entrenamiento.
    """
    rng = np.random.default_rng(seed)
    names = SIMULATED + ["G", "H"]
    stations = pd.DataFrame(
        {
            "station_id": names,
            "name": [f"Estación {name}" for name in names],
            "lat": 40.70 + rng.random(len(names)) * 0.01,
            "lon": -73.99 + rng.random(len(names)) * 0.01,
            "capacity": pd.array([8, 10, 12, 9, 14, 10, None, 12], dtype="Int64"),
            "station_type": pd.Series([None] * len(names), dtype="object"),
        }
    )
    n = 2600
    options = np.array(SIMULATED + ["G", None], dtype=object)
    started = ts("2026-08-30") + pd.to_timedelta(rng.integers(0, 8 * 24 * 3600, n), unit="s")
    trips = pd.DataFrame(
        {
            "ride_id": [f"R{i:05d}" for i in rng.permutation(n)],
            "started_at": stamps(started),
            "ended_at": stamps(started + pd.to_timedelta(rng.integers(60, 2400, n), unit="s")),
            "start_station_id": options[rng.choice(len(options), n, p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.07, 0.03])],
            "end_station_id": options[rng.choice(len(options), n, p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.07, 0.03])],
        }
    ).sort_values("started_at", kind="stable").reset_index(drop=True)

    hours = pd.date_range("2026-08-24", "2026-09-06 23:00", freq="h", tz=TZ)
    with_rows = [name for name in names if name not in ("F", "H")]
    station_hour = pd.DataFrame(
        {"station_id": np.repeat(with_rows, len(hours)), "ts_hour": stamps(np.tile(hours, len(with_rows)))}
    )
    station_hour["departures"] = rng.integers(0, 5, len(station_hour))
    station_hour["arrivals"] = rng.integers(0, 5, len(station_hour))
    station_hour["net_flow"] = station_hour["arrivals"] - station_hour["departures"]
    station_hour["departures_member"] = station_hour["departures"]
    station_hour["departures_casual"] = 0

    predicted = [name for name in names if name != "H"]
    predictions = pd.DataFrame(
        {"station_id": np.repeat(predicted, len(hours)), "ts_hour": stamps(np.tile(hours, len(predicted)))}
    )
    predictions = predictions[predictions["ts_hour"] >= VALIDATION.start].reset_index(drop=True)
    predictions["model"] = "baseline"
    predictions["split"] = np.where(predictions["ts_hour"] < FINAL.start, "val", "test")
    for prefix in ("dep", "arr"):
        middle = rng.random(len(predictions)) * 2.5
        predictions[f"{prefix}_q10"] = middle * 0.6
        predictions[f"{prefix}_q50"] = middle
        predictions[f"{prefix}_q90"] = middle * 1.5
    predictions = predictions[["station_id", "ts_hour", "model", "split", *QUANTILE_COLUMNS]]
    return {"stations": stations, "trips": trips, "station_hour": station_hour, "predictions": predictions}


def simulate_case(case: dict, tramo: Tramo = VALIDATION, settings=SETTINGS, cfg: dict | None = None, **options):
    history = case["station_hour"][case["station_hour"]["ts_hour"] < tramo.start]
    split = "val" if tramo.nombre == "validacion" else "test"
    predictions = case["predictions"]
    predictions = predictions[(predictions["split"] == split) & tramo.contains(predictions["ts_hour"])]
    return run_simulation(
        case["stations"], in_tramo(case["trips"], tramo), history[["station_id", "ts_hour", "net_flow"]],
        predictions, Reglas.from_operacion(cfg or make_cfg()), tramo, settings,
        forecast_model="baseline", prueba_final=FINAL, **options,
    )


@pytest.fixture(scope="module")
def case() -> dict:
    return make_case()


@pytest.fixture(scope="module")
def result(case):
    return simulate_case(case)


def row_of(report: dict, policy: str, budget: int) -> dict:
    (row,) = [
        r for r in report["resultados"] if r["politica"] == policy and r["presupuesto_del_dia_en_bicis"] == budget
    ]
    return row


# --------------------------------------------------------------------------- #
# a) Contratos y filas
# --------------------------------------------------------------------------- #

def test_las_tablas_cumplen_sus_contratos(result):
    assert validate_table(to_contract_table(result.recommendations, "recommendations"), "recommendations") == []
    assert validate_table(to_contract_table(result.policy_curve, "policy_curve"), "policy_curve") == []


def test_recommendations_trae_una_fila_por_estacion_franja_dia_evaluado_y_politica(result):
    table = result.recommendations
    assert result.network.station_ids == tuple(SIMULATED)
    assert len(table) == 6 * 2 * 3 * 3   # estaciones x franjas x días evaluados x políticas
    assert set(table["station_id"]) == set(SIMULATED) and set(table["policy"]) == set(POLICIES)
    # Franjas de 06:00 a 12:00 y de 15:00 a 21:00, hora de Nueva York, de los días 2, 3 y 4 (el 1 es calentamiento).
    windows = table[["window_start", "window_end"]].drop_duplicates().sort_values("window_start")
    expected = [
        (ts(f"2026-09-0{day} {start}"), ts(f"2026-09-0{day} {end}"))
        for day in (2, 3, 4)
        for start, end in (("06:00", "12:00"), ("15:00", "21:00"))
    ]
    assert list(map(tuple, windows.to_numpy())) == expected
    assert str(table["window_start"].dt.tz) == TZ
    assert (table.groupby(["window_start", "policy"]).size() == 6).all()


def test_policy_curve_trae_una_fila_por_politica_y_presupuesto(result):
    curve = result.policy_curve
    # 6 estaciones simuladas: 0, 0.5, 1 y 2 bicis por estación son 0, 3, 6 y 12 bicis al día.
    assert sorted(set(curve["budget"])) == [0, 3, 6, 12]
    assert len(curve) == 3 * 4 and not curve.duplicated(["policy", "budget"]).any()
    assert ((curve["ci_low"] <= curve["risk_station_hours"]) & (curve["risk_station_hours"] <= curve["ci_high"])).all()
    for row in curve.itertuples():
        risk = row_of(result.report, row.policy, row.budget)["riesgo_por_dia"]
        assert (row.risk_station_hours, row.ci_low, row.ci_high) == (risk["media"], risk["ic_bajo"], risk["ic_alto"])


def test_las_recomendaciones_dicen_lo_que_paso_en_cada_intervencion(result):
    table, network = result.recommendations, result.network
    reference = result.report["reglas"]["presupuesto_de_referencia_en_bicis"]
    assert reference == 6
    order = {name: i for i, name in enumerate(network.station_ids)}
    for policy in POLICIES:
        run = result.runs[(policy, reference)]
        records = [record for record in run.interventions if record.day_index >= 1]
        rows = table[table["policy"] == policy]
        assert len(records) == rows["window_start"].nunique() == 6
        for record, (_, group) in zip(records, rows.groupby("window_start", sort=True)):
            group = group.sort_values("station_id", key=lambda ids: ids.map(order))
            # bikes_to_move es lo que recibió (+) o entregó (-); el objetivo es el inventario justo después.
            assert group["bikes_to_move"].tolist() == record.net.tolist()
            assert group["start_inventory_target"].tolist() == (record.inventory_before + record.net).tolist()
            assert int(group["bikes_to_move"].sum()) == 0   # los traslados conservan las bicis
            assert int(group["bikes_to_move"].clip(lower=0).sum()) == record.budget_used
    assert (table.loc[table["policy"] == "none", "bikes_to_move"] == 0).all()
    assert (table.loc[table["policy"] != "none", "bikes_to_move"] != 0).any()   # las otras dos sí mueven
    assert (table["start_inventory_target"] <= table["station_id"].map(dict(zip(SIMULATED, network.capacity)))).all()
    # feasible es de la estación y la franja: vale lo mismo en las tres políticas.
    assert (table.groupby(["station_id", "window_start"])["feasible"].nunique() == 1).all()
    # F no tiene historia: no tiene información, queda en feasible verdadero y nadie la mueve.
    f_rows = table[table["station_id"] == "F"]
    assert f_rows["feasible"].all() and (f_rows["bikes_to_move"] == 0).all()
    assert result.report["informacion"]["estaciones_sin_historia"] == 1


# --------------------------------------------------------------------------- #
# b) Calentamiento
# --------------------------------------------------------------------------- #

def warmup_case():
    """Dos estaciones de 10 lugares; empiezan con 5 (ocupación de respaldo 0.5). Umbral: menos de 2.

    Día 1 (calentamiento). De A salen bicis hacia fuera a las 08:00, 08:10, 08:20 y 08:30: queda con 1 desde
      las 08:30 (riesgo). 09:00 sale otra (queda en 0). 09:30 otra salida: bloqueada.
      Riesgo del día 1: 08:30 a 24:00 = 15.5 h. Bloqueados: 1.
    Día 2 (cuenta). A sigue en 0. Salidas a las 03:00 y 04:00: bloqueadas (2). Llega una a las 10:00 (1 bici)
      y otra a las 11:00 (2 bicis, fin del riesgo).
      Riesgo del día 2: 00:00 a 11:00 = 11 h: 6 fuera de franja y 5 de la mañana.
      A B le llega una bici de fuera a las 12:00 (queda con 6). B nunca está en riesgo.
    """
    stations = pd.DataFrame(
        {"station_id": ["A", "B"], "name": ["a", "b"], "lat": [40.70, 40.705], "lon": [-73.99, -73.99],
         "capacity": pd.array([10, 10], dtype="Int64"), "station_type": pd.Series([None, None], dtype="object")}
    )
    rows = [(f"d{i}", f"2026-09-01 {hhmm}", f"2026-09-01 {hhmm}", "A", None)
            for i, hhmm in enumerate(["08:00", "08:10", "08:20", "08:30", "09:00", "09:30"])]
    rows += [("e1", "2026-09-02 03:00", "2026-09-02 03:05", "A", None),
             ("e2", "2026-09-02 04:00", "2026-09-02 04:05", "A", None),
             ("a1", "2026-09-02 09:50", "2026-09-02 10:00", None, "A"),
             ("a2", "2026-09-02 10:50", "2026-09-02 11:00", None, "A"),
             ("b1", "2026-09-02 11:50", "2026-09-02 12:00", None, "B")]
    history = pd.DataFrame({"station_id": ["B"], "ts_hour": stamps([ts("2026-08-31 08:00")]), "net_flow": [0]})
    predictions = pd.DataFrame(
        [("B", ts("2026-09-02 08:00"), "baseline", "val", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        columns=["station_id", "ts_hour", "model", "split", *QUANTILE_COLUMNS],
    )
    predictions["ts_hour"] = stamps(predictions["ts_hour"])
    return stations, make_trips(rows), history, predictions


def test_el_calentamiento_no_cuenta_pero_su_inventario_pasa_al_dia_siguiente():
    stations, trips, history, predictions = warmup_case()
    tramo = Tramo("validacion", date(2026, 9, 1), date(2026, 9, 2))
    out = run_simulation(
        stations, trips, history, predictions, Reglas.from_operacion(make_cfg()), tramo, SETTINGS,
        forecast_model="baseline", prueba_final=FINAL,
    )
    report = out.report
    assert report["tramo"]["dias_de_calentamiento"] == ["2026-09-01"]
    assert report["tramo"]["dias_evaluados"] == ["2026-09-02"]
    none = row_of(report, "none", 0)
    # Solo cuenta el día 2: 11 horas en riesgo y 2 viajes bloqueados.
    assert none["riesgo_por_dia"] == {"media": 11.0, "ic_bajo": 11.0, "ic_alto": 11.0}
    assert none["riesgo_de_cada_dia"] == [11.0]
    assert {label: stat["media"] for label, stat in none["riesgo_por_franja"].items()} == {
        "manana": 5.0, "tarde": 0.0, OUTSIDE: 6.0,
    }
    assert {kind: stat["media"] for kind, stat in none["riesgo_por_tipo"].items()} == {
        "pocas_bicis": 11.0, "pocos_lugares": 0.0,
    }
    assert none["viajes_bloqueados"] == {"por_dia": 2.0, "total": 2, "cada_dia": [2]}
    # Lo del día 1 queda aparte y marcado como que no cuenta.
    assert none["calentamiento_no_cuenta"]["horas_de_estacion_en_riesgo"] == 15.5
    assert none["calentamiento_no_cuenta"]["viajes_bloqueados"] == 1
    # Fronteras del día evaluado: entran 3 (las llegadas) y no sale ninguna (las dos salidas se bloquearon).
    assert none["frontera"]["entran"] == 3 and none["frontera"]["salen"] == 0 and none["frontera"]["neto"] == 3
    # Todo el tramo: 10 bicis al inicio, entran 3 y salen las 5 del día 1: quedan 8 (A con 2 y B con 6).
    assert none["bicis_en_la_red"] == {
        "al_inicio_del_tramo": 10, "al_final_del_tramo": 8,
        "entran_en_todo_el_tramo": 3, "salen_en_todo_el_tramo": 5,
    }
    assert out.policy_curve["risk_station_hours"].tolist() == [11.0] * len(out.policy_curve)

    # Recomendaciones: solo las dos franjas del día 2. El inventario viene del día 1, no del inicial.
    table = out.recommendations
    assert sorted(table["window_start"].unique()) == [ts("2026-09-02 06:00"), ts("2026-09-02 15:00")]
    none_rows = table[table["policy"] == "none"].set_index(["station_id", "window_start"])
    assert none_rows.loc[("A", ts("2026-09-02 06:00")), "start_inventory_target"] == 0   # empezó en 5
    assert none_rows.loc[("A", ts("2026-09-02 15:00")), "start_inventory_target"] == 2
    assert none_rows.loc[("B", ts("2026-09-02 06:00")), "start_inventory_target"] == 5
    assert none_rows.loc[("B", ts("2026-09-02 15:00")), "start_inventory_target"] == 6
    # A no tiene información: "alto" mientras está bajo el umbral y "bajo" cuando ya tiene 2 bicis.
    assert none_rows.loc[("A", ts("2026-09-02 06:00")), "risk_level"] == "alto"
    assert none_rows.loc[("A", ts("2026-09-02 15:00")), "risk_level"] == "bajo"
    assert none_rows["feasible"].all()


def test_las_llegadas_sin_lugar_del_calentamiento_no_cuentan():
    """Dos estaciones de 3 lugares que empiezan con 2 bicis (0.5 x 3 = 1.5, sube a 2).

    Día 1 (calentamiento): llegan tres bicis de fuera a A. La primera cabe; la segunda encuentra A llena y va
      a B; la tercera no cabe en ninguna. Desviadas: 2. Sin lugar en ninguna estación: 1.
    Día 2 (cuenta): sale una bici de A y otra de B. Ninguna llegada.
    """
    stations = pd.DataFrame(
        {"station_id": ["A", "B"], "name": ["a", "b"], "lat": [40.70, 40.705], "lon": [-73.99, -73.99],
         "capacity": pd.array([3, 3], dtype="Int64"), "station_type": pd.Series([None, None], dtype="object")}
    )
    trips = make_trips(
        [(f"n{i}", f"2026-09-01 08:0{i}", f"2026-09-01 08:1{i}", None, "A") for i in range(3)]
        + [("a1", "2026-09-02 09:00", "2026-09-02 09:05", "A", None),
           ("b1", "2026-09-02 10:00", "2026-09-02 10:05", "B", None)]
    )
    _, _, history, predictions = warmup_case()
    tramo = Tramo("validacion", date(2026, 9, 1), date(2026, 9, 2))
    out = run_simulation(
        stations, trips, history, predictions, Reglas.from_operacion(make_cfg()), tramo, SETTINGS,
        forecast_model="baseline", prueba_final=FINAL,
    )
    run = out.runs[("none", 0)]
    assert run.diverted.tolist() == [2, 0] and run.no_room.tolist() == [1, 0]
    none = row_of(out.report, "none", 0)
    # Las métricas son de los días evaluados: ni desviadas ni llegadas sin lugar.
    assert none["llegadas_desviadas"]["total"] == 0 and none["llegadas_sin_lugar_en_ninguna_estacion"] == 0
    # Lo del calentamiento queda aparte, como los demás conteos.
    assert none["calentamiento_no_cuenta"]["llegadas_desviadas"] == 2
    assert none["calentamiento_no_cuenta"]["llegadas_sin_lugar_en_ninguna_estacion"] == 1


def test_sin_calentamiento_cuentan_todos_los_dias():
    stations, trips, history, predictions = warmup_case()
    tramo = Tramo("validacion", date(2026, 9, 1), date(2026, 9, 2))
    settings = SimpleNamespace(dias_de_calentamiento=0, remuestreos=300, nivel_del_intervalo=0.9, semilla=7)
    out = run_simulation(
        stations, trips, history, predictions, Reglas.from_operacion(make_cfg()), tramo, settings,
        forecast_model="baseline", prueba_final=FINAL,
    )
    none = row_of(out.report, "none", 0)
    assert none["riesgo_de_cada_dia"] == [15.5, 11.0] and none["riesgo_por_dia"]["media"] == 13.25
    assert none["viajes_bloqueados"]["total"] == 3
    assert out.recommendations["window_start"].nunique() == 4


def test_el_calentamiento_no_puede_cubrir_todo_el_tramo(case):
    settings = SimpleNamespace(dias_de_calentamiento=4, remuestreos=300, nivel_del_intervalo=0.9, semilla=7)
    with pytest.raises(SimulacionError, match="dias_de_calentamiento"):
        simulate_case(case, settings=settings)


# --------------------------------------------------------------------------- #
# c) Presupuesto 0 y "none"
# --------------------------------------------------------------------------- #

def test_con_presupuesto_cero_las_tres_politicas_son_la_corrida_sin_intervencion(result):
    curve = result.policy_curve.set_index(["policy", "budget"])
    base = curve.loc[("none", 0)]
    for policy in POLICIES:
        assert curve.loc[(policy, 0)].equals(base)
        row = row_of(result.report, policy, 0)
        assert row["bicis_movidas"]["total"] == 0 and row["riesgo_simulado_evitado_por_dia"]["media"] == 0.0
        assert row["simulada"] == (policy == "none")   # las otras dos no se vuelven a simular
    # "none" da lo mismo con cualquier presupuesto.
    for budget in (3, 6, 12):
        assert curve.loc[("none", budget)].equals(base)
        assert row_of(result.report, "none", budget)["bicis_movidas"]["total"] == 0
    # Con presupuesto, las otras dos sí mueven bicis y nunca más de lo disponible.
    for policy in ("historical", "model"):
        for budget in (3, 6, 12):
            row = row_of(result.report, policy, budget)
            assert 0 < row["bicis_movidas"]["por_dia"] <= budget
            for franja in row["presupuesto_por_franja"].values():
                assert all(u <= a for u, a in zip(franja["usado_cada_dia"], franja["disponible_cada_dia"]))


def test_el_reporte_dice_el_presupuesto_disponible_y_usado_por_franja(result):
    # 6 bicis al día, mitad y mitad: 3 en la mañana; la tarde dispone de 3 más lo que no usó la mañana.
    for policy in ("historical", "model"):
        budget = row_of(result.report, policy, 6)["presupuesto_por_franja"]
        assert budget["manana"]["disponible_cada_dia"] == [3, 3, 3]
        for used, available in zip(budget["manana"]["usado_cada_dia"], budget["tarde"]["disponible_cada_dia"]):
            assert available == 3 + (3 - used)
        assert budget["tarde"]["usado_total"] == sum(budget["tarde"]["usado_cada_dia"])
    # Sin mover nada, todo el presupuesto del día queda disponible (y sin usar) en la última franja.
    unused = row_of(result.report, "none", 12)["presupuesto_por_franja"]
    assert unused["manana"]["disponible_cada_dia"] == [6, 6, 6] and unused["tarde"]["disponible_cada_dia"] == [12] * 3
    assert unused["manana"]["usado_total"] == unused["tarde"]["usado_total"] == 0
    assert result.report["reglas"]["reparto_por_franja"] == {
        "0": {"manana": 0, "tarde": 0}, "3": {"manana": 2, "tarde": 1},
        "6": {"manana": 3, "tarde": 3}, "12": {"manana": 6, "tarde": 6},
    }


def test_dos_presupuestos_que_dan_el_mismo_total_se_simulan_una_vez(case):
    # Con 6 estaciones, 0.05 por estación son 0.3 bicis: redondea a 0, igual que el presupuesto 0.
    out = simulate_case(case, cfg=make_cfg(presupuestos_por_estacion=[0, 0.05, 1.0]))
    assert sorted(set(out.policy_curve["budget"])) == [0, 6]
    assert out.report["reglas"]["presupuestos_con_el_mismo_total_que_otro"] == [0.05]


# --------------------------------------------------------------------------- #
# d) Resultado repetible
# --------------------------------------------------------------------------- #

def test_dos_corridas_con_las_mismas_entradas_dan_salidas_identicas(case, result):
    again = simulate_case(case)
    pd.testing.assert_frame_equal(again.recommendations, result.recommendations)
    pd.testing.assert_frame_equal(again.policy_curve, result.policy_curve)
    assert again.report == result.report
    assert {key: run.digest() for key, run in again.runs.items()} == {
        key: run.digest() for key, run in result.runs.items()
    }
    # El orden de las filas de entrada no cambia el resultado.
    shuffled = dict(case)
    shuffled["trips"] = case["trips"].sample(frac=1.0, random_state=1).reset_index(drop=True)
    shuffled["stations"] = case["stations"].iloc[::-1].reset_index(drop=True)
    other = simulate_case(shuffled)
    pd.testing.assert_frame_equal(other.policy_curve, result.policy_curve)
    pd.testing.assert_frame_equal(other.recommendations, result.recommendations)


def test_las_tres_politicas_parten_del_mismo_inventario_y_ven_los_mismos_viajes(result):
    runs = result.runs
    # Inventario inicial: ocupación de respaldo 0.5 por la capacidad (8, 10, 12, 9, 14, 10); 4.5 sube a 5.
    assert initial_inventory(result.network, Reglas.from_operacion(make_cfg())).tolist() == [4, 5, 6, 5, 7, 5]
    assert {run.initial_total for run in runs.values()} == {32}
    # Hasta la primera intervención nadie ha movido nada: todas las corridas ven el mismo inventario.
    first = {tuple(run.interventions[0].inventory_before.tolist()) for run in runs.values()}
    assert len(first) == 1
    assert result.report["inventario_inicial"] == {
        "bicis": 32, "lugares": 63, "franja_que_lo_define": "manana", "estaciones_sin_tipo": 6,
        "tipo_si_falta": "mixta",
    }
    # Los mismos viajes: las bicis que entran al conjunto son las mismas en todas las corridas.
    assert len({tuple(run.entered.tolist()) for run in runs.values()}) == 1
    for run in runs.values():
        assert run.final_total == run.initial_total + int(run.entered.sum()) - int(run.left.sum())
        assert (run.final_inventory >= 0).all() and (run.final_inventory <= result.network.capacity).all()


# --------------------------------------------------------------------------- #
# e) Candado de la prueba final (decisión 4)
# --------------------------------------------------------------------------- #

def test_la_prueba_final_no_se_simula_sin_pedirlo():
    with pytest.raises(FinalTestError, match="Decisión 4"):
        check_final_test(FINAL, FINAL, usar_prueba_final=False)
    # El candado mira las fechas, no solo el nombre: un tramo que toca la prueba final tampoco pasa.
    touching = Tramo("validacion", date(2026, 9, 1), date(2026, 9, 5))
    with pytest.raises(FinalTestError, match="Decisión 4"):
        check_final_test(touching, FINAL, usar_prueba_final=False)
    check_final_test(VALIDATION, FINAL, usar_prueba_final=False)   # validación sí
    check_final_test(FINAL, FINAL, usar_prueba_final=True)         # y la prueba final, solo pidiéndola


def test_run_simulation_rechaza_la_prueba_final_antes_de_tocar_los_datos():
    # Las tablas son None: si las tocara, fallaría con otro error.
    with pytest.raises(FinalTestError, match="Decisión 4"):
        run_simulation(None, None, None, None, Reglas.from_operacion(make_cfg()), FINAL, SETTINGS,
                       forecast_model="baseline", prueba_final=FINAL)


def test_la_historia_no_puede_traer_filas_del_tramo_simulado(case):
    history = case["station_hour"][["station_id", "ts_hour", "net_flow"]]
    upto = history[history["ts_hour"] <= VALIDATION.start]   # una sola hora de más: la primera del tramo
    with pytest.raises(SimulacionError, match="tramo simulado o posteriores"):
        run_simulation(
            case["stations"], in_tramo(case["trips"], VALIDATION), upto, case["predictions"],
            Reglas.from_operacion(make_cfg()), VALIDATION, SETTINGS, forecast_model="baseline", prueba_final=FINAL,
        )


# --------------------------------------------------------------------------- #
# f) Intervalo por remuestreo de días
# --------------------------------------------------------------------------- #

def test_el_remuestreo_se_repite_con_la_semilla_y_el_intervalo_contiene_a_la_media():
    draws = resample_days(13, 2000, semilla=20260901)
    assert draws.shape == (2000, 13) and draws.min() == 0 and draws.max() == 12
    assert (draws == resample_days(13, 2000, semilla=20260901)).all()
    assert (draws != resample_days(13, 2000, semilla=1)).any()
    values = np.array([410.0, 380.5, 395.0, 520.0, 300.0, 444.0, 401.5, 399.0, 610.0, 280.0, 350.0, 420.0, 390.0])
    mean, low, high = interval(values, draws, 0.90)
    assert mean == pytest.approx(values.mean()) and low < mean < high
    assert (mean, low, high) == interval(values, resample_days(13, 2000, semilla=20260901), 0.90)
    assert (low, high) != interval(values, resample_days(13, 2000, semilla=1), 0.90)[1:]
    # Un intervalo más ancho contiene al más angosto.
    _, wide_low, wide_high = interval(values, draws, 0.99)
    assert wide_low <= low and high <= wide_high


def test_el_intervalo_a_mano():
    # Tres días (1, 2 y 3) y tres sorteos: solo el primer día, solo el tercero, y uno de cada uno.
    draws = np.array([[0, 0, 0], [2, 2, 2], [0, 1, 2]])
    # Medias de los sorteos: 1, 3 y 2. Con nivel 0.5 el intervalo va del percentil 25 al 75: 1.5 a 2.5.
    assert interval([1.0, 2.0, 3.0], draws, 0.5) == (2.0, 1.5, 2.5)
    # Todos los días iguales: el intervalo no tiene ancho.
    assert interval([4.0, 4.0, 4.0], draws, 0.9) == (4.0, 4.0, 4.0)
    # Un solo día evaluado: tampoco.
    assert interval([7.5], resample_days(1, 100, semilla=3), 0.9) == (7.5, 7.5, 7.5)
    with pytest.raises(SimulacionError, match="No hay días"):
        resample_days(0, 100, semilla=3)


def test_los_intervalos_del_reporte_salen_de_los_dias_evaluados_y_la_diferencia_es_pareada(result):
    report = result.report
    assert report["intervalo"] == {"remuestreos": 300, "nivel": 0.9, "semilla": 7, "dias_remuestreados": 3}
    draws = resample_days(3, 300, semilla=7)
    base = np.array(row_of(report, "none", 0)["riesgo_de_cada_dia"])
    for row in report["resultados"]:
        days = np.array(row["riesgo_de_cada_dia"])
        mean, low, high = interval(days, draws, 0.9)
        assert row["riesgo_por_dia"] == {"media": mean, "ic_bajo": low, "ic_alto": high}
        assert low <= mean <= high
        # Riesgo simulado evitado: diferencia día por día contra "none", con los mismos días sorteados.
        mean, low, high = interval(base - days, draws, 0.9)
        assert row["riesgo_simulado_evitado_por_dia"] == {"media": mean, "ic_bajo": low, "ic_alto": high}
        # Las franjas suman el día y cada tipo de riesgo no pasa del total.
        parts = sum(stat["media"] for stat in row["riesgo_por_franja"].values())
        assert parts == pytest.approx(row["riesgo_por_dia"]["media"])
        assert all(stat["media"] <= row["riesgo_por_dia"]["media"] + 1e-9 for stat in row["riesgo_por_tipo"].values())


# --------------------------------------------------------------------------- #
# g) Conjunto simulado
# --------------------------------------------------------------------------- #

def test_se_simulan_las_estaciones_con_capacidad_y_con_viajes_en_el_tramo():
    stations = pd.DataFrame(
        {
            "station_id": ["A", "B", "C", "D", "E"],
            "name": list("abcde"),
            "lat": [40.70, 40.71, 40.72, 40.73, 40.74],
            "lon": [-73.99] * 5,
            "capacity": pd.array([10, 10, None, 10, None], dtype="Int64"),
            "station_type": pd.Series([None] * 5, dtype="object"),
        }
    )
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", "A", "C"),     # C no tiene capacidad
            ("r3", "2026-09-01 10:00", "2026-09-01 10:10", "C", "B"),
            ("r4", "2026-09-01 11:00", "2026-09-01 11:10", "C", None),    # sin estación de fin
            ("r5", "2026-09-01 12:00", "2026-09-01 12:10", "Z", "A"),     # Z no está en stations
            ("r6", "2026-08-31 12:00", "2026-08-31 12:10", "D", "D"),     # D solo tiene viajes fuera del tramo
            ("r7", "2026-09-02 12:00", "2026-09-02 12:10", "D", "A"),
            ("r8", "2026-09-01 23:55", "2026-09-02 00:05", "B", "D"),     # la llegada a D cae fuera del tramo
        ]
    )
    start, end = ts("2026-09-01").value // 1000, ts("2026-09-02").value // 1000
    chosen, report = select_stations(stations, trips, start, end)
    assert chosen["station_id"].tolist() == ["A", "B"]
    assert report == {
        "en_stations": 5,
        "simuladas": 2,
        # C (3 extremos de viaje en el tramo) y E (ninguno).
        "sin_capacidad": {
            "estaciones": 2, "de_ellas_sin_viajes_en_el_tramo": 1,
            "salidas_en_el_tramo": 2, "llegadas_en_el_tramo": 1,
        },
        "inactivas": {"estaciones": 1, "salidas_en_el_tramo": 0, "llegadas_en_el_tramo": 0},   # D
        "ids_de_viajes_que_no_estan_en_stations": {"ids": 1, "salidas_en_el_tramo": 1, "llegadas_en_el_tramo": 0},
        "extremos_sin_estacion": {"salidas_en_el_tramo": 0, "llegadas_en_el_tramo": 1},
        "salidas_en_el_tramo": 6,
        "llegadas_en_el_tramo": 5,
    }


def test_el_reporte_cuenta_las_estaciones_que_quedan_fuera(case, result):
    stations = result.report["estaciones"]
    assert (stations["en_stations"], stations["simuladas"]) == (8, 6)
    assert stations["sin_capacidad"]["estaciones"] == 1 and stations["inactivas"]["estaciones"] == 1   # G y H
    trips = in_tramo(case["trips"], VALIDATION)
    departs, arrives = VALIDATION.contains(trips["started_at"]), VALIDATION.contains(trips["ended_at"])
    without = stations["sin_capacidad"]
    assert without["salidas_en_el_tramo"] == int((trips.loc[departs, "start_station_id"] == "G").sum())
    assert without["llegadas_en_el_tramo"] == int((trips.loc[arrives, "end_station_id"] == "G").sum())
    counts = result.report["viajes"]
    assert counts["salidas"] == int(trips.loc[departs, "start_station_id"].isin(SIMULATED).sum())
    assert counts["llegadas"] == int(trips.loc[arrives, "end_station_id"].isin(SIMULATED).sum())
    assert counts["salidas"] == counts["viajes_internos"] + counts["salidas_que_salen"]
    assert counts["llegadas"] == counts["viajes_internos"] + counts["llegadas_que_entran"]


# --------------------------------------------------------------------------- #
# h) Nivel de riesgo y feasible
# --------------------------------------------------------------------------- #

def test_el_nivel_de_riesgo_sale_del_pronostico_y_del_inventario_antes_de_intervenir():
    """Umbral: menos de 2 bicis o menos de 2 lugares.

    Estación con información (20 lugares): la trayectoria central baja 7 y no sube; la envolvente baja 12.1
    y sube 0.24. Rango central: 2 + 7 = 9 a 20 - 2 = 18. Rango de la envolvente: 14.1 a 17.76.
    """
    stations = pd.DataFrame(
        {"station_id": list("ABCDEFG") + ["N1", "N2", "N3", "X1"], "name": ["x"] * 11,
         "lat": [40.70 + 0.001 * i for i in range(11)], "lon": [-73.99] * 11,
         "capacity": pd.array([20] * 7 + [10, 10, 10, 10], dtype="Int64"),
         "station_type": pd.Series([None] * 11, dtype="object")}
    )
    network = build_network(stations, "mixta")
    assert network.station_ids == ("A", "B", "C", "D", "E", "F", "G", "N1", "N2", "N3", "X1")
    info = np.array([True] * 7 + [False] * 3 + [True])
    window = ForecastWindow(
        has_info=info,
        center_dip=np.array([-7.0] * 7 + [0.0] * 3 + [-6.0]),
        center_rise=np.array([0.0] * 7 + [0.0] * 3 + [1.0]),
        lower_dip=np.array([-12.099] * 7 + [0.0] * 3 + [-6.0]),
        upper_rise=np.array([0.2426] * 7 + [0.0] * 3 + [7.0]),
    )
    before = [8, 9, 14, 15, 17, 18, 19, 1, 5, 9, 4]
    level, feasible = risk_levels(before, network, Reglas.from_operacion(make_cfg()), window)
    assert dict(zip(network.station_ids, level)) == {
        "A": "alto",    # 8 bicis: la trayectoria central baja de 2
        "B": "medio",   # 9: el centro aguanta, la envolvente no
        "C": "medio",   # 14 < 14.1
        "D": "bajo",    # 15 está dentro del rango de la envolvente
        "E": "bajo",    # 17
        "F": "medio",   # 18 > 17.76: la envolvente deja menos de 2 lugares
        "G": "alto",    # 19: ya tiene menos de 2 lugares libres
        "N1": "alto",   # sin información y ya bajo el umbral (1 bici)
        "N2": "bajo",   # sin información y sin estar bajo ningún umbral
        "N3": "alto",   # sin información y con 1 lugar libre
        "X1": "alto",   # sin solución: lo = 8 y hi = 1
    }
    assert feasible.tolist() == [True] * 10 + [False]


# --------------------------------------------------------------------------- #
# i) El script de punta a punta
# --------------------------------------------------------------------------- #

def write_inputs(folder, case: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_contract(case["stations"], "stations", folder / "stations.parquet")
    write_contract(case["predictions"], "predictions", folder / "predictions.parquet")
    for name in ("trips", "station_hour"):
        pq.write_table(pa.Table.from_pandas(case[name], preserve_index=False), folder / f"{name}.parquet")


@pytest.fixture()
def project(tmp_path, case):
    """Carpeta con las cuatro tablas de entrada y los dos YAML."""
    write_inputs(tmp_path / "datos", case)
    (tmp_path / "operacion.yaml").write_text(yaml.safe_dump(make_cfg(), sort_keys=False), encoding="utf-8")
    (tmp_path / "datos.yaml").write_text(DATOS_TEXT, encoding="utf-8")
    return tmp_path


def run_script(project, out: str = "salida", *extra: str) -> int:
    return main(
        ["--datos", str(project / "datos"), "--salida", str(project / out),
         "--operacion", str(project / "operacion.yaml"), "--config", str(project / "datos.yaml"), *extra]
    )


def test_el_script_escribe_las_tablas_y_el_reporte(project, case, result, capsys):
    assert run_script(project) == 0
    printed = capsys.readouterr()
    out = project / "salida"
    assert validate_file(out / "recommendations.parquet", "recommendations") == []
    assert validate_file(out / "policy_curve.parquet", "policy_curve") == []
    # Lo mismo que llamar a la función con las tablas en memoria: los lectores entregan lo debido.
    pd.testing.assert_frame_equal(
        read_contract("policy_curve", out / "policy_curve.parquet"), result.policy_curve, check_dtype=False
    )
    recommendations = read_contract("recommendations", out / "recommendations.parquet")
    assert len(recommendations) == len(result.recommendations)
    for column in ("station_id", "policy", "start_inventory_target", "bikes_to_move", "risk_level", "feasible"):
        assert recommendations[column].tolist() == result.recommendations[column].tolist(), column
    document = json.loads((out / "reports" / "simulacion.json").read_text(encoding="utf-8"))
    assert set(document["corrida"]) == {"generado_utc", "segundos", "datos", "operacion", "config", "huella"}
    corrida = document.pop("corrida")
    assert document == json.loads(json.dumps(result.report))
    # El reporte y la pantalla nombran el archivo de reglas con el que se corrió, no el del proyecto.
    assert Path(corrida["operacion"]) == (project / "operacion.yaml").resolve()
    text = (out / "reports" / "simulacion.md").read_text(encoding="utf-8")
    assert text == build_report(result.report, corrida["operacion"])
    assert f"(`{corrida['operacion']}`, `estado: provisional`)" in text and "config/operacion.yaml" not in text
    assert f"Reglas de operación PROVISIONALES ({corrida['operacion']})." in printed.out
    assert "config/operacion.yaml" not in printed.out
    # El resumen en pantalla.
    assert "Tramo de validación: 2026-09-01 a 2026-09-04." in printed.out
    assert "Estaciones simuladas: 6 de 8." in printed.out
    assert 'La política "model" se alimenta de: baseline.' in printed.out
    assert "La prueba final no se simuló (decisión 4)." in printed.out
    assert "policy_curve.parquet" in printed.out and "simulacion.md" in printed.out
    assert "pronóstico: baseline" in printed.err


def test_el_script_dos_veces_da_archivos_identicos(project):
    assert run_script(project, "uno") == 0 and run_script(project, "dos") == 0
    for name in ("recommendations.parquet", "policy_curve.parquet", "reports/simulacion.md"):
        assert (project / "uno" / name).read_bytes() == (project / "dos" / name).read_bytes(), name
    first, second = (
        json.loads((project / folder / "reports" / "simulacion.json").read_text(encoding="utf-8"))
        for folder in ("uno", "dos")
    )
    first.pop("corrida"), second.pop("corrida")
    assert first == second


def test_el_script_no_lee_filas_del_tramo_simulado_ni_posteriores_como_historia(project, case):
    assert run_script(project, "antes") == 0
    # Se cambian todas las filas de station_hour desde el primer día de validación: el resultado no cambia.
    altered = case["station_hour"].copy()
    later = altered["ts_hour"] >= VALIDATION.start
    altered.loc[later, "arrivals"] = altered.loc[later, "arrivals"] + 500
    altered["net_flow"] = altered["arrivals"] - altered["departures"]
    pq.write_table(pa.Table.from_pandas(altered, preserve_index=False), project / "datos" / "station_hour.parquet")
    # Y los viajes y los pronósticos de la prueba final: tampoco.
    trips = case["trips"].copy()
    final = trips["started_at"] >= FINAL.start
    trips.loc[final, "end_station_id"] = "A"
    pq.write_table(pa.Table.from_pandas(trips, preserve_index=False), project / "datos" / "trips.parquet")
    predictions = case["predictions"].copy()
    predictions.loc[predictions["split"] == "test", list(QUANTILE_COLUMNS)] = 99.0
    write_contract(predictions, "predictions", project / "datos" / "predictions.parquet")
    assert run_script(project, "despues") == 0
    for name in ("recommendations.parquet", "policy_curve.parquet", "reports/simulacion.md"):
        assert (project / "antes" / name).read_bytes() == (project / "despues" / name).read_bytes(), name


def test_los_lectores_entregan_solo_lo_que_toca(project, case):
    folder = project / "datos"
    trips = load_trips(folder / "trips.parquet", VALIDATION)
    expected = in_tramo(case["trips"], VALIDATION)
    assert sorted(trips["ride_id"]) == sorted(expected["ride_id"]) and 0 < len(trips) < len(case["trips"])
    history = load_history(folder / "station_hour.parquet", TRAINING, VALIDATION)
    assert history["ts_hour"].max() == ts("2026-08-31 23:00") and history["ts_hour"].min() == ts("2026-08-24 00:00")
    assert list(history.columns) == ["station_id", "ts_hour", "net_flow"]
    rows, model = load_forecast(folder / "predictions.parquet", VALIDATION, "val")
    assert model == "baseline" and set(rows["split"]) == {"val"}
    assert rows["ts_hour"].min() == ts("2026-09-01 00:00") and rows["ts_hour"].max() == ts("2026-09-04 23:00")


def test_el_modelo_se_alimenta_de_lgbm_si_existe_y_si_no_de_la_linea_base(project, case):
    path = project / "datos" / "predictions.parquet"
    assert load_forecast(path, VALIDATION, "val")[1] == "baseline"
    both = pd.concat([case["predictions"], case["predictions"].assign(model="lgbm")], ignore_index=True)
    write_contract(both, "predictions", path)
    rows, model = load_forecast(path, VALIDATION, "val")
    assert model == "lgbm" and set(rows["model"]) == {"lgbm"}
    assert len(rows) == int(((case["predictions"]["split"] == "val")).sum())
    # Sin filas de validación no hay con qué alimentar al modelo.
    write_contract(case["predictions"][case["predictions"]["split"] == "test"], "predictions", path)
    with pytest.raises(SimulacionError, match="run_baseline.py"):
        load_forecast(path, VALIDATION, "val")


def test_el_script_rechaza_la_prueba_final_y_no_escribe_nada(project, capsys):
    assert run_script(project, "salida", "--tramo", "prueba") == 1
    printed = capsys.readouterr().out
    assert printed.startswith("ERROR.") and "Decisión 4" in printed and "--usar-prueba-final" in printed
    assert not (project / "salida").exists()


def test_el_script_no_acepta_otros_tramos(project):
    with pytest.raises(SystemExit):
        run_script(project, "salida", "--tramo", "entrenamiento")
    assert not (project / "salida").exists()


def test_con_usar_prueba_final_escribe_aparte_y_lo_dice(project, capsys):
    """Datos sintéticos de esta prueba: no es la prueba final del proyecto."""
    assert run_script(project) == 0
    validation = {name: (project / "salida" / name).read_bytes() for name in ("policy_curve.parquet",)}
    capsys.readouterr()
    assert run_script(project, "salida", "--tramo", "prueba", "--usar-prueba-final") == 0
    printed = capsys.readouterr().out
    assert "Tramo de prueba final: 2026-09-05 a 2026-09-06." in printed and "no se simuló" not in printed
    final = project / "salida" / FINAL_FOLDER
    assert validate_file(final / "policy_curve.parquet", "policy_curve") == []
    document = json.loads((final / "reports" / "simulacion.json").read_text(encoding="utf-8"))
    assert document["prueba_final"]["simulada"] is True and document["tramo"]["nombre"] == "prueba"
    assert document["historia"]["antes_de"] == "2026-09-05"   # la historia llega hasta el fin de validación
    assert "Esta corrida usó la prueba final." in document["limitaciones"]
    # Los resultados de validación siguen intactos.
    assert (project / "salida" / "policy_curve.parquet").read_bytes() == validation["policy_curve.parquet"]


@pytest.mark.parametrize("missing", ["trips", "stations", "station_hour", "predictions"])
def test_el_script_dice_que_archivo_falta(project, missing, capsys):
    (project / "datos" / f"{missing}.parquet").unlink()
    assert run_script(project) == 1
    printed = capsys.readouterr().out
    assert "ERROR. No existe" in printed and f"{missing}.parquet" in printed and "Primero corre" in printed
    assert not (project / "salida").exists()


def test_el_script_dice_si_no_puede_abrir_las_reglas(project, capsys):
    (project / "operacion.yaml").unlink()
    assert run_script(project) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("ERROR. No se pudo abrir") and "operacion.yaml" in printed
    assert not (project / "salida").exists()


def test_el_script_no_corre_si_las_tablas_se_construyeron_con_otro_periodo(project, capsys):
    reports = project / "datos" / "reports"
    reports.mkdir()
    built = {"config": {"periodo": {"inicio": "2026-07-01", "fin": "2026-09-29"}}}
    (reports / "limpieza.json").write_text(json.dumps(built), encoding="utf-8")
    assert run_script(project) == 1
    assert "se construyeron con el periodo 2026-07-01 a 2026-09-29" in capsys.readouterr().out


def test_el_script_no_corre_sin_el_bloque_simulacion(project, capsys):
    text = "\n".join(line for line in DATOS_TEXT.splitlines() if not line.startswith("simulacion"))
    (project / "datos.yaml").write_text(text + "\n", encoding="utf-8")
    assert load_datos(project / "datos.yaml").simulacion is None
    assert run_script(project) == 1
    assert "no trae el bloque `simulacion`" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# j) Lenguaje del reporte
# --------------------------------------------------------------------------- #

def test_el_reporte_usa_el_lenguaje_honesto_y_declara_las_limitaciones(result):
    text = build_report(result.report)
    everything = (text + "\n".join(summary_lines(result.report))).lower()
    assert "viajes recuperados" not in everything and "demanda recuperada" not in everything
    for phrase in (
        "posible demanda insatisfecha",
        "riesgo simulado evitado",
        "línea recta: ignora ríos y puentes",
        "instantáneos y no hay camiones",
        "los días no son independientes",
        "solo registran demanda atendida",
        "no busca otra estación",
        "el inventario inicial es un supuesto",
        "los días de calentamiento no quitan la dependencia de ese supuesto",
        "los traslados reales del operador no están en los datos",
        "la capacidad es la del feed gbfs de hoy",
        "clásicas y eléctricas se cuentan juntas",
        "cerrada varios días se ve como demanda cero",
        "se alimenta hoy de la línea base",
        "lightgbm todavía no existe",
        "la prueba final no se simuló",
        "las reglas de operación son provisionales",
        "nadie mueve bicis",
    ):
        assert phrase in everything, phrase
    assert len(result.report["limitaciones"]) >= 12
    # Con otro pronóstico y reglas confirmadas, el texto cambia.
    other = copy.deepcopy(result.report)
    other["pronostico"]["modelo_que_alimenta_model"] = "lgbm"
    other["reglas"]["estado"] = "confirmado"
    other["limitaciones"] = limitations(other)
    text = build_report(other).lower()
    assert "línea base" not in text and 'pronóstico "lgbm"' in text and "provisionales" not in text


def test_el_reporte_trae_una_fila_por_politica_y_presupuesto_y_none_una_sola_vez(result):
    text = build_report(result.report)
    section = text.split("## Horas de estación en riesgo por día")[1].split("##")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "política" not in line]
    assert sum(line.startswith("| sin intervención") for line in rows) == 1
    assert sum(line.startswith("| regla histórica") for line in rows) == 4
    assert sum(line.startswith("| modelo") for line in rows) == 4
    # Y el riesgo de cada día evaluado con el presupuesto de referencia, una columna por política.
    daily = text.split("## Riesgo día por día")[1].split("##")[0]
    rows = [line for line in daily.splitlines() if line.startswith("| 2026-")]
    assert [line.split(" |")[0] for line in rows] == [
        "| 2026-09-02 (miércoles)", "| 2026-09-03 (jueves)", "| 2026-09-04 (viernes)",
    ]
    for line, day in zip(rows, range(3)):
        cells = [cell.strip() for cell in line.strip("|").split("|")][1:]
        expected = [row_of(result.report, policy, 6)["riesgo_de_cada_dia"][day] for policy in POLICIES]
        assert cells == [f"{value:,.1f}" for value in expected]


# --------------------------------------------------------------------------- #
# k) Lo que el reporte dice en palabras
# --------------------------------------------------------------------------- #

def section(text: str, title: str) -> str:
    """El cuerpo de una sección del reporte: de su título al título siguiente."""
    return text.split(f"## {title}\n")[1].split("\n## ")[0]


def number(cell: str) -> int:
    return int(cell.replace(",", ""))


def test_la_tabla_de_bicis_que_entran_y_salen_se_puede_comprobar(result):
    body = section(build_report(result.report), "Bicis que entran y salen del conjunto simulado")
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in body.splitlines()
        if line.startswith("| ") and not line.startswith(("| política", "| ---"))
    ]
    assert len(rows) == 1 + 2 * 4   # "none" una vez y cuatro presupuestos por cada una de las otras dos
    for cells in rows:
        entran, salen, neto, inicio, entran_todo, salen_todo, final = map(number, cells[3:])
        assert neto == entran - salen                       # días evaluados
        assert inicio + entran_todo - salen_todo == final   # todo el tramo, con el calentamiento
    # Con las columnas de los días evaluados la cuenta no cierra: falta el día de calentamiento.
    assert any(number(cells[6]) + number(cells[5]) != number(cells[9]) for cells in rows)
    none = row_of(result.report, "none", 0)["bicis_en_la_red"]
    keys = ("al_inicio_del_tramo", "entran_en_todo_el_tramo", "salen_en_todo_el_tramo", "al_final_del_tramo")
    assert rows[0][6:] == [f"{none[key]:,}" for key in keys]
    assert "inicio más entran menos salen da el final" in body


def reading(report: dict) -> dict:
    """Las líneas de la lectura que sigue a la primera tabla, por política."""
    body = section(build_report(report), "Horas de estación en riesgo por día")
    after_table = body.split("|\n")[-1].strip()   # lo que sigue a la última fila de la tabla
    assert after_table.startswith("Lectura de la tabla con el presupuesto de referencia (6 bicis al día; ")
    lines = {line.split(":")[0]: line for line in after_table.splitlines() if line.startswith("- ")}
    assert list(lines) == ["- Sin intervención", "- Regla histórica", "- Modelo"]
    return {"text": after_table, "none": lines["- Sin intervención"], "historical": lines["- Regla histórica"],
            "model": lines["- Modelo"]}


def test_la_lectura_de_la_primera_tabla_sale_de_los_numeros_de_la_corrida(result):
    report = result.report
    lines = reading(report)
    assert "días evaluados: 3). El intervalo es indicativo: los días no son independientes" in lines["text"]
    none, days = row_of(report, "none", 6), report["tramo"]["dias_evaluados"]
    base = none["riesgo_por_dia"]
    assert lines["none"] == (
        f"- Sin intervención: {base['media']:,.1f} horas de estación en riesgo por día "
        f"(intervalo {base['ic_bajo']:,.1f} a {base['ic_alto']:,.1f})."
    )
    for policy in ("historical", "model"):
        row, line = row_of(report, policy, 6), lines[policy]
        risk, avoided = row["riesgo_por_dia"], row["riesgo_simulado_evitado_por_dia"]
        assert (
            f"{risk['media']:,.1f} horas de estación en riesgo por día "
            f"(intervalo {risk['ic_bajo']:,.1f} a {risk['ic_alto']:,.1f})."
        ) in line
        assert (
            f"Riesgo simulado evitado contra no intervenir: {avoided['media']:,.1f} horas por día "
            f"({avoided['media'] / base['media'] * 100:.1f} % del riesgo sin intervención), "
            f"intervalo {avoided['ic_bajo']:,.1f} a {avoided['ic_alto']:,.1f}."
        ) in line
        # En cuántos días evaluados tuvo más riesgo que sin intervención, contados aquí.
        pairs = zip(days, row["riesgo_de_cada_dia"], none["riesgo_de_cada_dia"])
        worse = [day for day, mine, theirs in pairs if mine > theirs]
        count = f"Días evaluados con más riesgo que sin intervención: {len(worse)} de 3"
        assert count + (f" ({', '.join(worse)})." if worse else ".") in line
        assert (
            f"Viajes bloqueados por día (simulados): {row['viajes_bloqueados']['por_dia']:,.1f} contra "
            f"{none['viajes_bloqueados']['por_dia']:,.1f} sin intervención. Llegadas desviadas por día: "
            f"{row['llegadas_desviadas']['por_dia']:,.1f} contra {none['llegadas_desviadas']['por_dia']:,.1f}."
        ) in line
    # El caso distingue: una de las dos políticas tuvo algún día con más riesgo que sin intervención.
    assert "con más riesgo que sin intervención: 0 de 3." in lines["historical"]
    assert "con más riesgo que sin intervención: 2 de 3 (2026-09-03, 2026-09-04)." in lines["model"]


def test_la_lectura_dice_cuando_el_intervalo_del_riesgo_evitado_incluye_el_cero_o_queda_cerca():
    """Cifras puestas a mano en una copia del reporte: la lectura solo depende de ellas."""
    report = copy.deepcopy(simulate_case(make_case()).report)
    none = row_of(report, "none", 6)
    none["riesgo_por_dia"] = {"media": 1000.0, "ic_bajo": 900.0, "ic_alto": 1100.0}
    none["riesgo_de_cada_dia"] = [900.0, 1000.0, 1100.0]

    def line(policy: str, avoided: tuple, daily: list) -> str:
        row = row_of(report, policy, 6)
        row["riesgo_simulado_evitado_por_dia"] = dict(zip(("media", "ic_bajo", "ic_alto"), avoided))
        row["riesgo_de_cada_dia"] = daily
        return reading(report)[policy]

    # Límite inferior 200 y ancho 100: ni incluye el cero ni queda cerca. Ningún día con más riesgo.
    text = line("historical", (250.0, 200.0, 300.0), [700.0, 750.0, 800.0])
    assert (
        "250.0 horas por día (25.0 % del riesgo sin intervención), intervalo 200.0 a 300.0. "
        "Días evaluados con más riesgo que sin intervención: 0 de 3. Viajes bloqueados"
    ) in text
    assert "cero" not in text
    # Límite inferior 10.2 y ancho 179.9: queda cerca de cero. Más riesgo que sin intervención el primer día.
    text = line("model", (102.7, 10.2, 190.1), [900.5, 990.0, 1000.0])
    assert (
        "intervalo 10.2 a 190.1. El límite inferior de ese intervalo (10.2) es menor que su ancho (179.9): "
        "queda cerca de cero. Días evaluados con más riesgo que sin intervención: 1 de 3 (2026-09-02)."
    ) in text
    # El intervalo incluye el cero.
    text = line("model", (1.4, -2.6, 5.5), [901.0, 1001.0, 1090.0])
    assert "intervalo -2.6 a 5.5. Ese intervalo incluye el cero. Días evaluados" in text
    assert "más riesgo que sin intervención: 2 de 3 (2026-09-02, 2026-09-03)." in text
    # Todo el intervalo por debajo de cero, y más riesgo los tres días.
    text = line("model", (-5.0, -9.0, -1.0), [905.0, 1005.0, 1105.0])
    assert (
        "(-0.5 % del riesgo sin intervención), intervalo -9.0 a -1.0. "
        "Ese intervalo queda completo por debajo de cero."
    ) in text
    assert "más riesgo que sin intervención: 3 de 3 (2026-09-02, 2026-09-03, 2026-09-04)." in text


def differences_case():
    """Dos estaciones de 20 lugares y un solo día, el miércoles 2 de septiembre, sin calentamiento.

    Empiezan con 14 bicis (ocupación de respaldo de la mañana: 0.7). A las 03:00 sale una de B hacia fuera: a
    las 05:00 A tiene 14 y B tiene 13. Presupuesto de referencia: 1 por estación = 2 bicis al día, 1 por franja.

    Regla histórica, mañana (flujo neto medio de la historia, horas 5 a 11):
      A pierde 2 por hora -> baja 14: rango [16, 18]. Tiene 14: le faltan 2.
      B gana 1 por hora   -> sube 7:  rango [2, 11].  Tiene 13: le sobran 2.
      Con 1 bici de presupuesto, B da 1 a A. Quedan A = 15 y B = 12.
    Modelo, mañana (pronóstico sin dispersión):
      A recibe 1 por hora -> sube 7: rango [2, 11]. Tiene 14: le sobran 3.
      B: salen 6 a las 05:00 y llegan 17 a las 08:00 -> baja 6 y sube 11: lo = 8 y hi = 7, sin solución.
         Usa la ocupación de respaldo: 0.7 x 20 = 14. Tiene 13: le falta 1.
      Con 1 bici de presupuesto, A da 1 a B, que es una estación-franja sin solución. Quedan A = 13 y B = 14.
    Tarde (horas 14 a 20): flujo cero en la historia y en el pronóstico -> rango [2, 18] para las dos, en las
      dos corridas. Nadie pide nada.
    """
    stations = pd.DataFrame(
        {"station_id": ["A", "B"], "name": ["a", "b"], "lat": [40.70, 40.705], "lon": [-73.99, -73.99],
         "capacity": pd.array([20, 20], dtype="Int64"), "station_type": pd.Series([None, None], dtype="object")}
    )
    trips = make_trips(
        [("b0", "2026-09-02 03:00", "2026-09-02 03:10", "B", None),
         ("a0", "2026-09-02 22:00", "2026-09-02 22:10", "A", "B")]   # después de las dos intervenciones
    )
    morning, evening = range(5, 12), range(14, 21)
    flows = {("A", hour): -2 for hour in morning} | {("B", hour): 1 for hour in morning}
    flows |= {(station, hour): 0 for station in "AB" for hour in evening}
    history = pd.DataFrame(   # el jueves 27 de agosto: laborable, como el día simulado
        [(station, ts(f"2026-08-27 {hour:02d}:00"), flow) for (station, hour), flow in flows.items()],
        columns=["station_id", "ts_hour", "net_flow"],
    )
    history["ts_hour"] = stamps(history["ts_hour"])
    zero = (0.0,) * 6   # dep_q10, dep_q50, dep_q90, arr_q10, arr_q50, arr_q90
    forecast = {("A", hour): (0.0, 0.0, 0.0, 1.0, 1.0, 1.0) for hour in morning}
    forecast |= {("B", hour): zero for hour in morning}
    forecast |= {("B", 5): (6.0, 6.0, 6.0, 0.0, 0.0, 0.0), ("B", 8): (0.0, 0.0, 0.0, 17.0, 17.0, 17.0)}
    forecast |= {(station, hour): zero for station in "AB" for hour in evening}
    predictions = pd.DataFrame(
        [(station, ts(f"2026-09-02 {hour:02d}:00"), "baseline", "val", *values)
         for (station, hour), values in forecast.items()],
        columns=["station_id", "ts_hour", "model", "split", *QUANTILE_COLUMNS],
    )
    predictions["ts_hour"] = stamps(predictions["ts_hour"])
    strategy = {kind: {"prioridad": 2, "ocupacion_respaldo": {"manana": 0.7, "tarde": 0.5}} for kind in STATION_TYPES}
    settings = SimpleNamespace(dias_de_calentamiento=0, remuestreos=300, nivel_del_intervalo=0.9, semilla=7)
    return run_simulation(
        stations, trips, history, predictions, Reglas.from_operacion(make_cfg(estrategia_por_tipo=strategy)),
        Tramo("validacion", date(2026, 9, 2), date(2026, 9, 2)), settings,
        forecast_model="baseline", prueba_final=FINAL,
    )


def test_las_diferencias_entre_las_dos_politicas_a_mano():
    out = differences_case()
    report = out.report
    assert report["reglas"]["presupuesto_de_referencia_en_bicis"] == 2
    assert report["reglas"]["reparto_por_franja"]["2"] == {"manana": 1, "tarde": 1}
    # Lo que pasó: en la mañana la regla histórica llevó 1 bici de B a A y el modelo 1 de A a B. En la tarde, nada.
    table = out.recommendations.set_index(["policy", "window_start", "station_id"]).sort_index()
    morning, evening = ts("2026-09-02 06:00"), ts("2026-09-02 15:00")
    moved = table["bikes_to_move"]
    assert (moved[("historical", morning, "A")], moved[("historical", morning, "B")]) == (1, -1)
    assert (moved[("model", morning, "A")], moved[("model", morning, "B")]) == (-1, 1)
    assert (moved.xs(evening, level="window_start") == 0).all()
    # B en la mañana no tiene solución con la envolvente del pronóstico; lo demás sí.
    unsolved = table.loc["model", "feasible"]
    assert unsolved[~unsolved].index.tolist() == [(morning, "B")]
    assert report["diferencias_entre_politicas"] == {
        "presupuesto_del_dia_en_bicis": 2,
        "estacion_franja": 4,
        "historical": {
            "estacion_franja_con_informacion": 4,
            "estacion_franja_sin_solucion": 0,
            "bicis_que_faltan_por_dia": 2.0,    # a A le faltan 2 para llegar a 16
            "bicis_que_sobran_por_dia": 2.0,    # a B le sobran 2 sobre 11
            "estacion_franja_con_traslado": 2,
            "bicis_recibidas": 1,
            "bicis_entregadas": 1,
            "bicis_recibidas_en_franjas_sin_solucion": 0,
            "bicis_entregadas_en_franjas_sin_solucion": 0,
            "bicis_movidas_por_dia": 1.0,
        },
        "model": {
            "estacion_franja_con_informacion": 4,
            "estacion_franja_sin_solucion": 1,  # B en la mañana
            "bicis_que_faltan_por_dia": 1.0,    # a B le falta 1 para llegar a 14
            "bicis_que_sobran_por_dia": 3.0,    # a A le sobran 3 sobre 11
            "estacion_franja_con_traslado": 2,
            "bicis_recibidas": 1,
            "bicis_entregadas": 1,
            "bicis_recibidas_en_franjas_sin_solucion": 1,   # la que recibió B
            "bicis_entregadas_en_franjas_sin_solucion": 0,
            "bicis_movidas_por_dia": 1.0,
        },
    }
    # Nadie estuvo en riesgo: las dos evitan lo mismo y el título no dice que una evita menos.
    text = build_report(report)
    assert "Por qué la política model evita menos" not in text
    body = section(text, "En qué se diferencian la regla histórica y la política model")
    for phrase in (
        "con el presupuesto de referencia (2 bicis al día), en los días evaluados (4 estación-franja)",
        "Ninguna está probada como causa de esa diferencia.",
        'que es el rango de la política "model": 1 de 4 (25.0 %); tiene información en 4.',
        "Con el rango de la regla histórica (flujo neto medio de la historia): 0 de 4 (0.0 %); "
        "tiene información en 4.",
        "Regla histórica: faltan 2.0 y sobran 2.0. Modelo: faltan 1.0 y sobran 3.0. Presupuesto del día: 2 bicis.",
        "Bicis movidas por día: regla histórica 1.0, modelo 1.0.",
        "regla histórica 2, con 1.0 bicis recibidas o entregadas por cada una; modelo 2, con 1.0.",
        "Modelo: 1 de 1 (100.0 %) de las recibidas y 0 de 1 (0.0 %) de las entregadas.",
        "Regla histórica: 0 de 1 (0.0 %) de las recibidas y 0 de 1 (0.0 %) de las entregadas.",
    ):
        assert phrase in body, phrase


def test_la_seccion_de_diferencias_usa_las_cifras_de_la_corrida_y_solo_dice_que_model_evita_menos_si_paso(result):
    report = copy.deepcopy(result.report)
    historical, model = (
        row_of(report, policy, 6)["riesgo_simulado_evitado_por_dia"] for policy in ("historical", "model")
    )
    assert model["media"] < historical["media"]
    body = section(build_report(report), "Por qué la política model evita menos")
    assert f"regla histórica {historical['media']:,.1f} horas de estación, modelo {model['media']:,.1f}." in body
    assert "Ninguna está probada como causa de esa diferencia." in body
    assert "no de LightGBM: estas cifras no dicen nada de LightGBM." in body
    assert "recomend" not in body.lower()   # cifras, no recomendaciones
    # Las cifras de traslados son las de recommendations.parquet.
    facts, recs = report["diferencias_entre_politicas"], report["recomendaciones"]
    assert (facts["presupuesto_del_dia_en_bicis"], facts["estacion_franja"]) == (6, recs["estacion_franja"])
    for policy in ("historical", "model"):
        for key in ("estacion_franja_con_traslado", "bicis_recibidas", "bicis_entregadas"):
            assert facts[policy][key] == recs["por_politica"][policy][key]
        moved = row_of(report, policy, 6)["bicis_movidas"]["por_dia"]
        assert facts[policy]["bicis_movidas_por_dia"] == pytest.approx(moved)
        assert f"{facts[policy]['estacion_franja_con_traslado']:,}, con " in body
    # La envolvente del pronóstico es el rango de "model": sus franjas sin solución son las de `feasible`.
    assert facts["model"]["estacion_franja_sin_solucion"] == recs["estacion_franja_sin_solucion"] > 0
    for key in ("bicis_recibidas_en_franjas_sin_solucion", "bicis_entregadas_en_franjas_sin_solucion"):
        assert facts["model"][key] == recs["por_politica"]["model"][key]
    # F no tiene historia: 5 estaciones con información en las 6 franjas evaluadas.
    assert facts["historical"]["estacion_franja_con_informacion"] == 30
    assert facts["model"]["estacion_franja_con_informacion"] == 30

    # Si "model" evita más, el título cambia; con otro pronóstico no se habla de la línea base ni de LightGBM.
    model["media"] = historical["media"] + 1.0
    report["pronostico"]["modelo_que_alimenta_model"] = "lgbm"
    report["limitaciones"] = limitations(report)
    text = build_report(report)
    assert "Por qué la política model evita menos" not in text
    body = section(text, "En qué se diferencian la regla histórica y la política model")
    assert 'se alimenta del pronóstico "lgbm".' in body and "LightGBM" not in body and "línea base" not in body


def test_las_limitaciones_traen_las_cifras_del_inventario_inicial_y_de_las_bicis_que_entran_y_salen(case, result):
    report = result.report
    none = row_of(report, "none", 0)
    bikes, daily = none["bicis_en_la_red"], none["riesgo_de_cada_dia"]
    assert (bikes["al_inicio_del_tramo"], bikes["al_final_del_tramo"]) == (32, 47)
    (initial,) = [item for item in report["limitaciones"] if item.startswith("El inventario inicial es un supuesto")]
    assert "Los días de calentamiento no quitan la dependencia de ese supuesto" in initial
    assert "con cuántas bicis empieza el conjunto (32)" in initial and "termina el tramo con 47)" in initial
    assert (
        f"el primer día evaluado (2026-09-02) tiene {daily[0]:,.1f} horas de estación en riesgo y el último "
        f"(2026-09-04), {daily[-1]:,.1f}."
    ) in initial

    (drift,) = [item for item in report["limitaciones"] if item.startswith("Una bici que sale del conjunto")]
    assert "los traslados reales del operador no están en los datos" in drift
    entered, left = bikes["entran_en_todo_el_tramo"], bikes["salen_en_todo_el_tramo"]
    assert 32 + entered - left == 47
    assert f"empieza el tramo con 32 bicis y lo termina con 47 (+15): entran {entered} y salen {left}." in drift
    # Por qué entra o sale cada bici, contado aquí con los viajes del tramo. "G" no tiene capacidad: está fuera.
    trips = in_tramo(case["trips"], VALIDATION)
    start, end = trips["start_station_id"], trips["end_station_id"]
    starts_inside, ends_inside = VALIDATION.contains(trips["started_at"]), VALIDATION.contains(trips["ended_at"])
    departs, arrives = starts_inside & start.isin(SIMULATED), ends_inside & end.isin(SIMULATED)
    leaves, enters = departs & ~arrives, arrives & ~departs
    counts = report["viajes"]
    # De la punta que cae fuera del tramo no se mira la estación: solo que cae fuera.
    assert counts["salidas_que_salen_por_motivo"] == {
        "terminan_despues_del_tramo": int((leaves & ~ends_inside).sum()),
        "sin_estacion_de_fin": int((leaves & ends_inside & end.isna()).sum()),
        "a_una_estacion_fuera_del_conjunto": int((leaves & ends_inside & (end == "G")).sum()),
    }
    assert counts["llegadas_que_entran_por_motivo"] == {
        "salieron_antes_del_tramo": int((enters & ~starts_inside).sum()),
        "sin_estacion_de_inicio": int((enters & starts_inside & start.isna()).sum()),
        "de_una_estacion_fuera_del_conjunto": int((enters & starts_inside & (start == "G")).sum()),
    }
    leave, enter = counts["salidas_que_salen_por_motivo"], counts["llegadas_que_entran_por_motivo"]
    assert sum(leave.values()) == counts["salidas_que_salen"] and sum(enter.values()) == counts["llegadas_que_entran"]
    assert min(*leave.values(), *enter.values()) > 0   # el caso ejercita los tres motivos de cada lado
    assert (
        f"de las {counts['salidas_que_salen']} salidas que sacan una bici del conjunto, "
        f"{leave['sin_estacion_de_fin']} son de viajes sin estación de fin, "
        f"{leave['a_una_estacion_fuera_del_conjunto']} van a una estación fuera del conjunto y "
        f"{leave['terminan_despues_del_tramo']} terminan después del tramo"
    ) in drift
    assert (
        f"de las {counts['llegadas_que_entran']} llegadas que meten una, {enter['sin_estacion_de_inicio']} son "
        f"de viajes sin estación de inicio, {enter['de_una_estacion_fuera_del_conjunto']} vienen de una estación "
        f"fuera del conjunto y {enter['salieron_antes_del_tramo']} salieron antes del tramo."
    ) in drift
    # Las limitaciones van tal cual al reporte.
    text = build_report(report)
    assert f"- {initial}\n" in text and f"- {drift}\n" in text


def test_el_reporte_cuenta_las_estaciones_con_historia_que_se_quedan_sin_pronostico(case, result):
    """Las filas de `predictions` de B se cortan el jueves 3 a las 08:00, como si ahí cayera su último viaje."""
    rows = case["predictions"]
    cut = dict(case, predictions=rows[~((rows["station_id"] == "B") & (rows["ts_hour"] >= ts("2026-09-03 08:00")))])
    report = simulate_case(cut).report
    info = report["informacion"]
    # Días evaluados: 2, 3 y 4. B pierde las dos franjas del 3 (a la mañana le faltan las horas 8 a 11) y las del 4.
    assert info["con_historia_y_sin_pronostico"] == {"estacion_franja": 4, "estaciones": 1}
    # F no tiene historia: sus 6 estación-franja no entran en esa cuenta. La regla histórica sigue atendiendo a B.
    assert info["model"]["estacion_franja_sin_informacion"] == 6 + 4
    assert info["historical"]["estacion_franja_sin_informacion"] == 6
    (item,) = [item for item in report["limitaciones"] if item.startswith("`predictions` solo trae filas")]
    assert "4 estación-franja (estaciones con historia a las que les pasa: 1)" in item
    assert "depende de viajes posteriores a la intervención" in item and "la comparación no es pareja" in item
    text = build_report(report)
    assert "ningún viaje (salvedad: qué filas trae el pronóstico depende del último viaje de cada estación" in text
    assert "De ellas, 4 estación-franja son de estaciones con historia (estaciones: 1)" in text
    # Sin ese corte no hay ni salvedad ni limitación.
    assert result.report["informacion"]["con_historia_y_sin_pronostico"] == {"estacion_franja": 0, "estaciones": 0}
    assert not any(item.startswith("`predictions` solo trae filas") for item in result.report["limitaciones"])
    assert "y ningún viaje.\n" in build_report(result.report)


def test_el_reporte_dice_filas_de_pronostico_leidas_y_no_usadas(case, result):
    rows = case["predictions"]
    read = int(((rows["split"] == "val") & VALIDATION.contains(rows["ts_hour"])).sum())
    assert result.report["pronostico"]["filas_leidas"] == read and "filas_usadas" not in result.report["pronostico"]
    # Son más de las que usa la política: ahí van las de G, que no se simula, y las de F, que no tiene historia.
    assert read == 7 * 4 * 24
    text = build_report(result.report)
    assert f"- Filas de pronóstico leídas (todo el tramo, todas las estaciones): {read:,}. La política solo" in text
    assert "Filas de pronóstico usadas" not in text


# --------------------------------------------------------------------------- #
# l) La huella de las entradas: ¿las salidas siguen al día?
# --------------------------------------------------------------------------- #

def check_outputs_are_current(out_dir, data_dir=None, operacion=OPERACION_YAML, config=DATOS_YAML) -> None:
    """Falla, con el comando que hay que correr, si las salidas de la simulación en `out_dir` ya no corresponden.

    Compara la huella que la corrida dejó en simulacion.json con la de las tablas, las reglas y el config de
    hoy, y revisa que recommendations.parquet y simulacion.md sean los de ese simulacion.json. Lee lo mismo
    que lee la simulación: el tramo simulado y la historia anterior. Nada de la prueba final (decisión 4).
    """
    reports = out_dir / "reports"
    document = json.loads((reports / "simulacion.json").read_text(encoding="utf-8"))
    corrida = document.pop("corrida")
    current = current_fingerprint(data_dir or out_dir, operacion, config, document["tramo"]["nombre"])
    changed = stale_inputs(corrida.get("huella"), current)
    assert not changed, stale_message(changed)

    other = f"recommendations.parquet y simulacion.* no son de la misma corrida o son de otro simulador. {RERUN}"
    recommendations = read_contract("recommendations", out_dir / "recommendations.parquet")
    assert summarize_recommendations(recommendations) == document["recomendaciones"], other
    try:
        expected = build_report(document, corrida["operacion"])
    except KeyError as missing:
        raise AssertionError(f"A simulacion.json le falta {missing}. {other}") from None
    assert (reports / "simulacion.md").read_text(encoding="utf-8") == expected, other


def rewrite(folder, name: str, frame: pd.DataFrame) -> None:
    if name in ("stations", "predictions"):
        write_contract(frame, name, folder / f"{name}.parquet")
    else:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), folder / f"{name}.parquet")


def stale(*changed: str):
    """Espera el aviso con lo que cambió y el comando que hay que correr."""
    return pytest.raises(AssertionError, match=re.escape(f"Cambió: {'; '.join(changed)}. {RERUN}"))


def test_la_revision_dice_que_entrada_cambio_y_que_comando_correr(project, case):
    assert RERUN == "Vuelve a correr: python scripts/run_simulation.py"
    assert run_script(project) == 0
    out, data = project / "salida", project / "datos"
    files = {"operacion": project / "operacion.yaml", "config": project / "datos.yaml"}

    def check() -> None:
        check_outputs_are_current(out, data, **files)

    check()

    # stations: una capacidad distinta. El nombre no entra a la simulación.
    stations = case["stations"].copy()
    stations.loc[stations["station_id"] == "A", "capacity"] = 9
    rewrite(data, "stations", stations)
    with stale("stations.parquet"):
        check()
    rewrite(data, "stations", case["stations"].assign(name="otro nombre"))
    check()

    # trips: falta un viaje del tramo simulado.
    trips = case["trips"]
    inside = VALIDATION.contains(trips["started_at"]) & trips["start_station_id"].isin(SIMULATED)
    rewrite(data, "trips", trips.drop(index=trips.index[inside][0]))
    with stale("trips.parquet (los viajes del tramo simulado)"):
        check()
    rewrite(data, "trips", trips)
    check()

    # predictions: otro pronóstico para el tramo simulado.
    predictions = case["predictions"].copy()
    predictions.loc[predictions["split"] == "val", "dep_q90"] += 1.0
    rewrite(data, "predictions", predictions)
    with stale("predictions.parquet (el pronóstico del tramo simulado)"):
        check()
    rewrite(data, "predictions", case["predictions"])

    # station_hour: otra historia antes del tramo. Las filas del tramo y las posteriores no se leen.
    history = case["station_hour"].copy()
    before = history["ts_hour"] < VALIDATION.start
    history.loc[before, "arrivals"] += 1
    history["net_flow"] = history["arrivals"] - history["departures"]
    rewrite(data, "station_hour", history)
    with stale("station_hour.parquet (la historia anterior al tramo simulado)"):
        check()
    later = case["station_hour"].copy()
    later.loc[~before, "arrivals"] += 500
    later["net_flow"] = later["arrivals"] - later["departures"]
    rewrite(data, "station_hour", later)
    check()

    # config/operacion.yaml: un umbral, la ocupación de respaldo de un tipo o una prioridad.
    for change in (
        lambda cfg: cfg["umbral_riesgo"].update(bicis_minimas=3),
        lambda cfg: cfg["estrategia_por_tipo"]["mixta"]["ocupacion_respaldo"].update(manana=0.3),
        lambda cfg: cfg["estrategia_por_tipo"]["oficinas"].update(prioridad=3),
        lambda cfg: cfg.update(distancia_maxima_km=1.5),
    ):
        cfg = make_cfg()
        change(cfg)
        files["operacion"].write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        with stale("las reglas de config/operacion.yaml"):
            check()
    files["operacion"].write_text(yaml.safe_dump(make_cfg(), sort_keys=False), encoding="utf-8")
    check()

    # El bloque `simulacion` de config/datos.yaml: la semilla o los días de calentamiento.
    for old, new in (("semilla: 7", "semilla: 8"), ("dias_de_calentamiento: 1", "dias_de_calentamiento: 2")):
        assert old in DATOS_TEXT
        files["config"].write_text(DATOS_TEXT.replace(old, new), encoding="utf-8")
        with stale("el bloque `simulacion` de config/datos.yaml"):
            check()
    # Dos cosas a la vez: se dicen las dos.
    rewrite(data, "stations", stations)
    with stale("el bloque `simulacion` de config/datos.yaml", "stations.parquet"):
        check()

    # Al volver a correr la simulación con lo de hoy, todo queda al día otra vez.
    assert run_script(project) == 0
    check()


def test_la_huella_no_trae_nada_de_la_prueba_final(project, case):
    """Datos sintéticos de esta prueba: no es la prueba final del proyecto."""
    data = project / "datos"
    files = {"operacion": project / "operacion.yaml", "config": project / "datos.yaml"}
    # Un viaje que sale el último día de validación y termina ya en las fechas de la prueba final.
    crossing = make_trips([("ZZ9999", "2026-09-04 23:50", "2026-09-05 00:20", "A", "B")])
    trips = pd.concat([case["trips"], crossing], ignore_index=True)
    rewrite(data, "trips", trips)
    assert run_script(project, "antes") == 0
    check_outputs_are_current(project / "antes", data, **files)
    before = current_fingerprint(data, **files)

    # Cambia todo lo que tiene fecha de la prueba final: los viajes que empiezan ahí, la hora y la estación de
    # fin del viaje que cruza (ahora sin estación), el pronóstico del split "test" y las filas de station_hour
    # de esas fechas.
    changed = trips.copy()
    changed.loc[changed["started_at"] >= FINAL.start, "end_station_id"] = "A"
    crosses = changed["ride_id"] == "ZZ9999"
    changed.loc[crosses, "end_station_id"] = None
    changed.loc[crosses, "ended_at"] = ts("2026-09-05 03:00")
    rewrite(data, "trips", changed)
    predictions = case["predictions"].copy()
    predictions.loc[predictions["split"] == "test", list(QUANTILE_COLUMNS)] = 99.0
    rewrite(data, "predictions", predictions)
    hours = case["station_hour"].copy()
    hours.loc[hours["ts_hour"] >= FINAL.start, ["departures", "departures_member"]] += 7
    hours["net_flow"] = hours["arrivals"] - hours["departures"]
    rewrite(data, "station_hour", hours)

    # La huella es la misma, las salidas guardadas siguen al día y una corrida nueva da lo mismo.
    assert current_fingerprint(data, **files) == before
    check_outputs_are_current(project / "antes", data, **files)
    assert run_script(project, "despues") == 0
    for name in ("recommendations.parquet", "policy_curve.parquet", "reports/simulacion.md"):
        assert (project / "antes" / name).read_bytes() == (project / "despues" / name).read_bytes(), name
    # De la prueba final la huella solo guarda las fechas, y no se puede pedir la huella de ese tramo.
    assert before["fechas"]["prueba_final"] == {"inicio": "2026-09-05", "fin": "2026-09-06"}
    assert before["trips"]["filas"] == len(in_tramo(trips, VALIDATION))
    with pytest.raises(FinalTestError, match="Decisión 4"):
        current_fingerprint(data, **files, tramo="prueba")


def test_la_revision_pide_volver_a_correr_si_falta_la_huella_o_las_salidas_no_son_de_la_misma_corrida(project):
    assert run_script(project) == 0
    out, data = project / "salida", project / "datos"
    files = {"operacion": project / "operacion.yaml", "config": project / "datos.yaml"}
    check_outputs_are_current(out, data, **files)

    # Un simulacion.json escrito antes de que existiera la huella.
    path = out / "reports" / "simulacion.json"
    original = path.read_text(encoding="utf-8")
    document = json.loads(original)
    del document["corrida"]["huella"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with stale("la huella de las entradas (simulacion.json no la trae: lo escribió una versión anterior)"):
        check_outputs_are_current(out, data, **files)
    path.write_text(original, encoding="utf-8")
    check_outputs_are_current(out, data, **files)

    # Un simulacion.md que no es el de ese simulacion.json.
    report = out / "reports" / "simulacion.md"
    text = report.read_text(encoding="utf-8")
    report.write_text(text.replace("## Limitaciones", "## Limitaciones de otra corrida"), encoding="utf-8")
    with pytest.raises(AssertionError, match=re.escape(RERUN)):
        check_outputs_are_current(out, data, **files)
    report.write_text(text, encoding="utf-8")

    # Un recommendations.parquet que no es el de ese simulacion.json.
    recommendations = read_contract("recommendations", out / "recommendations.parquet")
    write_contract(recommendations.assign(risk_level="bajo"), "recommendations", out / "recommendations.parquet")
    with pytest.raises(AssertionError, match="no son de la misma corrida"):
        check_outputs_are_current(out, data, **files)


# --------------------------------------------------------------------------- #
# Archivos reales
# --------------------------------------------------------------------------- #

def test_las_salidas_reales_cumplen_los_contratos_y_cuadran_con_su_reporte():
    """Mira data/processed. Se salta si la simulación todavía no se ha corrido."""
    report_path = DATA_PROCESSED / "reports" / "simulacion.json"
    paths = [DATA_PROCESSED / "recommendations.parquet", DATA_PROCESSED / "policy_curve.parquet", report_path]
    if not all(path.exists() for path in paths):
        pytest.skip("Falta correr la simulación: python scripts/run_simulation.py")
    assert validate_file(paths[0], "recommendations") == []
    assert validate_file(paths[1], "policy_curve") == []
    document = json.loads(report_path.read_text(encoding="utf-8"))
    # Decisión 4: lo que hay en data/processed es de validación.
    assert document["tramo"]["nombre"] == "validacion" and document["prueba_final"]["simulada"] is False
    curve = read_contract("policy_curve", paths[1]).set_index(["policy", "budget"])
    assert len(curve) == len(document["resultados"])
    for row in document["resultados"]:
        line = curve.loc[(row["politica"], row["presupuesto_del_dia_en_bicis"])]
        assert line["risk_station_hours"] == row["riesgo_por_dia"]["media"]
        assert (line["ci_low"], line["ci_high"]) == (row["riesgo_por_dia"]["ic_bajo"], row["riesgo_por_dia"]["ic_alto"])
    none = curve.loc["none"]
    assert none["risk_station_hours"].nunique() == 1
    if 0 in none.index:
        for policy in POLICIES:
            assert curve.loc[(policy, 0)].equals(curve.loc[("none", 0)])
    recommendations = read_contract("recommendations", paths[0])
    evaluated = set(document["tramo"]["dias_evaluados"])
    assert set(recommendations["window_start"].dt.strftime("%Y-%m-%d")) == evaluated
    assert not set(document["tramo"]["dias_de_calentamiento"]) & evaluated
    assert recommendations["window_end"].max() < pd.Timestamp(document["prueba_final"]["inicio"], tz=TZ)
    assert recommendations["station_id"].nunique() == document["estaciones"]["simuladas"]
    assert len(recommendations) == document["recomendaciones"]["filas"]
    assert (recommendations.groupby(["station_id", "window_start"])["feasible"].nunique() == 1).all()


def test_las_salidas_reales_siguen_al_dia():
    """Las salidas reales de la simulación corresponden a las tablas, las reglas y el config de hoy.

    Lee lo mismo que lee la simulación: el tramo de validación y la historia anterior. Nada de la prueba final.
    """
    reports = DATA_PROCESSED / "reports"
    outputs = [DATA_PROCESSED / "recommendations.parquet", reports / "simulacion.json", reports / "simulacion.md"]
    if not all(path.exists() for path in outputs):
        pytest.skip("Falta correr la simulación: python scripts/run_simulation.py")
    inputs = [DATA_PROCESSED / f"{name}.parquet" for name in ("stations", "trips", "station_hour", "predictions")]
    if not all(path.exists() for path in inputs):
        pytest.skip("Faltan tablas de entrada en data/processed: no hay contra qué comparar.")
    check_outputs_are_current(DATA_PROCESSED)
