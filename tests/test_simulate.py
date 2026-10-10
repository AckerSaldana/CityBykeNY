"""Pruebas de la simulación completa (src/sim/simulate.py y scripts/run_simulation.py).

Con una red chica y datos hechos aquí mismo:

a) Las tablas cumplen sus contratos y traen las filas esperadas. Sin filas
   "lgbm" la política "model" no se simula y las tablas salen sin ella.
b) El día de calentamiento no cuenta en las métricas ni en las
   recomendaciones, pero su inventario sí pasa al día siguiente.
c) Con presupuesto 0 las políticas dan lo mismo que no intervenir, y "none" da
   lo mismo con cualquier presupuesto.
d) Dos corridas con las mismas entradas dan salidas idénticas, todas parten
   del mismo inventario y el total de bicis no cambia en ninguna.
e) El candado de la prueba final (decisión 4).
f) El intervalo por remuestreo de días se repite con la semilla y contiene a
   la media.
g) Qué estaciones se simulan y cuáles quedan fuera, con sus conteos.
h) Nivel de riesgo y `feasible` de las recomendaciones, a mano.
i) Inventario inicial: sale de la ocupación observada del archivo congelado
   de GBFS; lo que falta usa la media observada.
j) Banda de incertidumbre: se simulan todas las candidatas, gana la de menor
   riesgo medio entre políticas y se usa para todo lo demás.
k) Reporte por franja (mañana, tarde, noche, resto) y sensibilidad al
   inventario inicial, con el riesgo simulado evitado como porcentaje del
   riesgo sin intervención de cada inventario. Piso de azar: se corre con el
   presupuesto de referencia, va al reporte y no a las tablas. Tabla de
   viajes bloqueados y llegadas desviadas por política. El reporte termina
   con el registro del experimento descartado, tal cual.
l) El script de punta a punta, y lo que lee de cada archivo.
m) El reporte usa el lenguaje honesto y declara las limitaciones.
n) La huella de las entradas: con ella una prueba dice, con el comando que hay
   que correr, cuándo las salidas ya no corresponden a las tablas, las reglas,
   el archivo congelado de GBFS o el config de hoy.

Todo corre sin red y sin tocar data/processed. Las dos últimas pruebas miran
los archivos reales y se saltan si todavía no existen.
"""

import copy
import dataclasses
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
    validate_file,
    validate_table,
    to_contract_table,
    write_contract,
)
from src.data.datos_config import Tramo
from src.data.gbfs_frozen import FROZEN_FILE, Frozen, render
from src.models.metrics import FinalTestError
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML, TZ
from src.sim.engine import NIGHT, OUTSIDE, Reglas, SimulacionError, build_network, round_half_up
from src.sim.policies import ForecastWindow
from src.sim.simulate import (
    BAND_KEY,
    EXPERIMENT_FILE,
    FINAL_FOLDER,
    NO_MODEL,
    OBSERVED,
    RANDOM_LABEL,
    RERUN,
    avoided_percent,
    build_report,
    check_final_test,
    choose_band,
    current_fingerprint,
    daily_risk,
    interval,
    load_forecasts,
    load_history,
    load_trips,
    main,
    observed_occupancy,
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
SETTINGS = SimpleNamespace(
    dias_de_calentamiento=1, remuestreos=300, nivel_del_intervalo=0.9, semilla=7,
    bandas=(0.0, 0.5, 1.0), franja_noche=("22:00", "06:00"), sensibilidad_ocupacion_inicial=(0.4, 0.6),
    corridas_piso_de_azar=3,
)
SIMULATED = ["A", "B", "C", "D", "E", "F"]   # G no tiene capacidad y H no tiene viajes en validación
LABELS = ["manana", "tarde", NIGHT, OUTSIDE]

DATOS_TEXT = """\
meses_crudos: [202608, 202609]
periodo: {inicio: 2026-08-24, fin: 2026-09-06}
tramos:
  entrenamiento: {inicio: 2026-08-24, fin: 2026-08-31}
  validacion:    {inicio: 2026-09-01, fin: 2026-09-04}
  prueba:        {inicio: 2026-09-05, fin: 2026-09-06}
union_de_ids: {distancia_maxima_m: 1.0}
estaciones_excluidas: {prefijos: [HB], ids: []}
simulacion:
  dias_de_calentamiento: 1
  remuestreos: 300
  nivel_del_intervalo: 0.90
  semilla: 7
  bandas: [0, 0.5, 1]
  franja_noche: {inicio: "22:00", fin: "06:00"}
  sensibilidad_ocupacion_inicial: [0.4, 0.6]
  corridas_piso_de_azar: 3
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
        "estrategia_por_tipo": {
            kind: {"prioridad": 2, "ocupacion_respaldo": {"manana": 0.5, "tarde": 0.5}} for kind in STATION_TYPES
        },
    }
    cfg.update(copy.deepcopy(changes))
    return cfg


def settings(**changes) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(SETTINGS), **changes})


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ)


def stamps(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(values).as_unit("us")


def make_trips(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"])
    for column in ("started_at", "ended_at"):
        frame[column] = stamps([ts(value) for value in frame[column]])
    for column in ("start_lat", "start_lng", "end_lat", "end_lng"):
        frame[column] = np.nan
    return frame


def in_tramo(trips: pd.DataFrame, tramo: Tramo) -> pd.DataFrame:
    """Lo que entrega el lector: viajes con salida o llegada en el tramo."""
    keep = tramo.contains(trips["started_at"]) | tramo.contains(trips["ended_at"])
    return trips[keep].reset_index(drop=True)


def frozen_table(stations: pd.DataFrame, occupancy: dict | float = 0.5, drop: tuple = ()) -> pd.DataFrame:
    """Un archivo congelado de GBFS para esas estaciones. `occupancy` es un valor o uno por estación."""
    rows = stations[~stations["station_id"].isin(drop)]
    values = [occupancy.get(name, np.nan) if isinstance(occupancy, dict) else occupancy for name in rows["station_id"]]
    capacity = rows["capacity"].fillna(0).astype("int64")
    return pd.DataFrame(
        {
            "short_name": rows["station_id"].astype(str).to_numpy(),
            "capacity": pd.array(capacity.to_numpy(), dtype="Int64"),
            "ocupacion_inicial": np.where(capacity.to_numpy() > 0, np.array(values, dtype="float64"), np.nan),
            "fotos": np.where(capacity.to_numpy() > 0, 7, 0),
        }
    )


OCCUPANCY = {"A": 0.25, "B": 0.5, "C": 0.75, "D": 1.0, "E": 0.1, "H": 0.5}   # F: sin ocupación observada


def make_case(seed: int = 0) -> dict:
    """Ocho estaciones y dos semanas de datos al azar: entrenamiento, validación y prueba.

    A a F se simulan. G no tiene capacidad (y sí viajes). H tiene capacidad y
    ningún viaje en validación. F no tiene historia en entrenamiento ni
    ocupación observada en GBFS. Hay viajes sin estación, con y sin punto.
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

    def points() -> tuple[np.ndarray, np.ndarray]:
        missing = rng.random(n) < 0.3
        return (np.where(missing, np.nan, 40.70 + rng.random(n) * 0.01),
                np.where(missing, np.nan, -73.99 + rng.random(n) * 0.01))

    start_lat, start_lng = points()
    end_lat, end_lng = points()
    trips = pd.DataFrame(
        {
            "ride_id": [f"R{i:05d}" for i in rng.permutation(n)],
            "started_at": stamps(started),
            "ended_at": stamps(started + pd.to_timedelta(rng.integers(60, 2400, n), unit="s")),
            "start_station_id": options[rng.choice(len(options), n, p=[0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.07, 0.03])],
            "end_station_id": options[rng.choice(len(options), n, p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.07, 0.03])],
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
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
    parts = []
    for model in ("baseline", "lgbm"):
        part = pd.DataFrame(
            {"station_id": np.repeat(predicted, len(hours)), "ts_hour": stamps(np.tile(hours, len(predicted)))}
        )
        part = part[part["ts_hour"] >= VALIDATION.start].reset_index(drop=True)
        part["model"] = model
        part["split"] = np.where(part["ts_hour"] < FINAL.start, "val", "test")
        for prefix in ("dep", "arr"):
            middle = rng.random(len(part)) * 2.5
            part[f"{prefix}_q10"] = middle * 0.6
            part[f"{prefix}_q50"] = middle
            part[f"{prefix}_q90"] = middle * 1.5
        parts.append(part[["station_id", "ts_hour", "model", "split", *QUANTILE_COLUMNS]])
    return {
        "stations": stations,
        "trips": trips,
        "station_hour": station_hour,
        "predictions": parts[0],                                  # solo la línea base: hoy no hay LightGBM
        "predictions_with_lgbm": pd.concat(parts, ignore_index=True),
        "frozen": frozen_table(stations, OCCUPANCY),
    }


def forecasts_of(case: dict, tramo: Tramo, lgbm: bool) -> dict:
    split = "val" if tramo.nombre == "validacion" else "test"
    table = case["predictions_with_lgbm"] if lgbm else case["predictions"]
    table = table[(table["split"] == split) & tramo.contains(table["ts_hour"])]
    out = {"historical": table[table["model"] == "baseline"].reset_index(drop=True)}
    if lgbm:
        out["model"] = table[table["model"] == "lgbm"].reset_index(drop=True)
    return out


def simulate_case(
    case: dict, tramo: Tramo = VALIDATION, ajustes=SETTINGS, cfg: dict | None = None, lgbm: bool = False,
    frozen: pd.DataFrame | None = None, **options,
):
    history = case["station_hour"][case["station_hour"]["ts_hour"] < tramo.start]
    return run_simulation(
        case["stations"], in_tramo(case["trips"], tramo), history[["station_id", "ts_hour"]],
        forecasts_of(case, tramo, lgbm), case["frozen"] if frozen is None else frozen,
        Reglas.from_operacion(cfg or make_cfg()), tramo, ajustes, prueba_final=FINAL, **options,
    )


@pytest.fixture(scope="module")
def case() -> dict:
    return make_case()


@pytest.fixture(scope="module")
def result(case):
    """La corrida de hoy: sin filas "lgbm", así que sin la política "model"."""
    return simulate_case(case)


@pytest.fixture(scope="module")
def result_lgbm(case):
    return simulate_case(case, lgbm=True)


def row_of(report: dict, policy: str, budget: int) -> dict:
    (row,) = [
        r for r in report["resultados"] if r["politica"] == policy and r["presupuesto_del_dia_en_bicis"] == budget
    ]
    return row


# --------------------------------------------------------------------------- #
# a) Contratos, filas y políticas simuladas
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fixture", ["result", "result_lgbm"])
def test_las_tablas_cumplen_sus_contratos(fixture, request):
    outcome = request.getfixturevalue(fixture)
    for name, table in (("recommendations", outcome.recommendations), ("policy_curve", outcome.policy_curve)):
        assert validate_table(to_contract_table(table, name), name) == [], name


def test_sin_filas_lgbm_la_politica_model_no_se_simula_y_las_tablas_salen_sin_ella(result):
    assert set(result.recommendations["policy"]) == {"none", "historical"}
    assert set(result.policy_curve["policy"]) == {"none", "historical"}
    report = result.report
    assert report["pronostico"]["politicas_simuladas"] == ["none", "historical"]
    assert report["pronostico"]["politicas_con_pronostico"] == ["historical"]
    assert report["pronostico"]["model_simulada"] is False
    assert report["pronostico"]["modelo_que_alimenta_model"] == "ninguno"   # lo lee el dashboard
    assert report["pronostico"]["modelo_de_cada_politica"] == {"historical": "baseline"}
    assert not any(row["politica"] == "model" for row in report["resultados"])
    assert all(policy == "historical" for (policy, _) in result.runs if policy != "none")
    # Se dice sin rodeos, en el reporte y en la pantalla. Ya no se alimenta "model" con la línea base.
    text = build_report(report)
    assert f"**{NO_MODEL}**" in text and NO_MODEL in report["limitaciones"]
    assert 'NO se simuló' in NO_MODEL and 'Ya no se alimenta "model" con la línea base' in NO_MODEL
    assert any("NO se simuló" in line for line in summary_lines(report))
    assert "| modelo |" not in text


def test_con_filas_lgbm_se_simulan_las_tres_y_cada_una_usa_su_pronostico(result, result_lgbm):
    report = result_lgbm.report
    assert set(result_lgbm.recommendations["policy"]) == set(POLICIES)
    assert report["pronostico"]["politicas_simuladas"] == list(POLICIES)
    assert report["pronostico"]["modelo_de_cada_politica"] == {"historical": "baseline", "model": "lgbm"}
    assert report["pronostico"]["model_simulada"] is True and NO_MODEL not in build_report(report)
    assert report["pronostico"]["modelo_que_alimenta_model"] == "lgbm"
    budget = report["reglas"]["presupuesto_de_referencia_en_bicis"]
    # Con pronósticos distintos no deciden lo mismo.
    assert result_lgbm.runs[("historical", budget)].digest() != result_lgbm.runs[("model", budget)].digest()
    # Y si las dos reciben EL MISMO pronóstico deciden lo mismo: la maquinaria es una sola.
    same = make_case()
    twin = same["predictions"].assign(model="lgbm")
    same["predictions_with_lgbm"] = pd.concat([same["predictions"], twin], ignore_index=True)
    outcome = simulate_case(same, lgbm=True)
    for total in {b for (_, b) in outcome.runs}:
        one, other = outcome.runs[("historical", total)], outcome.runs[("model", total)]
        assert (one.risk_us == other.risk_us).all() and (one.final_inventory == other.final_inventory).all()
        assert [r.moves for r in one.interventions] == [r.moves for r in other.interventions]


def test_el_pronostico_de_la_regla_historica_es_obligatorio(case):
    with pytest.raises(SimulacionError, match="run_baseline.py"):
        simulate_case({**case, "predictions": case["predictions"].iloc[0:0]})
    forecasts = forecasts_of(case, VALIDATION, lgbm=False)
    history = case["station_hour"][case["station_hour"]["ts_hour"] < VALIDATION.start][["station_id", "ts_hour"]]
    with pytest.raises(SimulacionError, match="run_baseline.py"):
        run_simulation(
            case["stations"], in_tramo(case["trips"], VALIDATION), history, {"model": forecasts["historical"]},
            case["frozen"], Reglas.from_operacion(make_cfg()), VALIDATION, SETTINGS, prueba_final=FINAL,
        )


def test_recommendations_trae_una_fila_por_estacion_franja_dia_evaluado_y_politica(result, result_lgbm):
    for outcome, policies in ((result, 2), (result_lgbm, 3)):
        table = outcome.recommendations
        assert len(table) == len(SIMULATED) * 2 * 3 * policies   # 6 estaciones, 2 franjas, 3 días evaluados
        assert not table.duplicated(["station_id", "window_start", "policy"]).any()
        assert set(table["station_id"]) == set(SIMULATED)
        days = sorted(set(table["window_start"].dt.strftime("%Y-%m-%d")))
        assert days == ["2026-09-02", "2026-09-03", "2026-09-04"] == outcome.report["tramo"]["dias_evaluados"]
        assert set(table["window_start"].dt.strftime("%H:%M")) == {"06:00", "15:00"}
        assert outcome.report["recomendaciones"] == summarize_recommendations(table)


def test_policy_curve_trae_una_fila_por_politica_simulada_y_presupuesto(result, result_lgbm):
    for outcome, policies in ((result, 2), (result_lgbm, 3)):
        curve = outcome.policy_curve
        assert sorted(curve["budget"].unique().tolist()) == [0, 3, 6, 12]   # 0, 0.5, 1 y 2 por 6 estaciones
        assert len(curve) == 4 * policies
        for line in curve.itertuples():
            row = row_of(outcome.report, line.policy, line.budget)
            assert line.risk_station_hours == row["riesgo_por_dia"]["media"]
            assert (line.ci_low, line.ci_high) == (row["riesgo_por_dia"]["ic_bajo"], row["riesgo_por_dia"]["ic_alto"])
            assert line.ci_low <= line.risk_station_hours <= line.ci_high


def test_las_recomendaciones_dicen_lo_que_paso_en_cada_intervencion(result_lgbm):
    table, budget = result_lgbm.recommendations, 6
    ids = list(result_lgbm.network.station_ids)
    for policy in POLICIES:
        run = result_lgbm.runs[(policy, budget)]
        records = [r for r in run.interventions if r.day_index >= result_lgbm.calendar.warmup_days]
        rows = table[table["policy"] == policy]
        assert len(rows) == len(records) * len(ids)
        for record in records:
            start = pd.Timestamp(
                result_lgbm.calendar.window_start_us[record.day_index, record.franja_index], unit="us", tz="UTC"
            )
            window = rows[rows["window_start"] == start].set_index("station_id").loc[ids]
            assert window["bikes_to_move"].tolist() == record.net.tolist()
            assert window["start_inventory_target"].tolist() == (record.inventory_before + record.net).tolist()
            assert int(window["bikes_to_move"].sum()) == 0   # lo que reciben unas lo entregan otras
            assert int(window["bikes_to_move"].clip(lower=0).sum()) == record.budget_used <= record.budget_available
    assert (table.loc[table["policy"] == "none", "bikes_to_move"] == 0).all()
    assert (table.loc[table["policy"] != "none", "bikes_to_move"] != 0).any()


# --------------------------------------------------------------------------- #
# b) Calentamiento
# --------------------------------------------------------------------------- #

def test_el_calentamiento_no_cuenta_pero_su_inventario_pasa_al_dia_siguiente(case, result):
    whole = simulate_case(case, ajustes=settings(dias_de_calentamiento=0, bandas=(result.report["banda"]["elegida"],)))
    assert whole.report["tramo"]["dias_de_calentamiento"] == []
    assert len(whole.report["tramo"]["dias_evaluados"]) == 4
    assert result.report["tramo"]["dias_de_calentamiento"] == ["2026-09-01"]
    for policy, budget in (("none", 0), ("historical", 6), ("historical", 12)):
        all_days = row_of(whole.report, policy, budget)
        counted = row_of(result.report, policy, budget)
        # Los tres días que cuentan son los mismos, con el mismo inventario heredado del primero.
        assert counted["riesgo_de_cada_dia"] == all_days["riesgo_de_cada_dia"][1:]
        assert counted["viajes_bloqueados"]["cada_dia"] == all_days["viajes_bloqueados"]["cada_dia"][1:]
        assert counted["bicis_movidas"]["cada_dia"] == all_days["bicis_movidas"]["cada_dia"][1:]
        warm = counted["calentamiento_no_cuenta"]
        assert warm["horas_de_estacion_en_riesgo"] == pytest.approx(all_days["riesgo_de_cada_dia"][0])
        assert warm["viajes_bloqueados"] == all_days["viajes_bloqueados"]["cada_dia"][0]
        assert warm["bicis_movidas"] == all_days["bicis_movidas"]["cada_dia"][0]
    # El día de calentamiento sí pasa: sin él, el riesgo del segundo día sería otro.
    late = simulate_case(case, tramo=Tramo("validacion", date(2026, 9, 2), date(2026, 9, 4)),
                         ajustes=settings(dias_de_calentamiento=0))
    assert row_of(late.report, "none", 0)["riesgo_de_cada_dia"] != row_of(result.report, "none", 0)["riesgo_de_cada_dia"]


def test_el_calentamiento_no_puede_cubrir_todo_el_tramo(case):
    with pytest.raises(SimulacionError, match="dias_de_calentamiento"):
        simulate_case(case, ajustes=settings(dias_de_calentamiento=4))


# --------------------------------------------------------------------------- #
# c) Presupuesto cero y "none"
# --------------------------------------------------------------------------- #

def test_con_presupuesto_cero_las_politicas_son_la_corrida_sin_intervencion(result_lgbm):
    report = result_lgbm.report
    none = row_of(report, "none", 0)
    for policy in ("historical", "model"):
        zero = row_of(report, policy, 0)
        assert zero["simulada"] is False
        for key in ("riesgo_por_dia", "riesgo_de_cada_dia", "viajes_bloqueados", "llegadas_desviadas", "bicis"):
            assert zero[key] == none[key], key
        assert zero["riesgo_simulado_evitado_por_dia"] == {"media": 0.0, "ic_bajo": 0.0, "ic_alto": 0.0}
        assert zero["bicis_movidas"]["total"] == 0
    for budget in (3, 6, 12):
        other = row_of(report, "none", budget)
        assert other["riesgo_de_cada_dia"] == none["riesgo_de_cada_dia"] and other["bicis_movidas"]["total"] == 0
        # "none" tiene el presupuesto disponible y no lo usa.
        assert sum(f["usado_total"] for f in other["presupuesto_por_franja"].values()) == 0
        assert other["presupuesto_por_franja"]["tarde"]["disponible_por_dia"] == budget


def test_el_reporte_dice_el_presupuesto_disponible_y_usado_por_franja(result):
    row = row_of(result.report, "historical", 6)
    manana, tarde = row["presupuesto_por_franja"]["manana"], row["presupuesto_por_franja"]["tarde"]
    assert manana["disponible_cada_dia"] == [3, 3, 3]
    for available, used_before, used in zip(tarde["disponible_cada_dia"], manana["usado_cada_dia"], tarde["usado_cada_dia"]):
        assert available == 3 + (3 - used_before) and used <= available   # lo no usado pasa a la tarde
    assert row["bicis_movidas"]["cada_dia"] == [a + b for a, b in zip(manana["usado_cada_dia"], tarde["usado_cada_dia"])]
    assert result.report["reglas"]["reparto_por_franja"]["6"] == {"manana": 3, "tarde": 3}
    assert result.report["reglas"]["franjas"][0]["parte"] == 0.5


def test_dos_presupuestos_que_dan_el_mismo_total_se_simulan_una_vez(case):
    cfg = make_cfg(presupuestos_por_estacion=[0, 0.5, 0.55, 1.0])   # 0.5 y 0.55 por 6 estaciones dan 3
    outcome = simulate_case(case, cfg=cfg)
    assert outcome.report["reglas"]["presupuestos_con_el_mismo_total_que_otro"] == [0.55]
    assert sorted(outcome.policy_curve["budget"].unique().tolist()) == [0, 3, 6]
    assert "0.55" in build_report(outcome.report)


def test_un_presupuesto_de_referencia_que_da_cero_bicis_se_rechaza(case):
    cfg = make_cfg(presupuestos_por_estacion=[0, 0.01, 1.0], presupuesto_de_referencia=0.01)
    with pytest.raises(SimulacionError, match="0 bicis al día"):
        simulate_case(case, cfg=cfg)


# --------------------------------------------------------------------------- #
# d) Mismas entradas, mismo inventario y total constante
# --------------------------------------------------------------------------- #

def test_dos_corridas_con_las_mismas_entradas_dan_salidas_identicas(case, result):
    again = simulate_case(make_case())
    pd.testing.assert_frame_equal(again.recommendations, result.recommendations)
    pd.testing.assert_frame_equal(again.policy_curve, result.policy_curve)
    assert again.report == result.report
    assert build_report(again.report) == build_report(result.report)
    for key, run in result.runs.items():
        assert again.runs[key].digest() == run.digest()
    # El orden de las filas de entrada no cambia nada.
    shuffled = make_case()
    for name in ("stations", "trips", "predictions", "frozen"):
        shuffled[name] = shuffled[name].sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert simulate_case(shuffled).report == result.report


def test_todas_las_politicas_parten_del_mismo_inventario_y_el_total_de_bicis_no_cambia(result_lgbm):
    report = result_lgbm.report
    start = report["inventario_inicial"]["bicis"]
    assert {run.initial_total for run in result_lgbm.runs.values()} == {start}
    for row in report["resultados"]:
        bikes = row["bicis"]
        assert bikes["total_al_inicio"] == bikes["total_al_final"] == start
        assert bikes["en_estaciones_al_final"] + bikes["en_transito_al_final"] == start
        assert bikes["en_transito_al_final_de_cada_dia"][-1] == bikes["en_transito_al_final"]
        assert row["llegadas_sin_lugar_en_ninguna_estacion"] == 0
    assert row_of(report, "none", 0)["bicis"]["en_transito_al_final"] > 0   # hay viajes que siguen en camino
    assert any("total de bicis" in check for check in report["comprobaciones"])
    # La tabla del reporte: una fila por corrida mostrada, y en todas el total del inicio es el del final.
    text = build_report(report)
    table = text.split("## Viajes y total de bicis")[1].split("## Inventario inicial")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) == 1 + 4 * 2
    for line in rows:
        cells = [cell.strip().replace(",", "") for cell in line.strip("|").split("|")]
        assert int(cells[3]) == int(cells[6]) == int(cells[4]) + int(cells[5]) == start


def test_el_reporte_cuenta_los_viajes_reasignados_e_ignorados(case, result):
    trips = result.report["viajes"]
    inside = in_tramo(case["trips"], VALIDATION)
    assert trips["viajes_leidos"] == len(inside)
    starts = VALIDATION.contains(inside["started_at"])
    assert trips["ignorados"]["ya_iban_en_camino_al_empezar_el_tramo"] == int((inside["started_at"] < VALIDATION.start).sum())
    outside_start = starts & ~inside["start_station_id"].isin(SIMULATED)
    with_point = inside["start_lat"].notna()
    origins = trips["salidas_por_origen"]
    assert origins["de_la_estacion_mas_cercana_al_punto_de_inicio"] == int((outside_start & with_point).sum()) > 0
    assert trips["ignorados"]["empiezan_fuera_del_conjunto_y_sin_punto_de_inicio"] == int((outside_start & ~with_point).sum()) > 0
    assert trips["salidas"] == int(starts.sum()) - int((outside_start & ~with_point).sum())
    assert trips["salidas"] == trips["llegadas"] + trips["siguen_en_camino_al_terminar_el_tramo"]
    ends = trips["llegadas_por_destino"]
    assert ends["a_la_estacion_mas_cercana_al_punto_de_fin"] > 0 and ends["de_vuelta_a_la_estacion_de_salida"] > 0
    assert sum(ends[k] for k in ("a_su_estacion", "a_la_estacion_mas_cercana_al_punto_de_fin",
                                 "de_vuelta_a_la_estacion_de_salida")) == trips["llegadas"]


# --------------------------------------------------------------------------- #
# e) Candado de la prueba final (decisión 4)
# --------------------------------------------------------------------------- #

def test_la_prueba_final_no_se_simula_sin_pedirlo():
    with pytest.raises(FinalTestError, match="Decisión 4"):
        check_final_test(FINAL, FINAL, usar_prueba_final=False)
    # Tampoco un tramo con otro nombre que toque sus fechas.
    with pytest.raises(FinalTestError, match="--usar-prueba-final"):
        check_final_test(Tramo("validacion", date(2026, 9, 1), date(2026, 9, 5)), FINAL, usar_prueba_final=False)
    check_final_test(VALIDATION, FINAL, usar_prueba_final=False)
    check_final_test(FINAL, FINAL, usar_prueba_final=True)


def test_run_simulation_rechaza_la_prueba_final_antes_de_tocar_los_datos():
    with pytest.raises(FinalTestError):
        run_simulation(None, None, None, None, None, None, FINAL, None, prueba_final=FINAL)


def test_la_historia_no_puede_traer_filas_del_tramo_simulado(case):
    leak = dict(case)
    extra = case["station_hour"][case["station_hour"]["ts_hour"] == VALIDATION.start]
    with pytest.raises(SimulacionError, match="filas del tramo simulado"):
        run_simulation(
            case["stations"], in_tramo(case["trips"], VALIDATION),
            pd.concat([leak["station_hour"][leak["station_hour"]["ts_hour"] < VALIDATION.start], extra])[
                ["station_id", "ts_hour"]
            ],
            forecasts_of(case, VALIDATION, False), case["frozen"], Reglas.from_operacion(make_cfg()), VALIDATION,
            SETTINGS, prueba_final=FINAL,
        )


# --------------------------------------------------------------------------- #
# f) Intervalos
# --------------------------------------------------------------------------- #

def test_el_remuestreo_se_repite_con_la_semilla_y_el_intervalo_contiene_a_la_media():
    draws = resample_days(5, 200, 7)
    assert draws.shape == (200, 5) and draws.min() >= 0 and draws.max() <= 4
    assert (resample_days(5, 200, 7) == draws).all() and (resample_days(5, 200, 8) != draws).any()
    values = [10.0, 12.0, 30.0, 11.0, 9.0]
    mean, low, high = interval(values, draws, 0.9)
    assert mean == pytest.approx(14.4) and low <= mean <= high and low < high
    with pytest.raises(SimulacionError, match="No hay días"):
        resample_days(0, 100, 7)


def test_el_intervalo_a_mano():
    # Dos días y tres sorteos: (día 0, día 0), (día 0, día 1) y (día 1, día 1). Medias 10, 15 y 20.
    draws = np.array([[0, 0], [0, 1], [1, 1]])
    mean, low, high = interval([10.0, 20.0], draws, 0.5)
    assert (mean, low, high) == (15.0, 12.5, 17.5)   # percentiles 25 y 75 de (10, 15, 20)
    # Un valor constante da un intervalo de ancho cero.
    assert interval([7.0, 7.0], draws, 0.9) == (7.0, 7.0, 7.0)


def test_los_intervalos_del_reporte_salen_de_los_dias_evaluados_y_la_diferencia_es_pareada(result):
    draws = resample_days(3, SETTINGS.remuestreos, SETTINGS.semilla)
    none, row = row_of(result.report, "none", 0), row_of(result.report, "historical", 6)
    mean, low, high = interval(row["riesgo_de_cada_dia"], draws, 0.9)
    assert row["riesgo_por_dia"] == {"media": mean, "ic_bajo": low, "ic_alto": high}
    avoided = np.array(none["riesgo_de_cada_dia"]) - np.array(row["riesgo_de_cada_dia"])
    mean, low, high = interval(avoided, draws, 0.9)
    assert row["riesgo_simulado_evitado_por_dia"] == {"media": mean, "ic_bajo": low, "ic_alto": high}
    assert result.report["intervalo"] == {"remuestreos": 300, "nivel": 0.9, "semilla": 7, "dias_remuestreados": 3}


# --------------------------------------------------------------------------- #
# g) Estaciones simuladas
# --------------------------------------------------------------------------- #

def test_se_simulan_las_estaciones_con_capacidad_y_con_viajes_en_el_tramo():
    stations = pd.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "name": list("abcd"),
            "lat": [40.70, 40.71, 40.72, 40.73],
            "lon": [-73.99] * 4,
            "capacity": pd.array([10, None, 12, 0], dtype="Int64"),   # B sin capacidad; D con 0
            "station_type": pd.Series([None] * 4, dtype="object"),
        }
    )
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", "B", "D"),
            ("r3", "2026-09-01 10:00", "2026-09-01 10:10", "Z", None),      # Z no está en stations
            ("r4", "2026-08-31 23:00", "2026-08-31 23:30", "C", "C"),      # antes del tramo: C queda inactiva
        ]
    )
    start, end = (int(v) for v in pd.DatetimeIndex([ts("2026-09-01"), ts("2026-09-02")]).as_unit("us").asi8)
    chosen, report = select_stations(stations, trips, start, end)
    assert chosen["station_id"].tolist() == ["A"]
    assert (report["en_stations"], report["simuladas"]) == (4, 1)
    assert report["sin_capacidad"] == {
        "estaciones": 2, "de_ellas_sin_viajes_en_el_tramo": 0, "salidas_en_el_tramo": 1, "llegadas_en_el_tramo": 2,
    }
    assert report["inactivas"]["estaciones"] == 1
    assert report["ids_de_viajes_que_no_estan_en_stations"] == {"ids": 1, "salidas_en_el_tramo": 1, "llegadas_en_el_tramo": 0}
    assert report["extremos_sin_estacion"] == {"salidas_en_el_tramo": 0, "llegadas_en_el_tramo": 1}
    assert (report["salidas_en_el_tramo"], report["llegadas_en_el_tramo"]) == (3, 3)


def test_el_reporte_cuenta_las_estaciones_que_quedan_fuera(result):
    stations = result.report["estaciones"]
    assert (stations["en_stations"], stations["simuladas"]) == (8, 6)
    assert stations["sin_capacidad"]["estaciones"] == 1 and stations["inactivas"]["estaciones"] == 1
    assert result.network.station_ids == tuple(SIMULATED)
    assert "Estaciones simuladas: 6 de 8." in "\n".join(summary_lines(result.report))
    info = result.report["informacion"]
    assert info["estaciones_sin_historia"] == 1                       # F
    assert info["historical"]["estaciones_sin_informacion_en_todas"] == 1
    assert info["historical"]["estacion_franja_sin_informacion"] == 6   # F, 2 franjas, 3 días


# --------------------------------------------------------------------------- #
# h) Nivel de riesgo y `feasible`, a mano
# --------------------------------------------------------------------------- #

def test_el_nivel_de_riesgo_sale_del_pronostico_y_del_inventario_antes_de_intervenir():
    stations = pd.DataFrame(
        {
            "station_id": list("ABCDEF"),
            "name": list("abcdef"),
            "lat": [40.70 + 0.005 * i for i in range(6)],
            "lon": [-73.99] * 6,
            "capacity": pd.array([20] * 6, dtype="Int64"),
            "station_type": pd.Series([None] * 6, dtype="object"),
        }
    )
    network = build_network(stations)
    reglas = Reglas.from_operacion(make_cfg())   # menos de 2 bicis o menos de 2 lugares
    info = np.array([True, True, True, True, False, False])
    zeros = np.zeros(6)
    # La trayectoria central baja 5 en todas (lo central = 7, hi = 18).
    center_dip = np.full(6, -5.0)
    # Banda con la que decide la política: baja 8 (lo = 10) y sube 2 (hi = 16). En D no hay solución.
    window = ForecastWindow(info, center_dip, zeros, np.array([-8.0, -8.0, -8.0, -15.0, 0, 0]),
                            np.array([2.0, 2.0, 2.0, 12.0, 0, 0]))
    # Banda completa: baja 12 (lo = 14) y sube 4 (hi = 14).
    full = ForecastWindow(info, center_dip, zeros, np.full(6, -12.0), np.full(6, 4.0))
    before = [5, 12, 14, 12, 1, 10]
    level, feasible = risk_levels(before, network, reglas, window, full)
    # A: 5 < 7, la trayectoria central ya cruza -> alto.  B: 12 aguanta la central y no la banda completa -> medio.
    # C: 14 aguanta todo -> bajo.  D: como B.  E: sin información y ya bajo el umbral -> alto.  F: sin información -> bajo.
    assert level.tolist() == ["alto", "medio", "bajo", "medio", "alto", "bajo"]
    # `feasible` usa la banda de la política: en D lo = 17 y hi = 6. Sin información queda en verdadero.
    assert feasible.tolist() == [True, True, True, False, True, True]
    # El nivel no depende de la banda de la política: con banda 0 (la central) "medio" sigue existiendo.
    zero_band = ForecastWindow(info, center_dip, zeros, center_dip, zeros)
    level, feasible = risk_levels(before, network, reglas, zero_band, full)
    assert level.tolist() == ["alto", "medio", "bajo", "medio", "alto", "bajo"] and feasible.all()
    # Sin la banda completa se usa la de la política.
    assert risk_levels(before, network, reglas, zero_band)[0].tolist() == ["alto", "bajo", "bajo", "bajo", "alto", "bajo"]


# --------------------------------------------------------------------------- #
# i) Inventario inicial: ocupación observada en GBFS
# --------------------------------------------------------------------------- #

def test_la_ocupacion_inicial_sale_del_archivo_congelado_y_lo_que_falta_usa_la_media(case, result):
    stations = case["stations"]
    network = build_network(stations[stations["station_id"].isin(SIMULATED)])
    occupancy, report = observed_occupancy(network, case["frozen"])
    mean = (0.25 + 0.5 + 0.75 + 1.0 + 0.1) / 5
    assert occupancy.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0, 0.1, mean])   # F no tiene: la media
    assert report == {
        "estaciones_con_ocupacion_observada": 5,
        "estaciones_con_la_media": 1,
        "de_ellas_fuera_del_archivo": 0,
        "ocupacion_media_observada": pytest.approx(mean),
    }
    # Una estación que no está en el archivo también usa la media, y se cuenta aparte.
    _, missing = observed_occupancy(network, frozen_table(stations, OCCUPANCY, drop=("A",)))
    assert (missing["estaciones_con_la_media"], missing["de_ellas_fuera_del_archivo"]) == (2, 1)
    # El inventario de la corrida es ocupación por capacidad, redondeado: 8, 10, 12, 9, 14 y 10 lugares.
    expected = [2, 5, 9, 9, 1, 5]   # 0.25x8, 0.5x10, 0.75x12, 1x9, 0.1x14 = 1.4, 0.52x10 = 5.2
    assert round_half_up(occupancy * network.capacity).tolist() == expected
    start = result.report["inventario_inicial"]
    assert start["bicis"] == sum(expected) and start["lugares"] == 63
    assert start["estaciones_con_la_media"] == 1 and start["estaciones_sin_tipo"] == 6
    first = result.runs[("none", 0)].interventions[0]
    assert int(first.inventory_before.sum()) <= sum(expected)   # menos las que van en camino a las 05:00
    assert {run.initial_total for run in result.runs.values()} == {sum(expected)}


def test_si_la_capacidad_del_archivo_congelado_no_es_la_de_stations_la_simulacion_no_corre(case):
    other = case["frozen"].copy()
    other.loc[other["short_name"] == "B", "capacity"] = 11
    with pytest.raises(SimulacionError, match=r"B: 10 contra 11.*build_tables.py"):
        simulate_case(case, frozen=other)
    nothing = frozen_table(case["stations"], {})
    with pytest.raises(SimulacionError, match="no trae la ocupación inicial de ninguna estación"):
        simulate_case(case, frozen=nothing)


def test_el_reporte_dice_que_la_captura_es_de_octubre_y_no_de_los_dias_simulados(result):
    text = build_report(result.report)
    section = text.split("## Inventario inicial")[1].split("## Pronóstico")[0]
    assert "**La captura es de octubre de 2026, no de los días simulados**" in section
    assert "config/gbfs_congelado.csv" in section and "00:00 y las 00:59" in section
    assert any("La captura es de octubre de 2026, no de los días simulados" in item for item in result.report["limitaciones"])
    assert any("octubre de 2026" in line for line in summary_lines(result.report))


# --------------------------------------------------------------------------- #
# j) Banda de incertidumbre
# --------------------------------------------------------------------------- #

def test_se_elige_la_banda_de_menor_riesgo_medio_entre_politicas_y_un_empate_va_a_la_primera():
    risk = {0.0: {"historical": 10.0, "model": 30.0}, 0.5: {"historical": 22.0, "model": 14.0},
            1.0: {"historical": 5.0, "model": 40.0}}
    # Promedios 20, 18 y 22.5: gana 0.5 aunque cada política por separado prefiera otra.
    assert choose_band((0.0, 0.5, 1.0), risk) == 0.5
    assert choose_band((1.0, 0.0), risk) == 0.0
    tie = {0.0: {"historical": 12.0}, 0.5: {"historical": 12.0}, 1.0: {"historical": 12.5}}
    assert choose_band((0.0, 0.5, 1.0), tie) == 0.0 and choose_band((0.5, 0.0, 1.0), tie) == 0.5


@pytest.mark.parametrize("fixture", ["result", "result_lgbm"])
def test_la_banda_se_elige_en_validacion_y_se_aplica_igual_a_todas_las_politicas(fixture, request, case):
    outcome = request.getfixturevalue(fixture)
    band, names = outcome.report["banda"], outcome.report["pronostico"]["politicas_con_pronostico"]
    reference = outcome.report["reglas"]["presupuesto_de_referencia_en_bicis"]
    assert band["candidatas"] == [0.0, 0.5, 1.0] and band["presupuesto_del_dia_en_bicis"] == reference == 6
    rows = band["resultados"]
    assert [row["banda"] for row in rows] == [0.0, 0.5, 1.0] and all(set(row["por_politica"]) == set(names) for row in rows)
    # Cada candidata, simulada aparte con esa sola banda, da el mismo riesgo: los tres resultados son corridas reales.
    for row in rows:
        alone = simulate_case(case, lgbm=fixture == "result_lgbm", ajustes=settings(bandas=(row["banda"],)))
        for name in names:
            assert row["por_politica"][name]["riesgo_por_dia"] == row_of(alone.report, name, reference)["riesgo_por_dia"]["media"]
        mean = np.mean([row["por_politica"][name]["riesgo_por_dia"] for name in names])
        assert row["riesgo_medio_entre_politicas"] == pytest.approx(mean)
    # Gana la de menor riesgo medio entre políticas (la primera si empatan), y es una sola para todas.
    best = min(rows, key=lambda row: row["riesgo_medio_entre_politicas"])
    assert band["elegida"] == best["banda"] and [row["elegida"] for row in rows] == [row is best for row in rows]
    # La curva y las recomendaciones usan esa banda: el punto de referencia de la curva es el de la tabla de bandas.
    for name in names:
        assert row_of(outcome.report, name, reference)["riesgo_por_dia"]["media"] == best["por_politica"][name]["riesgo_por_dia"]
        avoided = row_of(outcome.report, name, reference)["riesgo_simulado_evitado_por_dia"]
        assert best["por_politica"][name]["riesgo_simulado_evitado_por_dia"] == avoided
    text = build_report(outcome.report)
    section = text.split("## Banda de incertidumbre")[1].split("## Estaciones sin información")[0]
    assert f"**Se eligió {band['elegida']:g}**" in section and "(elegida)" in section
    assert len([line for line in section.splitlines() if line.startswith("| ") and "---" not in line]) == 1 + 3 * len(names)


def test_la_banda_cambia_el_resultado_y_con_mas_banda_hay_mas_franjas_sin_solucion(result):
    rows = result.report["banda"]["resultados"]
    assert len({row["por_politica"]["historical"]["riesgo_por_dia"] for row in rows}) > 1
    unsolved = [row["por_politica"]["historical"]["estacion_franja_sin_solucion"] for row in rows]
    assert unsolved == sorted(unsolved) and unsolved[-1] > unsolved[0]
    # Las estaciones no tienen tipo: en una franja sin solución no se tocan.
    chosen = next(row for row in rows if row["elegida"])
    facts = result.report["diferencias_entre_politicas"]["historical"]
    assert facts["estacion_franja_sin_solucion"] == chosen["por_politica"]["historical"]["estacion_franja_sin_solucion"]
    assert facts["de_ellas_sin_tipo_no_se_tocan"] == facts["estacion_franja_sin_solucion"]
    assert facts["bicis_recibidas_en_franjas_sin_solucion"] == facts["bicis_entregadas_en_franjas_sin_solucion"] == 0
    table = result.recommendations
    assert (table.loc[~table["feasible"], "bikes_to_move"] == 0).all()
    assert result.report["recomendaciones"]["por_politica"]["historical"]["estacion_franja_sin_solucion"] == int(
        (~table.loc[table["policy"] == "historical", "feasible"]).sum()
    )


# --------------------------------------------------------------------------- #
# k) Reporte por franja y sensibilidad al inventario inicial
# --------------------------------------------------------------------------- #

def test_el_riesgo_se_reporta_por_manana_tarde_noche_y_resto_y_por_dia(result):
    assert result.report["reglas"]["etiquetas_del_reporte"] == LABELS
    assert result.report["reglas"]["franja_noche"] == ["22:00", "06:00"]
    for row in result.report["resultados"]:
        assert list(row["riesgo_por_franja"]) == LABELS == list(row["riesgo_por_franja_de_cada_dia"])
        total = sum(row["riesgo_por_franja"][label]["media"] for label in LABELS)
        assert total == pytest.approx(row["riesgo_por_dia"]["media"])
        by_day = np.sum([row["riesgo_por_franja_de_cada_dia"][label] for label in LABELS], axis=0)
        assert by_day.tolist() == pytest.approx(row["riesgo_de_cada_dia"])
        assert len(row["riesgo_de_cada_dia"]) == 3
    none = row_of(result.report, "none", 0)
    assert none["riesgo_por_franja"][NIGHT]["media"] > 0 and none["riesgo_por_franja"][OUTSIDE]["media"] > 0
    # Seis estaciones en riesgo toda la noche (8 horas) serían 48 horas de estación: la noche no puede pasar de ahí.
    assert max(none["riesgo_por_franja_de_cada_dia"][NIGHT]) <= 6 * 8
    text = build_report(result.report)
    assert "| manana | tarde | noche | resto |" in text and "## Riesgo día por día" in text
    daily = text.split("## Riesgo día por día")[1].split("## Riesgo por franja")[0]
    assert "2026-09-02 (miércoles)" in daily and "2026-09-04 (viernes)" in daily and "2026-09-01" not in daily


def test_otra_franja_de_noche_cambia_el_reparto_y_no_el_total(case, result):
    other = simulate_case(case, ajustes=settings(franja_noche=("00:00", "05:00")))
    for policy, budget in (("none", 0), ("historical", 6)):
        one, two = row_of(result.report, policy, budget), row_of(other.report, policy, budget)
        assert one["riesgo_de_cada_dia"] == pytest.approx(two["riesgo_de_cada_dia"])
        assert one["riesgo_por_franja"]["manana"] == two["riesgo_por_franja"]["manana"]
        assert one["riesgo_por_franja"][NIGHT]["media"] > two["riesgo_por_franja"][NIGHT]["media"]


def test_la_sensibilidad_repite_la_corrida_de_referencia_con_otros_inventarios_iniciales(case, result_lgbm):
    report = result_lgbm.report
    block = report["sensibilidad_al_inventario_inicial"]
    assert block["presupuesto_del_dia_en_bicis"] == 6 and block["banda"] == report["banda"]["elegida"]
    rows = block["filas"]
    assert [row["inventario_inicial"] for row in rows] == [OBSERVED, "todas al 40 %", "todas al 60 %"]
    assert all(set(row["por_politica"]) == {"historical", "model"} for row in rows)
    # La primera fila es la corrida principal.
    main_row, none = rows[0], row_of(report, "none", 0)
    assert main_row["bicis"] == report["inventario_inicial"]["bicis"]
    assert main_row["riesgo_sin_intervencion_por_dia"] == none["riesgo_por_dia"]["media"]
    for name in ("historical", "model"):
        row = row_of(report, name, 6)
        assert main_row["por_politica"][name]["riesgo_por_dia"] == row["riesgo_por_dia"]["media"]
        assert main_row["por_politica"][name]["riesgo_simulado_evitado_por_dia"] == pytest.approx(
            row["riesgo_simulado_evitado_por_dia"]["media"]
        )
    # El porcentaje evitado de cada fila es contra el riesgo sin intervención de ESE inventario inicial.
    assert len({row["riesgo_sin_intervencion_por_dia"] for row in rows}) == len(rows)   # las tres filas difieren
    for row in rows:
        for name in ("historical", "model"):
            cell = row["por_politica"][name]
            assert cell["riesgo_simulado_evitado_pct"] == pytest.approx(
                cell["riesgo_simulado_evitado_por_dia"] / row["riesgo_sin_intervencion_por_dia"] * 100
            )
            assert cell["riesgo_simulado_evitado_pct"] == pytest.approx(
                (1 - cell["riesgo_por_dia"] / row["riesgo_sin_intervencion_por_dia"]) * 100
            )
    # Las otras: todas las estaciones a esa fracción. Lo mismo que una corrida aparte con ese inventario.
    capacity = np.array([8, 10, 12, 9, 14, 10])
    for flat, row in zip((0.4, 0.6), rows[1:]):
        assert row["bicis"] == int(round_half_up(flat * capacity).sum())
        alone = simulate_case(
            case, lgbm=True, frozen=frozen_table(case["stations"], flat),
            ajustes=settings(bandas=(block["banda"],), sensibilidad_ocupacion_inicial=()),
        )
        assert alone.report["inventario_inicial"]["bicis"] == row["bicis"]
        assert row["riesgo_sin_intervencion_por_dia"] == row_of(alone.report, "none", 0)["riesgo_por_dia"]["media"]
        for name in ("historical", "model"):
            assert row["por_politica"][name]["riesgo_por_dia"] == row_of(alone.report, name, 6)["riesgo_por_dia"]["media"]
        assert [r["inventario_inicial"] for r in alone.report["sensibilidad_al_inventario_inicial"]["filas"]] == [OBSERVED]
    # No alimenta ni la curva ni las recomendaciones.
    assert {run.initial_total for run in result_lgbm.runs.values()} == {main_row["bicis"]}
    section = build_report(report).split("## Sensibilidad al inventario inicial")[1].split("## Métricas secundarias")[0]
    assert "todas al 40 %" in section and "regla histórica: riesgo simulado evitado" in section
    assert "modelo: riesgo simulado evitado" in section
    assert "regla histórica: % evitado" in section and "modelo: % evitado" in section
    assert "de ESE MISMO inventario inicial" in section
    lines = {line.split("|")[1].strip(): line for line in section.splitlines() if line.startswith("| ")}
    for row in rows:
        for name in ("historical", "model"):
            assert f"| {row['por_politica'][name]['riesgo_simulado_evitado_pct']:.1f} % |" in lines[row["inventario_inicial"]]


def test_el_porcentaje_evitado_a_mano():
    # 25 horas evitadas de 100 sin intervención: 25 %. Más riesgo que sin intervenir: negativo.
    assert avoided_percent(25.0, 100.0) == 25.0 and avoided_percent(-10.0, 40.0) == -25.0
    # Sin riesgo sin intervención no hay contra qué dividir.
    assert avoided_percent(0.0, 0.0) is None


# --------------------------------------------------------------------------- #
# k2) Piso de azar, tabla de viajes bloqueados y anexo del reporte
# --------------------------------------------------------------------------- #

def test_el_piso_de_azar_se_corre_con_el_presupuesto_de_referencia_y_no_entra_a_las_tablas(case, result_lgbm):
    report = result_lgbm.report
    floor = report["piso_de_azar"]
    assert (floor["corridas"], floor["semilla"], floor["presupuesto_del_dia_en_bicis"]) == (3, 7, 6)
    runs, calendar = result_lgbm.random_runs, result_lgbm.calendar
    assert len(runs) == 3 and {run.policy for run in runs} == {"random"} and {run.daily_budget for run in runs} == {6}
    start = report["inventario_inicial"]["bicis"]
    none = result_lgbm.runs[("none", 0)]
    for run, row in zip(runs, floor["cada_corrida"]):
        # Mismo inventario inicial que las demás, total de bicis constante y presupuesto dentro de lo disponible.
        assert run.initial_total == run.final_total == start
        assert (run.budget_used <= run.budget_available).all() and int(run.budget_used.sum()) > 0
        assert row["riesgo_por_dia"] == pytest.approx(daily_risk(run, calendar).mean())
        assert row["riesgo_simulado_evitado_por_dia"] == pytest.approx(
            (daily_risk(none, calendar) - daily_risk(run, calendar)).mean()
        )
        assert sum(row["riesgo_por_franja"].values()) == pytest.approx(row["riesgo_por_dia"])
        assert row["viajes_bloqueados_por_dia"] == pytest.approx(run.blocked[calendar.evaluated].mean())
        assert row["llegadas_desviadas_por_dia"] == pytest.approx(run.diverted[calendar.evaluated].mean())
    # Media y rango entre corridas.
    for key in ("riesgo_por_dia", "riesgo_simulado_evitado_por_dia", "viajes_bloqueados_por_dia",
                "llegadas_desviadas_por_dia", "bicis_movidas_por_dia"):
        values = [row[key] for row in floor["cada_corrida"]]
        assert floor[key] == {"media": pytest.approx(np.mean(values)), "minimo": min(values), "maximo": max(values)}
    assert len({row["riesgo_por_dia"] for row in floor["cada_corrida"]}) > 1   # cada corrida tiene su sorteo
    base = row_of(report, "none", 0)["riesgo_por_dia"]["media"]
    assert floor["riesgo_simulado_evitado_pct"] == pytest.approx(
        floor["riesgo_simulado_evitado_por_dia"]["media"] / base * 100
    )
    assert set(floor["riesgo_por_franja"]) == set(LABELS)

    # No es una política del contrato: no está en las tablas ni entre los resultados de la curva.
    assert "random" not in POLICIES
    assert set(result_lgbm.policy_curve["policy"]) == set(POLICIES)
    assert set(result_lgbm.recommendations["policy"]) == set(POLICIES)
    assert {row["politica"] for row in report["resultados"]} == set(POLICIES)
    assert {policy for policy, _ in result_lgbm.runs} == set(POLICIES)
    assert report["pronostico"]["politicas_simuladas"] == list(POLICIES)
    assert any(RANDOM_LABEL in check for check in report["comprobaciones"])

    # Con la misma semilla sale lo mismo; con otra, otro sorteo.
    again = simulate_case(case, lgbm=True)
    assert again.report["piso_de_azar"] == floor
    assert [run.digest() for run in again.random_runs] == [run.digest() for run in runs]
    other = simulate_case(case, lgbm=True, ajustes=settings(semilla=8))
    assert [run.digest() for run in other.random_runs] != [run.digest() for run in runs]


def test_el_reporte_muestra_el_piso_de_azar_junto_a_las_politicas(result_lgbm):
    report = result_lgbm.report
    floor = report["piso_de_azar"]
    text = build_report(report)
    section = text.split("## Piso de azar: mover bicis sin información")[1].split("## Riesgo día por día")[0]
    assert "**no una política del contrato**" in section and "`recommendations.parquet`" in section
    assert "no es un intervalo de confianza" in section and "son simulados" in section
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert [line.split("|")[1].strip() for line in rows] == [
        "sin intervención", f"{RANDOM_LABEL} (media de 3 corridas)", "regla histórica", "modelo",
    ]
    cells = [cell.strip() for cell in rows[1].strip("|").split("|")]
    assert cells[1] == f"{floor['riesgo_por_dia']['media']:,.1f}"
    assert cells[2] == f"{floor['riesgo_por_dia']['minimo']:,.1f} a {floor['riesgo_por_dia']['maximo']:,.1f}"
    assert cells[3] == f"{floor['riesgo_simulado_evitado_por_dia']['media']:,.1f}"
    assert cells[4] == f"{floor['riesgo_simulado_evitado_pct']:.1f} %"
    model = row_of(report, "model", 6)
    assert f"modelo, {floor['riesgo_por_dia']['media'] - model['riesgo_por_dia']['media']:,.1f}" in section
    assert any(line.startswith("Piso de azar") for line in summary_lines(report))
    # En ninguna otra tabla de políticas aparece como una política más.
    curve = text.split("## Horas de estación en riesgo por día")[1].split("Lectura de la tabla")[0]
    assert RANDOM_LABEL not in curve
    assert RANDOM_LABEL not in text.split("## Recomendaciones")[1].split("## Comprobaciones de la corrida")[0]


def test_con_cero_corridas_no_hay_piso_de_azar(case, result):
    without = simulate_case(case, ajustes=settings(corridas_piso_de_azar=0))
    assert without.report["piso_de_azar"] == {"corridas": 0, "semilla": 7, "presupuesto_del_dia_en_bicis": 6}
    assert without.random_runs == ()
    text = build_report(without.report) + "\n".join(summary_lines(without.report))
    assert "## Piso de azar" not in text and f"{RANDOM_LABEL} (media" not in text
    assert "Piso de azar (mover bicis al azar" not in text
    assert without.report["corridas_simuladas"] == result.report["corridas_simuladas"] - 3
    # Lo demás no cambia: el piso de azar no toca ni la banda, ni la curva, ni las recomendaciones.
    pd.testing.assert_frame_equal(without.policy_curve, result.policy_curve)
    pd.testing.assert_frame_equal(without.recommendations, result.recommendations)
    assert without.report["banda"] == result.report["banda"]


def test_el_reporte_trae_la_tabla_de_viajes_bloqueados_y_llegadas_desviadas_por_politica(result_lgbm):
    report = result_lgbm.report
    text = build_report(report)
    section = text.split("## Viajes bloqueados y llegadas desviadas por política (simulados)")[1].split(
        "## Presupuesto disponible y usado por franja"
    )[0]
    assert "**Son resultados de la simulación, no mediciones**" in section
    assert "demanda atendida" in section and "posible demanda insatisfecha" in section
    rows = [[cell.strip() for cell in line.strip("|").split("|")]
            for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    assert rows[0] == ["política", "bicis por estación", "bicis al día", "viajes bloqueados por día",
                       "diferencia contra sin intervención", "llegadas desviadas por día",
                       "diferencia contra sin intervención"]
    body = rows[1:]
    # Sin intervención una vez, cada política con cada presupuesto y, al final, el piso de azar.
    assert len(body) == 1 + 4 * 2 + 1
    none = row_of(report, "none", 0)
    blocked, diverted = none["viajes_bloqueados"]["por_dia"], none["llegadas_desviadas"]["por_dia"]
    assert body[0] == ["sin intervención", "cualquiera", "cualquiera", f"{blocked:,.1f}", "+0.0",
                       f"{diverted:,.1f}", "+0.0"]
    labels = {"historical": "regla histórica", "model": "modelo"}
    expected = []
    for name in ("historical", "model"):
        for per_station, budget in ((0, 0), (0.5, 3), (1.0, 6), (2.0, 12)):
            row = row_of(report, name, budget)
            mine, theirs = row["viajes_bloqueados"]["por_dia"], row["llegadas_desviadas"]["por_dia"]
            # Lo que dice la tabla es lo que pasó en la corrida: salidas bloqueadas y llegadas desviadas por día.
            run = result_lgbm.runs[(name, budget)]
            evaluated = result_lgbm.calendar.evaluated
            assert mine == run.blocked[evaluated].mean() and theirs == run.diverted[evaluated].mean()
            expected.append([labels[name], f"{per_station:g}", f"{budget:,}", f"{mine:,.1f}", f"{mine - blocked:+,.1f}",
                             f"{theirs:,.1f}", f"{theirs - diverted:+,.1f}"])
    assert body[1:-1] == expected
    floor = report["piso_de_azar"]
    mine, theirs = floor["viajes_bloqueados_por_dia"]["media"], floor["llegadas_desviadas_por_dia"]["media"]
    assert body[-1] == [f"{RANDOM_LABEL} (media de 3 corridas)", "1", "6", f"{mine:,.1f}", f"{mine - blocked:+,.1f}",
                        f"{theirs:,.1f}", f"{theirs - diverted:+,.1f}"]
    # Con presupuesto 0 la diferencia es cero: es la corrida sin intervención.
    assert body[1][3:] == body[0][3:]


def test_el_reporte_termina_con_el_registro_del_experimento_tal_cual(result, tmp_path):
    report = result.report
    plain = build_report(report, anexo=None)
    assert plain.endswith("\n") and "Experimento descartado" not in plain
    # Un archivo cualquiera: va al final, sin cambiarle ni un espacio.
    record = tmp_path / "registro.md"
    content = "## Registro de prueba\n\n| a | b |\n|---|---:|\n| x | 1,234.5 |\n\n  Línea con  espacios raros.  \n"
    record.write_text(content, encoding="utf-8")
    assert build_report(report, anexo=record) == plain + "\n" + content
    # Si el archivo no existe, el reporte sale sin anexo.
    assert build_report(report, anexo=tmp_path / "no_existe.md") == plain
    # Por defecto, el registro del experimento de mediodía (docs/experimento_mediodia.md), si está.
    assert EXPERIMENT_FILE.parts[-2:] == ("docs", "experimento_mediodia.md")
    if EXPERIMENT_FILE.exists():
        assert build_report(report) == plain + "\n" + EXPERIMENT_FILE.read_text(encoding="utf-8")
    else:
        assert build_report(report) == plain


# --------------------------------------------------------------------------- #
# l) El script, de punta a punta
# --------------------------------------------------------------------------- #

def write_frozen(path: Path, table: pd.DataFrame) -> None:
    meta = {
        "captura_desde_utc": "2026-10-10T04:05:04Z", "captura_hasta_utc": "2026-10-10T17:11:35Z",
        "fotos_de_estado": 155, "estados_de_informacion": 13, "hora_local_del_inventario": 0, "fotos_en_esa_hora": 7,
    }
    path.write_text(render(Frozen(table=table, meta=meta)), encoding="utf-8")


def write_inputs(folder, case: dict, lgbm: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_contract(case["stations"], "stations", folder / "stations.parquet")
    predictions = case["predictions_with_lgbm"] if lgbm else case["predictions"]
    write_contract(predictions, "predictions", folder / "predictions.parquet")
    for name in ("trips", "station_hour"):
        pq.write_table(pa.Table.from_pandas(case[name], preserve_index=False), folder / f"{name}.parquet")


@pytest.fixture()
def project(tmp_path, case):
    """Carpeta con las cuatro tablas de entrada, el archivo congelado de GBFS y los dos YAML."""
    write_inputs(tmp_path / "datos", case)
    write_frozen(tmp_path / "gbfs.csv", case["frozen"])
    (tmp_path / "operacion.yaml").write_text(yaml.safe_dump(make_cfg(), sort_keys=False), encoding="utf-8")
    (tmp_path / "datos.yaml").write_text(DATOS_TEXT, encoding="utf-8")
    return tmp_path


def files_of(project) -> dict:
    return {"operacion": project / "operacion.yaml", "config": project / "datos.yaml", "gbfs": project / "gbfs.csv"}


def run_script(project, out: str = "salida", *extra: str) -> int:
    return main(
        ["--datos", str(project / "datos"), "--salida", str(project / out), "--gbfs", str(project / "gbfs.csv"),
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
    assert set(document["corrida"]) == {"generado_utc", "segundos", "datos", "operacion", "config", "gbfs", "huella"}
    corrida = document.pop("corrida")
    # El script agrega de dónde salió el inventario inicial; lo demás es lo que da la función.
    start = document["inventario_inicial"]
    assert Path(start.pop("archivo")) == (project / "gbfs.csv").resolve()
    assert start.pop("captura")["captura_desde_utc"] == "2026-10-10T04:05:04Z"
    assert document == json.loads(json.dumps(result.report))
    assert corrida["huella"][BAND_KEY] == document["banda"]["elegida"]
    # El reporte y la pantalla nombran el archivo de reglas con el que se corrió, no el del proyecto.
    assert Path(corrida["operacion"]) == (project / "operacion.yaml").resolve()
    text = (out / "reports" / "simulacion.md").read_text(encoding="utf-8")
    assert f"(`{corrida['operacion']}`, `estado: provisional`)" in text and "config/operacion.yaml" not in text
    assert "Captura usada: del 2026-10-10T04:05:04Z al 2026-10-10T17:11:35Z (UTC), 155 fotos" in text
    if EXPERIMENT_FILE.exists():   # el reporte escrito termina con el registro del experimento, tal cual
        assert text.endswith("\n\n" + EXPERIMENT_FILE.read_text(encoding="utf-8"))
    assert "Piso de azar (mover bicis al azar, 6 bicis al día, 3 corridas" in printed.out
    assert f"Reglas de operación PROVISIONALES ({corrida['operacion']})." in printed.out
    assert "config/operacion.yaml" not in printed.out
    # El resumen en pantalla.
    assert "Tramo de validación: 2026-09-01 a 2026-09-04." in printed.out
    assert "Estaciones simuladas: 6 de 8." in printed.out
    assert 'La política "model" NO se simuló' in printed.out and 'La política "model" NO se simuló' in printed.err
    assert "(elegida)" in printed.out and "| modelo |" not in printed.out
    assert "El total de bicis (en estaciones más en tránsito) no cambió en ninguna corrida." in printed.out
    assert "La prueba final no se simuló (decisión 4)." in printed.out
    assert "policy_curve.parquet" in printed.out and "simulacion.md" in printed.out


def test_el_script_dos_veces_da_archivos_identicos(project):
    assert run_script(project, "uno") == 0 and run_script(project, "dos") == 0
    for name in ("recommendations.parquet", "policy_curve.parquet", "reports/simulacion.md"):
        assert (project / "uno" / name).read_bytes() == (project / "dos" / name).read_bytes(), name
    one, two = (json.loads((project / out / "reports" / "simulacion.json").read_text(encoding="utf-8")) for out in ("uno", "dos"))
    for document in (one, two):
        del document["corrida"]["generado_utc"], document["corrida"]["segundos"]
    assert one == two


def test_los_lectores_entregan_solo_lo_que_toca(project, case):
    data = project / "datos"
    trips = load_trips(data / "trips.parquet", VALIDATION)
    assert len(trips) == len(in_tramo(case["trips"], VALIDATION))
    assert (VALIDATION.contains(trips["started_at"]) | VALIDATION.contains(trips["ended_at"])).all()
    assert {"start_lat", "start_lng", "end_lat", "end_lng"} <= set(trips.columns)
    history = load_history(data / "station_hour.parquet", TRAINING, VALIDATION)
    assert list(history.columns) == ["station_id", "ts_hour"]
    assert history["ts_hour"].min() == TRAINING.start and history["ts_hour"].max() < VALIDATION.start
    forecasts = load_forecasts(data / "predictions.parquet", VALIDATION, "val")
    assert set(forecasts) == {"historical"}
    rows = forecasts["historical"]
    assert set(rows["split"]) == {"val"} and set(rows["model"]) == {"baseline"}
    assert rows["ts_hour"].min() == VALIDATION.start and rows["ts_hour"].max() < VALIDATION.end
    # Una tabla de viajes sin las coordenadas es de antes de este simulador: hay que reconstruirla.
    old = case["trips"].drop(columns=["end_lat", "end_lng"])
    pq.write_table(pa.Table.from_pandas(old, preserve_index=False), data / "trips.parquet")
    with pytest.raises(SimulacionError, match=r"end_lat.*build_tables.py"):
        load_trips(data / "trips.parquet", VALIDATION)


def test_model_se_alimenta_solo_de_lgbm_y_si_no_hay_no_se_simula(project, case, result_lgbm, capsys):
    data = project / "datos"
    assert set(load_forecasts(data / "predictions.parquet", VALIDATION, "val")) == {"historical"}
    write_inputs(data, case, lgbm=True)
    forecasts = load_forecasts(data / "predictions.parquet", VALIDATION, "val")
    assert set(forecasts) == {"historical", "model"}
    assert set(forecasts["historical"]["model"]) == {"baseline"} and set(forecasts["model"]["model"]) == {"lgbm"}
    capsys.readouterr()
    assert run_script(project, "con_lgbm") == 0
    printed = capsys.readouterr()
    assert "NO se simuló" not in printed.out + printed.err and 'Modelo: filas "lgbm".' in printed.err
    curve = read_contract("policy_curve", project / "con_lgbm" / "policy_curve.parquet")
    pd.testing.assert_frame_equal(curve, result_lgbm.policy_curve, check_dtype=False)
    assert set(curve["policy"]) == set(POLICIES)
    # Sin filas de la línea base no hay regla histórica: el script lo dice.
    only = case["predictions_with_lgbm"]
    write_contract(only[only["model"] == "lgbm"], "predictions", data / "predictions.parquet")
    assert run_script(project, "sin_base") == 1
    assert 'no trae filas "baseline"' in capsys.readouterr().out and not (project / "sin_base").exists()


def test_el_script_rechaza_la_prueba_final_y_no_escribe_nada(project, capsys):
    assert run_script(project, "salida", "--tramo", "prueba") == 1
    assert "La prueba final no se simula aquí" in capsys.readouterr().out
    assert not (project / "salida").exists()


def test_el_script_no_acepta_otros_tramos(project):
    with pytest.raises(SystemExit):
        run_script(project, "salida", "--tramo", "entrenamiento")


def test_con_usar_prueba_final_escribe_aparte_y_lo_dice(project, capsys):
    """Datos sintéticos de esta prueba: no es la prueba final del proyecto."""
    assert run_script(project, "salida", "--tramo", "prueba", "--usar-prueba-final") == 0
    out = project / "salida" / FINAL_FOLDER
    assert (out / "recommendations.parquet").exists() and not (project / "salida" / "recommendations.parquet").exists()
    text = (out / "reports" / "simulacion.md").read_text(encoding="utf-8")
    assert "tramo de prueba final" in text and "Esta corrida usó la prueba final." in text
    assert "La prueba final no se simuló" not in capsys.readouterr().out


def test_con_banda_la_banda_no_se_elige_y_solo_se_simula_esa(project, capsys):
    """`--banda` fija el multiplicador (para la prueba final: el elegido en validación). Sin ella, se elige."""
    assert run_script(project, "fija", "--banda", "1") == 0
    band = json.loads((project / "fija" / "reports" / "simulacion.json").read_text(encoding="utf-8"))["banda"]
    assert band["elegida"] == 1.0 and band["candidatas"] == [1.0] and band["fijada"] is True
    assert [row["banda"] for row in band["resultados"]] == [1.0]
    text = (project / "fija" / "reports" / "simulacion.md").read_text(encoding="utf-8")
    assert "La banda llegó fijada de antemano en 1" in text and "Se eligió con estos mismos días" not in text
    # Sin la opción todo sigue igual: se simulan las tres candidatas y la clave "fijada" no aparece.
    assert run_script(project, "libre") == 0
    free = json.loads((project / "libre" / "reports" / "simulacion.json").read_text(encoding="utf-8"))["banda"]
    assert free["candidatas"] == [0.0, 0.5, 1.0] and "fijada" not in free
    # Una banda que no es candidata se rechaza y no escribe nada.
    capsys.readouterr()
    assert run_script(project, "mala", "--banda", "0.3") == 1
    assert "no es una de las candidatas" in capsys.readouterr().out and not (project / "mala").exists()


@pytest.mark.parametrize("missing", ["trips", "stations", "station_hour", "predictions"])
def test_el_script_dice_que_archivo_falta(project, missing, capsys):
    (project / "datos" / f"{missing}.parquet").unlink()
    assert run_script(project) == 1
    printed = capsys.readouterr().out
    step = "run_baseline.py" if missing == "predictions" else "build_tables.py"
    assert f"{missing}.parquet" in printed and f"python scripts/{step}" in printed
    assert not (project / "salida").exists()


def test_el_script_dice_si_falta_el_archivo_congelado_de_gbfs_o_no_se_puede_leer(project, capsys):
    (project / "gbfs.csv").unlink()
    assert run_script(project) == 1
    assert "gbfs.csv" in (printed := capsys.readouterr().out) and "python scripts/freeze_gbfs.py" in printed
    (project / "gbfs.csv").write_text("short_name,capacity\nA,8\n", encoding="utf-8")
    assert run_script(project) == 1
    assert "le faltan columnas" in capsys.readouterr().out and not (project / "salida").exists()


def test_el_script_dice_si_no_puede_abrir_las_reglas(project, capsys):
    (project / "operacion.yaml").unlink()
    assert run_script(project) == 1
    assert "No se pudo abrir" in capsys.readouterr().out


def test_el_script_no_acepta_unas_reglas_con_tipo_si_falta(project, capsys):
    cfg = make_cfg()
    cfg["tipo_si_falta"] = "mixta"
    (project / "operacion.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert run_script(project) == 1
    assert "tipo_si_falta: esta clave ya no se usa" in capsys.readouterr().out


def test_el_script_no_corre_si_las_tablas_se_construyeron_con_otro_periodo(project, capsys):
    reports = project / "datos" / "reports"
    reports.mkdir()
    built = {"config": {"periodo": {"inicio": "2026-07-01", "fin": "2026-09-06"}}}
    (reports / "limpieza.json").write_text(json.dumps(built), encoding="utf-8")
    assert run_script(project) == 1
    assert "se construyeron con el periodo 2026-07-01 a 2026-09-06" in capsys.readouterr().out


def test_el_script_no_corre_sin_el_bloque_simulacion(project, capsys):
    text = DATOS_TEXT[: DATOS_TEXT.index("simulacion:")]
    (project / "datos.yaml").write_text(text, encoding="utf-8")
    assert run_script(project) == 1
    assert "no trae el bloque `simulacion`" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# m) Lenguaje honesto y limitaciones
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fixture", ["result", "result_lgbm"])
def test_el_reporte_usa_el_lenguaje_honesto_y_declara_las_limitaciones(fixture, request):
    report = request.getfixturevalue(fixture).report
    text = build_report(report) + "\n".join(summary_lines(report))
    lowered = text.lower()
    assert "riesgo simulado evitado" in lowered and "posible demanda insatisfecha" in lowered
    for forbidden in ("viajes recuperados", "viajes salvados", "demanda recuperada", "viajes perdidos"):
        assert forbidden not in lowered, forbidden
    assert "**Las reglas de operación son provisionales**" in text
    limits = "\n".join(report["limitaciones"])
    for expected in (
        "línea recta", "demanda atendida", "Los días no son independientes", "octubre de 2026",
        "El total de bicis no cambia durante la corrida", "**Es una interpretación de programación**",
        '"riesgo evitado por bici", **no se implementó**', "La banda de incertidumbre se eligió con los mismos días",
        "La prueba final no se simuló (decisión 4).", "promedio histórico",
    ):
        assert expected in limits, expected
    rules = text.split("## Reglas usadas")[1].split("## Horas de estación en riesgo por día")[0]
    assert "primero la estación a la que le falta o le sobra MENOS" in rules
    assert '**No se implementó "riesgo evitado por bici"**' in rules
    assert "una estación sin tipo no se toca en esa franja" in rules and "tipo_si_falta" not in text
    assert "la misma maquinaria" in text


def test_el_reporte_trae_una_fila_por_politica_y_presupuesto_y_none_una_sola_vez(result_lgbm):
    text = build_report(result_lgbm.report)
    section = text.split("## Horas de estación en riesgo por día")[1].split("Lectura de la tabla")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) == 1 + 4 + 4
    assert sum(line.startswith("| sin intervención |") for line in rows) == 1
    assert sum(line.startswith("| modelo |") for line in rows) == 4
    reading = text.split("Lectura de la tabla")[1].split("## Riesgo día por día")[0]
    for name in ("Sin intervención", "Regla histórica", "Modelo"):
        assert f"- {name}: " in reading
    row = row_of(result_lgbm.report, "model", 6)
    assert f"{row['riesgo_simulado_evitado_por_dia']['media']:,.1f} horas por día" in reading


# --------------------------------------------------------------------------- #
# n) La huella de las entradas
# --------------------------------------------------------------------------- #

def check_outputs_are_current(out_dir, data_dir=None, operacion=OPERACION_YAML, config=DATOS_YAML, gbfs=FROZEN_FILE) -> None:
    """Falla, con el comando que hay que correr, si las salidas de la simulación en `out_dir` ya no corresponden.

    Compara la huella que la corrida dejó en simulacion.json con la de las tablas, las reglas, el archivo
    congelado de GBFS y el config de hoy, y revisa que recommendations.parquet y simulacion.md sean los de
    ese simulacion.json. Lee lo mismo que lee la simulación: el tramo simulado y la historia anterior. Nada
    de la prueba final (decisión 4).
    """
    reports = out_dir / "reports"
    document = json.loads((reports / "simulacion.json").read_text(encoding="utf-8"))
    corrida = document.pop("corrida")
    current = current_fingerprint(data_dir or out_dir, operacion, config, document["tramo"]["nombre"], gbfs)
    changed = stale_inputs(corrida.get("huella"), current)
    assert not changed, stale_message(changed)

    other = f"recommendations.parquet y simulacion.* no son de la misma corrida o son de otro simulador. {RERUN}"
    assert corrida["huella"][BAND_KEY] == document["banda"]["elegida"], other
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
    out, data, files = project / "salida", project / "datos", files_of(project)
    check_outputs_are_current(out, data, **files)

    # stations: otra coordenada.
    stations = case["stations"].copy()
    stations.loc[0, "lat"] += 0.001
    rewrite(data, "stations", stations)
    with stale("stations.parquet"):
        check_outputs_are_current(out, data, **files)
    rewrite(data, "stations", case["stations"])

    # trips: un viaje del tramo cambia de destino; otro, solo de punto de fin.
    inside = np.nonzero(VALIDATION.contains(case["trips"]["started_at"]).to_numpy()
                        & VALIDATION.contains(case["trips"]["ended_at"]).to_numpy())[0]
    for column, value in (("end_station_id", "A"), ("end_lat", 40.7055)):
        trips = case["trips"].copy()
        row = next(i for i in inside if trips.loc[i, column] != value)
        trips.loc[row, column] = value
        rewrite(data, "trips", trips)
        with stale("trips.parquet (los viajes del tramo simulado)"):
            check_outputs_are_current(out, data, **files)
    rewrite(data, "trips", case["trips"])

    # station_hour: una estación pierde su historia. Cambiar solo los conteos no importa: ya no se usan.
    hours = case["station_hour"]
    rewrite(data, "station_hour", hours[(hours["station_id"] != "B") | (hours["ts_hour"] >= VALIDATION.start)])
    with stale("station_hour.parquet (la historia anterior al tramo simulado)"):
        check_outputs_are_current(out, data, **files)
    rewrite(data, "station_hour", hours.assign(departures=hours["departures"] + 1, net_flow=hours["net_flow"] - 1,
                                               departures_member=hours["departures_member"] + 1))
    check_outputs_are_current(out, data, **files)
    rewrite(data, "station_hour", hours)

    # predictions: cambia un cuantil del tramo; y aparecen filas "lgbm" (ahora "model" sí se simularía).
    predictions = case["predictions"].copy()
    predictions.loc[predictions["split"] == "val", "dep_q90"] += 0.5
    rewrite(data, "predictions", predictions)
    with stale("predictions.parquet (el pronóstico del tramo simulado)"):
        check_outputs_are_current(out, data, **files)
    rewrite(data, "predictions", case["predictions_with_lgbm"])
    with stale("predictions.parquet (el pronóstico del tramo simulado)"):
        check_outputs_are_current(out, data, **files)
    rewrite(data, "predictions", case["predictions"])

    # El archivo congelado de GBFS: otra ocupación inicial.
    write_frozen(files["gbfs"], frozen_table(case["stations"], {**OCCUPANCY, "A": 0.3}))
    with stale("config/gbfs_congelado.csv (capacidad e inventario inicial congelados de GBFS)"):
        check_outputs_are_current(out, data, **files)
    write_frozen(files["gbfs"], case["frozen"])

    # Las reglas.
    files["operacion"].write_text(yaml.safe_dump(make_cfg(distancia_maxima_km=2.0), sort_keys=False), encoding="utf-8")
    with stale("las reglas de config/operacion.yaml"):
        check_outputs_are_current(out, data, **files)
    files["operacion"].write_text(yaml.safe_dump(make_cfg(), sort_keys=False), encoding="utf-8")

    # El bloque `simulacion`: otra semilla, otra franja de noche, y unas bandas que ya no traen la elegida.
    chosen = json.loads((out / "reports" / "simulacion.json").read_text(encoding="utf-8"))["banda"]["elegida"]
    for old, new in (("semilla: 7", "semilla: 8"), ('inicio: "22:00"', 'inicio: "23:00"'),
                     ("sensibilidad_ocupacion_inicial: [0.4, 0.6]", "sensibilidad_ocupacion_inicial: [0.5]"),
                     ("corridas_piso_de_azar: 3", "corridas_piso_de_azar: 4")):
        files["config"].write_text(DATOS_TEXT.replace(old, new), encoding="utf-8")
        with stale("el bloque `simulacion` de config/datos.yaml"):
            check_outputs_are_current(out, data, **files)
    others = [band for band in (0, 0.5, 1) if band != chosen]
    files["config"].write_text(DATOS_TEXT.replace("bandas: [0, 0.5, 1]", f"bandas: {others}"), encoding="utf-8")
    with stale("el bloque `simulacion` de config/datos.yaml",
               "la banda de incertidumbre elegida (no está en la huella o ya no es una candidata)"):
        check_outputs_are_current(out, data, **files)

    # Las fechas.
    files["config"].write_text(DATOS_TEXT.replace("fin: 2026-08-31}\n  validacion:    {inicio: 2026-09-01",
                                                  "fin: 2026-08-30}\n  validacion:    {inicio: 2026-08-31"), encoding="utf-8")
    with pytest.raises(AssertionError, match=re.escape("las fechas de los tramos en config/datos.yaml")):
        check_outputs_are_current(out, data, **files)
    files["config"].write_text(DATOS_TEXT, encoding="utf-8")
    check_outputs_are_current(out, data, **files)


def test_la_huella_guarda_la_banda_elegida(project):
    assert run_script(project) == 0
    path = project / "salida" / "reports" / "simulacion.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = document["corrida"]["huella"]
    assert fingerprint[BAND_KEY] == document["banda"]["elegida"] and fingerprint[BAND_KEY] in (0.0, 0.5, 1.0)
    assert fingerprint["simulacion"]["bandas"] == [0.0, 0.5, 1.0]
    assert fingerprint["simulacion"]["franja_noche"] == ["22:00", "06:00"]
    assert set(fingerprint["predictions"]) == {"baseline"} and "gbfs_congelado" in fingerprint
    current = current_fingerprint(project / "datos", **files_of(project))
    assert BAND_KEY not in current and stale_inputs(fingerprint, current) == []
    # Una huella sin la banda (de una versión anterior) se da por vieja.
    del fingerprint[BAND_KEY]
    assert stale_inputs(fingerprint, current) == [
        "la banda de incertidumbre elegida (no está en la huella o ya no es una candidata)"
    ]


def test_la_huella_no_trae_nada_de_la_prueba_final(project, case):
    """Datos sintéticos de esta prueba: no es la prueba final del proyecto."""
    data, files = project / "datos", files_of(project)
    # Un viaje que sale el último día de validación y termina ya en las fechas de la prueba final.
    crossing = make_trips([("ZZ9999", "2026-09-04 23:50", "2026-09-05 00:20", "A", "B")])
    crossing[["end_lat", "end_lng"]] = [40.7051, -73.9851]
    trips = pd.concat([case["trips"], crossing], ignore_index=True)
    rewrite(data, "trips", trips)
    assert run_script(project, "antes") == 0
    check_outputs_are_current(project / "antes", data, **files)
    before = current_fingerprint(data, **files)

    # Cambia todo lo que tiene fecha de la prueba final: los viajes que empiezan ahí, la hora, la estación y
    # el punto de fin del viaje que cruza, el pronóstico del split "test" y las filas de station_hour de esas
    # fechas. También los viajes que salen en validación y terminan después: de su llegada no se usa nada.
    changed = trips.copy()
    changed.loc[changed["started_at"] >= FINAL.start, "end_station_id"] = "A"
    late = changed["ended_at"] >= FINAL.start
    changed.loc[late, "end_station_id"] = None
    changed.loc[late, ["end_lat", "end_lng"]] = [40.7001, -73.9801]
    changed.loc[changed["ride_id"] == "ZZ9999", "ended_at"] = ts("2026-09-05 03:00")
    rewrite(data, "trips", changed)
    predictions = case["predictions"].copy()
    predictions.loc[predictions["split"] == "test", list(QUANTILE_COLUMNS)] = 99.0
    rewrite(data, "predictions", predictions)
    hours = case["station_hour"]
    rewrite(data, "station_hour", hours[(hours["ts_hour"] < FINAL.start) | (hours["station_id"] == "A")])

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
    out, data, files = project / "salida", project / "datos", files_of(project)
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
    if "banda" not in document:
        pytest.fail(f"simulacion.json es de un simulador anterior. {RERUN}", pytrace=False)
    # Decisión 4: lo que hay en data/processed es de validación.
    assert document["tramo"]["nombre"] == "validacion" and document["prueba_final"]["simulada"] is False
    simulated = document["pronostico"]["politicas_simuladas"]
    curve = read_contract("policy_curve", paths[1]).set_index(["policy", "budget"])
    assert len(curve) == len(document["resultados"])
    assert set(curve.index.get_level_values("policy")) == set(simulated)
    for row in document["resultados"]:
        line = curve.loc[(row["politica"], row["presupuesto_del_dia_en_bicis"])]
        assert line["risk_station_hours"] == row["riesgo_por_dia"]["media"]
        assert (line["ci_low"], line["ci_high"]) == (row["riesgo_por_dia"]["ic_bajo"], row["riesgo_por_dia"]["ic_alto"])
        # El total de bicis no cambia en ninguna corrida.
        bikes = row["bicis"]
        assert bikes["total_al_inicio"] == bikes["total_al_final"] == document["inventario_inicial"]["bicis"]
        assert bikes["en_estaciones_al_final"] + bikes["en_transito_al_final"] == bikes["total_al_inicio"]
    none = curve.loc["none"]
    assert none["risk_station_hours"].nunique() == 1
    if 0 in none.index:
        for policy in simulated:
            assert curve.loc[(policy, 0)].equals(curve.loc[("none", 0)])
    # "model" está si y solo si predictions trae filas "lgbm" del tramo.
    assert ("model" in simulated) == document["pronostico"]["model_simulada"]
    assert document["banda"]["elegida"] in document["banda"]["candidatas"]
    recommendations = read_contract("recommendations", paths[0])
    assert set(recommendations["policy"]) == set(simulated)
    evaluated = set(document["tramo"]["dias_evaluados"])
    assert set(recommendations["window_start"].dt.strftime("%Y-%m-%d")) == evaluated
    assert not set(document["tramo"]["dias_de_calentamiento"]) & evaluated
    assert recommendations["window_end"].max() < pd.Timestamp(document["prueba_final"]["inicio"], tz=TZ)
    assert recommendations["station_id"].nunique() == document["estaciones"]["simuladas"]
    assert len(recommendations) == document["recomendaciones"]["filas"]


def test_las_salidas_reales_siguen_al_dia():
    """Las salidas reales de la simulación corresponden a las tablas, las reglas y el config de hoy.

    Lee lo mismo que lee la simulación: el tramo de validación y la historia anterior. Nada de la prueba final.
    """
    reports = DATA_PROCESSED / "reports"
    outputs = [DATA_PROCESSED / "recommendations.parquet", reports / "simulacion.json", reports / "simulacion.md"]
    if not all(path.exists() for path in outputs):
        pytest.skip("Falta correr la simulación: python scripts/run_simulation.py")
    inputs = [DATA_PROCESSED / f"{name}.parquet" for name in ("stations", "trips", "station_hour", "predictions")]
    if not all(path.exists() for path in inputs) or not FROZEN_FILE.exists():
        pytest.skip("Faltan tablas de entrada en data/processed o config/gbfs_congelado.csv: no hay contra qué comparar.")
    check_outputs_are_current(DATA_PROCESSED)


# --------------------------------------------------------------------------- #
# Tipo de estación, estaciones con valet y grupos aislados
# --------------------------------------------------------------------------- #

def test_el_riesgo_por_tipo_de_estacion_suma_el_total_y_el_reporte_lo_trae():
    case = make_case()
    case["stations"] = case["stations"].assign(
        station_type=pd.Series(["oficinas", "oficinas", "residencial", "mixta", None, None, None, None], dtype="object")
    )
    result = simulate_case(case, lgbm=True)
    report = result.report
    assert report["estaciones"]["simuladas_por_tipo"] == {
        "oficinas": 2, "residencial": 1, "recreativa": 0, "mixta": 1, "sin tipo": 2,
    }
    for row in report["resultados"]:
        by_type = row["riesgo_por_tipo_de_estacion"]
        assert list(by_type) == ["oficinas", "residencial", "recreativa", "mixta", "sin tipo"]
        total = sum(cell["riesgo_por_dia"]["media"] for cell in by_type.values())
        assert total == pytest.approx(row["riesgo_por_dia"]["media"])
        avoided = sum(cell["riesgo_simulado_evitado_por_dia"]["media"] for cell in by_type.values())
        assert avoided == pytest.approx(row["riesgo_simulado_evitado_por_dia"]["media"])
        each_day = np.sum([cell["riesgo_de_cada_dia"] for cell in by_type.values()], axis=0)
        assert each_day == pytest.approx(row["riesgo_de_cada_dia"])
    text = build_report(report, anexo=None)
    assert "## Riesgo por tipo de estación" in text and "| sin tipo | 2 |" in text and "| todas | 6 |" in text


def test_valet_y_grupos_aislados_cambian_la_corrida_y_el_reporte_dice_cuanto():
    case = make_case()
    plain = simulate_case(case)
    # Días evaluados: del 2 al 4 de septiembre de 2026 (miércoles a viernes). Valet de A: miércoles y jueves.
    cfg = make_cfg(
        excepciones_valet={"dias": ["miercoles", "jueves"], "estaciones": {"A": [["07:00", "11:00"]], "ZZ": [["07:00", "08:00"]]}},
        grupos_aislados={"isla": ["B", "C", "ZZ"]},
    )
    result = simulate_case(case, cfg=cfg)
    facts = result.report["excepciones"]
    valet = facts["valet"]
    assert valet["estaciones_en_el_archivo"] == ["A", "ZZ"] and valet["estaciones_simuladas"] == ["A"]
    days = [date.fromisoformat(day).weekday() for day in result.report["tramo"]["dias_evaluados"]]
    with_valet = sum(day in (2, 3) for day in days)
    assert valet["dias_evaluados_con_valet"] == with_valet
    assert valet["horas_de_estacion_excluidas"]["total"] == 4.0 * with_valet
    assert valet["estacion_intervencion_dentro_de_un_intervalo"] == 0   # se interviene a las 05:00 y a las 14:00
    group = facts["grupos_aislados"]["isla"]
    assert group["estaciones_simuladas"] == ["B", "C"] and group["vecinas_de_fuera_dentro_de_la_distancia_maxima"] == 4
    # Ningún traslado cruza el grupo, en ninguna corrida.
    inside = {result.network.station_ids.index(name) for name in ("B", "C")}
    moves = [move for run in result.runs.values() for record in run.interventions for move in record.moves]
    assert moves and all((move.origin in inside) == (move.destination in inside) for move in moves)
    crossing = [
        move for run in plain.runs.values() for record in run.interventions for move in record.moves
        if (move.origin in inside) != (move.destination in inside)
    ]
    assert crossing, "sin la regla el caso sí tiene traslados que cruzan: la prueba distingue"
    # Sin intervención lo único que cambia es el tiempo con valet, que ya no cuenta.
    none, none_plain = result.runs[("none", 0)], plain.runs[("none", 0)]
    assert none.risk_us[:, :, 0].sum() + none.valet_risk_us.sum() == none_plain.risk_us[:, :, 0].sum()
    text = build_report(result.report, anexo=None)
    assert "Estaciones con valet (`excepciones_valet`): 2 en el archivo, 1 simuladas" in text
    assert "Grupo aislado `isla`" in text and "no miran el grupo" in text
    rules = dataclasses.asdict(Reglas.from_operacion(cfg))
    assert rules["valet_dias"] == (2, 3) and json.loads(json.dumps(rules))["grupos_aislados"] == [["isla", ["B", "C", "ZZ"]]]
