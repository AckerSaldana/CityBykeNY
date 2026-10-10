"""Simulación de inventario de punta a punta: corridas, métricas, tablas y reporte.

Compara tres políticas (sin intervención, regla histórica y modelo) con varios
presupuestos sobre el tramo de validación (decisiones 7 y 8). Las tres parten
del mismo inventario y enfrentan los mismos viajes. El motor está en
`src/sim/engine.py` y las políticas en `src/sim/policies.py`.

Entradas (carpeta `--datos`): stations.parquet, trips.parquet (tabla interna
que escribe build_tables), station_hour.parquet (solo las filas anteriores al
tramo simulado: la historia de la regla histórica) y predictions.parquet (el
pronóstico del tramo: "lgbm" si existe, si no "baseline").

Salidas (carpeta `--salida`): recommendations.parquet y policy_curve.parquet
(contratos, escritos con `write_contract`), reports/simulacion.json y
reports/simulacion.md.

simulacion.json guarda una huella de lo que se leyó (`corrida.huella`: reglas,
bloque `simulacion`, fechas y un número de control de cada tabla). Con ella
`pytest` avisa, con el comando que hay que correr, cuando las salidas ya no
corresponden a las tablas, las reglas o el config de hoy (`current_fingerprint`
y `stale_inputs`). La huella no guarda nada de la prueba final, salvo sus fechas.

Qué cuenta y qué no:

- Los primeros `dias_de_calentamiento` días se simulan igual que los demás
  (las políticas actúan y gastan presupuesto), pero no entran en ninguna
  métrica ni en las recomendaciones.
- El riesgo es el tiempo exacto que cada estación pasa con menos bicis o menos
  lugares libres que el umbral, sumado sobre las 24 horas.
- El intervalo sale de remuestrear días completos. Los días no son
  independientes: el intervalo es indicativo.

La prueba final no se simula (decisión 4). `--tramo prueba` se rechaza salvo
que se pida con `--usar-prueba-final`, y entonces escribe en la subcarpeta
`prueba_final/` para no pisar los resultados de validación.

Los datos solo registran demanda atendida (decisión 9). Un viaje bloqueado es
un resultado de la simulación: posible demanda insatisfecha, no demanda medida.

Uso:  python scripts/run_simulation.py --help
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.contracts import (
    MODELS,
    POLICIES,
    QUANTILE_COLUMNS,
    RISK_LEVELS,
    TS,
    read_contract,
    validate_file,
    write_contract,
)
from src.data.datos_config import SPLIT_NAMES, TRAMOS, Tramo, load_datos
from src.models.metrics import FinalTestError
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML, ROOT, TZ
from src.sim.engine import (
    EPS,
    KINDS,
    TRIP_COLUMNS,
    Calendar,
    Network,
    Reglas,
    RunResult,
    SimulacionError,
    CheckError,
    InterventionContext,
    build_calendar,
    build_network,
    daily_budget,
    initial_inventory,
    prepare_events,
    simulate,
    split_budget,
    to_timestamp,
)
from src.sim.operacion import load_operacion
from src.sim.policies import (
    ForecastInfo,
    ForecastWindow,
    HistoricalPolicy,
    HistoricalProfile,
    ModelPolicy,
    NoIntervention,
    day_type,
    safe_range,
)

REPORT_NAME = "simulacion"
TRIPS_FILE = "trips.parquet"
FINAL_TRAMO = "prueba"
FINAL_FOLDER = "prueba_final"
POLICY_LABELS = {"none": "sin intervención", "historical": "regla histórica", "model": "modelo"}
TRAMO_LABELS = {"entrenamiento": "entrenamiento", "validacion": "validación", "prueba": "prueba final"}
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
KIND_LABELS = {"pocas_bicis": "pocas bicis", "pocos_lugares": "pocos lugares libres"}
MODEL_PREFERENCE = ("lgbm", "baseline")  # el primero que tenga filas alimenta la política "model"


@dataclass(frozen=True)
class Resultado:
    """Lo que produce una simulación completa."""

    recommendations: pd.DataFrame
    policy_curve: pd.DataFrame
    report: dict
    runs: dict          # (política, presupuesto del día en bicis) -> RunResult
    network: Network
    calendar: Calendar


# --------------------------------------------------------------------------- #
# Candado de la prueba final (decisión 4)
# --------------------------------------------------------------------------- #

def check_final_test(tramo: Tramo, prueba_final: Tramo | None, usar_prueba_final: bool) -> None:
    """Falla si el tramo pedido es la prueba final, o toca sus fechas, y no se pidió de forma explícita."""
    if usar_prueba_final:
        return
    by_name = tramo.nombre == FINAL_TRAMO
    by_date = prueba_final is not None and tramo.fin >= prueba_final.inicio
    if by_name or by_date:
        raise FinalTestError(
            "La prueba final no se simula aquí. Decisión 4: las decisiones se toman con validación y la "
            "prueba final se usa una sola vez. Si de verdad es el momento de usarla, corre: "
            "python scripts/run_simulation.py --tramo prueba --usar-prueba-final"
        )


# --------------------------------------------------------------------------- #
# Conjunto simulado
# --------------------------------------------------------------------------- #

def _epoch_us(column: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(column).as_unit("us").asi8


def select_stations(
    stations: pd.DataFrame, trips: pd.DataFrame, start_us: int, end_us: int
) -> tuple[pd.DataFrame, dict]:
    """Elige las estaciones que se simulan y cuenta las que quedan fuera.

    Se simula una estación con capacidad (no nula y mayor que 0) y con al menos
    una salida o una llegada dentro del tramo. Devuelve esas filas de
    `stations` y el conteo de lo que quedó fuera, por motivo.
    """
    started, ended = _epoch_us(trips["started_at"]), _epoch_us(trips["ended_at"])
    departs = (started >= start_us) & (started < end_us)
    arrives = (ended >= start_us) & (ended < end_us)
    departures = trips.loc[departs, "start_station_id"].value_counts()
    arrivals = trips.loc[arrives, "end_station_id"].value_counts()

    ids = stations["station_id"].astype(str)
    out = departures.reindex(ids).fillna(0).to_numpy(dtype="int64")
    back = arrivals.reindex(ids).fillna(0).to_numpy(dtype="int64")
    capacity = stations["capacity"]
    has_capacity = (capacity.notna() & (capacity.fillna(0) > 0)).to_numpy()
    active = (out + back) > 0
    chosen = has_capacity & active
    known = set(ids)
    unknown_out = departures[[station not in known for station in departures.index]]
    unknown_back = arrivals[[station not in known for station in arrivals.index]]
    report = {
        "en_stations": int(len(stations)),
        "simuladas": int(chosen.sum()),
        "sin_capacidad": {
            "estaciones": int((~has_capacity).sum()),
            "de_ellas_sin_viajes_en_el_tramo": int((~has_capacity & ~active).sum()),
            "salidas_en_el_tramo": int(out[~has_capacity].sum()),
            "llegadas_en_el_tramo": int(back[~has_capacity].sum()),
        },
        "inactivas": {
            "estaciones": int((has_capacity & ~active).sum()),
            "salidas_en_el_tramo": 0,
            "llegadas_en_el_tramo": 0,
        },
        "ids_de_viajes_que_no_estan_en_stations": {
            "ids": int(len(set(unknown_out.index) | set(unknown_back.index))),
            "salidas_en_el_tramo": int(unknown_out.sum()),
            "llegadas_en_el_tramo": int(unknown_back.sum()),
        },
        "extremos_sin_estacion": {
            "salidas_en_el_tramo": int(trips.loc[departs, "start_station_id"].isna().sum()),
            "llegadas_en_el_tramo": int(trips.loc[arrives, "end_station_id"].isna().sum()),
        },
        "salidas_en_el_tramo": int(departs.sum()),
        "llegadas_en_el_tramo": int(arrives.sum()),
    }
    return stations.loc[chosen].copy(), report


# --------------------------------------------------------------------------- #
# Intervalos por remuestreo de días
# --------------------------------------------------------------------------- #

def resample_days(n_days: int, remuestreos: int, semilla: int) -> np.ndarray:
    """Sorteos de días con repetición: una fila por remuestreo, con `n_days` índices.

    Con la misma semilla salen los mismos sorteos. Se usan los mismos para
    todas las políticas y presupuestos, así las diferencias quedan pareadas.
    """
    if n_days < 1:
        raise SimulacionError("No hay días evaluados para remuestrear.")
    return np.random.default_rng(semilla).integers(0, n_days, size=(remuestreos, n_days))


def interval(values, draws: np.ndarray, nivel: float) -> tuple[float, float, float]:
    """Media por día y su intervalo percentil al remuestrear días completos.

    Los días no son independientes (el inventario pasa de un día al siguiente
    y hay patrón semanal), así que el intervalo es indicativo. Si por redondeo
    el intervalo no contuviera a la media, se abre hasta contenerla.
    """
    values = np.asarray(values, dtype="float64")
    mean = float(values.mean())
    means = values[draws].mean(axis=1)
    tail = (1.0 - nivel) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return mean, float(min(low, mean)), float(max(high, mean))


# --------------------------------------------------------------------------- #
# Recomendaciones
# --------------------------------------------------------------------------- #

def risk_levels(before, network: Network, reglas: Reglas, window: ForecastWindow) -> tuple[np.ndarray, np.ndarray]:
    """Nivel de riesgo y `feasible` de cada estación en una franja.

    Usa el pronóstico de la franja y el inventario ANTES de intervenir:
    "alto" si la trayectoria central cruza un umbral o el inventario ya está
    bajo uno; "medio" si solo lo cruza la envolvente; "bajo" en otro caso. Una
    estación sin información es "alto" si ya está bajo un umbral y "bajo" si
    no. `feasible` es falso cuando ningún inventario evita los dos umbrales
    (lo > hi con la envolvente); una estación sin información queda en verdadero.
    """
    before = np.asarray(before, dtype=np.int64)
    low, high = safe_range(network, reglas, window.lower_dip, window.upper_rise)
    low_center, high_center = safe_range(network, reglas, window.center_dip, window.center_rise)
    info = window.has_info
    under = (before < reglas.bicis_minimas) | (network.capacity - before < reglas.lugares_libres_minimos)
    center = info & ((before < low_center - EPS) | (before > high_center + EPS))
    envelope = info & ((before < low - EPS) | (before > high + EPS))
    level = np.where(under | center, "alto", np.where(envelope, "medio", "bajo"))
    return level, ~info | (low <= high + EPS)


def build_recommendations(
    runs: dict, budget: int, network: Network, calendar: Calendar, reglas: Reglas, forecast: ForecastInfo
) -> pd.DataFrame:
    """Una fila por estación, franja, día evaluado y política, con el presupuesto de referencia."""
    ids = np.array(network.station_ids, dtype=object)
    parts = []
    for policy in POLICIES:
        for record in runs[(policy, budget)].interventions:
            d, f = record.day_index, record.franja_index
            if d < calendar.warmup_days:
                continue
            window = forecast.window(calendar.days[d], reglas.franjas[f].nombre)
            level, feasible = risk_levels(record.inventory_before, network, reglas, window)
            parts.append(
                pd.DataFrame(
                    {
                        "station_id": ids,
                        "window_start": calendar.window_start_us[d, f],
                        "window_end": calendar.window_end_us[d, f],
                        "policy": policy,
                        "start_inventory_target": record.inventory_before + record.net,
                        "bikes_to_move": record.net,
                        "risk_level": level,
                        "feasible": feasible,
                    }
                )
            )
    table = pd.concat(parts, ignore_index=True)
    for column in ("window_start", "window_end"):
        table[column] = pd.to_datetime(table[column], unit="us", utc=True).dt.tz_convert(TZ)
    table["station_id"] = table["station_id"].astype("str")
    return table


def summarize_recommendations(table: pd.DataFrame) -> dict:
    """Cuenta lo que traen las recomendaciones: traslados, niveles de riesgo y franjas sin solución."""
    per_window = table.drop_duplicates(["station_id", "window_start"])
    out = {
        "filas": int(len(table)),
        "estacion_franja": int(len(per_window)),
        "estacion_franja_sin_solucion": int((~per_window["feasible"]).sum()),
        "por_politica": {},
    }
    for policy in POLICIES:
        rows = table[table["policy"] == policy]
        bikes, unsolved = rows["bikes_to_move"], ~rows["feasible"]
        levels = rows["risk_level"].value_counts()
        out["por_politica"][policy] = {
            "estacion_franja_con_traslado": int((bikes != 0).sum()),
            "bicis_recibidas": int(bikes[bikes > 0].sum()),
            "bicis_entregadas": int(-bikes[bikes < 0].sum()),
            "bicis_recibidas_en_franjas_sin_solucion": int(bikes[(bikes > 0) & unsolved].sum()),
            "bicis_entregadas_en_franjas_sin_solucion": int(-bikes[(bikes < 0) & unsolved].sum()),
            "nivel_de_riesgo": {level: int(levels.get(level, 0)) for level in RISK_LEVELS},
        }
    return out


# --------------------------------------------------------------------------- #
# Métricas de una corrida
# --------------------------------------------------------------------------- #

def _per_day(values: np.ndarray) -> dict:
    return {
        "por_dia": float(values.mean()),
        "total": int(values.sum()),
        "cada_dia": [int(v) for v in values],
    }


def summarize(result: RunResult, none: RunResult, calendar: Calendar, reglas: Reglas, draws, nivel: float) -> dict:
    """Métricas de una corrida sobre los días evaluados (sin el calentamiento)."""
    days = calendar.evaluated

    def stat(values) -> dict:
        mean, low, high = interval(values, draws, nivel)
        return {"media": mean, "ic_bajo": low, "ic_alto": high}

    hours = result.risk_hours[days]
    total = hours[:, :, 0].sum(axis=1)
    base = none.risk_hours[days][:, :, 0].sum(axis=1)
    moved = result.budget_used[days].sum(axis=1)
    entered, left = result.entered[days], result.left[days]
    warm = np.arange(calendar.warmup_days)
    return {
        "riesgo_por_dia": stat(total),
        "riesgo_simulado_evitado_por_dia": stat(base - total),
        "riesgo_por_franja": {label: stat(hours[:, i, 0]) for i, label in enumerate(calendar.labels)},
        "riesgo_por_tipo": {kind: stat(hours[:, :, k].sum(axis=1)) for k, kind in enumerate(KINDS) if k},
        "riesgo_por_franja_y_tipo": {
            label: {kind: float(hours[:, i, k].mean()) for k, kind in enumerate(KINDS) if k}
            for i, label in enumerate(calendar.labels)
        },
        "riesgo_de_cada_dia": [float(v) for v in total],
        "viajes_bloqueados": _per_day(result.blocked[days]),
        "llegadas_desviadas": _per_day(result.diverted[days]),
        "llegadas_sin_lugar_en_ninguna_estacion": int(result.no_room[days].sum()),
        "bicis_movidas": _per_day(moved),
        "presupuesto_por_franja": {
            franja.nombre: {
                "disponible_por_dia": float(result.budget_available[days, f].mean()),
                "usado_por_dia": float(result.budget_used[days, f].mean()),
                "disponible_total": int(result.budget_available[days, f].sum()),
                "usado_total": int(result.budget_used[days, f].sum()),
                "disponible_cada_dia": [int(v) for v in result.budget_available[days, f]],
                "usado_cada_dia": [int(v) for v in result.budget_used[days, f]],
            }
            for f, franja in enumerate(reglas.franjas)
        },
        "frontera": {
            "entran": int(entered.sum()),
            "salen": int(left.sum()),
            "neto": int(entered.sum() - left.sum()),
            "entran_por_dia": float(entered.mean()),
            "salen_por_dia": float(left.mean()),
            "neto_por_dia": float((entered - left).mean()),
        },
        "bicis_en_la_red": {
            "al_inicio_del_tramo": int(result.initial_total),
            "al_final_del_tramo": int(result.final_total),
            "entran_en_todo_el_tramo": int(result.entered.sum()),
            "salen_en_todo_el_tramo": int(result.left.sum()),
        },
        "calentamiento_no_cuenta": {
            "bicis_movidas": int(result.budget_used[warm].sum()),
            "viajes_bloqueados": int(result.blocked[warm].sum()),
            "llegadas_desviadas": int(result.diverted[warm].sum()),
            "llegadas_sin_lugar_en_ninguna_estacion": int(result.no_room[warm].sum()),
            "horas_de_estacion_en_riesgo": float(result.risk_hours[warm][:, :, 0].sum()),
        },
    }


# --------------------------------------------------------------------------- #
# Simulación completa
# --------------------------------------------------------------------------- #

def _as_unused(none: RunResult, policy: str, budget: int, reglas: Reglas) -> RunResult:
    """La corrida sin intervención vista como otra política o con otro presupuesto.

    Sin mover bicis el resultado es idéntico; solo cambia el presupuesto que
    quedó disponible (y sin usar) en cada franja. No se vuelve a simular.
    """
    available = np.cumsum(split_budget(budget, reglas.franjas))  # lo no usado pasa a la franja siguiente
    records = tuple(
        dataclasses.replace(record, budget_available=int(available[record.franja_index]))
        for record in none.interventions
    )
    return dataclasses.replace(
        none,
        policy=policy,
        daily_budget=int(budget),
        budget_available=np.tile(available, (len(none.blocked), 1)),
        interventions=records,
    )


def _missing_info(calendar: Calendar, reglas: Reglas, profile: HistoricalProfile, forecast: ForecastInfo) -> dict:
    """Cuántas estaciones y cuántas estación-franja quedan sin información en los días evaluados."""
    n = len(profile.with_history)
    windows = 0
    without = {"historical": np.zeros(n, dtype=np.int64), "model": np.zeros(n, dtype=np.int64)}
    gap = np.zeros(n, dtype=np.int64)  # con historia y sin pronóstico de alguna hora de la franja
    for d in calendar.evaluated:
        day = calendar.days[d]
        for franja in reglas.franjas:
            windows += 1
            missing = ~forecast.window(day, franja.nombre).has_info
            without["historical"] += ~profile.has_info[(franja.nombre, day_type(day))]
            without["model"] += missing
            gap += profile.with_history & missing
    return {
        "estaciones_sin_historia": int((~profile.with_history).sum()),
        "franjas_evaluadas_por_estacion": int(windows),
        **{
            policy: {
                "estacion_franja_sin_informacion": int(count.sum()),
                "estaciones_sin_informacion_en_alguna_franja": int((count > 0).sum()),
                "estaciones_sin_informacion_en_todas": int((count == windows).sum()),
            }
            for policy, count in without.items()
        },
        # `predictions` solo trae filas hasta la hora del último viaje de cada estación: desde ahí el modelo
        # deja de tener información de una estación que sí tiene historia.
        "con_historia_y_sin_pronostico": {"estacion_franja": int(gap.sum()), "estaciones": int((gap > 0).sum())},
    }


def compare_policies(runs: dict, budget: int, policies: dict, calendar: Calendar, reglas: Reglas) -> dict:
    """Qué pidió y qué movió cada política con un presupuesto, en los días evaluados. Solo cuenta.

    Vuelve a preguntar a cada política por su rango en cada intervención, con el inventario que vio en ella
    (`targets` no mueve nada ni cambia la corrida). "Sin solución" es aquí con el rango de la propia política:
    la envolvente del pronóstico para "model" y el flujo neto medio para la regla histórica.

    - bicis que faltan y que sobran: lo que le falta a cada estación para llegar al mínimo de su rango y lo que
      le sobra por encima del máximo, sumado sobre las estaciones y las intervenciones del día.
    - estación-franja con traslado, bicis recibidas y entregadas, y cuántas de ellas tocan una estación-franja
      sin solución.
    """
    days = len(calendar.evaluated)
    out = {}
    for name, policy in policies.items():
        count = dict.fromkeys(
            ("con_informacion", "sin_solucion", "faltan", "sobran", "con_traslado", "recibidas", "entregadas",
             "recibidas_sin_solucion", "entregadas_sin_solucion"), 0,
        )
        for record in runs[(name, budget)].interventions:
            if record.day_index < calendar.warmup_days:
                continue
            before, net = record.inventory_before, record.net
            targets = policy.targets(
                InterventionContext(
                    instant=to_timestamp(record.instant_us),
                    day=calendar.days[record.day_index],
                    franja=reglas.franjas[record.franja_index],
                    inventory=before,
                    budget=record.budget_available,
                )
            )
            unsolved = ~targets.feasible  # una estación sin información queda como "con solución"
            count["con_informacion"] += int(targets.has_info.sum())
            count["sin_solucion"] += int(unsolved.sum())
            count["faltan"] += int(np.maximum(targets.low_int - before, 0).sum())
            count["sobran"] += int(np.maximum(before - targets.high_int, 0).sum())
            count["con_traslado"] += int((net != 0).sum())
            count["recibidas"] += int(net[net > 0].sum())
            count["entregadas"] += int(-net[net < 0].sum())
            count["recibidas_sin_solucion"] += int(net[(net > 0) & unsolved].sum())
            count["entregadas_sin_solucion"] += int(-net[(net < 0) & unsolved].sum())
        out[name] = {
            "estacion_franja_con_informacion": count["con_informacion"],
            "estacion_franja_sin_solucion": count["sin_solucion"],
            "bicis_que_faltan_por_dia": count["faltan"] / days,
            "bicis_que_sobran_por_dia": count["sobran"] / days,
            "estacion_franja_con_traslado": count["con_traslado"],
            "bicis_recibidas": count["recibidas"],
            "bicis_entregadas": count["entregadas"],
            "bicis_recibidas_en_franjas_sin_solucion": count["recibidas_sin_solucion"],
            "bicis_entregadas_en_franjas_sin_solucion": count["entregadas_sin_solucion"],
            "bicis_movidas_por_dia": count["recibidas"] / days,
        }
    return out


def run_simulation(
    stations: pd.DataFrame,
    trips: pd.DataFrame,
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    reglas: Reglas,
    tramo: Tramo,
    ajustes,
    *,
    forecast_model: str,
    prueba_final: Tramo | None = None,
    usar_prueba_final: bool = False,
    progress=None,
) -> Resultado:
    """Corre las tres políticas con todos los presupuestos y arma tablas y reporte.

    - `trips`: viajes con salida o llegada en el tramo (columnas de TRIP_COLUMNS).
    - `history`: filas de `station_hour` ANTERIORES al tramo (falla si llega una posterior).
    - `predictions`: filas del tramo de un solo modelo, el que alimenta a "model".
    - `ajustes`: el bloque `simulacion` de config/datos.yaml (`load_datos().simulacion`), con
      dias_de_calentamiento, remuestreos, nivel_del_intervalo y semilla.
    """
    check_final_test(tramo, prueba_final, usar_prueba_final)
    say = progress or (lambda message: None)
    calendar = build_calendar(tramo.inicio, tramo.fin, reglas.franjas, int(ajustes.dias_de_calentamiento))
    chosen, station_report = select_stations(stations, trips, calendar.start_us, calendar.end_us)
    network = build_network(chosen, reglas.tipo_si_falta)
    n = network.size
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    start = initial_inventory(network, reglas)
    profile = HistoricalProfile.from_station_hour(history, network, reglas, before=tramo.start)
    forecast = ForecastInfo.from_predictions(predictions, network, reglas, calendar.days, profile.with_history)
    neighbors = network.neighbors_within(reglas.distancia_maxima_km)
    policies = {
        "historical": HistoricalPolicy(network, reglas, profile, neighbors),
        "model": ModelPolicy(network, reglas, forecast, neighbors),
    }

    # Presupuestos del día en bicis. Dos valores por estación que dan el mismo total se simulan una vez.
    budgets: dict[int, float] = {}
    repeated = []
    for per_station in reglas.presupuestos_por_estacion:
        total = daily_budget(per_station, n)
        if total in budgets:
            repeated.append(per_station)
        else:
            budgets[total] = per_station
    reference = daily_budget(reglas.presupuesto_de_referencia, n)
    if reference == 0:
        raise SimulacionError(
            f"presupuesto_de_referencia ({reglas.presupuesto_de_referencia} por estación) da 0 bicis al día con "
            f"{n:,} estaciones simuladas: las recomendaciones saldrían sin ningún traslado. Sube ese valor."
        )

    say(f"sin intervención ({len(events):,} eventos)")
    none = simulate(network, events, calendar, reglas, NoIntervention(), 0, start)
    if int(none.budget_used.sum()) or any(record.moves for record in none.interventions):
        raise CheckError('La política "none" movió bicis.')
    runs: dict[tuple[str, int], RunResult] = {}
    for total in budgets:
        runs[("none", total)] = _as_unused(none, "none", total, reglas)
        for name, policy in policies.items():
            if total == 0:
                runs[(name, total)] = _as_unused(none, name, total, reglas)
                continue
            say(f"{POLICY_LABELS[name]}, {total:,} bicis al día")
            runs[(name, total)] = simulate(network, events, calendar, reglas, policy, total, start)

    # Entradas iguales dan salidas iguales: se repite una corrida y se compara la huella.
    again_policy, again_budget = ("model", reference) if reference > 0 else ("none", 0)
    say("repetición para comprobar que el resultado no cambia")
    if again_budget:
        again = simulate(network, events, calendar, reglas, policies[again_policy], again_budget, start)
        same = again.digest() == runs[(again_policy, again_budget)].digest()
    else:
        same = simulate(network, events, calendar, reglas, NoIntervention(), 0, start).digest() == none.digest()
    if not same:
        raise CheckError("Dos corridas con las mismas entradas dieron resultados distintos.")
    if len({result.initial_total for result in runs.values()}) != 1:
        raise CheckError("Las políticas no partieron del mismo inventario.")

    evaluated = calendar.evaluated
    draws = resample_days(len(evaluated), int(ajustes.remuestreos), int(ajustes.semilla))
    nivel = float(ajustes.nivel_del_intervalo)
    results = []
    curve = []
    for policy in POLICIES:
        for total, per_station in budgets.items():
            summary = summarize(runs[(policy, total)], none, calendar, reglas, draws, nivel)
            results.append(
                {
                    "politica": policy,
                    "presupuesto_por_estacion": per_station,
                    "presupuesto_del_dia_en_bicis": total,
                    "simulada": bool(policy == "none" or total > 0),
                    **summary,
                }
            )
            risk = summary["riesgo_por_dia"]
            curve.append(
                {
                    "policy": policy,
                    "budget": total,
                    "risk_station_hours": risk["media"],
                    "ci_low": risk["ic_bajo"],
                    "ci_high": risk["ic_alto"],
                }
            )
    policy_curve = pd.DataFrame(curve)
    recommendations = build_recommendations(runs, reference, network, calendar, reglas, forecast)
    recommended = summarize_recommendations(recommendations)

    report = {
        "tramo": {
            "nombre": tramo.nombre,
            "inicio": tramo.inicio.isoformat(),
            "fin": tramo.fin.isoformat(),
            "dias": [day.isoformat() for day in calendar.days],
            "dias_de_calentamiento": [day.isoformat() for day in calendar.days[: calendar.warmup_days]],
            "dias_evaluados": [calendar.days[d].isoformat() for d in evaluated],
        },
        "prueba_final": {
            "simulada": bool(tramo.nombre == FINAL_TRAMO),
            "inicio": prueba_final.inicio.isoformat() if prueba_final else None,
            "fin": prueba_final.fin.isoformat() if prueba_final else None,
        },
        "estaciones": station_report,
        "viajes": events.counts,
        "inventario_inicial": {
            "bicis": int(start.sum()),
            "lugares": int(network.capacity.sum()),
            "franja_que_lo_define": reglas.first_franja.nombre,
            "estaciones_sin_tipo": int((~network.typed).sum()),
            "tipo_si_falta": reglas.tipo_si_falta,
        },
        "pronostico": {
            "modelo_que_alimenta_model": forecast_model,
            "filas_leidas": int(len(predictions)),
            "estacion_franja_sin_solucion": recommended["estacion_franja_sin_solucion"],
        },
        "informacion": _missing_info(calendar, reglas, profile, forecast),
        "historia": {"filas": int(len(history)), "antes_de": tramo.inicio.isoformat()},
        "reglas": {
            "estado": reglas.estado,
            "franjas": [dataclasses.asdict(franja) for franja in reglas.franjas],
            "bicis_minimas": reglas.bicis_minimas,
            "lugares_libres_minimos": reglas.lugares_libres_minimos,
            "distancia_maxima_km": reglas.distancia_maxima_km,
            "presupuestos_por_estacion": list(reglas.presupuestos_por_estacion),
            "presupuesto_de_referencia": reglas.presupuesto_de_referencia,
            "presupuesto_de_referencia_en_bicis": int(reference),
            "presupuestos_con_el_mismo_total_que_otro": repeated,
            "reparto_por_franja": {
                str(total): dict(zip((f.nombre for f in reglas.franjas), split_budget(total, reglas.franjas)))
                for total in budgets
            },
        },
        "intervalo": {
            "remuestreos": int(ajustes.remuestreos),
            "nivel": nivel,
            "semilla": int(ajustes.semilla),
            "dias_remuestreados": int(len(evaluated)),
        },
        "resultados": results,
        "recomendaciones": recommended,
        "diferencias_entre_politicas": {
            "presupuesto_del_dia_en_bicis": int(reference),
            "estacion_franja": recommended["estacion_franja"],
            **compare_policies(runs, reference, policies, calendar, reglas),
        },
        "comprobaciones": [
            "El inventario estuvo siempre entre 0 y la capacidad en todas las corridas.",
            "En cada corrida el total final de bicis es el inicial más las que entran menos las que salen.",
            "Los traslados no crearon ni destruyeron bicis.",
            "Ninguna franja usó más presupuesto del disponible.",
            "Todos los traslados quedaron dentro de la distancia máxima.",
            'La política "none" no movió bicis.',
            f"Se repitió la corrida de {POLICY_LABELS[again_policy]} con {again_budget:,} bicis al día y dio "
            "el mismo resultado.",
        ],
        "corridas_simuladas": 1 + sum(1 for total in budgets if total > 0) * len(policies),
    }
    report["limitaciones"] = limitations(report)
    return Resultado(recommendations, policy_curve, report, runs, network, calendar)


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #

def limitations(report: dict) -> list[str]:
    """Limitaciones de la simulación, dichas como hechos."""
    model = report["pronostico"]["modelo_que_alimenta_model"]
    fed = (
        'La política "model" se alimenta hoy de la línea base (promedio histórico por estación, día de la '
        "semana y hora), porque LightGBM todavía no existe. Su q50 es una media, no una mediana."
        if model == "baseline"
        else f'La política "model" se alimenta del pronóstico "{model}".'
    )
    final = (
        "Esta corrida usó la prueba final."
        if report["prueba_final"]["simulada"]
        else "La prueba final no se simuló (decisión 4)."
    )
    none = next(row for row in report["resultados"] if row["politica"] == "none")
    bikes = none["bicis_en_la_red"]
    drift = bikes["al_final_del_tramo"] - bikes["al_inicio_del_tramo"]
    trips = report["viajes"]
    leave, enter = trips["salidas_que_salen_por_motivo"], trips["llegadas_que_entran_por_motivo"]
    days, daily = report["tramo"]["dias_evaluados"], none["riesgo_de_cada_dia"]
    items = [
        "La distancia entre estaciones es en línea recta: ignora ríos y puentes. Dos estaciones con un río en "
        "medio pueden quedar dentro de la distancia máxima.",
        "Los viajes no traen los traslados que el operador hizo de verdad en esos días (camiones, bicis "
        "llevadas a taller). La política sin intervención es un sistema en el que nadie mueve bicis: no es lo "
        "que pasó en la realidad.",
        f"Una bici que sale del conjunto simulado solo vuelve si otro viaje la trae: los traslados reales del "
        f"operador no están en los datos y nada corrige el total de bicis durante el tramo. Sin intervención, "
        f"el conjunto empieza el tramo con {bikes['al_inicio_del_tramo']:,} bicis y lo termina con "
        f"{bikes['al_final_del_tramo']:,} ({drift:+,}): entran {bikes['entran_en_todo_el_tramo']:,} y salen "
        f"{bikes['salen_en_todo_el_tramo']:,}. En los viajes tal como pasaron, de las "
        f"{trips['salidas_que_salen']:,} salidas que sacan una bici del conjunto, "
        f"{leave['sin_estacion_de_fin']:,} son de viajes sin estación de fin, "
        f"{leave['a_una_estacion_fuera_del_conjunto']:,} van a una estación fuera del conjunto y "
        f"{leave['terminan_despues_del_tramo']:,} terminan después del tramo; de las "
        f"{trips['llegadas_que_entran']:,} llegadas que meten una, {enter['sin_estacion_de_inicio']:,} son de "
        f"viajes sin estación de inicio, {enter['de_una_estacion_fuera_del_conjunto']:,} vienen de una "
        f"estación fuera del conjunto y {enter['salieron_antes_del_tramo']:,} salieron antes del tramo. En la "
        "simulación, una salida bloqueada no saca la bici.",
        "Los traslados son instantáneos y no hay camiones: no se modela su número, su capacidad ni su ruta. "
        "El único límite de esfuerzo es el presupuesto de bicis movidas.",
        "Los días no son independientes: el inventario pasa de un día al siguiente, hay patrón semanal y son "
        "pocos días. Los intervalos son indicativos, no una garantía estadística.",
        "Los datos solo registran demanda atendida. Los viajes bloqueados son un resultado de la simulación "
        "(posible demanda insatisfecha), no demanda medida. Un usuario bloqueado no busca otra estación ni "
        "vuelve a intentar.",
        "El inventario inicial es un supuesto (ocupacion_respaldo por capacidad). No hay inventario real del "
        "periodo: la captura de GBFS empezó después. Los días de calentamiento no quitan la dependencia de ese "
        f"supuesto: `ocupacion_respaldo` decide con cuántas bicis empieza el conjunto "
        f"({bikes['al_inicio_del_tramo']:,}) y después ese total solo cambia por las bicis que entran y salen "
        f"(sin intervención termina el tramo con {bikes['al_final_del_tramo']:,}). El nivel de riesgo y el "
        "riesgo simulado evitado dependen de ese valor en todos los días evaluados, también en el último. Sin "
        f"intervención, el primer día evaluado ({days[0]}) tiene {_n(daily[0])} horas de estación en riesgo y "
        f"el último ({days[-1]}), {_n(daily[-1])}.",
        "La capacidad es la del feed GBFS de hoy, no la del periodo simulado.",
        "Bicis clásicas y eléctricas se cuentan juntas.",
        "Una estación cerrada varios días se ve como demanda cero: la simulación no sabe que estaba cerrada.",
        "El modelo trata las horas como independientes entre sí y las salidas como independientes de las "
        "llegadas al sumar la incertidumbre.",
        "Una llegada a una estación llena va a la estación con lugar más cercana en línea recta. No se modela "
        "a qué estación iría de verdad el usuario.",
        "El conjunto simulado se define con los viajes del propio tramo (estaciones con al menos una salida o "
        "llegada). Las bicis que entran y salen del conjunto se toman tal como pasaron.",
    ]
    gap = report["informacion"]["con_historia_y_sin_pronostico"]
    if gap["estacion_franja"]:
        items.append(
            "`predictions` solo trae filas hasta la hora del último viaje de cada estación (`station_hour` es "
            "densa entre el primer y el último viaje). Por eso el modelo deja de pedir y de prestar bicis en "
            "estaciones que sí tienen historia, desde la franja en que cae su último viaje: "
            f"{gap['estacion_franja']:,} estación-franja (estaciones con historia a las que les pasa: "
            f"{gap['estaciones']:,}). Ahí lo que decide depende de viajes posteriores a la intervención. La "
            "regla histórica sí las atiende; en esas estación-franja la comparación no es pareja."
        )
    return [*items, fed, final]


def _n(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def _table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    align = ["---"] * text_columns + ["---:"] * (len(header) - text_columns)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _budget_text(row: dict) -> list[str]:
    return [f"{row['presupuesto_por_estacion']:g}", f"{row['presupuesto_del_dia_en_bicis']:,}"]


def _shown(report: dict) -> list[dict]:
    """Filas que se muestran: "none" una sola vez (da lo mismo con cualquier presupuesto)."""
    seen_none = False
    rows = []
    for row in report["resultados"]:
        if row["politica"] == "none":
            if seen_none:
                continue
            seen_none = True
        rows.append(row)
    return rows


def _first_cells(row: dict) -> list[str]:
    if row["politica"] == "none":
        return [POLICY_LABELS["none"], "cualquiera", "cualquiera"]
    return [POLICY_LABELS[row["politica"]], *_budget_text(row)]


def _daily_rows(report: dict) -> list[list[str]]:
    """Riesgo de cada día evaluado con el presupuesto de referencia: una fila por día y una columna por política."""
    reference = report["reglas"]["presupuesto_de_referencia_en_bicis"]
    series = {
        row["politica"]: row["riesgo_de_cada_dia"]
        for row in report["resultados"]
        if row["presupuesto_del_dia_en_bicis"] == reference
    }
    rows = []
    for i, day in enumerate(report["tramo"]["dias_evaluados"]):
        weekday = WEEKDAYS[datetime.fromisoformat(day).weekday()]
        rows.append([f"{day} ({weekday})", *[_n(series[policy][i]) for policy in POLICIES]])
    return rows


def _at_reference(report: dict) -> dict:
    """Fila de resultados de cada política con el presupuesto de referencia."""
    reference = report["reglas"]["presupuesto_de_referencia_en_bicis"]
    return {
        row["politica"]: row for row in report["resultados"] if row["presupuesto_del_dia_en_bicis"] == reference
    }


def _range(stat: dict) -> str:
    return f"{_n(stat['ic_bajo'])} a {_n(stat['ic_alto'])}"


def _share(part: int, whole: int) -> str:
    """`part` de `whole`, con el porcentaje cuando se puede calcular."""
    return f"{part:,} de {whole:,}" + (f" ({part / whole * 100:.1f} %)" if whole else "")


def _reading(report: dict) -> list[str]:
    """La primera tabla dicha en palabras, con el presupuesto de referencia. Solo cifras de `report`.

    Por política: riesgo por día, riesgo simulado evitado contra no intervenir con su intervalo, cuántos días
    evaluados tuvo más riesgo que sin intervención, y viajes bloqueados y llegadas desviadas. Del intervalo
    del riesgo evitado dice si queda completo por debajo de cero, si incluye el cero o si su límite inferior
    es menor que su propio ancho (queda cerca de cero). No califica a ninguna política.
    """
    rows, days = _at_reference(report), report["tramo"]["dias_evaluados"]
    none = rows["none"]
    base = none["riesgo_por_dia"]
    lines = [
        f"Lectura de la tabla con el presupuesto de referencia "
        f"({report['reglas']['presupuesto_de_referencia_en_bicis']:,} bicis al día; días evaluados: {len(days)}). "
        "El intervalo es indicativo: los días no son independientes y puede salir más angosto de lo debido. Un "
        "intervalo de riesgo simulado evitado cuyo límite inferior queda cerca de cero no alcanza para decir "
        "que la política reduce el riesgo.",
        "",
        f"- {POLICY_LABELS['none'].capitalize()}: {_n(base['media'])} horas de estación en riesgo por día "
        f"(intervalo {_range(base)}).",
    ]
    for policy in POLICIES:
        if policy == "none":
            continue
        row = rows[policy]
        avoided = row["riesgo_simulado_evitado_por_dia"]
        low, high = avoided["ic_bajo"], avoided["ic_alto"]
        part = ""
        if base["media"]:
            part = f" ({avoided['media'] / base['media'] * 100:.1f} % del riesgo sin intervención)"
        if high < 0:
            note = " Ese intervalo queda completo por debajo de cero."
        elif low <= 0:
            note = " Ese intervalo incluye el cero."
        elif low < high - low:
            note = (
                f" El límite inferior de ese intervalo ({_n(low)}) es menor que su ancho ({_n(high - low)}): "
                "queda cerca de cero."
            )
        else:
            note = ""
        worse = [
            day
            for day, mine, theirs in zip(days, row["riesgo_de_cada_dia"], none["riesgo_de_cada_dia"])
            if mine > theirs
        ]
        lines.append(
            f"- {POLICY_LABELS[policy].capitalize()}: {_n(row['riesgo_por_dia']['media'])} horas de estación en "
            f"riesgo por día (intervalo {_range(row['riesgo_por_dia'])}). Riesgo simulado evitado contra no "
            f"intervenir: {_n(avoided['media'])} horas por día{part}, intervalo {_range(avoided)}.{note} Días "
            f"evaluados con más riesgo que sin intervención: {len(worse)} de {len(days)}"
            + (f" ({', '.join(worse)})" if worse else "")
            + f". Viajes bloqueados por día (simulados): {_n(row['viajes_bloqueados']['por_dia'])} contra "
            f"{_n(none['viajes_bloqueados']['por_dia'])} sin intervención. Llegadas desviadas por día: "
            f"{_n(row['llegadas_desviadas']['por_dia'])} contra {_n(none['llegadas_desviadas']['por_dia'])}."
        )
    return lines


def _differences(report: dict) -> list[str]:
    """Sección con lo que pidió y movió cada política con el presupuesto de referencia.

    Solo cifras de `report` (`diferencias_entre_politicas` y los resultados). No afirma ninguna causa ni
    recomienda nada. El título depende de cuál de las dos políticas evitó menos riesgo en la corrida.
    """
    facts, rows = report["diferencias_entre_politicas"], _at_reference(report)
    reference, windows = facts["presupuesto_del_dia_en_bicis"], facts["estacion_franja"]
    historical, model = facts["historical"], facts["model"]
    avoided = {name: rows[name]["riesgo_simulado_evitado_por_dia"]["media"] for name in ("historical", "model")}
    title = (
        "Por qué la política model evita menos"
        if avoided["model"] < avoided["historical"]
        else "En qué se diferencian la regla histórica y la política model"
    )

    def per_window(row: dict) -> str:
        """Bicis recibidas o entregadas por cada estación-franja con traslado."""
        touched = row["estacion_franja_con_traslado"]
        return _n((row["bicis_recibidas"] + row["bicis_entregadas"]) / touched) if touched else "0"

    def unsolved(row: dict) -> str:
        return (
            f"{_share(row['bicis_recibidas_en_franjas_sin_solucion'], row['bicis_recibidas'])} de las recibidas "
            f"y {_share(row['bicis_entregadas_en_franjas_sin_solucion'], row['bicis_entregadas'])} de las "
            "entregadas"
        )

    fed = report["pronostico"]["modelo_que_alimenta_model"]
    return [
        f"## {title}",
        "",
        f"Hechos de esta corrida con el presupuesto de referencia ({reference:,} bicis al día), en los días "
        f"evaluados ({windows:,} estación-franja). Riesgo simulado evitado por día: regla histórica "
        f"{_n(avoided['historical'])} horas de estación, modelo {_n(avoided['model'])}. Las cifras de abajo dicen "
        "qué pidió y qué movió cada política. Ninguna está probada como causa de esa diferencia.",
        "",
        "- Estación-franja sin solución (hay información y ningún inventario evita a la vez los dos umbrales). "
        f"Con la envolvente del pronóstico, que es el rango de la política \"model\": "
        f"{_share(model['estacion_franja_sin_solucion'], windows)}; tiene información en "
        f"{model['estacion_franja_con_informacion']:,}. Con el rango de la regla histórica (flujo neto medio de "
        f"la historia): {_share(historical['estacion_franja_sin_solucion'], windows)}; tiene información en "
        f"{historical['estacion_franja_con_informacion']:,}.",
        "- Bicis que pide cada política en un día: las que faltan a las estaciones que están debajo de su rango "
        "y las que sobran a las que están arriba, con el inventario que la política vio en cada intervención, "
        f"sumando las intervenciones del día. Regla histórica: faltan {_n(historical['bicis_que_faltan_por_dia'])} "
        f"y sobran {_n(historical['bicis_que_sobran_por_dia'])}. Modelo: faltan "
        f"{_n(model['bicis_que_faltan_por_dia'])} y sobran {_n(model['bicis_que_sobran_por_dia'])}. Presupuesto "
        f"del día: {reference:,} bicis. Bicis movidas por día: regla histórica "
        f"{_n(historical['bicis_movidas_por_dia'])}, modelo {_n(model['bicis_movidas_por_dia'])}.",
        f"- Estación-franja con traslado: regla histórica {historical['estacion_franja_con_traslado']:,}, con "
        f"{per_window(historical)} bicis recibidas o entregadas por cada una; modelo "
        f"{model['estacion_franja_con_traslado']:,}, con {per_window(model)}.",
        "- Bicis que llegan a una estación-franja sin solución o salen de ella, con el rango de la propia "
        f"política. Modelo: {unsolved(model)}. Regla histórica: {unsolved(historical)}.",
        (
            '- En esta corrida la política "model" se alimenta de la línea base (promedio histórico por estación, '
            "día de la semana y hora), no de LightGBM: estas cifras no dicen nada de LightGBM."
            if fed == "baseline"
            else f'- En esta corrida la política "model" se alimenta del pronóstico "{fed}".'
        ),
    ]


def build_report(report: dict, operacion: str = "config/operacion.yaml") -> str:
    """El reporte en Markdown. Todos los números salen de `report`.

    `operacion` es el archivo de reglas con el que se corrió, tal como se muestra en el reporte.
    """
    tramo, rules, stations = report["tramo"], report["reglas"], report["estaciones"]
    level = f"{report['intervalo']['nivel'] * 100:g} %"
    shown = _shown(report)
    recs = report["recomendaciones"]
    labels = list(report["resultados"][0]["riesgo_por_franja"])
    gap = report["informacion"]["con_historia_y_sin_pronostico"]
    lines = [
        f"# Simulación de inventario: tramo de {TRAMO_LABELS.get(tramo['nombre'], tramo['nombre'])}",
        "",
        "Generado por `python scripts/run_simulation.py`. Todos los números salen de la corrida.",
        "",
    ]
    if rules["estado"] != "confirmado":
        lines += [
            f"**Las reglas de operación son provisionales** (`{operacion}`, `estado: {rules['estado']}`). "
            "Franjas, umbrales, presupuestos, distancia y ocupación de respaldo son valores de arranque: "
            "estos resultados cambian cuando logística los confirme.",
            "",
        ]
    lines += [
        "## Qué se simuló",
        "",
        f"- Tramo: {TRAMO_LABELS.get(tramo['nombre'], tramo['nombre'])}, del {tramo['inicio']} al "
        f"{tramo['fin']} ({len(tramo['dias'])} días), "
        "de corrido: el inventario pasa de un día al siguiente y no hay reinicios.",
        f"- Días de calentamiento (se simulan, gastan presupuesto y no cuentan): "
        f"{', '.join(tramo['dias_de_calentamiento']) or 'ninguno'}.",
        f"- Días evaluados: {len(tramo['dias_evaluados'])}, del {tramo['dias_evaluados'][0]} al "
        f"{tramo['dias_evaluados'][-1]}.",
        f"- Corridas simuladas: {report['corridas_simuladas']}. Con presupuesto 0 las tres políticas son "
        "la corrida sin intervención y no se vuelven a simular.",
        "- Las tres políticas parten del mismo inventario y enfrentan los mismos viajes. En cada intervención "
        "una política ve el inventario simulado de ese momento y ningún viaje"
        + (
            " (salvedad: qué filas trae el pronóstico depende del último viaje de cada estación; ver "
            "Limitaciones)."
            if gap["estacion_franja"]
            else "."
        ),
    ]
    if not report["prueba_final"]["simulada"] and report["prueba_final"]["inicio"]:
        lines.append(
            f"- La prueba final ({report['prueba_final']['inicio']} a {report['prueba_final']['fin']}) "
            "no se simuló (decisión 4). De esas fechas no se leyó ninguna fila de `station_hour` ni de "
            "`predictions`. De los viajes solo se leyeron los que salieron dentro del tramo simulado y "
            "terminaron después, para contarlos como bicis que salen del conjunto."
        )
    without, idle = stations["sin_capacidad"], stations["inactivas"]
    unknown, loose = stations["ids_de_viajes_que_no_estan_en_stations"], stations["extremos_sin_estacion"]
    lines += [
        "",
        "## Estaciones",
        "",
        _table(
            ["grupo", "estaciones", "salidas en el tramo", "llegadas en el tramo"],
            [
                ["En `stations`", f"{stations['en_stations']:,}", "", ""],
                ["Simuladas", f"{stations['simuladas']:,}", f"{report['viajes']['salidas']:,}",
                 f"{report['viajes']['llegadas']:,}"],
                ["Fuera: sin capacidad", f"{without['estaciones']:,}", f"{without['salidas_en_el_tramo']:,}",
                 f"{without['llegadas_en_el_tramo']:,}"],
                ["Fuera: con capacidad y sin viajes en el tramo (inactivas)", f"{idle['estaciones']:,}", "0", "0"],
                ["Ids de viajes que no están en `stations`", f"{unknown['ids']:,}",
                 f"{unknown['salidas_en_el_tramo']:,}", f"{unknown['llegadas_en_el_tramo']:,}"],
                ["Extremos de viaje sin estación", "", f"{loose['salidas_en_el_tramo']:,}",
                 f"{loose['llegadas_en_el_tramo']:,}"],
            ],
        ),
        "",
        f"Estaciones sin capacidad: {without['estaciones']:,}; de ellas, "
        f"{without['de_ellas_sin_viajes_en_el_tramo']:,} tampoco tienen viajes en el tramo. "
        "Los viajes de las estaciones que quedan fuera siguen contando del lado simulado: son bicis que "
        "entran o salen del conjunto.",
        "",
        "## Viajes",
        "",
        f"- Viajes con salida o llegada en el tramo: {report['viajes']['viajes_leidos']:,}.",
        f"- Salidas simuladas: {report['viajes']['salidas']:,}. Llegadas simuladas: "
        f"{report['viajes']['llegadas']:,}.",
        f"- Viajes con las dos puntas simuladas: {report['viajes']['viajes_internos']:,}.",
        f"- Llegadas de viajes que vienen de fuera del conjunto (o de antes del tramo): "
        f"{report['viajes']['llegadas_que_entran']:,}.",
        f"- Salidas de viajes que terminan fuera del conjunto (o después del tramo): "
        f"{report['viajes']['salidas_que_salen']:,}.",
        "",
        "## Inventario inicial",
        "",
        f"- {report['inventario_inicial']['bicis']:,} bicis en {report['inventario_inicial']['lugares']:,} "
        f"lugares, al primer instante del tramo. Es un supuesto: `ocupacion_respaldo` de la franja "
        f"\"{report['inventario_inicial']['franja_que_lo_define']}\" por la capacidad de cada estación.",
        f"- Estaciones sin tipo: {report['inventario_inicial']['estaciones_sin_tipo']:,}. Usan la estrategia "
        f"del tipo \"{report['inventario_inicial']['tipo_si_falta']}\" (`tipo_si_falta`).",
        "",
        '## Pronóstico que alimenta la política "model"',
        "",
    ]
    model = report["pronostico"]["modelo_que_alimenta_model"]
    if model == "baseline":
        lines.append(
            'Hoy es la **línea base** (`model = "baseline"` en `predictions`): el promedio histórico por '
            "estación, día de la semana y hora. LightGBM todavía no existe. La política \"model\" de esta "
            "corrida no es un modelo de aprendizaje."
        )
    else:
        lines.append(f'Es el pronóstico `model = "{model}"` de `predictions`.')
    info = report["informacion"]
    lines += [
        "",
        f"- Filas de pronóstico leídas (todo el tramo, todas las estaciones): "
        f"{report['pronostico']['filas_leidas']:,}. La política solo usa las de estaciones simuladas con "
        "historia, en las horas que van de la intervención al fin de cada franja.",
        f"- Estación-franja sin solución en los días evaluados (ningún inventario evita los dos umbrales "
        f"con la envolvente del pronóstico): {report['pronostico']['estacion_franja_sin_solucion']:,}.",
        "",
        "## Estaciones sin información",
        "",
        "Una estación sin información no pide ni presta bicis en esa franja.",
        "",
        f"- Estaciones simuladas sin ninguna fila en la historia (antes del {report['historia']['antes_de']}): "
        f"{info['estaciones_sin_historia']:,}. Ninguna de las dos políticas las toca.",
        f"- Regla histórica: {info['historical']['estacion_franja_sin_informacion']:,} estación-franja sin "
        f"información de {stations['simuladas'] * info['franjas_evaluadas_por_estacion']:,}; "
        f"{info['historical']['estaciones_sin_informacion_en_alguna_franja']:,} estaciones en alguna franja.",
        f"- Modelo: {info['model']['estacion_franja_sin_informacion']:,} estación-franja sin información "
        f"(sin historia, o sin pronóstico de alguna hora de la franja); "
        f"{info['model']['estaciones_sin_informacion_en_alguna_franja']:,} estaciones en alguna franja. "
        f"De ellas, {gap['estacion_franja']:,} estación-franja son de estaciones con historia "
        f"(estaciones: {gap['estaciones']:,}): el pronóstico se corta en la hora de su último viaje.",
        "",
        "## Reglas usadas",
        "",
        _table(
            ["franja", "intervención", "inicio", "fin", "parte del presupuesto"],
            [
                [f["nombre"], f["momento_intervencion"], f["inicio"], f["fin"], f"{f['parte_del_presupuesto']:g}"]
                for f in rules["franjas"]
            ],
            text_columns=4,
        ),
        "",
        f"- En riesgo: menos de {rules['bicis_minimas']} bicis o menos de {rules['lugares_libres_minimos']} "
        "lugares libres.",
        f"- Distancia máxima de un traslado: {rules['distancia_maxima_km']:g} km en línea recta.",
        f"- Presupuestos, en bicis por estación por día: "
        f"{', '.join(f'{b:g}' for b in rules['presupuestos_por_estacion'])}. Se multiplican por las "
        f"{stations['simuladas']:,} estaciones simuladas.",
        f"- Presupuesto de referencia (el de las recomendaciones): {rules['presupuesto_de_referencia']:g} por "
        f"estación, {rules['presupuesto_de_referencia_en_bicis']:,} bicis al día.",
    ]
    if rules["presupuestos_con_el_mismo_total_que_otro"]:
        lines.append(
            "- Presupuestos que dan el mismo total de bicis que otro y no se repiten: "
            f"{', '.join(f'{b:g}' for b in rules['presupuestos_con_el_mismo_total_que_otro'])}."
        )
    lines += [
        "",
        "## Horas de estación en riesgo por día",
        "",
        f"Media por día evaluado, sobre las 24 horas. Intervalo del {level} por remuestreo de días. "
        "\"Riesgo simulado evitado\" es la diferencia contra no intervenir, día por día, con su propio "
        "intervalo (los mismos días sorteados para las dos corridas). Positivo es menos riesgo.",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", "horas en riesgo por día", f"intervalo {level}",
             "riesgo simulado evitado por día", f"intervalo {level}"],
            [
                [
                    *_first_cells(row),
                    _n(row["riesgo_por_dia"]["media"]),
                    f"{_n(row['riesgo_por_dia']['ic_bajo'])} a {_n(row['riesgo_por_dia']['ic_alto'])}",
                    _n(row["riesgo_simulado_evitado_por_dia"]["media"]),
                    f"{_n(row['riesgo_simulado_evitado_por_dia']['ic_bajo'])} a "
                    f"{_n(row['riesgo_simulado_evitado_por_dia']['ic_alto'])}",
                ]
                for row in shown
            ],
        ),
        "",
        *_reading(report),
        "",
        "## Riesgo día por día",
        "",
        f"Horas de estación en riesgo de cada día evaluado, con el presupuesto de referencia "
        f"({rules['presupuesto_de_referencia_en_bicis']:,} bicis al día). Los días no se parecen entre sí: "
        "la media por día y su intervalo resumen esta tabla, no la reemplazan.",
        "",
        _table(["día", *[POLICY_LABELS[policy] for policy in POLICIES]], _daily_rows(report)),
        "",
        "## Riesgo por franja y por tipo",
        "",
        "Horas de estación en riesgo por día evaluado. Las franjas cuentan el tiempo entre su inicio y su "
        "fin; el resto del día es \"fuera de franja\".",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", *labels, *KIND_LABELS.values()],
            [
                [
                    *_first_cells(row),
                    *[_n(row["riesgo_por_franja"][label]["media"]) for label in labels],
                    *[_n(row["riesgo_por_tipo"][kind]["media"]) for kind in KIND_LABELS],
                ]
                for row in shown
            ],
        ),
        "",
        "## Métricas secundarias",
        "",
        "Por día evaluado. Un viaje bloqueado es una salida que en la simulación encontró la estación sin "
        "bicis: posible demanda insatisfecha, simulada, no medida. Una llegada desviada encontró la estación "
        "llena y la bici fue a la estación con lugar más cercana.",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", "viajes bloqueados por día",
             "llegadas desviadas por día", "bicis movidas por día"],
            [
                [
                    *_first_cells(row),
                    _n(row["viajes_bloqueados"]["por_dia"]),
                    _n(row["llegadas_desviadas"]["por_dia"]),
                    _n(row["bicis_movidas"]["por_dia"]),
                ]
                for row in shown
            ],
        ),
        "",
        "## Presupuesto disponible y usado por franja",
        "",
        "Bicis por día evaluado. Lo que una franja no usa pasa a la siguiente del mismo día; lo que sobra al "
        "final del día se pierde. Mover una bici de una estación a otra cuesta 1. El detalle de cada día y los "
        "totales están en `simulacion.json` (`presupuesto_por_franja`).",
        "",
    ]
    names = [f["nombre"] for f in rules["franjas"]]
    lines += [
        _table(
            ["política", "bicis por estación", "bicis al día",
             *[f"{name}: {what}" for name in names for what in ("disponible", "usado")]],
            [
                [
                    POLICY_LABELS[row["politica"]],
                    *_budget_text(row),
                    *[
                        _n(row["presupuesto_por_franja"][name][key])
                        for name in names
                        for key in ("disponible_por_dia", "usado_por_dia")
                    ],
                ]
                for row in report["resultados"]
                if row["politica"] != "none"
            ],
        ),
        "",
        "## Bicis que entran y salen del conjunto simulado",
        "",
        "Entra una bici cuando llega un viaje que salió de fuera del conjunto, sin estación o antes del "
        "tramo. Sale cuando un viaje termina fuera del conjunto, sin estación o después del tramo. El neto es "
        "entran menos salen. Las columnas \"entran\", \"salen\" y \"neto\" son totales de los días evaluados. "
        "Las cuatro últimas son de todo el tramo, con el calentamiento: inicio más entran menos salen da el "
        "final.",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", "entran", "salen", "neto",
             "bicis al inicio del tramo", "entran en todo el tramo", "salen en todo el tramo",
             "bicis al final del tramo"],
            [
                [
                    *_first_cells(row),
                    f"{row['frontera']['entran']:,}",
                    f"{row['frontera']['salen']:,}",
                    f"{row['frontera']['neto']:,}",
                    f"{row['bicis_en_la_red']['al_inicio_del_tramo']:,}",
                    f"{row['bicis_en_la_red']['entran_en_todo_el_tramo']:,}",
                    f"{row['bicis_en_la_red']['salen_en_todo_el_tramo']:,}",
                    f"{row['bicis_en_la_red']['al_final_del_tramo']:,}",
                ]
                for row in shown
            ],
        ),
        "",
        "## Recomendaciones",
        "",
        f"`recommendations.parquet` sale de las corridas con el presupuesto de referencia "
        f"({rules['presupuesto_de_referencia_en_bicis']:,} bicis al día): {recs['filas']:,} filas, una por "
        f"estación, franja, día evaluado y política ({recs['estacion_franja']:,} estación-franja). "
        f"En {recs['estacion_franja_sin_solucion']:,} de ellas `feasible` es falso: con la envolvente del "
        "pronóstico ningún inventario evita a la vez los dos umbrales. El nivel de riesgo se calcula con el "
        "pronóstico y el inventario que cada política tenía ANTES de intervenir.",
        "",
        _table(
            ["política", "estación-franja con traslado", "bicis recibidas", "de ellas, en franjas sin solución",
             "bicis entregadas", "de ellas, en franjas sin solución", "riesgo alto", "riesgo medio", "riesgo bajo"],
            [
                [
                    POLICY_LABELS[policy],
                    f"{row['estacion_franja_con_traslado']:,}",
                    f"{row['bicis_recibidas']:,}",
                    f"{row['bicis_recibidas_en_franjas_sin_solucion']:,}",
                    f"{row['bicis_entregadas']:,}",
                    f"{row['bicis_entregadas_en_franjas_sin_solucion']:,}",
                    *[f"{row['nivel_de_riesgo'][level]:,}" for level in ("alto", "medio", "bajo")],
                ]
                for policy, row in recs["por_politica"].items()
            ],
        ),
        "",
        *_differences(report),
        "",
        "## Comprobaciones de la corrida",
        "",
        *[f"- {check}" for check in report["comprobaciones"]],
        "",
        "## Cómo leer los intervalos",
        "",
        f"Se sortean, con repetición, los {report['intervalo']['dias_remuestreados']} días evaluados, se "
        f"promedia, y se repite {report['intervalo']['remuestreos']:,} veces (semilla "
        f"{report['intervalo']['semilla']}). El intervalo va del percentil "
        f"{(1 - report['intervalo']['nivel']) / 2 * 100:g} al {(1 + report['intervalo']['nivel']) / 2 * 100:g} "
        "de esas medias. **Los días no son independientes**: el inventario de un día es el punto de partida "
        "del siguiente, hay patrón semanal y son pocos días. El intervalo es indicativo y puede salir más "
        "angosto de lo debido.",
        "",
        "## Limitaciones",
        "",
        *[f"- {item}" for item in report["limitaciones"]],
        "",
    ]
    return "\n".join(lines)


def summary_lines(report: dict, operacion: str = "config/operacion.yaml") -> list[str]:
    """Resumen corto para la pantalla. `operacion` es el archivo de reglas con el que se corrió."""
    tramo, stations = report["tramo"], report["estaciones"]
    level = f"{report['intervalo']['nivel'] * 100:g} %"
    model = report["pronostico"]["modelo_que_alimenta_model"]
    lines = [
        f"Simulación de inventario. Tramo de {TRAMO_LABELS.get(tramo['nombre'], tramo['nombre'])}: "
        f"{tramo['inicio']} a {tramo['fin']}.",
        f"Días evaluados: {len(tramo['dias_evaluados'])}. Calentamiento (no cuenta): "
        f"{', '.join(tramo['dias_de_calentamiento']) or 'ninguno'}.",
        f"Estaciones simuladas: {stations['simuladas']:,} de {stations['en_stations']:,}. Fuera: "
        f"{stations['sin_capacidad']['estaciones']:,} sin capacidad y {stations['inactivas']['estaciones']:,} "
        "sin viajes en el tramo.",
        f"Inventario inicial: {report['inventario_inicial']['bicis']:,} bicis en "
        f"{report['inventario_inicial']['lugares']:,} lugares (supuesto).",
        f'La política "model" se alimenta de: {model}.'
        + (" Es la línea base: LightGBM todavía no existe." if model == "baseline" else ""),
    ]
    if report["reglas"]["estado"] != "confirmado":
        lines.append(f"Reglas de operación PROVISIONALES ({operacion}).")
    lines += [
        "",
        f"Horas de estación en riesgo por día evaluado (intervalo del {level}, indicativo: los días no son "
        "independientes):",
        _table(
            ["política", "bicis al día", "en riesgo por día", "intervalo", "riesgo simulado evitado",
             "bloqueados por día", "movidas por día"],
            [
                [
                    POLICY_LABELS[row["politica"]],
                    "cualquiera" if row["politica"] == "none" else f"{row['presupuesto_del_dia_en_bicis']:,}",
                    _n(row["riesgo_por_dia"]["media"]),
                    f"{_n(row['riesgo_por_dia']['ic_bajo'])} a {_n(row['riesgo_por_dia']['ic_alto'])}",
                    _n(row["riesgo_simulado_evitado_por_dia"]["media"]),
                    _n(row["viajes_bloqueados"]["por_dia"]),
                    _n(row["bicis_movidas"]["por_dia"]),
                ]
                for row in _shown(report)
            ],
        ),
        "",
        "Los viajes bloqueados son simulados (posible demanda insatisfecha), no demanda medida.",
    ]
    if not report["prueba_final"]["simulada"]:
        lines.append("La prueba final no se simuló (decisión 4).")
    return lines


# --------------------------------------------------------------------------- #
# Lectura de archivos
# --------------------------------------------------------------------------- #

def _short(path: Path) -> str:
    """Ruta relativa a la raíz del repositorio, para mensajes más cortos."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _instant(value: pd.Timestamp) -> pa.Scalar:
    return pa.scalar(value.to_pydatetime(), type=TS)


def load_trips(path: Path, tramo: Tramo) -> pd.DataFrame:
    """Viajes con salida o llegada dentro del tramo. Del resto del archivo no se lee nada."""
    schema = pq.read_schema(path)
    missing = [column for column in TRIP_COLUMNS if column not in schema.names]
    if missing:
        raise SimulacionError(f"A {_short(path)} le faltan columnas: {missing}.")
    start, end = _instant(tramo.start), _instant(tramo.end)
    inside = lambda column: (pc.field(column) >= start) & (pc.field(column) < end)  # noqa: E731
    table = pq.read_table(path, columns=list(TRIP_COLUMNS), filters=inside("started_at") | inside("ended_at"))
    return table.to_pandas()


def load_history(path: Path, first: Tramo, tramo: Tramo) -> pd.DataFrame:
    """Filas de `station_hour` anteriores al tramo simulado. Las del tramo y las posteriores no se leen."""
    start, end = _instant(first.start), _instant(tramo.start)
    table = pq.read_table(
        path,
        columns=["station_id", "ts_hour", "net_flow"],
        filters=(pc.field("ts_hour") >= start) & (pc.field("ts_hour") < end),
    )
    return table.to_pandas()


def load_forecast(path: Path, tramo: Tramo, split: str) -> tuple[pd.DataFrame, str]:
    """Pronóstico del tramo: "lgbm" si tiene filas, si no "baseline". Devuelve las filas y cuál fue."""
    start, end = _instant(tramo.start), _instant(tramo.end)
    table = pq.read_table(
        path,
        filters=(pc.field("split") == split) & (pc.field("ts_hour") >= start) & (pc.field("ts_hour") < end),
    )
    present = set(pc.unique(table["model"]).to_pylist())
    for model in MODEL_PREFERENCE:
        if model in present:
            return table.filter(pc.equal(table["model"], model)).to_pandas(), model
    raise SimulacionError(
        f"{_short(path)} no trae pronóstico de {list(MODELS)} con split={split!r} para el tramo. "
        "Primero corre: python scripts/run_baseline.py"
    )


def _check_built_period(data_dir: Path, datos, config_path: str) -> None:
    """Falla si build_tables construyó las tablas con otro periodo (lo deja en reports/limpieza.json)."""
    cleaning = data_dir / "reports" / "limpieza.json"
    if not cleaning.exists():
        return
    try:
        built = json.loads(cleaning.read_text(encoding="utf-8"))["config"]["periodo"]
        built = (str(built["inicio"]), str(built["fin"]))
    except (ValueError, KeyError, TypeError):
        raise SimulacionError(
            f"No se pudo leer el periodo en {_short(cleaning)}. Vuelve a correr: python scripts/build_tables.py"
        ) from None
    mine = (datos.periodo.inicio.isoformat(), datos.periodo.fin.isoformat())
    if built != mine:
        raise SimulacionError(
            f"Las tablas de {_short(data_dir)} se construyeron con el periodo {built[0]} a {built[1]} y "
            f"{config_path} dice {mine[0]} a {mine[1]}. Primero corre: python scripts/build_tables.py"
        )


def read_inputs(data_dir: Path, datos, tramo: Tramo) -> tuple:
    """Las cuatro tablas tal como las lee la simulación, y qué pronóstico alimenta a "model".

    Devuelve stations, los viajes del tramo, la historia anterior al tramo, el pronóstico del tramo y su modelo.
    """
    stations = read_contract("stations", data_dir / "stations.parquet")
    trips = load_trips(data_dir / TRIPS_FILE, tramo)
    history = load_history(data_dir / "station_hour.parquet", datos.tramos[TRAMOS[0]], tramo)
    predictions, model = load_forecast(data_dir / "predictions.parquet", tramo, SPLIT_NAMES[tramo.nombre])
    return stations, trips, history, predictions, model


# --------------------------------------------------------------------------- #
# ¿Las salidas siguen al día?
# --------------------------------------------------------------------------- #

RERUN = "Vuelve a correr: python scripts/run_simulation.py"
# Partes de la huella, con las palabras que se le muestran a quien corre las pruebas.
FINGERPRINT_PARTS = {
    "reglas": "las reglas de config/operacion.yaml",
    "simulacion": "el bloque `simulacion` de config/datos.yaml",
    "fechas": "las fechas de los tramos en config/datos.yaml",
    "stations": "stations.parquet",
    "trips": "trips.parquet (los viajes del tramo simulado)",
    "station_hour": "station_hour.parquet (la historia anterior al tramo simulado)",
    "predictions": "predictions.parquet (el pronóstico del tramo simulado)",
}


def _control(columns: dict) -> dict:
    """Cuántas filas trae una tabla y un número de control de su contenido. No depende del orden de las filas."""
    frame = pd.DataFrame(columns)
    hashes = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy()
    return {"filas": int(len(frame)), "control": int(hashes.sum())}


def _text(column: pd.Series, keep=None) -> np.ndarray:
    """Columna de texto lista para el número de control: vacío donde no hay valor o donde `keep` es falso."""
    values = column.astype("object").where(column.notna(), "")
    return (values if keep is None else values.where(keep, "")).to_numpy(dtype=object)


def input_fingerprint(
    stations: pd.DataFrame,
    trips: pd.DataFrame,
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    forecast_model: str,
    reglas: Reglas,
    datos,
    tramo: Tramo,
) -> dict:
    """Huella de lo que lee una simulación. Sirve para saber si sus salidas siguen al día.

    Guarda las reglas de operación que usa el simulador, el bloque `simulacion`, las fechas y, de cada
    tabla, cuántas filas se leyeron y un número de control de su contenido:

    - stations: todas las filas, con las columnas que usa el simulador (sin el nombre).
    - trips: los viajes con salida o llegada en el tramo. De la punta que cae fuera del tramo no entra ni
      la hora ni la estación, porque la simulación tampoco las usa. De un viaje que termina después del
      tramo no se guarda nada de su llegada.
    - station_hour: la historia anterior al tramo (estación, hora y flujo neto).
    - predictions: las filas del tramo del modelo que alimenta a "model".

    De la prueba final solo van las fechas (decisión 4), salvo que el tramo simulado sea ella misma.
    """
    start_us, end_us = (int(value) for value in pd.DatetimeIndex([tramo.start, tramo.end]).as_unit("us").asi8)
    started, ended = _epoch_us(trips["started_at"]), _epoch_us(trips["ended_at"])
    departs = (started >= start_us) & (started < end_us)
    arrives = (ended >= start_us) & (ended < end_us)
    final = datos.tramos[FINAL_TRAMO]
    return {
        "reglas": dataclasses.asdict(reglas),
        "simulacion": {
            name: getattr(datos.simulacion, name)
            for name in ("dias_de_calentamiento", "remuestreos", "nivel_del_intervalo", "semilla")
        },
        "fechas": {
            "tramo": tramo.nombre,
            "inicio": tramo.inicio.isoformat(),
            "fin": tramo.fin.isoformat(),
            "historia_desde": datos.tramos[TRAMOS[0]].inicio.isoformat(),
            "prueba_final": {"inicio": final.inicio.isoformat(), "fin": final.fin.isoformat()},
        },
        "stations": _control(
            {
                "station_id": _text(stations["station_id"]),
                "lat": stations["lat"].to_numpy(dtype="float64"),
                "lon": stations["lon"].to_numpy(dtype="float64"),
                "capacity": stations["capacity"].astype("float64").to_numpy(),  # sin capacidad: NaN
                "station_type": _text(stations["station_type"]),
            }
        ),
        "trips": _control(
            {
                "ride_id": _text(trips["ride_id"]),
                "started_at": np.where(departs, started, -1),
                "ended_at": np.where(arrives, ended, -1),
                "start_station_id": _text(trips["start_station_id"], departs),
                "end_station_id": _text(trips["end_station_id"], arrives),
            }
        ),
        "station_hour": _control(
            {
                "station_id": _text(history["station_id"]),
                "ts_hour": _epoch_us(history["ts_hour"]),
                "net_flow": history["net_flow"].to_numpy(dtype="int64"),
            }
        ),
        "predictions": {
            "modelo": forecast_model,
            **_control(
                {
                    "station_id": _text(predictions["station_id"]),
                    "ts_hour": _epoch_us(predictions["ts_hour"]),
                    **{column: predictions[column].to_numpy(dtype="float64") for column in QUANTILE_COLUMNS},
                }
            ),
        },
    }


def current_fingerprint(
    data_dir: str | Path = DATA_PROCESSED,
    operacion: str | Path = OPERACION_YAML,
    config: str | Path = DATOS_YAML,
    tramo: str = "validacion",
) -> dict:
    """La huella que tendría una corrida hecha ahora con estas tablas, estas reglas y este config.

    Lee lo mismo que lee la simulación y nada más. No acepta la prueba final (decisión 4).
    """
    datos = load_datos(config)
    chosen = datos.tramos[tramo]
    check_final_test(chosen, datos.tramos[FINAL_TRAMO], usar_prueba_final=False)
    if datos.simulacion is None:
        raise SimulacionError(f"{config} no trae el bloque `simulacion` (calentamiento y remuestreo).")
    reglas = Reglas.from_operacion(load_operacion(operacion))
    stations, trips, history, predictions, model = read_inputs(Path(data_dir), datos, chosen)
    return input_fingerprint(stations, trips, history, predictions, model, reglas, datos, chosen)


def stale_inputs(stored: dict | None, current: dict) -> list[str]:
    """Qué entradas ya no son las de la corrida que guardó la huella `stored`. Lista vacía: siguen al día."""
    if not isinstance(stored, dict):
        return ["la huella de las entradas (simulacion.json no la trae: lo escribió una versión anterior)"]
    current = json.loads(json.dumps(current))  # como queda al guardarse: las tuplas pasan a listas
    return [label for part, label in FINGERPRINT_PARTS.items() if stored.get(part) != current[part]]


def stale_message(changed: list[str]) -> str:
    """El aviso para quien corre las pruebas: qué cambió y qué comando volver a correr."""
    return (
        "Las recomendaciones, la curva y el reporte de la simulación no salieron de las reglas o las tablas "
        f"de hoy. Cambió: {'; '.join(changed)}. {RERUN} (si también falla la revisión de la línea base, corre "
        "antes: python scripts/run_baseline.py)"
    )


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_simulation.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python scripts/run_simulation.py [opciones]",
        description=(
            "Simulación de inventario: sin intervención, regla histórica y modelo, con varios presupuestos.\n"
            "Escribe recommendations.parquet, policy_curve.parquet y reports/simulacion.json y .md.\n"
            "Simula el tramo de VALIDACIÓN. La prueba final no se simula (decisión 4)."
        ),
    )
    options = parser.add_argument_group("opciones")
    options.add_argument(
        "--datos", metavar="DIR", default=str(DATA_PROCESSED),
        help=(
            "carpeta con stations.parquet, trips.parquet, station_hour.parquet y predictions.parquet "
            "(por defecto data/processed)"
        ),
    )
    options.add_argument(
        "--salida", metavar="DIR",
        help="carpeta donde se escriben las tablas y, en reports/, el reporte (por defecto, la misma de --datos)",
    )
    options.add_argument(
        "--operacion", metavar="FILE", default=str(OPERACION_YAML),
        help="archivo con las reglas de operación (por defecto config/operacion.yaml)",
    )
    options.add_argument(
        "--config", metavar="FILE", default=str(DATOS_YAML),
        help="archivo con los tramos y los ajustes de la simulación (por defecto config/datos.yaml)",
    )
    options.add_argument(
        "--tramo", choices=[name for name in TRAMOS if name in SPLIT_NAMES], default="validacion",
        help="tramo que se simula (por defecto validacion). 'prueba' se rechaza sin --usar-prueba-final",
    )
    options.add_argument(
        "--usar-prueba-final", action="store_true",
        help=(
            "permite --tramo prueba. Decisión 4: la prueba final se usa una sola vez. "
            f"Escribe en la subcarpeta {FINAL_FOLDER}/ de --salida"
        ),
    )
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve 0 si todo salió bien y 1 si no se pudo correr."""
    args = build_parser().parse_args(argv)
    started = time.time()
    data_dir = Path(args.datos).expanduser()
    out_dir = Path(args.salida).expanduser() if args.salida else data_dir
    try:
        datos = load_datos(args.config)
        tramo = datos.tramos[args.tramo]
        # El candado va antes de abrir cualquier archivo de datos.
        check_final_test(tramo, datos.tramos[FINAL_TRAMO], args.usar_prueba_final)
        if tramo.nombre == FINAL_TRAMO:
            out_dir = out_dir / FINAL_FOLDER
        if getattr(datos, "simulacion", None) is None:
            raise SimulacionError(f"{args.config} no trae el bloque `simulacion` (calentamiento y remuestreo).")
        try:
            reglas = Reglas.from_operacion(load_operacion(args.operacion))
        except OSError as error:  # el archivo de reglas no existe o no se puede leer
            raise SimulacionError(f"No se pudo abrir {args.operacion}: {error}") from None

        paths = {name: data_dir / f"{name}.parquet" for name in ("stations", "station_hour", "predictions")}
        paths["trips"] = data_dir / TRIPS_FILE
        for name, path in paths.items():
            if not path.exists():
                step = "run_baseline.py" if name == "predictions" else "build_tables.py"
                raise SimulacionError(f"No existe {_short(path)}. Primero corre: python scripts/{step}")
        _check_built_period(data_dir, datos, args.config)
        problems = validate_file(paths["stations"], "stations")
        if problems:
            raise SimulacionError(f"{_short(paths['stations'])} no cumple su contrato:\n- " + "\n- ".join(problems))

        stations, trips, history, predictions, model = read_inputs(data_dir, datos, tramo)
        print(f'La política "model" se alimenta del pronóstico: {model}.', file=sys.stderr)
        result = run_simulation(
            stations, trips, history, predictions, reglas, tramo, datos.simulacion,
            forecast_model=model,
            prueba_final=datos.tramos[FINAL_TRAMO],
            usar_prueba_final=args.usar_prueba_final,
            progress=lambda message: print(f"  corriendo: {message}", file=sys.stderr),
        )
        # La huella de lo que se leyó: con ella una prueba sabe si estas salidas siguen al día.
        fingerprint = input_fingerprint(stations, trips, history, predictions, model, reglas, datos, tramo)
        seconds = time.time() - started
        operacion = _short(Path(args.operacion))
        document = {
            "corrida": {
                "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "segundos": round(seconds, 1),
                "datos": _short(data_dir),
                "operacion": operacion,
                "config": _short(Path(args.config)),
                "huella": fingerprint,
            },
            **result.report,
        }
        written = [
            write_contract(result.recommendations, "recommendations", out_dir / "recommendations.parquet"),
            write_contract(result.policy_curve, "policy_curve", out_dir / "policy_curve.parquet"),
        ]
        reports = out_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        written.append(reports / f"{REPORT_NAME}.json")
        written[-1].write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(reports / f"{REPORT_NAME}.md")
        written[-1].write_text(build_report(result.report, operacion), encoding="utf-8")
    except FinalTestError as error:
        print(f"ERROR. {error}")
        return 1
    except (ValueError, OSError) as error:  # config, reglas, contrato o archivo ilegible
        print(f"ERROR. {error}")
        return 1

    print("\n".join(summary_lines(result.report, operacion)))
    print("\nEscrito:")
    for path in written:
        print(f"  {_short(path)}")
    print(f"Listo en {seconds:.1f} s.")
    return 0
