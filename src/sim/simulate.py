"""Simulación de inventario de punta a punta: corridas, métricas, tablas y reporte.

Compara las políticas (sin intervención, regla histórica y modelo) con varios
presupuestos sobre el tramo de validación (decisiones 7 y 8). Todas parten del
mismo inventario y enfrentan los mismos viajes. El motor está en
`src/sim/engine.py` y las políticas en `src/sim/policies.py`.

La regla histórica y el modelo usan la misma maquinaria; solo cambia su
pronóstico: las filas `model = "baseline"` de `predictions` para la regla
histórica y las filas `model = "lgbm"` para el modelo. **Si no hay filas
"lgbm", la política "model" no se simula**: las tablas salen sin filas de
"model" y el reporte lo dice.

Entradas (carpeta `--datos`): stations.parquet, trips.parquet (tabla interna
que escribe build_tables, con las coordenadas de cada punta),
station_hour.parquet (solo qué estaciones tienen historia anterior al tramo) y
predictions.parquet. Además `config/gbfs_congelado.csv` (`--gbfs`): la
ocupación observada en la captura de GBFS, que da el inventario inicial.

Salidas (carpeta `--salida`): recommendations.parquet y policy_curve.parquet
(contratos, escritos con `write_contract`), reports/simulacion.json y
reports/simulacion.md.

Orden de la corrida:

1. Sin intervención.
2. Banda de incertidumbre: con el presupuesto de referencia se simula cada
   multiplicador candidato (`simulacion.bandas` de config/datos.yaml) con cada
   política que tiene pronóstico. Se elige el de menor riesgo simulado medio,
   promediado entre esas políticas. Se usa para todo lo demás.
3. Curva: los demás presupuestos, con la banda elegida.
4. Sensibilidad: la corrida de referencia con otros inventarios iniciales.

simulacion.json guarda una huella de lo que se leyó (`corrida.huella`: reglas,
bloque `simulacion`, fechas, un número de control de cada tabla y del archivo
congelado de GBFS, y la banda elegida). Con ella `pytest` avisa, con el
comando que hay que correr, cuando las salidas ya no corresponden a las tablas,
las reglas o el config de hoy (`current_fingerprint` y `stale_inputs`). La
huella no guarda nada de la prueba final, salvo sus fechas.

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

from src.contracts import POLICIES, QUANTILE_COLUMNS, RISK_LEVELS, TS, read_contract, validate_file, write_contract
from src.data.datos_config import SPLIT_NAMES, TRAMOS, Tramo, load_datos
from src.data.gbfs_frozen import FROZEN_FILE, GbfsFrozenError, load_frozen
from src.models.metrics import FinalTestError
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML, ROOT, TZ
from src.sim.engine import (
    EPS,
    KINDS,
    POINT_COLUMNS,
    TRIP_COLUMNS,
    Calendar,
    CheckError,
    InterventionContext,
    Network,
    Reglas,
    RunResult,
    SimulacionError,
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
    ForecastPolicy,
    ForecastWindow,
    NoIntervention,
    safe_range,
    stations_with_history,
)

REPORT_NAME = "simulacion"
TRIPS_FILE = "trips.parquet"
FINAL_TRAMO = "prueba"
FINAL_FOLDER = "prueba_final"
POLICY_LABELS = {"none": "sin intervención", "historical": "regla histórica", "model": "modelo"}
TRAMO_LABELS = {"entrenamiento": "entrenamiento", "validacion": "validación", "prueba": "prueba final"}
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
KIND_LABELS = {"pocas_bicis": "pocas bicis", "pocos_lugares": "pocos lugares libres"}
# Qué filas de `predictions` alimentan a cada política con pronóstico. Es lo ÚNICO que las distingue.
FORECAST_OF = {"historical": "baseline", "model": "lgbm"}
OBSERVED = "observado en GBFS"
NOT_SIMULATED = "ninguno"  # lo que dice el reporte del pronóstico de "model" cuando no se simuló
FULL_BAND = 1.0  # la banda completa, de q10 a q90: la que separa el riesgo "medio" del "bajo"


@dataclass(frozen=True)
class Resultado:
    """Lo que produce una simulación completa."""

    recommendations: pd.DataFrame
    policy_curve: pd.DataFrame
    report: dict
    runs: dict          # (política, presupuesto del día en bicis) -> RunResult, con la banda elegida
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
# Conjunto simulado e inventario inicial
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


def observed_occupancy(network: Network, frozen: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Ocupación inicial de cada estación simulada, del archivo congelado de GBFS.

    `frozen` trae short_name, capacity y ocupacion_inicial. Una estación que no
    está en el archivo, o que está sin ocupación, usa la media de la ocupación
    observada en las estaciones simuladas que sí la tienen (respaldo declarado).
    Falla si la capacidad del archivo no es la de `stations`: las tablas se
    construyeron con otro archivo congelado.
    """
    table = frozen.drop_duplicates("short_name").set_index("short_name")
    ids = list(network.station_ids)
    capacity = pd.to_numeric(table["capacity"].reindex(ids), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    occupancy = pd.to_numeric(table["ocupacion_inicial"].reindex(ids), errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    in_file = np.isfinite(capacity)
    differ = in_file & (capacity != network.capacity)
    if differ.any():
        first = int(np.nonzero(differ)[0][0])
        raise SimulacionError(
            f"La capacidad de {int(differ.sum()):,} estaciones en stations.parquet no es la del archivo congelado "
            f"de GBFS (por ejemplo {ids[first]}: {int(network.capacity[first])} contra {int(capacity[first])}). "
            "Primero corre: python scripts/build_tables.py"
        )
    known = np.isfinite(occupancy)
    if not known.any():
        raise SimulacionError(
            "El archivo congelado de GBFS no trae la ocupación inicial de ninguna estación simulada. "
            "Revisa config/gbfs_congelado.csv (se crea con: python scripts/freeze_gbfs.py)."
        )
    mean = float(occupancy[known].mean())
    filled = np.clip(np.where(known, occupancy, mean), 0.0, 1.0)
    return filled, {
        "estaciones_con_ocupacion_observada": int(known.sum()),
        "estaciones_con_la_media": int((~known).sum()),
        "de_ellas_fuera_del_archivo": int((~in_file).sum()),
        "ocupacion_media_observada": mean,
    }


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

def risk_levels(
    before, network: Network, reglas: Reglas, window: ForecastWindow, full: ForecastWindow | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Nivel de riesgo y `feasible` de cada estación en una franja.

    `window` es el pronóstico de la franja con la banda con la que decide la
    política; `full`, el mismo pronóstico con la banda completa (de q10 a q90).
    Si no se da `full` se usa `window`.

    Nivel, con el inventario ANTES de intervenir: "alto" si la trayectoria
    central cruza un umbral o el inventario ya está bajo uno; "medio" si solo
    lo cruza la banda completa; "bajo" en otro caso. Una estación sin
    información es "alto" si ya está bajo un umbral y "bajo" si no. El nivel
    no depende de la banda elegida: así "medio" sigue existiendo cuando la
    banda elegida es 0.

    `feasible` es falso cuando ningún inventario evita los dos umbrales con la
    banda de la política (lo > hi); una estación sin información queda en
    verdadero.
    """
    full = full or window
    before = np.asarray(before, dtype=np.int64)
    low, high = safe_range(network, reglas, window.lower_dip, window.upper_rise)
    low_full, high_full = safe_range(network, reglas, full.lower_dip, full.upper_rise)
    low_center, high_center = safe_range(network, reglas, window.center_dip, window.center_rise)
    info = window.has_info
    under = (before < reglas.bicis_minimas) | (network.capacity - before < reglas.lugares_libres_minimos)
    center = info & ((before < low_center - EPS) | (before > high_center + EPS))
    envelope = info & ((before < low_full - EPS) | (before > high_full + EPS))
    level = np.where(under | center, "alto", np.where(envelope, "medio", "bajo"))
    return level, ~info | (low <= high + EPS)


def build_recommendations(
    runs: dict, budget: int, network: Network, calendar: Calendar, reglas: Reglas, forecasts: dict, band: float
) -> pd.DataFrame:
    """Una fila por estación, franja, día evaluado y política simulada, con el presupuesto de referencia.

    `forecasts` trae el pronóstico de cada política que lo tiene. El nivel de
    riesgo y `feasible` de cada política salen de SU pronóstico (`feasible`
    con la banda elegida; el nivel "medio", con la banda completa); los de
    "none", del pronóstico de la regla histórica.
    """
    ids = np.array(network.station_ids, dtype=object)
    parts = []
    for policy in POLICIES:
        if (policy, budget) not in runs:
            continue
        forecast = forecasts.get(policy, forecasts["historical"])
        for record in runs[(policy, budget)].interventions:
            d, f = record.day_index, record.franja_index
            if d < calendar.warmup_days:
                continue
            day, franja = calendar.days[d], reglas.franjas[f].nombre
            level, feasible = risk_levels(
                record.inventory_before, network, reglas,
                forecast.window(day, franja, band), forecast.window(day, franja, FULL_BAND),
            )
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
    out = {
        "filas": int(len(table)),
        "estacion_franja": int(len(table.drop_duplicates(["station_id", "window_start"]))),
        "por_politica": {},
    }
    for policy in POLICIES:
        rows = table[table["policy"] == policy]
        if rows.empty:
            continue
        bikes, unsolved = rows["bikes_to_move"], ~rows["feasible"]
        levels = rows["risk_level"].value_counts()
        out["por_politica"][policy] = {
            "estacion_franja_sin_solucion": int(unsolved.sum()),
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


def daily_risk(result: RunResult, calendar: Calendar) -> np.ndarray:
    """Horas de estación en riesgo de cada día evaluado."""
    return result.risk_hours[calendar.evaluated][:, :, 0].sum(axis=1)


def summarize(result: RunResult, none: RunResult, calendar: Calendar, reglas: Reglas, draws, nivel: float) -> dict:
    """Métricas de una corrida sobre los días evaluados (sin el calentamiento)."""
    days = calendar.evaluated

    def stat(values) -> dict:
        mean, low, high = interval(values, draws, nivel)
        return {"media": mean, "ic_bajo": low, "ic_alto": high}

    hours = result.risk_hours[days]
    total = hours[:, :, 0].sum(axis=1)
    base = daily_risk(none, calendar)
    moved = result.budget_used[days].sum(axis=1)
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
        "riesgo_por_franja_de_cada_dia": {
            label: [float(v) for v in hours[:, i, 0]] for i, label in enumerate(calendar.labels)
        },
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
        "bicis": {
            "total_al_inicio": int(result.initial_total),
            "en_estaciones_al_final": int(result.final_docked),
            "en_transito_al_final": int(result.final_in_transit),
            "total_al_final": int(result.final_total),
            "en_transito_al_final_de_cada_dia": [int(v) for v in result.in_transit],
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


def _missing_info(calendar: Calendar, reglas: Reglas, with_history: np.ndarray, forecasts: dict) -> dict:
    """Cuántas estaciones y cuántas estación-franja quedan sin información en los días evaluados, por política."""
    n = len(with_history)
    windows = len(calendar.evaluated) * len(reglas.franjas)
    out = {"estaciones_sin_historia": int((~with_history).sum()), "franjas_evaluadas_por_estacion": int(windows)}
    for policy, forecast in forecasts.items():
        without = np.zeros(n, dtype=np.int64)
        gap = np.zeros(n, dtype=np.int64)  # con historia y sin pronóstico de alguna hora de la franja
        for d in calendar.evaluated:
            for franja in reglas.franjas:
                missing = ~forecast.window(calendar.days[d], franja.nombre, 0.0).has_info
                without += missing
                gap += with_history & missing
        out[policy] = {
            "estacion_franja_sin_informacion": int(without.sum()),
            "estaciones_sin_informacion_en_alguna_franja": int((without > 0).sum()),
            "estaciones_sin_informacion_en_todas": int((without == windows).sum()),
            # `predictions` solo trae filas hasta la hora del último viaje de cada estación: desde ahí la
            # política deja de tener información de una estación que sí tiene historia.
            "con_historia_y_sin_pronostico": {
                "estacion_franja": int(gap.sum()),
                "estaciones": int((gap > 0).sum()),
            },
        }
    return out


def compare_policies(runs: dict, budget: int, policies: dict, calendar: Calendar, reglas: Reglas) -> dict:
    """Qué pidió y qué movió cada política con un presupuesto, en los días evaluados. Solo cuenta.

    Vuelve a preguntar a cada política por su rango en cada intervención, con el inventario que vio en ella
    (`targets` no mueve nada ni cambia la corrida). "Sin solución" es con el rango de la propia política.

    - bicis que faltan y que sobran: lo que le falta a cada estación para llegar al mínimo de su rango y lo que
      le sobra por encima del máximo, sumado sobre las estaciones y las intervenciones del día.
    - estación-franja con traslado, bicis recibidas y entregadas, y cuántas de ellas tocan una estación-franja
      sin solución.
    """
    days = len(calendar.evaluated)
    out = {}
    for name, policy in policies.items():
        count = dict.fromkeys(
            ("con_informacion", "sin_solucion", "sin_solucion_sin_tipo", "faltan", "sobran", "con_traslado",
             "recibidas", "entregadas", "recibidas_sin_solucion", "entregadas_sin_solucion"), 0,
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
            count["sin_solucion_sin_tipo"] += int((unsolved & ~policy.network.typed).sum())
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
            "de_ellas_sin_tipo_no_se_tocan": count["sin_solucion_sin_tipo"],
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


def choose_band(candidates, mean_risk: dict) -> float:
    """El multiplicador con menor riesgo medio entre políticas. Si empatan, el que está antes en la lista.

    `mean_risk[banda][política]` es el riesgo simulado medio por día evaluado.
    """
    def score(band: float) -> float:
        return float(np.mean(list(mean_risk[band].values())))

    return min(candidates, key=score)  # `min` devuelve el primero de los empatados


def run_simulation(
    stations: pd.DataFrame,
    trips: pd.DataFrame,
    history: pd.DataFrame,
    forecasts: dict,
    ocupacion: pd.DataFrame,
    reglas: Reglas,
    tramo: Tramo,
    ajustes,
    *,
    prueba_final: Tramo | None = None,
    usar_prueba_final: bool = False,
    progress=None,
) -> Resultado:
    """Corre las políticas con todos los presupuestos y arma tablas y reporte.

    - `trips`: viajes con salida o llegada en el tramo (TRIP_COLUMNS y, si las hay, POINT_COLUMNS).
    - `history`: filas de `station_hour` ANTERIORES al tramo (falla si llega una posterior). Solo se usa
      para saber qué estaciones tienen historia.
    - `forecasts`: por política ("historical", "model"), las filas de `predictions` del tramo que la
      alimentan. "historical" es obligatoria. Sin "model", esa política no se simula.
    - `ocupacion`: el archivo congelado de GBFS (short_name, capacity, ocupacion_inicial).
    - `ajustes`: el bloque `simulacion` de config/datos.yaml (`load_datos().simulacion`).
    """
    check_final_test(tramo, prueba_final, usar_prueba_final)
    say = progress or (lambda message: None)
    unknown = [name for name in forecasts if name not in FORECAST_OF]
    if unknown or "historical" not in forecasts:
        raise SimulacionError(
            'Hace falta el pronóstico de la regla histórica (filas "baseline" de predictions) y solo se aceptan '
            f"{list(FORECAST_OF)}. Primero corre: python scripts/run_baseline.py"
        )
    names = [name for name in FORECAST_OF if name in forecasts and len(forecasts[name])]
    if "historical" not in names:
        raise SimulacionError(
            'predictions no trae filas "baseline" del tramo. Primero corre: python scripts/run_baseline.py'
        )
    bands = tuple(float(band) for band in ajustes.bandas)
    calendar = build_calendar(
        tramo.inicio, tramo.fin, reglas.franjas, int(ajustes.dias_de_calentamiento), tuple(ajustes.franja_noche)
    )
    chosen, station_report = select_stations(stations, trips, calendar.start_us, calendar.end_us)
    network = build_network(chosen)
    n = network.size
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    occupancy, occupancy_report = observed_occupancy(network, ocupacion)
    start = initial_inventory(network, occupancy)
    with_history = stations_with_history(history, network, before=tramo.start)
    info = {
        name: ForecastInfo.from_predictions(forecasts[name], network, reglas, calendar.days, with_history)
        for name in names
    }
    neighbors = network.neighbors_within(reglas.distancia_maxima_km)

    def policy(name: str, band: float) -> ForecastPolicy:
        return ForecastPolicy(name, network, reglas, info[name], band, neighbors)

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

    simulated = 1
    say(f"sin intervención ({len(events):,} eventos)")
    none = simulate(network, events, calendar, reglas, NoIntervention(), 0, start)
    if int(none.budget_used.sum()) or any(record.moves for record in none.interventions):
        raise CheckError('La política "none" movió bicis.')
    base_risk = daily_risk(none, calendar)

    # Banda de incertidumbre: todos los candidatos con el presupuesto de referencia; gana el de menor riesgo
    # simulado medio, promediado entre las políticas con pronóstico.
    by_band: dict[float, dict[str, RunResult]] = {}
    for band in bands:
        by_band[band] = {}
        for name in names:
            say(f"banda {band:g}: {POLICY_LABELS[name]}, {reference:,} bicis al día")
            by_band[band][name] = simulate(network, events, calendar, reglas, policy(name, band), reference, start)
            simulated += 1
    mean_risk = {
        band: {name: float(daily_risk(result, calendar).mean()) for name, result in results.items()}
        for band, results in by_band.items()
    }
    band = choose_band(bands, mean_risk)
    policies = {name: policy(name, band) for name in names}

    runs: dict[tuple[str, int], RunResult] = {}
    for total in budgets:
        runs[("none", total)] = _as_unused(none, "none", total, reglas)
        for name in names:
            if total == 0:
                runs[(name, total)] = _as_unused(none, name, total, reglas)
            elif total == reference:
                runs[(name, total)] = by_band[band][name]
            else:
                say(f"{POLICY_LABELS[name]}, {total:,} bicis al día")
                runs[(name, total)] = simulate(network, events, calendar, reglas, policies[name], total, start)
                simulated += 1

    # Entradas iguales dan salidas iguales: se repite una corrida y se compara la huella.
    say("repetición para comprobar que el resultado no cambia")
    again = simulate(network, events, calendar, reglas, policies[names[0]], reference, start)
    if again.digest() != runs[(names[0], reference)].digest():
        raise CheckError("Dos corridas con las mismas entradas dieron resultados distintos.")
    every = [none, *runs.values(), *(result for results in by_band.values() for result in results.values())]
    if len({result.initial_total for result in every}) != 1:
        raise CheckError("Las políticas no partieron del mismo inventario.")
    if any(result.final_total != result.initial_total for result in every):
        raise CheckError("El total de bicis (en estaciones más en tránsito) cambió en alguna corrida.")

    # Sensibilidad al inventario inicial: la corrida de referencia con todas las estaciones a la misma ocupación.
    def sensitivity_row(label: str, mean_occupancy: float, base: RunResult, results: dict) -> dict:
        without = daily_risk(base, calendar)
        return {
            "inventario_inicial": label,
            "ocupacion_media": float(mean_occupancy),
            "bicis": int(base.initial_total),
            "riesgo_sin_intervencion_por_dia": float(without.mean()),
            "por_politica": {
                name: {
                    "riesgo_por_dia": float(daily_risk(result, calendar).mean()),
                    "riesgo_simulado_evitado_por_dia": float((without - daily_risk(result, calendar)).mean()),
                    "viajes_bloqueados_por_dia": float(result.blocked[calendar.evaluated].mean()),
                }
                for name, result in results.items()
            },
        }

    sensitivity = [
        sensitivity_row(OBSERVED, float(start.sum() / network.capacity.sum()), none,
                        {name: runs[(name, reference)] for name in names})
    ]
    for flat in ajustes.sensibilidad_ocupacion_inicial:
        other = initial_inventory(network, float(flat))
        say(f"sensibilidad: todas las estaciones al {float(flat) * 100:g} % de su capacidad")
        other_none = simulate(network, events, calendar, reglas, NoIntervention(), 0, other)
        other_runs = {
            name: simulate(network, events, calendar, reglas, policies[name], reference, other) for name in names
        }
        simulated += 1 + len(names)
        for result in (other_none, *other_runs.values()):
            if result.final_total != result.initial_total:
                raise CheckError("El total de bicis cambió en una corrida de sensibilidad.")
        sensitivity.append(
            sensitivity_row(f"todas al {float(flat) * 100:g} %", float(other.sum() / network.capacity.sum()),
                            other_none, other_runs)
        )

    evaluated = calendar.evaluated
    draws = resample_days(len(evaluated), int(ajustes.remuestreos), int(ajustes.semilla))
    nivel = float(ajustes.nivel_del_intervalo)
    present = [name for name in POLICIES if name == "none" or name in names]
    results = []
    curve = []
    for name in present:
        for total, per_station in budgets.items():
            summary = summarize(runs[(name, total)], none, calendar, reglas, draws, nivel)
            results.append(
                {
                    "politica": name,
                    "presupuesto_por_estacion": per_station,
                    "presupuesto_del_dia_en_bicis": total,
                    "simulada": bool(name == "none" or total > 0),
                    **summary,
                }
            )
            risk = summary["riesgo_por_dia"]
            curve.append(
                {
                    "policy": name,
                    "budget": total,
                    "risk_station_hours": risk["media"],
                    "ci_low": risk["ic_bajo"],
                    "ci_high": risk["ic_alto"],
                }
            )
    policy_curve = pd.DataFrame(curve)
    recommendations = build_recommendations(runs, reference, network, calendar, reglas, info, band)
    recommended = summarize_recommendations(recommendations)

    def band_row(candidate: float) -> dict:
        per_policy = {}
        for name, result in by_band[candidate].items():
            risk = daily_risk(result, calendar)
            mean, low, high = interval(base_risk - risk, draws, nivel)
            unsolved = 0
            for d in evaluated:
                for franja in reglas.franjas:
                    window = info[name].window(calendar.days[d], franja.nombre, candidate)
                    low_edge, high_edge = safe_range(network, reglas, window.lower_dip, window.upper_rise)
                    unsolved += int((window.has_info & (low_edge > high_edge + EPS)).sum())
            per_policy[name] = {
                "riesgo_por_dia": float(risk.mean()),
                "riesgo_simulado_evitado_por_dia": {"media": mean, "ic_bajo": low, "ic_alto": high},
                "viajes_bloqueados_por_dia": float(result.blocked[evaluated].mean()),
                "bicis_movidas_por_dia": float(result.budget_used[evaluated].sum(axis=1).mean()),
                "estacion_franja_sin_solucion": unsolved,
            }
        return {
            "banda": candidate,
            "elegida": bool(candidate == band),
            "riesgo_medio_entre_politicas": float(np.mean(list(mean_risk[candidate].values()))),
            "por_politica": per_policy,
        }

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
            "ocupacion_del_conjunto": float(start.sum() / network.capacity.sum()),
            "estaciones_sin_tipo": int((~network.typed).sum()),
            **occupancy_report,
        },
        "pronostico": {
            "politicas_simuladas": present,
            "politicas_con_pronostico": names,
            "modelo_de_cada_politica": {name: FORECAST_OF[name] for name in names},
            "model_simulada": bool("model" in names),
            # Lo lee el dashboard (app/datos.py): "lgbm" si "model" se simuló y NOT_SIMULATED si no.
            "modelo_que_alimenta_model": FORECAST_OF["model"] if "model" in names else NOT_SIMULATED,
            "filas_leidas": {name: int(len(forecasts[name])) for name in names},
        },
        "banda": {
            "candidatas": list(bands),
            "elegida": band,
            "presupuesto_del_dia_en_bicis": int(reference),
            "resultados": [band_row(candidate) for candidate in bands],
        },
        "informacion": _missing_info(calendar, reglas, with_history, info),
        "historia": {"filas": int(len(history)), "antes_de": tramo.inicio.isoformat()},
        "reglas": {
            "estado": reglas.estado,
            "franjas": [
                {"nombre": f.nombre, "momento_intervencion": f.momento_intervencion, "inicio": f.inicio,
                 "fin": f.fin, "parte": f.share}
                for f in reglas.franjas
            ],
            "franja_noche": list(ajustes.franja_noche),
            "etiquetas_del_reporte": list(calendar.labels),
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
        "sensibilidad_al_inventario_inicial": {
            "presupuesto_del_dia_en_bicis": int(reference),
            "banda": band,
            "filas": sensitivity,
        },
        "recomendaciones": recommended,
        "diferencias_entre_politicas": {
            "presupuesto_del_dia_en_bicis": int(reference),
            "estacion_franja": recommended["estacion_franja"],
            **compare_policies(runs, reference, policies, calendar, reglas),
        },
        "comprobaciones": [
            "El inventario estuvo siempre entre 0 y la capacidad en todas las corridas.",
            "En todas las corridas el total de bicis (en estaciones más en tránsito) fue el inicial en cada "
            "intervención, en cada cambio de día y al final.",
            "Los traslados no crearon ni destruyeron bicis.",
            "Ninguna franja usó más presupuesto del disponible.",
            "Todos los traslados quedaron dentro de la distancia máxima.",
            'La política "none" no movió bicis.',
            f"Se repitió la corrida de {POLICY_LABELS[names[0]]} con {reference:,} bicis al día y dio el mismo "
            "resultado.",
        ],
        "corridas_simuladas": simulated,
    }
    report["limitaciones"] = limitations(report)
    return Resultado(recommendations, policy_curve, report, runs, network, calendar)


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #

NO_MODEL = (
    'La política "model" NO se simuló: `predictions` no trae filas `model = "lgbm"` (LightGBM todavía no '
    'existe). Las tablas `recommendations` y `policy_curve` salen sin filas de "model". Ya no se alimenta '
    '"model" con la línea base.'
)


def limitations(report: dict) -> list[str]:
    """Limitaciones de la simulación, dichas como hechos."""
    final = (
        "Esta corrida usó la prueba final."
        if report["prueba_final"]["simulada"]
        else "La prueba final no se simuló (decisión 4)."
    )
    none = next(row for row in report["resultados"] if row["politica"] == "none")
    bikes, start = none["bicis"], report["inventario_inicial"]
    trips = report["viajes"]
    ends, origins, ignored = trips["llegadas_por_destino"], trips["salidas_por_origen"], trips["ignorados"]
    days, daily = report["tramo"]["dias_evaluados"], none["riesgo_de_cada_dia"]
    rows = report["sensibilidad_al_inventario_inicial"]["filas"]
    spread = ", ".join(
        f"{row['inventario_inicial']}: {_n(row['riesgo_sin_intervencion_por_dia'])}" for row in rows
    )
    items = [
        "La distancia entre estaciones es en línea recta: ignora ríos y puentes. Dos estaciones con un río en "
        "medio pueden quedar dentro de la distancia máxima.",
        "Los viajes no traen los traslados que el operador hizo de verdad en esos días (camiones, bicis "
        "llevadas a taller). La política sin intervención es un sistema en el que nadie mueve bicis: no es lo "
        "que pasó en la realidad.",
        f"El total de bicis no cambia durante la corrida ({bikes['total_al_inicio']:,}): es un supuesto, no un "
        "hecho. En la realidad entran y salen bicis del sistema (taller, bicis nuevas, viajes a estaciones que "
        "aquí no se simulan). Para conservar el total, un viaje que termina fuera del conjunto simulado "
        f"devuelve la bici a la estación simulada más cercana al punto donde terminó "
        f"({ends['a_la_estacion_mas_cercana_al_punto_de_fin']:,} viajes) o, si no trae ese punto, a su estación "
        f"de salida ({ends['de_vuelta_a_la_estacion_de_salida']:,} viajes). Esas bicis no volvieron ahí en la "
        "realidad.",
        "Regla simétrica para los viajes que empiezan fuera del conjunto simulado: salen de la estación "
        "simulada más cercana a su punto de inicio "
        f"({origins['de_la_estacion_mas_cercana_al_punto_de_inicio']:,} viajes; bloqueados si está vacía) y, sin "
        f"punto de inicio, se ignoran ({ignored['empiezan_fuera_del_conjunto_y_sin_punto_de_inicio']:,}). **Es "
        "una interpretación de programación** de \"el total no debe cambiar\": la decisión del usuario solo "
        "describe los viajes que salen del conjunto. Carga a estaciones simuladas demanda que no fue suya.",
        f"Los {ignored['ya_iban_en_camino_al_empezar_el_tramo']:,} viajes que ya iban en camino al empezar el "
        f"tramo se ignoran. Los {trips['siguen_en_camino_al_terminar_el_tramo']:,} que siguen en camino al "
        "terminar dejan su bici en tránsito, dentro del total (una salida bloqueada en la simulación no cuenta).",
        "Los traslados son instantáneos y no hay camiones: no se modela su número, su capacidad ni su ruta. "
        "El único límite de esfuerzo es el presupuesto de bicis movidas.",
        "Los días no son independientes: el inventario pasa de un día al siguiente, hay patrón semanal y son "
        "pocos días. Los intervalos son indicativos, no una garantía estadística.",
        "Los datos solo registran demanda atendida. Los viajes bloqueados son un resultado de la simulación "
        "(posible demanda insatisfecha), no demanda medida. Un usuario bloqueado no busca otra estación ni "
        "vuelve a intentar.",
        "El inventario inicial es un supuesto: la ocupación de cada estación en la captura de GBFS entre las "
        "00:00 y las 00:59 de Nueva York, por su capacidad. **La captura es de octubre de 2026, no de los días "
        f"simulados**: no hay inventario real del periodo. Da {start['bicis']:,} bicis en {start['lugares']:,} "
        f"lugares ({start['ocupacion_del_conjunto'] * 100:.1f} %). Las {start['estaciones_con_la_media']:,} "
        "estaciones simuladas sin ocupación observada usan la media de las demás "
        f"({start['ocupacion_media_observada'] * 100:.1f} %). Los días de calentamiento no quitan la dependencia "
        "de ese supuesto: el nivel de riesgo y el riesgo simulado evitado cambian con él en todos los días "
        f"evaluados. Riesgo sin intervención por día según el inventario inicial: {spread}. Sin intervención, "
        f"el primer día evaluado ({days[0]}) tiene {_n(daily[0])} horas de estación en riesgo y el último "
        f"({days[-1]}), {_n(daily[-1])}.",
        "La capacidad es la más frecuente de la captura de GBFS de octubre de 2026, no la del periodo simulado.",
        "La banda de incertidumbre se eligió con los mismos días de validación con los que se reporta el "
        "resultado: el riesgo de la banda elegida sale favorecido. La diferencia entre políticas no, porque "
        "todas usan la misma banda.",
        "Reparto del presupuesto: dentro de la misma prioridad se atiende primero la diferencia más pequeña. "
        "La regla que pidió el usuario, \"riesgo evitado por bici\", **no se implementó**: necesita un modelo "
        "del riesgo esperado por estación y franja que no existe. La diferencia más pequeña es su sustituto.",
        "Bicis clásicas y eléctricas se cuentan juntas.",
        "Una estación cerrada varios días se ve como demanda cero: la simulación no sabe que estaba cerrada.",
        "La banda trata las horas como independientes entre sí y las salidas como independientes de las "
        "llegadas al sumar la incertidumbre.",
        "Una llegada a una estación llena va a la estación con lugar más cercana en línea recta. No se modela "
        "a qué estación iría de verdad el usuario.",
        "El conjunto simulado se define con los viajes del propio tramo (estaciones con al menos una salida o "
        "llegada).",
    ]
    for name in report["pronostico"]["politicas_con_pronostico"]:
        gap = report["informacion"][name]["con_historia_y_sin_pronostico"]
        if gap["estacion_franja"]:
            items.append(
                f"`predictions` solo trae filas hasta la hora del último viaje de cada estación (`station_hour` "
                f"es densa entre el primer y el último viaje). Por eso la política \"{name}\" deja de pedir y de "
                "prestar bicis en estaciones que sí tienen historia, desde la franja en que cae su último viaje: "
                f"{gap['estacion_franja']:,} estación-franja (estaciones: {gap['estaciones']:,}). Ahí lo que "
                "decide depende de viajes posteriores a la intervención."
            )
    if report["pronostico"]["model_simulada"]:
        items.append('La política "model" se alimenta de las filas `model = "lgbm"` de `predictions`.')
    else:
        items.append(NO_MODEL)
    items.append(
        'La regla histórica se alimenta de las filas `model = "baseline"` de `predictions`: promedio histórico '
        "por estación, día de la semana y hora en q50 (es una media, no una mediana) y cuantiles empíricos en "
        "q10 y q90."
    )
    return [*items, final]


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


def _policies(report: dict) -> list[str]:
    """Políticas simuladas, en el orden del contrato."""
    return list(report["pronostico"]["politicas_simuladas"])


def _daily_rows(report: dict) -> list[list[str]]:
    """Riesgo de cada día evaluado con el presupuesto de referencia: una fila por día y una columna por política."""
    series = {name: row["riesgo_de_cada_dia"] for name, row in _at_reference(report).items()}
    rows = []
    for i, day in enumerate(report["tramo"]["dias_evaluados"]):
        weekday = WEEKDAYS[datetime.fromisoformat(day).weekday()]
        rows.append([f"{day} ({weekday})", *[_n(series[policy][i]) for policy in _policies(report)]])
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
    for policy in _policies(report):
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
    if not report["pronostico"]["model_simulada"]:
        lines.append(f"- {NO_MODEL}")
    return lines


def _differences(report: dict) -> list[str]:
    """Sección con lo que pidió y movió cada política con el presupuesto de referencia. No afirma causas."""
    facts = report["diferencias_entre_politicas"]
    names = report["pronostico"]["politicas_con_pronostico"]

    def per_window(row: dict) -> str:
        touched = row["estacion_franja_con_traslado"]
        return _n((row["bicis_recibidas"] + row["bicis_entregadas"]) / touched) if touched else "0"

    return [
        "## Qué pidió y qué movió cada política",
        "",
        f"Con el presupuesto de referencia ({facts['presupuesto_del_dia_en_bicis']:,} bicis al día) y la banda "
        f"elegida, en los días evaluados ({facts['estacion_franja']:,} estación-franja). \"Faltan\" y \"sobran\" "
        "son las bicis que separan a cada estación del borde más cercano de su rango aceptado, con el "
        "inventario que la política vio en cada intervención, sumadas en el día. Son cuentas de la corrida: "
        "ninguna está probada como causa de la diferencia entre políticas.",
        "",
        _table(
            ["política", "estación-franja con información", "sin solución", "de ellas, sin tipo (no se tocan)",
             "bicis que faltan por día", "bicis que sobran por día", "bicis movidas por día",
             "estación-franja con traslado", "bicis por estación-franja con traslado"],
            [
                [
                    POLICY_LABELS[name],
                    f"{facts[name]['estacion_franja_con_informacion']:,}",
                    f"{facts[name]['estacion_franja_sin_solucion']:,}",
                    f"{facts[name]['de_ellas_sin_tipo_no_se_tocan']:,}",
                    _n(facts[name]["bicis_que_faltan_por_dia"]),
                    _n(facts[name]["bicis_que_sobran_por_dia"]),
                    _n(facts[name]["bicis_movidas_por_dia"]),
                    f"{facts[name]['estacion_franja_con_traslado']:,}",
                    per_window(facts[name]),
                ]
                for name in names
            ],
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
    labels = list(rules["etiquetas_del_reporte"])
    forecast, band, start, trips = report["pronostico"], report["banda"], report["inventario_inicial"], report["viajes"]
    names = forecast["politicas_con_pronostico"]
    policies = _policies(report)
    reference = rules["presupuesto_de_referencia_en_bicis"]
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
    if not forecast["model_simulada"]:
        lines += [f"**{NO_MODEL}**", ""]
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
        f"- Políticas simuladas: {', '.join(POLICY_LABELS[name] for name in policies)}.",
        f"- Corridas simuladas: {report['corridas_simuladas']} (sin intervención, las de la banda, las de la "
        "curva y las de la sensibilidad). Con presupuesto 0 todas las políticas son la corrida sin intervención "
        "y no se vuelven a simular.",
        "- Todas las políticas parten del mismo inventario y enfrentan los mismos viajes. En cada intervención "
        "una política ve el inventario simulado de ese momento y ningún viaje.",
        "- La regla histórica y el modelo son la misma maquinaria: lo único que cambia es el pronóstico que "
        "reciben.",
    ]
    if not report["prueba_final"]["simulada"] and report["prueba_final"]["inicio"]:
        lines.append(
            f"- La prueba final ({report['prueba_final']['inicio']} a {report['prueba_final']['fin']}) "
            "no se simuló (decisión 4). De esas fechas no se leyó ninguna fila de `station_hour` ni de "
            "`predictions`. De los viajes que salieron dentro del tramo simulado y terminaron después solo se "
            "usa la salida: su bici queda en tránsito y de su llegada no se mira nada."
        )
    without, idle = stations["sin_capacidad"], stations["inactivas"]
    unknown, loose = stations["ids_de_viajes_que_no_estan_en_stations"], stations["extremos_sin_estacion"]
    origins, ends, ignored = trips["salidas_por_origen"], trips["llegadas_por_destino"], trips["ignorados"]
    far = trips["distancia_a_la_estacion_mas_cercana"]

    def distance(side: dict) -> str:
        if not side["viajes"]:
            return "ninguno"
        return (
            f"mediana {side['mediana_km']:.2f} km, percentil 90 {side['p90_km']:.2f} km, percentil 99 "
            f"{side['p99_km']:.2f} km, máximo {side['maximo_km']:.2f} km"
        )

    lines += [
        "",
        "## Estaciones",
        "",
        _table(
            ["grupo", "estaciones", "salidas en el tramo", "llegadas en el tramo"],
            [
                ["En `stations`", f"{stations['en_stations']:,}", "", ""],
                ["Simuladas", f"{stations['simuladas']:,}", f"{trips['salidas']:,}", f"{trips['llegadas']:,}"],
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
        "En la fila \"Simuladas\" van las salidas y llegadas que se simulan, ya con los viajes que se "
        "reasignan a la estación simulada más cercana (sección siguiente). Las demás filas cuentan los viajes "
        "tal como vienen en los datos.",
        "",
        "## Viajes y total de bicis",
        "",
        "**El total de bicis no cambia durante la corrida**: las que están en estaciones más las que van en un "
        "viaje dan siempre el total inicial. Se comprueba en cada intervención, en cada cambio de día y al "
        "final de cada corrida.",
        "",
        f"- Viajes leídos (con salida o llegada en el tramo): {trips['viajes_leidos']:,}.",
        f"- Salidas simuladas: {trips['salidas']:,}. De su estación: {origins['de_su_estacion']:,}. De la "
        "estación simulada más cercana a su punto de inicio, porque el viaje empezó fuera del conjunto: "
        f"{origins['de_la_estacion_mas_cercana_al_punto_de_inicio']:,} "
        f"({origins['de_ellas_sin_estacion_en_la_tabla']:,} sin estación en la tabla de viajes y "
        f"{origins['de_ellas_con_estacion_que_no_se_simula']:,} con una estación que no se simula). Distancia "
        f"del punto a esa estación: {distance(far['punto_de_inicio'])}. **Esta regla es una interpretación de "
        "programación**, simétrica a la de las llegadas.",
        f"- Llegadas simuladas: {trips['llegadas']:,}. A su estación: {ends['a_su_estacion']:,}. A la estación "
        "simulada más cercana al punto donde terminó el viaje, porque terminó fuera del conjunto: "
        f"{ends['a_la_estacion_mas_cercana_al_punto_de_fin']:,}. De vuelta a la estación de salida, porque el "
        f"viaje no trae punto de fin: {ends['de_vuelta_a_la_estacion_de_salida']:,}. (De los que terminan fuera, "
        f"{ends['fuera_del_conjunto_sin_estacion_en_la_tabla']:,} no traen estación en la tabla de viajes y "
        f"{ends['fuera_del_conjunto_con_estacion_que_no_se_simula']:,} traen una que no se simula.) Distancia "
        f"del punto a esa estación: {distance(far['punto_de_fin'])}.",
        f"- Viajes ignorados: {ignored['ya_iban_en_camino_al_empezar_el_tramo']:,} que ya iban en camino al "
        f"empezar el tramo y {ignored['empiezan_fuera_del_conjunto_y_sin_punto_de_inicio']:,} que empiezan fuera "
        "del conjunto y no traen punto de inicio.",
        f"- Viajes que siguen en camino al terminar el tramo: {trips['siguen_en_camino_al_terminar_el_tramo']:,}. "
        "Su bici queda \"en tránsito\", dentro del total.",
        "- Una salida que la simulación bloquea no saca la bici ni genera llegada.",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", "total al inicio", "en estaciones al final",
             "en tránsito al final", "total al final"],
            [
                [
                    *_first_cells(row),
                    f"{row['bicis']['total_al_inicio']:,}",
                    f"{row['bicis']['en_estaciones_al_final']:,}",
                    f"{row['bicis']['en_transito_al_final']:,}",
                    f"{row['bicis']['total_al_final']:,}",
                ]
                for row in shown
            ],
        ),
        "",
        "## Inventario inicial",
        "",
        f"- {start['bicis']:,} bicis en {start['lugares']:,} lugares ({start['ocupacion_del_conjunto'] * 100:.1f} "
        "% de ocupación), al primer instante del tramo.",
        "- Sale de `config/gbfs_congelado.csv`: la ocupación media de cada estación en las fotos de GBFS "
        "tomadas entre las 00:00 y las 00:59 de Nueva York, por su capacidad, redondeado. **La captura es de "
        "octubre de 2026, no de los días simulados**: es un supuesto, no el inventario real de esas fechas.",
        *(
            [
                f"- Captura usada: del {start['captura'].get('captura_desde_utc')} al "
                f"{start['captura'].get('captura_hasta_utc')} (UTC), {start['captura'].get('fotos_de_estado')} fotos "
                f"de estado; {start['captura'].get('fotos_en_esa_hora')} de ellas en esa hora. Archivo: "
                f"`{start.get('archivo')}`."
            ]
            if start.get("captura")
            else []
        ),
        f"- Estaciones con ocupación observada: {start['estaciones_con_ocupacion_observada']:,}. Sin ella: "
        f"{start['estaciones_con_la_media']:,} ({start['de_ellas_fuera_del_archivo']:,} no están en el archivo); "
        f"usan la media observada de las demás, {start['ocupacion_media_observada'] * 100:.1f} %.",
        f"- Estaciones sin tipo: {start['estaciones_sin_tipo']:,}. No usan ninguna `ocupacion_respaldo`.",
        "",
        "## Pronóstico de cada política",
        "",
        "- Regla histórica: filas `model = \"baseline\"` de `predictions` (promedio histórico en q50, cuantiles "
        f"empíricos en q10 y q90). Filas leídas: {forecast['filas_leidas']['historical']:,}.",
    ]
    if forecast["model_simulada"]:
        lines.append(
            f"- Modelo: filas `model = \"lgbm\"` de `predictions`. Filas leídas: {forecast['filas_leidas']['model']:,}."
        )
    else:
        lines.append(f"- Modelo: {NO_MODEL}")
    lines += [
        "- Una política solo usa las filas de estaciones simuladas con historia, en las horas que van de la "
        "intervención al fin de cada franja.",
        "",
        "## Banda de incertidumbre",
        "",
        "El rango de inventario que acepta una estación se calcula con la trayectoria central del pronóstico y "
        "una banda alrededor: la dispersión del pronóstico (de q10 a q90) por un multiplicador. Con 0 no hay "
        "banda; con 1 es la banda completa. Si el inventario ya está dentro del rango no se mueve nada; si "
        "está fuera, va solo hasta el borde más cercano.",
        "",
        f"Se simularon los multiplicadores {', '.join(f'{b:g}' for b in band['candidatas'])} "
        f"(`simulacion.bandas` de config/datos.yaml) con el presupuesto de referencia ({reference:,} bicis al "
        "día). **Se eligió "
        f"{band['elegida']:g}**: el de menor riesgo simulado medio, promediado entre las políticas con "
        "pronóstico. Ese valor se aplica igual a todas y es el de la curva y las recomendaciones. Se eligió "
        "con estos mismos días de validación.",
        "",
        _table(
            ["banda", "política", "horas en riesgo por día", "riesgo simulado evitado por día", f"intervalo {level}",
             "viajes bloqueados por día", "bicis movidas por día", "estación-franja sin solución"],
            [
                [
                    f"{row['banda']:g}" + (" (elegida)" if row["elegida"] else ""),
                    POLICY_LABELS[name],
                    _n(cell["riesgo_por_dia"]),
                    _n(cell["riesgo_simulado_evitado_por_dia"]["media"]),
                    _range(cell["riesgo_simulado_evitado_por_dia"]),
                    _n(cell["viajes_bloqueados_por_dia"]),
                    _n(cell["bicis_movidas_por_dia"]),
                    f"{cell['estacion_franja_sin_solucion']:,}",
                ]
                for row in band["resultados"]
                for name, cell in row["por_politica"].items()
            ],
            text_columns=2,
        ),
        "",
        f"Sin intervención: {_n(_at_reference(report)['none']['riesgo_por_dia']['media'])} horas en riesgo por "
        "día.",
        "",
        "## Estaciones sin información",
        "",
        "Una estación sin información no pide ni presta bicis en esa franja.",
        "",
        f"- Estaciones simuladas sin ninguna fila en la historia (antes del {report['historia']['antes_de']}): "
        f"{report['informacion']['estaciones_sin_historia']:,}. Ninguna política las toca: su pronóstico es un "
        "marcador, no un pronóstico suyo.",
    ]
    for name in names:
        info = report["informacion"][name]
        gap = info["con_historia_y_sin_pronostico"]
        lines.append(
            f"- {POLICY_LABELS[name].capitalize()}: {info['estacion_franja_sin_informacion']:,} estación-franja sin "
            f"información de {stations['simuladas'] * report['informacion']['franjas_evaluadas_por_estacion']:,} "
            f"(sin historia, o sin pronóstico de alguna hora de la franja); "
            f"{info['estaciones_sin_informacion_en_alguna_franja']:,} estaciones en alguna franja. De ellas, "
            f"{gap['estacion_franja']:,} estación-franja son de estaciones con historia (estaciones: "
            f"{gap['estaciones']:,}): el pronóstico se corta en la hora de su último viaje."
        )
    lines += [
        "",
        "## Reglas usadas",
        "",
        _table(
            ["franja", "intervención", "inicio", "fin", "parte del presupuesto"],
            [
                [f["nombre"], f["momento_intervencion"], f["inicio"], f["fin"], f"{f['parte']:g}"]
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
        f"estación, {reference:,} bicis al día.",
        "- Reparto del presupuesto: primero el tipo con más prioridad (una estación sin tipo va después de "
        "todas las que tienen); dentro de la misma prioridad, primero la estación a la que le falta o le sobra "
        "MENOS para llegar a su rango, y después por id. La contraparte es la estación más cercana, dentro de la "
        "distancia máxima, con la necesidad contraria (a una le faltan bicis y a la otra le sobran); una "
        "estación que ya está dentro de su rango no recibe ni presta. **No se implementó \"riesgo evitado por bici\"**, que era la primera opción del "
        "usuario: necesita un modelo del riesgo esperado por estación y franja.",
        "- Franja sin solución (ningún inventario evita los dos umbrales con la banda): una estación con tipo "
        "va a `ocupacion_respaldo` de su tipo por su capacidad, sin quedar bajo ningún umbral; una estación sin "
        "tipo no se toca en esa franja.",
        f"- Franja de noche del reporte: {rules['franja_noche'][0]} a {rules['franja_noche'][1]} "
        "(`simulacion.franja_noche` de config/datos.yaml). No es una franja de operación.",
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
        f"Media por día evaluado, sobre las 24 horas, con la banda elegida ({band['elegida']:g}). Intervalo del "
        f"{level} por remuestreo de días. \"Riesgo simulado evitado\" es la diferencia contra no intervenir, día "
        "por día, con su propio intervalo (los mismos días sorteados para las dos corridas). Positivo es menos "
        "riesgo.",
        "",
        _table(
            ["política", "bicis por estación", "bicis al día", "horas en riesgo por día", f"intervalo {level}",
             "riesgo simulado evitado por día", f"intervalo {level}"],
            [
                [
                    *_first_cells(row),
                    _n(row["riesgo_por_dia"]["media"]),
                    _range(row["riesgo_por_dia"]),
                    _n(row["riesgo_simulado_evitado_por_dia"]["media"]),
                    _range(row["riesgo_simulado_evitado_por_dia"]),
                ]
                for row in shown
            ],
        ),
        "",
        *_reading(report),
        "",
        "## Riesgo día por día",
        "",
        f"Horas de estación en riesgo de cada día evaluado, con el presupuesto de referencia ({reference:,} "
        "bicis al día). Los días no se parecen entre sí: la media por día y su intervalo resumen esta tabla, no "
        "la reemplazan.",
        "",
        _table(["día", *[POLICY_LABELS[policy] for policy in policies]], _daily_rows(report)),
        "",
        "## Riesgo por franja del día y por tipo",
        "",
        "Horas de estación en riesgo por día evaluado. Las franjas de operación cuentan el tiempo entre su "
        f"inicio y su fin; \"noche\" es de {rules['franja_noche'][0]} a {rules['franja_noche'][1]}; \"resto\" es "
        "lo que no cae en ninguna. Las columnas de franja suman el total del día.",
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
        "El riesgo de cada franja en cada día está en `simulacion.json` (`riesgo_por_franja_de_cada_dia`).",
        "",
        "## Sensibilidad al inventario inicial",
        "",
        f"La corrida con el presupuesto de referencia ({reference:,} bicis al día) y la banda elegida, repetida "
        "con otros inventarios iniciales: todas las estaciones a la misma fracción de su capacidad "
        "(`simulacion.sensibilidad_ocupacion_inicial` de config/datos.yaml). La primera fila es la corrida "
        "principal. Horas de estación en riesgo por día evaluado. Estas corridas no alimentan las "
        "recomendaciones ni la curva.",
        "",
        _table(
            ["inventario inicial", "bicis", "ocupación", "sin intervención",
             *[f"{POLICY_LABELS[name]}: {what}" for name in names
               for what in ("en riesgo", "riesgo simulado evitado", "bloqueados por día")]],
            [
                [
                    row["inventario_inicial"],
                    f"{row['bicis']:,}",
                    f"{row['ocupacion_media'] * 100:.1f} %",
                    _n(row["riesgo_sin_intervencion_por_dia"]),
                    *[
                        _n(row["por_politica"][name][key])
                        for name in names
                        for key in ("riesgo_por_dia", "riesgo_simulado_evitado_por_dia", "viajes_bloqueados_por_dia")
                    ],
                ]
                for row in report["sensibilidad_al_inventario_inicial"]["filas"]
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
    franja_names = [f["nombre"] for f in rules["franjas"]]
    lines += [
        _table(
            ["política", "bicis por estación", "bicis al día",
             *[f"{name}: {what}" for name in franja_names for what in ("disponible", "usado")]],
            [
                [
                    POLICY_LABELS[row["politica"]],
                    *_budget_text(row),
                    *[
                        _n(row["presupuesto_por_franja"][name][key])
                        for name in franja_names
                        for key in ("disponible_por_dia", "usado_por_dia")
                    ],
                ]
                for row in report["resultados"]
                if row["politica"] != "none"
            ],
        ),
        "",
        "## Recomendaciones",
        "",
        f"`recommendations.parquet` sale de las corridas con el presupuesto de referencia ({reference:,} bicis "
        f"al día) y la banda elegida: {recs['filas']:,} filas, una por estación, franja, día evaluado y política "
        f"simulada ({recs['estacion_franja']:,} estación-franja). `feasible` es falso cuando, con el pronóstico "
        "y la banda de esa política, ningún inventario evita a la vez los dos umbrales. El nivel de riesgo se "
        "calcula con ese pronóstico y el inventario que la política tenía ANTES de intervenir: alto si la "
        "trayectoria central cruza un umbral (o la estación ya está bajo uno), medio si solo lo cruza la banda "
        "completa de q10 a q90, bajo en otro caso; no depende de la banda elegida. Las filas de "
        "\"sin intervención\" usan el pronóstico de la regla histórica.",
        "",
        _table(
            ["política", "estación-franja sin solución", "estación-franja con traslado", "bicis recibidas",
             "bicis entregadas", "riesgo alto", "riesgo medio", "riesgo bajo"],
            [
                [
                    POLICY_LABELS[policy],
                    f"{row['estacion_franja_sin_solucion']:,}",
                    f"{row['estacion_franja_con_traslado']:,}",
                    f"{row['bicis_recibidas']:,}",
                    f"{row['bicis_entregadas']:,}",
                    *[f"{row['nivel_de_riesgo'][level_name]:,}" for level_name in ("alto", "medio", "bajo")],
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
    tramo, stations, band = report["tramo"], report["estaciones"], report["banda"]
    level = f"{report['intervalo']['nivel'] * 100:g} %"
    start = report["inventario_inicial"]
    lines = [
        f"Simulación de inventario. Tramo de {TRAMO_LABELS.get(tramo['nombre'], tramo['nombre'])}: "
        f"{tramo['inicio']} a {tramo['fin']}.",
        f"Días evaluados: {len(tramo['dias_evaluados'])}. Calentamiento (no cuenta): "
        f"{', '.join(tramo['dias_de_calentamiento']) or 'ninguno'}.",
        f"Estaciones simuladas: {stations['simuladas']:,} de {stations['en_stations']:,}. Fuera: "
        f"{stations['sin_capacidad']['estaciones']:,} sin capacidad y {stations['inactivas']['estaciones']:,} "
        "sin viajes en el tramo.",
        f"Inventario inicial: {start['bicis']:,} bicis en {start['lugares']:,} lugares. Supuesto: ocupación de "
        "la captura de GBFS de octubre de 2026, no de los días simulados.",
        "El total de bicis (en estaciones más en tránsito) no cambió en ninguna corrida.",
        "Regla histórica: filas \"baseline\" de predictions."
        + (' Modelo: filas "lgbm".' if report["pronostico"]["model_simulada"] else ""),
    ]
    if not report["pronostico"]["model_simulada"]:
        lines.append(f"AVISO. {NO_MODEL}".replace("`", ""))
    if report["reglas"]["estado"] != "confirmado":
        lines.append(f"Reglas de operación PROVISIONALES ({operacion}).")
    lines += [
        "",
        f"Banda de incertidumbre, con {band['presupuesto_del_dia_en_bicis']:,} bicis al día (se elige la de menor "
        "riesgo medio):",
        _table(
            ["banda", "política", "en riesgo por día", "riesgo simulado evitado"],
            [
                [
                    f"{row['banda']:g}" + (" (elegida)" if row["elegida"] else ""),
                    POLICY_LABELS[name],
                    _n(cell["riesgo_por_dia"]),
                    _n(cell["riesgo_simulado_evitado_por_dia"]["media"]),
                ]
                for row in band["resultados"]
                for name, cell in row["por_politica"].items()
            ],
            text_columns=2,
        ),
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
                    _range(row["riesgo_por_dia"]),
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
    missing = [column for column in (*TRIP_COLUMNS, *POINT_COLUMNS) if column not in schema.names]
    if missing:
        raise SimulacionError(
            f"A {_short(path)} le faltan columnas: {missing}. Vuelve a correr: python scripts/build_tables.py"
        )
    start, end = _instant(tramo.start), _instant(tramo.end)
    inside = lambda column: (pc.field(column) >= start) & (pc.field(column) < end)  # noqa: E731
    table = pq.read_table(
        path, columns=[*TRIP_COLUMNS, *POINT_COLUMNS], filters=inside("started_at") | inside("ended_at")
    )
    return table.to_pandas()


def load_history(path: Path, first: Tramo, tramo: Tramo) -> pd.DataFrame:
    """Filas de `station_hour` anteriores al tramo simulado. Las del tramo y las posteriores no se leen."""
    start, end = _instant(first.start), _instant(tramo.start)
    table = pq.read_table(
        path,
        columns=["station_id", "ts_hour"],
        filters=(pc.field("ts_hour") >= start) & (pc.field("ts_hour") < end),
    )
    return table.to_pandas()


def load_forecasts(path: Path, tramo: Tramo, split: str) -> dict:
    """Pronóstico del tramo de cada política que lo tiene: "historical" (baseline) y, si hay filas, "model" (lgbm).

    Sin filas "lgbm" el resultado no trae "model": esa política no se simula. Sin filas "baseline" falla.
    """
    start, end = _instant(tramo.start), _instant(tramo.end)
    table = pq.read_table(
        path,
        filters=(pc.field("split") == split) & (pc.field("ts_hour") >= start) & (pc.field("ts_hour") < end),
    )
    out = {}
    for policy, model in FORECAST_OF.items():
        rows = table.filter(pc.equal(table["model"], model))
        if rows.num_rows:
            out[policy] = rows.to_pandas()
    if "historical" not in out:
        raise SimulacionError(
            f'{_short(path)} no trae filas "baseline" con split={split!r} para el tramo. '
            "Primero corre: python scripts/run_baseline.py"
        )
    return out


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


def read_inputs(data_dir: Path, datos, tramo: Tramo, gbfs: str | Path = FROZEN_FILE) -> tuple:
    """Las entradas tal como las lee la simulación.

    Devuelve stations, los viajes del tramo, la historia anterior al tramo, el pronóstico de cada política
    y el archivo congelado de GBFS (tabla y datos de la captura).
    """
    stations = read_contract("stations", data_dir / "stations.parquet")
    trips = load_trips(data_dir / TRIPS_FILE, tramo)
    history = load_history(data_dir / "station_hour.parquet", datos.tramos[TRAMOS[0]], tramo)
    forecasts = load_forecasts(data_dir / "predictions.parquet", tramo, SPLIT_NAMES[tramo.nombre])
    try:
        frozen = load_frozen(gbfs)
    except GbfsFrozenError as error:
        raise SimulacionError(str(error)) from None
    return stations, trips, history, forecasts, frozen


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
    "gbfs_congelado": "config/gbfs_congelado.csv (capacidad e inventario inicial congelados de GBFS)",
}
BAND_KEY = "banda_elegida"
SETTINGS = (
    "dias_de_calentamiento", "remuestreos", "nivel_del_intervalo", "semilla", "bandas", "franja_noche",
    "sensibilidad_ocupacion_inicial",
)


def _control(columns: dict) -> dict:
    """Cuántas filas trae una tabla y un número de control de su contenido. No depende del orden de las filas."""
    frame = pd.DataFrame(columns)
    hashes = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy()
    return {"filas": int(len(frame)), "control": int(hashes.sum())}


def _text(column: pd.Series, keep=None) -> np.ndarray:
    """Columna de texto lista para el número de control: vacío donde no hay valor o donde `keep` es falso."""
    values = column.astype("object").where(column.notna(), "")
    return (values if keep is None else values.where(keep, "")).to_numpy(dtype=object)


def _number(frame: pd.DataFrame, column: str, keep) -> np.ndarray:
    """Columna decimal para el número de control: NaN donde no hay valor, no existe o `keep` es falso."""
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return np.where(keep, values, np.nan)


def input_fingerprint(
    stations: pd.DataFrame,
    trips: pd.DataFrame,
    history: pd.DataFrame,
    forecasts: dict,
    frozen: pd.DataFrame,
    reglas: Reglas,
    datos,
    tramo: Tramo,
) -> dict:
    """Huella de lo que lee una simulación. Sirve para saber si sus salidas siguen al día.

    Guarda las reglas de operación que usa el simulador, el bloque `simulacion`, las fechas y, de cada
    tabla, cuántas filas se leyeron y un número de control de su contenido:

    - stations: todas las filas, con las columnas que usa el simulador (sin el nombre).
    - trips: los viajes con salida o llegada en el tramo. De un viaje solo entra lo que la simulación usa:
      la salida (hora, estación y punto de inicio) si empieza dentro del tramo, y la llegada (hora, estación
      y punto de fin) si además termina dentro. De un viaje que termina después del tramo no se guarda nada
      de su llegada.
    - station_hour: qué estaciones tienen historia anterior al tramo (estación y hora).
    - predictions: las filas del tramo de cada modelo que alimenta a una política.
    - gbfs_congelado: capacidad y ocupación inicial de cada estación del archivo congelado.

    De la prueba final solo van las fechas (decisión 4), salvo que el tramo simulado sea ella misma.
    """
    start_us, end_us = (int(value) for value in pd.DatetimeIndex([tramo.start, tramo.end]).as_unit("us").asi8)
    started, ended = _epoch_us(trips["started_at"]), _epoch_us(trips["ended_at"])
    departs = (started >= start_us) & (started < end_us)
    arrives = departs & (ended < end_us)
    final = datos.tramos[FINAL_TRAMO]
    settings = {}
    for name in SETTINGS:
        value = getattr(datos.simulacion, name)
        settings[name] = list(value) if isinstance(value, tuple) else value
    return {
        "reglas": dataclasses.asdict(reglas),
        "simulacion": settings,
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
                "start_lat": _number(trips, "start_lat", departs),
                "start_lng": _number(trips, "start_lng", departs),
                "end_lat": _number(trips, "end_lat", arrives),
                "end_lng": _number(trips, "end_lng", arrives),
            }
        ),
        "station_hour": _control(
            {"station_id": _text(history["station_id"]), "ts_hour": _epoch_us(history["ts_hour"])}
        ),
        "predictions": {
            FORECAST_OF[policy]: _control(
                {
                    "station_id": _text(rows["station_id"]),
                    "ts_hour": _epoch_us(rows["ts_hour"]),
                    **{column: rows[column].to_numpy(dtype="float64") for column in QUANTILE_COLUMNS},
                }
            )
            for policy, rows in forecasts.items()
        },
        "gbfs_congelado": _control(
            {
                "short_name": _text(frozen["short_name"]),
                "capacity": frozen["capacity"].astype("float64").to_numpy(),
                "ocupacion_inicial": frozen["ocupacion_inicial"].astype("float64").to_numpy(),
            }
        ),
    }


def current_fingerprint(
    data_dir: str | Path = DATA_PROCESSED,
    operacion: str | Path = OPERACION_YAML,
    config: str | Path = DATOS_YAML,
    tramo: str = "validacion",
    gbfs: str | Path = FROZEN_FILE,
) -> dict:
    """La huella que tendría una corrida hecha ahora con estas tablas, estas reglas y este config.

    Lee lo mismo que lee la simulación y nada más. No acepta la prueba final (decisión 4). No trae la
    banda elegida: esa sale de simular (ver `stale_inputs`).
    """
    datos = load_datos(config)
    chosen = datos.tramos[tramo]
    check_final_test(chosen, datos.tramos[FINAL_TRAMO], usar_prueba_final=False)
    if datos.simulacion is None:
        raise SimulacionError(f"{config} no trae el bloque `simulacion` (calentamiento, remuestreo y bandas).")
    reglas = Reglas.from_operacion(load_operacion(operacion))
    stations, trips, history, forecasts, frozen = read_inputs(Path(data_dir), datos, chosen, gbfs)
    return input_fingerprint(stations, trips, history, forecasts, frozen.table, reglas, datos, chosen)


def stale_inputs(stored: dict | None, current: dict) -> list[str]:
    """Qué entradas ya no son las de la corrida que guardó la huella `stored`. Lista vacía: siguen al día.

    La banda elegida es un resultado de la corrida y va guardada en la huella. Aquí se comprueba que esté y
    que siga siendo una de las candidatas de config/datos.yaml.
    """
    if not isinstance(stored, dict):
        return ["la huella de las entradas (simulacion.json no la trae: lo escribió una versión anterior)"]
    current = json.loads(json.dumps(current))  # como queda al guardarse: las tuplas pasan a listas
    changed = [label for part, label in FINGERPRINT_PARTS.items() if stored.get(part) != current[part]]
    if stored.get(BAND_KEY) not in current["simulacion"]["bandas"]:
        changed.append("la banda de incertidumbre elegida (no está en la huella o ya no es una candidata)")
    return changed


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
            'La política "model" solo se simula si predictions trae filas "lgbm".\n'
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
        "--gbfs", metavar="FILE", default=str(FROZEN_FILE),
        help=(
            "archivo congelado de GBFS con la ocupación inicial, el que escribe scripts/freeze_gbfs.py "
            "(por defecto config/gbfs_congelado.csv)"
        ),
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
    gbfs = Path(args.gbfs).expanduser()
    try:
        datos = load_datos(args.config)
        tramo = datos.tramos[args.tramo]
        # El candado va antes de abrir cualquier archivo de datos.
        check_final_test(tramo, datos.tramos[FINAL_TRAMO], args.usar_prueba_final)
        if tramo.nombre == FINAL_TRAMO:
            out_dir = out_dir / FINAL_FOLDER
        if getattr(datos, "simulacion", None) is None:
            raise SimulacionError(
                f"{args.config} no trae el bloque `simulacion` (calentamiento, remuestreo y bandas)."
            )
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
        if not gbfs.exists():
            raise SimulacionError(f"No existe {_short(gbfs)}. Primero corre: python scripts/freeze_gbfs.py")
        _check_built_period(data_dir, datos, args.config)
        problems = validate_file(paths["stations"], "stations")
        if problems:
            raise SimulacionError(f"{_short(paths['stations'])} no cumple su contrato:\n- " + "\n- ".join(problems))

        stations, trips, history, forecasts, frozen = read_inputs(data_dir, datos, tramo, gbfs)
        if "model" in forecasts:
            print('Regla histórica: filas "baseline". Modelo: filas "lgbm".', file=sys.stderr)
        else:
            print(f"AVISO. {NO_MODEL}".replace("`", ""), file=sys.stderr)
        result = run_simulation(
            stations, trips, history, forecasts, frozen.table, reglas, tramo, datos.simulacion,
            prueba_final=datos.tramos[FINAL_TRAMO],
            usar_prueba_final=args.usar_prueba_final,
            progress=lambda message: print(f"  corriendo: {message}", file=sys.stderr),
        )
        result.report["inventario_inicial"]["archivo"] = _short(gbfs)
        result.report["inventario_inicial"]["captura"] = frozen.meta
        # La huella de lo que se leyó: con ella una prueba sabe si estas salidas siguen al día.
        fingerprint = input_fingerprint(stations, trips, history, forecasts, frozen.table, reglas, datos, tramo)
        fingerprint[BAND_KEY] = result.report["banda"]["elegida"]
        seconds = time.time() - started
        operacion = _short(Path(args.operacion))
        document = {
            "corrida": {
                "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "segundos": round(seconds, 1),
                "datos": _short(data_dir),
                "operacion": operacion,
                "config": _short(Path(args.config)),
                "gbfs": _short(gbfs),
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
