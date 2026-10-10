"""Políticas de rebalanceo: sin intervención, regla histórica y modelo.

Una política recibe en cada intervención un `InterventionContext` (instante,
franja, inventario simulado de su propia corrida y presupuesto disponible) y
devuelve traslados. Los datos fijos de las estaciones, las reglas de
config/operacion.yaml y su fuente de información los tiene desde que se crea.
Ninguna recibe viajes.

Fuente de información de cada una:

- "none": ninguna. No mueve bicis.
- "historical" (regla histórica): el flujo neto medio por hora (llegadas menos
  salidas) de cada estación, por franja y tipo de día (laborable = lunes a
  viernes, o fin de semana), calculado solo con la historia anterior al tramo
  simulado. No usa `ocupacion_respaldo` (decisión 14).
- "model": el pronóstico de cada hora (cuantiles 10, 50 y 90 de salidas y
  llegadas), hecho también solo con historia anterior.

Las dos últimas hacen lo mismo con su información:

1. Trayectoria esperada del inventario entre la intervención y el fin de la
   franja, acumulando el flujo neto hora por hora.
2. Rango seguro del inventario al intervenir. `lo` es el mínimo de bicis para
   no bajar de `bicis_minimas` en ningún momento; `hi` es el máximo para no
   dejar menos de `lugares_libres_minimos`.
3. Si `lo <= hi`, el inventario deseado es el actual llevado a ese rango. Si
   no hay solución (`lo > hi`): la regla histórica usa el punto medio y el
   modelo usa `ocupacion_respaldo` de la franja por la capacidad.
4. Los inventarios deseados se convierten en traslados con `match_moves`.

Una estación sin información no pide ni presta bicis.

Supuestos del modelo (declarados en el reporte): las horas son independientes
entre sí y las salidas son independientes de las llegadas, así que las
dispersiones se suman en cuadratura.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.contracts import QUANTILE_COLUMNS
from src.paths import TZ
from src.sim.engine import EPS, InterventionContext, Move, Network, Reglas, SimulacionError, round_half_up

WEEKDAY, WEEKEND = "laborable", "fin de semana"
DAY_TYPES = (WEEKDAY, WEEKEND)


def day_type(day: date) -> str:
    """Laborable de lunes a viernes; sábado y domingo, fin de semana."""
    return WEEKDAY if day.weekday() < 5 else WEEKEND


# --------------------------------------------------------------------------- #
# Del flujo esperado al inventario deseado
# --------------------------------------------------------------------------- #

def safe_range(network: Network, reglas: Reglas, dip, rise) -> tuple[np.ndarray, np.ndarray]:
    """Rango seguro (lo, hi) del inventario al intervenir, por estación.

    `dip` es lo más que baja la trayectoria respecto al inventario inicial (0
    o negativo) y `rise` lo más que sube (0 o positivo).
    """
    low = reglas.bicis_minimas - np.minimum(0.0, dip)
    high = network.capacity - reglas.lugares_libres_minimos - np.maximum(0.0, rise)
    return low, high


def integer_range(inventory, low, high, has_info, fallback, capacity) -> tuple[np.ndarray, np.ndarray]:
    """Rango de inventarios enteros que una estación acepta, por estación.

    - Con solución: los enteros dentro de [lo, hi].
    - Sin solución, o sin ningún entero dentro: un solo valor, `fallback`,
      recortado a [0, capacidad].
    - Sin información: su inventario actual. No pide ni presta.
    """
    low_int = np.ceil(np.where(has_info, low, 0.0) - EPS).astype(np.int64)
    high_int = np.floor(np.where(has_info, high, 0.0) + EPS).astype(np.int64)
    point = has_info & (low_int > high_int)
    target = np.clip(fallback, 0, capacity).astype(np.int64)
    low_int = np.where(point, target, low_int)
    high_int = np.where(point, target, high_int)
    inventory = np.asarray(inventory, dtype=np.int64)
    return np.where(has_info, low_int, inventory), np.where(has_info, high_int, inventory)


def match_moves(inventory, low_int, high_int, priority, neighbors, budget: int) -> list[Move]:
    """Convierte inventarios deseados en traslados, sin pasar del presupuesto.

    A una estación le faltan bicis si está debajo de su rango y le sobran si
    está arriba. Se atienden por prioridad (la más alta primero), después por
    la diferencia más grande y después por id. Para cada una se buscan
    contrapartes entre sus vecinas, de la más cercana a la más lejana:

    1. primero las que tienen la necesidad contraria;
    2. después las que pueden prestar bicis o recibirlas sin salir de su rango.

    `neighbors[s]` ya trae solo las estaciones dentro de la distancia máxima.
    Mover una bici cuesta 1. Se detiene cuando se acaba el presupuesto.
    """
    current = np.asarray(inventory, dtype=np.int64).copy()
    gap = np.where(current < low_int, low_int - current, np.where(current > high_int, current - high_int, 0))
    pending = np.nonzero(gap)[0]
    order = pending[np.lexsort((pending, -gap[pending], -np.asarray(priority)[pending]))]
    moves: list[Move] = []
    left = int(budget)
    for s in order.tolist():
        if left <= 0:
            break
        near = neighbors[s]
        if near.size == 0:
            continue
        if current[s] < low_int[s]:
            receives = True
        elif current[s] > high_int[s]:
            receives = False
        else:
            continue  # quedó dentro de su rango al atender a otra
        for step in (1, 2):
            need = int(low_int[s] - current[s]) if receives else int(current[s] - high_int[s])
            if need <= 0 or left <= 0:
                break
            if receives:  # bicis que cada vecina puede dar
                spare = current[near] - (high_int if step == 1 else low_int)[near]
            else:         # lugares que cada vecina puede ocupar
                spare = (low_int if step == 1 else high_int)[near] - current[near]
            for position in np.nonzero(spare > 0)[0].tolist():
                other = int(near[position])
                bikes = min(need, int(spare[position]), left)
                origin, destination = (other, s) if receives else (s, other)
                moves.append(Move(origin, destination, bikes))
                current[origin] -= bikes
                current[destination] += bikes
                need -= bikes
                left -= bikes
                if need == 0 or left == 0:
                    break
    return moves


# --------------------------------------------------------------------------- #
# Fuentes de información
# --------------------------------------------------------------------------- #

def _local(ts_hour: pd.Series) -> pd.Series:
    """Hora de reloj de Nueva York, sin zona."""
    if not isinstance(ts_hour.dtype, pd.DatetimeTZDtype):
        raise SimulacionError(f"ts_hour debe traer zona horaria (tipo recibido: {ts_hour.dtype}).")
    return ts_hour.dt.tz_convert(TZ).dt.tz_localize(None)


def _extremes(cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lo más que baja (0 o menos) y lo más que sube (0 o más) una trayectoria. Eje 0 = horas."""
    return np.minimum(0.0, cumulative.min(axis=0)), np.maximum(0.0, cumulative.max(axis=0))


@dataclass(frozen=True)
class HistoricalProfile:
    """Información de la regla histórica: flujo neto medio por hora.

    Por franja y tipo de día guarda, por estación, lo más que baja y lo más
    que sube la trayectoria acumulada de la intervención al fin de la franja.
    """

    dip: dict        # (franja, tipo de día) -> lo más que baja, por estación
    rise: dict       # (franja, tipo de día) -> lo más que sube, por estación
    has_info: dict   # (franja, tipo de día) -> la estación tiene media en todas las horas
    with_history: np.ndarray  # la estación tiene al menos una fila en la historia

    @classmethod
    def from_station_hour(
        cls, history: pd.DataFrame, network: Network, reglas: Reglas, before: pd.Timestamp
    ) -> "HistoricalProfile":
        """Calcula el perfil con filas de `station_hour` anteriores a `before`.

        `before` es el primer instante del tramo simulado. Si llega una fila de
        ese instante o posterior, falla: sería ver el futuro.
        """
        if len(history) and (history["ts_hour"] >= before).any():
            raise SimulacionError(
                "La historia de la regla histórica trae filas del tramo simulado o posteriores. "
                "Solo puede usar filas anteriores (decisión 8)."
            )
        n = network.size
        station = network.index.get_indexer(history["station_id"].astype("object"))
        keep = station >= 0
        station = station[keep]
        local = _local(history["ts_hour"])[keep]
        hour = local.dt.hour.to_numpy()
        weekend = (local.dt.dayofweek.to_numpy() >= 5).astype(np.int64)
        net = history["net_flow"].to_numpy(dtype="float64")[keep]
        with_history = np.zeros(n, dtype=bool)
        with_history[station] = True

        cell = (weekend * 24 + hour) * n + station
        total = np.bincount(cell, weights=net, minlength=2 * 24 * n).reshape(2, 24, n)
        count = np.bincount(cell, minlength=2 * 24 * n).reshape(2, 24, n)
        dip, rise, has_info = {}, {}, {}
        for franja in reglas.franjas:
            hours = list(franja.hours)
            for kind_index, kind in enumerate(DAY_TYPES):
                seen = count[kind_index][hours]
                known = (seen > 0).all(axis=0)
                mean = np.where(seen > 0, total[kind_index][hours] / np.maximum(seen, 1.0), 0.0)
                low, high = _extremes(np.cumsum(mean, axis=0))
                key = (franja.nombre, kind)
                dip[key], rise[key], has_info[key] = low, high, known
        return cls(dip=dip, rise=rise, has_info=has_info, with_history=with_history)


@dataclass(frozen=True)
class ForecastWindow:
    """Pronóstico de una fecha y una franja, por estación."""

    has_info: np.ndarray     # hay pronóstico de todas las horas y la estación tiene historia
    center_dip: np.ndarray   # lo más que baja la trayectoria central
    center_rise: np.ndarray  # lo más que sube la trayectoria central
    lower_dip: np.ndarray    # lo más que baja la envolvente inferior
    upper_rise: np.ndarray   # lo más que sube la envolvente superior


@dataclass(frozen=True)
class ForecastInfo:
    """Información de la política "model": el pronóstico de cada franja de cada día."""

    windows: dict  # (fecha, franja) -> ForecastWindow

    def window(self, day: date, franja: str) -> ForecastWindow:
        try:
            return self.windows[(day, franja)]
        except KeyError:
            raise SimulacionError(f"No hay pronóstico preparado para {day} {franja}.") from None

    @classmethod
    def from_predictions(
        cls,
        predictions: pd.DataFrame,
        network: Network,
        reglas: Reglas,
        days: tuple[date, ...],
        with_history: np.ndarray,
    ) -> "ForecastInfo":
        """Prepara las envolventes con las filas de `predictions` de un solo modelo y tramo.

        Por hora: centro m = arr_q50 - dep_q50; dispersión hacia abajo
        d = raíz((arr_q50 - arr_q10)² + (dep_q90 - dep_q50)²); hacia arriba
        u = raíz((arr_q90 - arr_q50)² + (dep_q50 - dep_q10)²). Acumulado hasta
        la hora t: inferior = suma de m - raíz(suma de d²); superior = suma de
        m + raíz(suma de u²).

        Una estación sin historia (`with_history` falso) queda sin información
        aunque tenga filas: su pronóstico es un marcador, no un pronóstico suyo.

        Limitación conocida: `predictions` solo trae filas hasta la hora del
        último viaje de cada estación. Desde la franja en que falta alguna hora
        la estación queda sin información, así que ahí la decisión depende de
        qué filas existen. El reporte de la simulación cuenta esos casos
        (`informacion.con_historia_y_sin_pronostico`) y lo declara.
        """
        n, n_days = network.size, len(days)
        station = network.index.get_indexer(predictions["station_id"].astype("object"))
        local = _local(predictions["ts_hour"])
        first = pd.Timestamp(days[0])
        day = ((local.dt.normalize() - first) // pd.Timedelta(days=1)).to_numpy()
        keep = (station >= 0) & (day >= 0) & (day < n_days)
        station, day = station[keep], day[keep].astype(np.int64)
        hour = local.dt.hour.to_numpy()[keep]
        q = {column: predictions[column].to_numpy(dtype="float64")[keep] for column in QUANTILE_COLUMNS}
        center = q["arr_q50"] - q["dep_q50"]
        down2 = (q["arr_q50"] - q["arr_q10"]) ** 2 + (q["dep_q90"] - q["dep_q50"]) ** 2
        up2 = (q["arr_q90"] - q["arr_q50"]) ** 2 + (q["dep_q50"] - q["dep_q10"]) ** 2

        cell = (day * 24 + hour) * n + station
        shape, size = (n_days, 24, n), n_days * 24 * n
        cube = {
            "center": np.bincount(cell, weights=center, minlength=size).reshape(shape),
            "down2": np.bincount(cell, weights=down2, minlength=size).reshape(shape),
            "up2": np.bincount(cell, weights=up2, minlength=size).reshape(shape),
            "count": np.bincount(cell, minlength=size).reshape(shape),
        }

        windows = {}
        for franja in reglas.franjas:
            hours = list(franja.hours)
            for d, current in enumerate(days):
                known = (cube["count"][d][hours] > 0).all(axis=0) & with_history
                path = np.cumsum(cube["center"][d][hours], axis=0)
                lower = path - np.sqrt(np.cumsum(cube["down2"][d][hours], axis=0))
                upper = path + np.sqrt(np.cumsum(cube["up2"][d][hours], axis=0))
                center_dip, center_rise = _extremes(path)
                lower_dip, _ = _extremes(lower)
                _, upper_rise = _extremes(upper)
                windows[(current, franja.nombre)] = ForecastWindow(
                    has_info=known,
                    center_dip=center_dip,
                    center_rise=center_rise,
                    lower_dip=lower_dip,
                    upper_rise=upper_rise,
                )
        return cls(windows=windows)


# --------------------------------------------------------------------------- #
# Políticas
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Targets:
    """Lo que una política quiere en una intervención, por estación."""

    low: np.ndarray        # lo
    high: np.ndarray       # hi
    has_info: np.ndarray
    feasible: np.ndarray   # lo <= hi; verdadero si no hay información
    low_int: np.ndarray    # rango entero aceptado
    high_int: np.ndarray
    desired: np.ndarray    # inventario deseado: el actual llevado al rango entero


def _targets(inventory, low, high, has_info, fallback, capacity) -> Targets:
    low_int, high_int = integer_range(inventory, low, high, has_info, fallback, capacity)
    return Targets(
        low=low,
        high=high,
        has_info=has_info,
        feasible=~has_info | (low <= high + EPS),
        low_int=low_int,
        high_int=high_int,
        desired=np.clip(np.asarray(inventory, dtype=np.int64), low_int, high_int),
    )


class NoIntervention:
    """Sin intervención: nunca mueve bicis."""

    name = "none"

    def decide(self, context: InterventionContext) -> list[Move]:
        return []


class _RangePolicy:
    """Lo común de la regla histórica y el modelo: del rango deseado a los traslados."""

    name = ""

    def __init__(self, network: Network, reglas: Reglas, neighbors: list | None = None):
        self.network = network
        self.reglas = reglas
        self.neighbors = neighbors if neighbors is not None else network.neighbors_within(reglas.distancia_maxima_km)
        self.priority = np.array([reglas.prioridad[kind] for kind in network.station_type], dtype=np.int64)

    def targets(self, context: InterventionContext) -> Targets:
        raise NotImplementedError

    def decide(self, context: InterventionContext) -> list[Move]:
        if context.budget <= 0:
            return []
        targets = self.targets(context)
        return match_moves(
            context.inventory, targets.low_int, targets.high_int, self.priority, self.neighbors, context.budget
        )


class HistoricalPolicy(_RangePolicy):
    """Regla histórica: rango seguro con el flujo neto medio de la historia.

    Sin solución usa el punto medio de `lo` y `hi`, recortado a [0, capacidad].
    No usa `ocupacion_respaldo`.
    """

    name = "historical"

    def __init__(self, network: Network, reglas: Reglas, profile: HistoricalProfile, neighbors: list | None = None):
        super().__init__(network, reglas, neighbors)
        self.profile = profile

    def targets(self, context: InterventionContext) -> Targets:
        key = (context.franja.nombre, day_type(context.day))
        low, high = safe_range(self.network, self.reglas, self.profile.dip[key], self.profile.rise[key])
        midpoint = round_half_up((low + high) / 2)
        return _targets(context.inventory, low, high, self.profile.has_info[key], midpoint, self.network.capacity)


class ModelPolicy(_RangePolicy):
    """Modelo: rango seguro con la envolvente del pronóstico.

    Sin solución usa `ocupacion_respaldo` de la franja, según el tipo de la
    estación (o `tipo_si_falta`), por la capacidad. Con solución pero sin
    ningún entero dentro del rango usa el punto medio.
    """

    name = "model"

    def __init__(self, network: Network, reglas: Reglas, forecast: ForecastInfo, neighbors: list | None = None):
        super().__init__(network, reglas, neighbors)
        self.forecast = forecast

    def targets(self, context: InterventionContext) -> Targets:
        window = self.forecast.window(context.day, context.franja.nombre)
        low, high = safe_range(self.network, self.reglas, window.lower_dip, window.upper_rise)
        occupancy = np.array(
            [self.reglas.ocupacion_respaldo[kind][context.franja.nombre] for kind in self.network.station_type]
        )
        respaldo = round_half_up(occupancy * self.network.capacity)
        fallback = np.where(low <= high + EPS, round_half_up((low + high) / 2), respaldo)
        return _targets(context.inventory, low, high, window.has_info, fallback, self.network.capacity)
