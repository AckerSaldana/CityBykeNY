"""Políticas de rebalanceo: sin intervención, regla histórica y modelo.

Una política recibe en cada intervención un `InterventionContext` (instante,
franja, inventario simulado de su propia corrida y presupuesto disponible) y
devuelve traslados. Los datos fijos de las estaciones, las reglas de
config/operacion.yaml y su pronóstico los tiene desde que se crea. Ninguna
recibe viajes.

**La regla histórica y el modelo son la misma maquinaria** (`ForecastPolicy`).
Lo único que cambia entre ellas es el pronóstico que reciben:

- "none": ninguno. No mueve bicis.
- "historical" (regla histórica): los cuantiles históricos, es decir, las
  filas de `predictions` con `model = "baseline"` (promedio histórico en q50 y
  cuantiles empíricos en q10 y q90).
- "model": las filas de `predictions` con `model = "lgbm"`.

Qué hacen con su pronóstico, las dos igual:

1. Trayectoria esperada del inventario entre la intervención y el fin de la
   franja, acumulando el flujo neto hora por hora, con una banda de
   incertidumbre alrededor. La banda es la dispersión del pronóstico por un
   multiplicador (`band`): 0 es sin banda, 1 es de q10 a q90. El multiplicador
   se elige con validación y es el mismo para las dos políticas.
2. Rango seguro del inventario al intervenir. `lo` es el mínimo de bicis para
   no bajar de `bicis_minimas` en ningún momento; `hi` es el máximo para no
   dejar menos de `lugares_libres_minimos`.
3. Rango aceptado. Si `lo <= hi`, los enteros dentro de [lo, hi]: una estación
   que ya está dentro no se mueve, y una que está fuera va solo hasta el borde
   más cercano. Si no hay solución (`lo > hi`), una estación CON tipo va a
   `ocupacion_respaldo` de su tipo por su capacidad, sin quedar bajo ningún
   umbral; una estación SIN tipo no se toca en esa franja.
4. Los rangos se convierten en traslados con `match_moves`, que reparte el
   presupuesto: primero el tipo con más prioridad y, dentro de la misma
   prioridad, la estación a la que le falta o le sobra menos. Se atiende solo
   a las estaciones que están fuera de su rango. Su contraparte es primero
   una estación con la necesidad contraria y después una que ya está dentro
   de su rango, que presta o recibe sin salir de él.

Una estación sin información no pide ni presta bicis.

Aparte está el **piso de azar** (`RandomPolicy`, "random"): gasta el mismo
presupuesto por franja moviendo bicis al azar, sin pronóstico. No es una de
las políticas del contrato: sirve de referencia de lo que logra mover bicis
sin información.

Supuestos de la banda (declarados en el reporte): las horas son independientes
entre sí y las salidas son independientes de las llegadas, así que las
dispersiones se suman en cuadratura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.contracts import QUANTILE_COLUMNS
from src.paths import TZ
from src.sim.engine import EPS, InterventionContext, Move, Network, Reglas, SimulacionError, round_half_up

# Prioridad de una estación sin tipo en el reparto: va después de todas las que tienen tipo
# (las prioridades de config/operacion.yaml son 1, 2 o 3).
UNTYPED_PRIORITY = 0


# --------------------------------------------------------------------------- #
# Del flujo esperado al rango aceptado
# --------------------------------------------------------------------------- #

def safe_range(network: Network, reglas: Reglas, dip, rise) -> tuple[np.ndarray, np.ndarray]:
    """Rango seguro (lo, hi) del inventario al intervenir, por estación.

    `dip` es lo más que baja la trayectoria respecto al inventario inicial (0
    o negativo) y `rise` lo más que sube (0 o positivo).
    """
    low = reglas.bicis_minimas - np.minimum(0.0, dip)
    high = network.capacity - reglas.lugares_libres_minimos - np.maximum(0.0, rise)
    return low, high


def respaldo_inventory(network: Network, reglas: Reglas, franja: str) -> np.ndarray:
    """Inventario de respaldo de cada estación en una franja; -1 en las estaciones sin tipo.

    Es `ocupacion_respaldo` del tipo de la estación por su capacidad, llevado
    a [bicis_minimas, capacidad - lugares_libres_minimos] para no dejar la
    estación bajo un umbral. Si la estación es tan chica que ese rango no
    existe, manda bicis_minimas, sin pasar de la capacidad.
    """
    out = np.full(network.size, -1, dtype=np.int64)
    for s, kind in enumerate(network.station_type):
        if kind is None:
            continue
        places = int(network.capacity[s])
        value = int(round_half_up(reglas.ocupacion_respaldo[kind][franja] * places))
        value = max(reglas.bicis_minimas, min(value, places - reglas.lugares_libres_minimos))
        out[s] = max(0, min(value, places))
    return out


def accepted_range(inventory, low, high, has_info, respaldo, capacity) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rango de inventarios enteros que acepta cada estación, y si tiene solución.

    - Con solución (lo <= hi): los enteros dentro de [lo, hi]. Si no cabe
      ningún entero, el punto medio redondeado.
    - Sin solución y con tipo (`respaldo` >= 0): un solo valor, `respaldo`.
    - Sin solución y sin tipo (`respaldo` < 0): su inventario actual. No se toca.
    - Sin información: su inventario actual. No pide ni presta.

    Devuelve (mínimo, máximo, tiene solución). Una estación sin información
    cuenta como "con solución".
    """
    inventory = np.asarray(inventory, dtype=np.int64)
    capacity = np.asarray(capacity, dtype=np.int64)
    feasible = ~has_info | (low <= high + EPS)
    low_int = np.ceil(np.where(has_info, low, 0.0) - EPS).astype(np.int64)
    high_int = np.floor(np.where(has_info, high, 0.0) + EPS).astype(np.int64)
    middle = np.clip(round_half_up(np.where(has_info, (low + high) / 2, 0.0)), 0, capacity)
    narrow = feasible & (low_int > high_int)
    low_int = np.where(narrow, middle, low_int)
    high_int = np.where(narrow, middle, high_int)
    # Sin solución: respaldo si la estación tiene tipo; si no, se queda como está.
    point = np.where(np.asarray(respaldo) >= 0, respaldo, inventory)
    low_int = np.where(feasible, low_int, point)
    high_int = np.where(feasible, high_int, point)
    return np.where(has_info, low_int, inventory), np.where(has_info, high_int, inventory), feasible


def match_moves(inventory, low_int, high_int, priority, neighbors, budget: int) -> list[Move]:
    """Convierte rangos aceptados en traslados, sin pasar del presupuesto.

    A una estación le faltan bicis si está debajo de su rango y le sobran si
    está arriba; lo que le falta o le sobra es su diferencia con el borde más
    cercano. Se atienden por prioridad (la más alta primero), después por la
    diferencia MÁS PEQUEÑA y después por id.

    Solo se atiende a las estaciones que están fuera de su rango, y cada una
    llega como mucho al borde más cercano. Sus contrapartes, en este orden:

    1. Vecinas con la necesidad contraria (a una le faltan y a la otra le
       sobran), de la más cercana a la más lejana. Cada una va como mucho al
       borde más cercano de su rango.
    2. Si todavía falta, vecinas que están dentro de su rango, de la más
       cercana a la más lejana. **Una estación dentro de su rango presta o
       recibe solo mientras no salga de él**: presta hasta quedar en su
       mínimo y recibe hasta quedar en su máximo. Aquí entran también las del
       paso 1 que ya llegaron a su borde.

    Una estación sin información, o sin solución y sin tipo, tiene por rango
    su inventario actual: no le queda margen y no presta ni recibe.

    `neighbors[s]` ya trae solo las estaciones dentro de la distancia máxima.
    Mover una bici cuesta 1. Se detiene cuando se acaba el presupuesto.
    """
    current = np.asarray(inventory, dtype=np.int64).copy()
    gap = np.where(current < low_int, low_int - current, np.where(current > high_int, current - high_int, 0))
    pending = np.nonzero(gap)[0]
    order = pending[np.lexsort((pending, gap[pending], -np.asarray(priority)[pending]))]
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
            need = int(low_int[s] - current[s])
        elif current[s] > high_int[s]:
            receives = False
            need = int(current[s] - high_int[s])
        else:
            continue  # quedó dentro de su rango al atender a otra
        # Paso 1: vecinas con la necesidad contraria. Paso 2: vecinas dentro de su rango, sin sacarlas de él.
        for inside in (False, True):
            if need == 0 or left == 0:
                break
            theirs = current[near]
            if inside:
                in_range = (theirs >= low_int[near]) & (theirs <= high_int[near])
                margin = theirs - low_int[near] if receives else high_int[near] - theirs
                spare = np.where(in_range, margin, 0)
            else:
                # Bicis que le sobran a cada vecina (si `s` recibe) o que le faltan (si `s` entrega).
                spare = theirs - high_int[near] if receives else low_int[near] - theirs
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
# Pronóstico
# --------------------------------------------------------------------------- #

def _local(ts_hour: pd.Series) -> pd.Series:
    """Hora de reloj de Nueva York, sin zona."""
    if not isinstance(ts_hour.dtype, pd.DatetimeTZDtype):
        raise SimulacionError(f"ts_hour debe traer zona horaria (tipo recibido: {ts_hour.dtype}).")
    return ts_hour.dt.tz_convert(TZ).dt.tz_localize(None)


def _extremes(cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lo más que baja (0 o menos) y lo más que sube (0 o más) una trayectoria. Eje 0 = horas."""
    return np.minimum(0.0, cumulative.min(axis=0)), np.maximum(0.0, cumulative.max(axis=0))


def stations_with_history(history: pd.DataFrame, network: Network, before: pd.Timestamp) -> np.ndarray:
    """Qué estaciones de la red tienen al menos una fila de `station_hour` anterior a `before`.

    `before` es el primer instante del tramo simulado. Si llega una fila de
    ese instante o posterior, falla: sería ver el futuro. Una estación sin
    historia no tiene pronóstico propio (el de `predictions` es un marcador:
    el promedio del sistema), así que las políticas no la tocan.
    """
    if len(history) and (history["ts_hour"] >= before).any():
        raise SimulacionError(
            "La historia trae filas del tramo simulado o posteriores. Solo puede usar filas anteriores "
            "(decisión 8)."
        )
    known = np.zeros(network.size, dtype=bool)
    station = network.index.get_indexer(history["station_id"].astype("object"))
    known[station[station >= 0]] = True
    return known


@dataclass(frozen=True)
class ForecastWindow:
    """Pronóstico de una fecha y una franja con una banda, por estación."""

    has_info: np.ndarray     # hay pronóstico de todas las horas y la estación tiene historia
    center_dip: np.ndarray   # lo más que baja la trayectoria central
    center_rise: np.ndarray  # lo más que sube la trayectoria central
    lower_dip: np.ndarray    # lo más que baja la envolvente inferior
    upper_rise: np.ndarray   # lo más que sube la envolvente superior


@dataclass(frozen=True)
class ForecastInfo:
    """El pronóstico de cada franja de cada día, listo para aplicarle cualquier banda."""

    paths: dict   # (fecha, franja) -> (tiene información, trayectoria central, dispersión abajo, dispersión arriba)
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def window(self, day: date, franja: str, band: float = 1.0) -> ForecastWindow:
        """La franja de ese día con la banda `band` (multiplicador de la dispersión)."""
        key = (day, franja, float(band))
        if key not in self._cache:
            try:
                known, path, down, up = self.paths[(day, franja)]
            except KeyError:
                raise SimulacionError(f"No hay pronóstico preparado para {day} {franja}.") from None
            center_dip, center_rise = _extremes(path)
            lower_dip, _ = _extremes(path - float(band) * down)
            _, upper_rise = _extremes(path + float(band) * up)
            self._cache[key] = ForecastWindow(known, center_dip, center_rise, lower_dip, upper_rise)
        return self._cache[key]

    @classmethod
    def from_predictions(
        cls,
        predictions: pd.DataFrame,
        network: Network,
        reglas: Reglas,
        days: tuple[date, ...],
        with_history: np.ndarray,
    ) -> "ForecastInfo":
        """Prepara las trayectorias con las filas de `predictions` de un solo modelo y tramo.

        Por hora: centro m = arr_q50 - dep_q50; dispersión hacia abajo
        d = raíz((arr_q50 - arr_q10)² + (dep_q90 - dep_q50)²); hacia arriba
        u = raíz((arr_q90 - arr_q50)² + (dep_q50 - dep_q10)²). Acumulado hasta
        la hora t, con la banda b: inferior = suma de m - b x raíz(suma de d²);
        superior = suma de m + b x raíz(suma de u²).

        Una estación sin historia (`with_history` falso) queda sin información
        aunque tenga filas: su pronóstico es un marcador, no un pronóstico suyo.

        Limitación conocida: `predictions` solo trae filas hasta la hora del
        último viaje de cada estación. Desde la franja en que falta alguna hora
        la estación queda sin información, así que ahí la decisión depende de
        qué filas existen. El reporte de la simulación cuenta esos casos y lo
        declara.
        """
        if predictions["model"].nunique() > 1:
            raise SimulacionError("El pronóstico de una política debe traer filas de un solo modelo.")
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
        paths = {}
        for franja in reglas.franjas:
            hours = list(franja.hours)
            for d, current in enumerate(days):
                known = (cube["count"][d][hours] > 0).all(axis=0) & with_history
                path = np.cumsum(cube["center"][d][hours], axis=0)
                down = np.sqrt(np.cumsum(cube["down2"][d][hours], axis=0))
                up = np.sqrt(np.cumsum(cube["up2"][d][hours], axis=0))
                paths[(current, franja.nombre)] = (known, path, down, up)
        return cls(paths=paths)


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
    desired: np.ndarray    # inventario deseado: el actual llevado al borde más cercano del rango


class NoIntervention:
    """Sin intervención: nunca mueve bicis."""

    name = "none"

    def decide(self, context: InterventionContext) -> list[Move]:
        return []


class RandomPolicy:
    """Piso de azar: gasta el presupuesto de cada franja moviendo bicis al azar, sin ninguna información.

    No es una de las políticas del contrato (no va a `recommendations` ni a
    `policy_curve`). Es una referencia: lo que se obtiene con el mismo
    presupuesto cuando los traslados no usan pronóstico.

    En cada intervención mueve una bici a la vez hasta gastar el presupuesto:

    1. Sortea el origen entre las estaciones que tienen al menos una bici y
       alguna vecina dentro de la distancia máxima con lugar libre (todas con
       la misma probabilidad).
    2. Sortea el destino entre esas vecinas con lugar libre.

    Así cada traslado respeta los cuatro límites: bicis en el origen, lugar en
    el destino, distancia máxima y presupuesto. Se detiene antes solo si ya no
    queda ningún traslado posible. No mira umbrales ni rangos: puede vaciar o
    llenar una estación.

    El sorteo depende solo de `seed`, del día y de la franja: la misma
    intervención con el mismo inventario da siempre los mismos traslados.
    """

    name = "random"

    def __init__(self, network: Network, reglas: Reglas, seed, neighbors: list | None = None):
        self.network = network
        self.reglas = reglas
        self.seed = tuple(int(value) for value in np.atleast_1d(seed))
        self.neighbors = neighbors if neighbors is not None else network.neighbors_within(reglas.distancia_maxima_km)
        self._has_neighbors = np.array([len(near) > 0 for near in self.neighbors], dtype=bool)

    def decide(self, context: InterventionContext) -> list[Move]:
        left = int(context.budget)
        if left <= 0:
            return []
        franja = self.reglas.franjas.index(context.franja)
        rng = np.random.default_rng([*self.seed, context.day.toordinal(), franja])
        current = np.asarray(context.inventory, dtype=np.int64).copy()
        capacity = self.network.capacity
        stuck = np.zeros(self.network.size, dtype=bool)  # orígenes sin ninguna vecina con lugar, por ahora
        moves: list[Move] = []
        while left > 0:
            origins = np.nonzero(self._has_neighbors & ~stuck & (current > 0))[0]
            if origins.size == 0:
                break
            origin = int(origins[rng.integers(origins.size)])
            near = self.neighbors[origin]
            free = near[current[near] < capacity[near]]
            if free.size == 0:
                stuck[origin] = True
                continue
            destination = int(free[rng.integers(free.size)])
            moves.append(Move(origin, destination, 1))
            current[origin] -= 1
            current[destination] += 1
            left -= 1
            stuck[:] = False  # el origen ganó un lugar libre: otra estación puede volver a tener destino
        return moves


class ForecastPolicy:
    """Regla histórica o modelo: rango aceptado con la banda del pronóstico y reparto del presupuesto.

    `name` es "historical" o "model"; lo único que las distingue es el
    `forecast` que reciben. `band` es el multiplicador de la dispersión.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        reglas: Reglas,
        forecast: ForecastInfo,
        band: float,
        neighbors: list | None = None,
    ):
        self.name = name
        self.network = network
        self.reglas = reglas
        self.forecast = forecast
        self.band = float(band)
        self.neighbors = neighbors if neighbors is not None else network.neighbors_within(reglas.distancia_maxima_km)
        self.priority = np.array(
            [UNTYPED_PRIORITY if kind is None else reglas.prioridad[kind] for kind in network.station_type],
            dtype=np.int64,
        )
        self._respaldo = {
            franja.nombre: respaldo_inventory(network, reglas, franja.nombre) for franja in reglas.franjas
        }

    def targets(self, context: InterventionContext) -> Targets:
        window = self.forecast.window(context.day, context.franja.nombre, self.band)
        low, high = safe_range(self.network, self.reglas, window.lower_dip, window.upper_rise)
        low_int, high_int, feasible = accepted_range(
            context.inventory, low, high, window.has_info, self._respaldo[context.franja.nombre],
            self.network.capacity,
        )
        return Targets(
            low=low,
            high=high,
            has_info=window.has_info,
            feasible=feasible,
            low_int=low_int,
            high_int=high_int,
            desired=np.clip(np.asarray(context.inventory, dtype=np.int64), low_int, high_int),
        )

    def decide(self, context: InterventionContext) -> list[Move]:
        if context.budget <= 0:
            return []
        targets = self.targets(context)
        return match_moves(
            context.inventory, targets.low_int, targets.high_int, self.priority, self.neighbors, context.budget
        )
