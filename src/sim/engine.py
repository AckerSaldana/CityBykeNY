"""Motor del simulador de inventario: red, calendario, eventos y corrida.

Qué hace, en el orden en que pasa:

1. `Reglas.from_operacion` toma los números de config/operacion.yaml. Aquí no
   hay ningún número de operación escrito.
2. `build_network` arma la red simulada: capacidad, coordenadas, tipo y la
   distancia en línea recta entre cada par de estaciones.
3. `build_calendar` arma los días del tramo, las franjas de cada día y el
   instante de cada intervención, en hora local de Nueva York.
4. `prepare_events` convierte los viajes en salidas y llegadas ordenadas.
5. `simulate` recorre los eventos uno por uno, llama a la política en cada
   intervención, valida y aplica sus movimientos, y lleva las cuentas.

Reglas del recorrido (puntos 11 a 16 de la especificación):

- El inventario pasa de un día al siguiente. No hay reinicios.
- Una salida de una estación con cero bicis es un viaje bloqueado: se cuenta
  y no genera llegada. Es un resultado de la simulación, no demanda medida
  (los datos solo registran demanda atendida).
- Una llegada a una estación llena se cuenta como desviada y la bici va a la
  estación con lugar más cercana, en línea recta.
- El total de bicis no cambia: las que están en estaciones más las que van en
  un viaje dan siempre el total inicial. Se comprueba en cada intervención,
  en cada cambio de día y al final. Para eso:
  - un viaje que termina fuera del conjunto simulado (sin estación de fin, o
    con una estación que no se simula) devuelve la bici, a la hora en que
    terminó, a la estación simulada más cercana al punto donde terminó; si no
    trae ese punto, a su estación de origen. Si está llena, se desvía como
    cualquier llegada;
  - un viaje que empieza fuera del conjunto y trae el punto de inicio sale de
    la estación simulada más cercana a ese punto (bloqueado si está vacía);
    sin punto de inicio, el viaje se ignora y se cuenta;
  - un viaje que ya iba en camino al empezar el tramo se ignora y se cuenta;
  - un viaje que sigue en camino al terminar el tramo deja su bici "en
    tránsito", dentro del total. De su llegada no se mira nada.
- En instantes iguales van primero las llegadas y después las salidas; el
  desempate es el `ride_id`. La llegada de un viaje de duración cero va
  después de su propia salida.
- Una intervención ve el inventario antes de los viajes de ese mismo
  instante. Sus movimientos se aplican al momento: es una simplificación.
- La política recibe solo un `InterventionContext`. Nunca recibe viajes.
- El riesgo se mide en tiempo exacto: una estación está en riesgo mientras
  tiene menos bicis que `bicis_minimas` o menos lugares libres que
  `lugares_libres_minimos`.
- Estaciones con valet (`excepciones_valet`): en los días de la semana y las
  horas que dice el archivo, el tiempo de esa estación no cuenta como riesgo
  (en ninguna política), y en una intervención cuyo instante cae dentro de uno
  de sus intervalos [inicio, fin) la estación no entrega ni recibe bicis: el
  motor rechaza el traslado. El día se toma por su nombre; un festivo no se
  trata distinto.
- Grupos aislados (`grupos_aislados`): un traslado entre una estación de un
  grupo y cualquier estación de fuera se rechaza. Las llegadas desviadas y
  las bicis que vuelven de un viaje que termina fuera del conjunto siguen su
  regla de siempre (la estación con lugar más cercana): no miran el grupo.

Las horas se guardan como microsegundos enteros desde 1970 (UTC), así las
sumas de tiempo son exactas.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd

from src.contracts import STATION_TYPES
from src.paths import TZ
from src.sim.operacion import CLAVE_PARTE, DIAS_SEMANA, presupuesto_en_bicis

US_PER_HOUR = 3_600_000_000
EARTH_RADIUS_KM = 6371.0088

# Etiquetas del reporte para el tiempo fuera de las franjas de operación: la franja de noche
# (config/datos.yaml) y lo que no cae en ninguna.
NIGHT = "noche"
OUTSIDE = "resto"
# Tipos de riesgo que se reportan. "total" es estar en cualquiera de los dos.
KINDS = ("total", "pocas_bicis", "pocos_lugares")

# Códigos de evento. Toda llegada es de un viaje cuya salida también se simula.
ARRIVAL = 0         # la bici de un viaje vuelve a una estación
DEPARTURE = 2       # salida de un viaje que termina dentro del tramo
DEPARTURE_OPEN = 3  # salida de un viaje que termina después del tramo: la bici queda en tránsito

# Estaciones vecinas que se guardan como lista para desviar llegadas rápido.
# Si ninguna tiene lugar se revisa la fila completa.
_NEAR = 16
# Holgura para comparar decimales que deberían ser iguales.
EPS = 1e-9
_CHUNK = 20_000
_POINTS_CHUNK = 2_000  # puntos por tanda al buscar la estación más cercana


class SimulacionError(ValueError):
    """La simulación no se puede correr con estas entradas o estas reglas."""


class MoveError(SimulacionError):
    """Una política pidió un movimiento que rompe un límite."""


class CheckError(RuntimeError):
    """Falló una comprobación interna: el resultado no es confiable."""


def round_half_up(values):
    """Redondea al entero más cercano; un medio exacto sube (7.5 da 8)."""
    return np.floor(np.asarray(values, dtype="float64") + 0.5 + EPS).astype(np.int64)


# --------------------------------------------------------------------------- #
# Reglas de operación
# --------------------------------------------------------------------------- #

def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


@dataclass(frozen=True)
class Franja:
    """Una franja de config/operacion.yaml. Horas "HH:MM", hora de Nueva York."""

    nombre: str
    momento_intervencion: str
    inicio: str
    fin: str
    share: float   # parte del presupuesto del día que le toca (las de todas las franjas suman 1)

    @property
    def intervencion_min(self) -> int:
        return _minutes(self.momento_intervencion)

    @property
    def inicio_min(self) -> int:
        return _minutes(self.inicio)

    @property
    def fin_min(self) -> int:
        return _minutes(self.fin)

    @property
    def hours(self) -> tuple[int, ...]:
        """Horas de reloj que tocan el periodo de la intervención al fin de la franja.

        De 05:00 a 12:00 son las horas 5 a 11. Una hora que entra a medias
        cuenta completa.
        """
        return tuple(range(self.intervencion_min // 60, math.ceil(self.fin_min / 60)))


@dataclass(frozen=True)
class Reglas:
    """Lo que el simulador usa de config/operacion.yaml."""

    franjas: tuple[Franja, ...]          # en el orden en que se interviene
    bicis_minimas: int
    lugares_libres_minimos: int
    distancia_maxima_km: float
    prioridad: dict                      # tipo -> 1, 2 o 3
    ocupacion_respaldo: dict             # tipo -> franja -> fracción de la capacidad
    presupuestos_por_estacion: tuple[float, ...]
    presupuesto_de_referencia: float
    estado: str = "provisional"
    # Estaciones con valet: días de la semana (lunes = 0) y, por estación, sus intervalos ("HH:MM", "HH:MM").
    valet_dias: tuple = ()
    valet_estaciones: tuple = ()         # ((station_id, ((inicio, fin), ...)), ...), ordenado por id
    # Grupos aislados: ((nombre, (station_id, ...)), ...). Una estación está en un grupo como mucho.
    grupos_aislados: tuple = ()

    @classmethod
    def from_operacion(cls, cfg: dict) -> "Reglas":
        """Arma las reglas con lo que devuelve `load_operacion()`."""
        needed = (
            "franjas", "umbral_riesgo", "distancia_maxima_km", "estrategia_por_tipo",
            "presupuestos_por_estacion", "presupuesto_de_referencia",
        )
        missing = [key for key in needed if key not in cfg]
        for i, franja in enumerate(cfg.get("franjas") or []):
            if CLAVE_PARTE not in franja:
                missing.append(f"franjas[{i}].{CLAVE_PARTE}")
        if missing:
            raise SimulacionError(
                f"A las reglas de operación les falta: {', '.join(missing)}. El simulador lo necesita."
            )
        franjas = tuple(
            sorted(
                (
                    Franja(
                        nombre=f["nombre"],
                        momento_intervencion=f["momento_intervencion"],
                        inicio=f["inicio"],
                        fin=f["fin"],
                        share=float(f[CLAVE_PARTE]),
                    )
                    for f in cfg["franjas"]
                ),
                key=lambda f: (f.intervencion_min, f.inicio_min),
            )
        )
        if not franjas:
            raise SimulacionError("Las reglas de operación no traen ninguna franja.")
        budgets = tuple(float(value) for value in cfg["presupuestos_por_estacion"])
        reference = float(cfg["presupuesto_de_referencia"])
        if reference not in budgets:
            raise SimulacionError("presupuesto_de_referencia debe ser uno de presupuestos_por_estacion.")
        strategy = cfg["estrategia_por_tipo"]
        valet = cfg.get("excepciones_valet") or {}
        groups = cfg.get("grupos_aislados") or {}
        members = [station for stations in groups.values() for station in stations]
        if len(set(members)) != len(members):
            raise SimulacionError("grupos_aislados: una estación está en más de un grupo.")
        return cls(
            franjas=franjas,
            bicis_minimas=int(cfg["umbral_riesgo"]["bicis_minimas"]),
            lugares_libres_minimos=int(cfg["umbral_riesgo"]["lugares_libres_minimos"]),
            distancia_maxima_km=float(cfg["distancia_maxima_km"]),
            prioridad={tipo: int(rule["prioridad"]) for tipo, rule in strategy.items()},
            ocupacion_respaldo={
                tipo: {name: float(value) for name, value in rule["ocupacion_respaldo"].items()}
                for tipo, rule in strategy.items()
            },
            presupuestos_por_estacion=budgets,
            presupuesto_de_referencia=reference,
            estado=str(cfg.get("estado", "provisional")),
            valet_dias=tuple(sorted(DIAS_SEMANA.index(dia) for dia in valet.get("dias") or ())),
            valet_estaciones=tuple(
                (str(station), tuple((str(a), str(b)) for a, b in intervals))
                for station, intervals in sorted((valet.get("estaciones") or {}).items())
            ),
            grupos_aislados=tuple(
                (str(name), tuple(str(station) for station in stations)) for name, stations in groups.items()
            ),
        )


def daily_budget(per_station: float, stations: int) -> int:
    """Presupuesto del día en bicis: bicis por estación por día por estaciones simuladas."""
    return int(presupuesto_en_bicis(per_station, stations))


def split_budget(daily: int, franjas: tuple[Franja, ...]) -> tuple[int, ...]:
    """Reparte el presupuesto del día entre las franjas, en bicis enteras.

    A cada franja le toca la parte entera de `daily x parte`. Las bicis que
    faltan para completar el total se dan de una en una a las franjas con la
    fracción más grande; si empatan, a la que interviene primero. La suma
    siempre es `daily`.
    """
    total = math.fsum(f.share for f in franjas)
    exact = [daily * f.share / total for f in franjas]
    parts = [int(math.floor(value + EPS)) for value in exact]
    pending = daily - sum(parts)
    order = sorted(range(len(franjas)), key=lambda i: (-(exact[i] - parts[i]), i))
    for i in order[:pending]:
        parts[i] += 1
    return tuple(parts)


# --------------------------------------------------------------------------- #
# Red
# --------------------------------------------------------------------------- #

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta sobre la esfera, en kilómetros. No sabe de ríos ni de puentes."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


@dataclass(frozen=True)
class Network:
    """Datos fijos de las estaciones simuladas. No trae nada de los viajes."""

    station_ids: tuple[str, ...]   # ordenados; la posición es el índice de la estación
    capacity: np.ndarray           # lugares (int64)
    lat: np.ndarray
    lon: np.ndarray
    station_type: tuple            # tipo de cada estación; None si no tiene
    typed: np.ndarray              # True si la estación tiene tipo
    dist_km: np.ndarray            # distancia en línea recta entre cada par
    nearest: np.ndarray            # por estación, las demás de la más cercana a la más lejana

    @property
    def size(self) -> int:
        return len(self.station_ids)

    @property
    def index(self) -> pd.Index:
        return pd.Index(self.station_ids)

    def neighbors_within(self, km: float) -> list[np.ndarray]:
        """Por estación, las que están a `km` o menos, de la más cercana a la más lejana."""
        within = (np.take_along_axis(self.dist_km, self.nearest, axis=1) <= km).sum(axis=1)
        return [self.nearest[s, : within[s]].astype(np.int64) for s in range(self.size)]


def build_network(stations: pd.DataFrame) -> Network:
    """Arma la red con las filas de `stations` que se van a simular.

    Todas deben traer capacidad mayor que 0. El orden es el de `station_id`,
    para que el resultado no dependa del orden de las filas.
    """
    if stations.empty:
        raise SimulacionError("No quedó ninguna estación para simular.")
    if stations["station_id"].duplicated().any():
        raise SimulacionError("stations trae ids repetidos.")
    stations = stations.sort_values("station_id", kind="stable").reset_index(drop=True)
    if stations["capacity"].isna().any() or (stations["capacity"] <= 0).any():
        raise SimulacionError("Todas las estaciones simuladas deben tener capacidad mayor que 0.")
    capacity = stations["capacity"].astype("int64").to_numpy()
    lat = stations["lat"].astype("float64").to_numpy()
    lon = stations["lon"].astype("float64").to_numpy()
    raw_type = [None if pd.isna(value) else str(value) for value in stations["station_type"]]
    unknown = sorted({value for value in raw_type if value is not None and value not in STATION_TYPES})
    if unknown:
        raise SimulacionError(f"station_type con valores que no están en las reglas: {unknown}.")

    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    keyed = dist.copy()
    np.fill_diagonal(keyed, -1.0)  # la propia estación queda primera y se quita
    nearest = np.argsort(keyed, axis=1, kind="stable")[:, 1:].astype(np.int32)
    np.fill_diagonal(dist, 0.0)
    for array in (capacity, lat, lon, dist, nearest):
        array.setflags(write=False)
    typed = np.array([value is not None for value in raw_type], dtype=bool)
    typed.setflags(write=False)
    return Network(
        station_ids=tuple(stations["station_id"].astype(str)),
        capacity=capacity,
        lat=lat,
        lon=lon,
        station_type=tuple(raw_type),
        typed=typed,
        dist_km=dist,
        nearest=nearest,
    )


def initial_inventory(network: Network, occupancy) -> np.ndarray:
    """Inventario del primer instante: ocupación (fracción de la capacidad) por capacidad, redondeado.

    `occupancy` es un número para todas las estaciones o uno por estación.
    """
    occupancy = np.broadcast_to(np.asarray(occupancy, dtype="float64"), (network.size,))
    if not np.isfinite(occupancy).all() or (occupancy < 0).any() or (occupancy > 1).any():
        raise SimulacionError("La ocupación inicial debe estar entre 0 y 1 en todas las estaciones.")
    return np.clip(round_half_up(occupancy * network.capacity), 0, network.capacity)


def nearest_station(network: Network, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Estación de la red más cercana a cada punto, en línea recta, y su distancia en km.

    Si dos estaciones quedan a la misma distancia gana la de id menor.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    index = np.empty(len(lat), dtype=np.int64)
    km = np.empty(len(lat), dtype="float64")
    if not len(lat):
        return index, km
    points, inverse = np.unique(np.stack([lat, lon], axis=1), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    best = np.empty(len(points), dtype=np.int64)
    best_km = np.empty(len(points), dtype="float64")
    for first in range(0, len(points), _POINTS_CHUNK):
        chunk = points[first : first + _POINTS_CHUNK]
        dist = haversine_km(chunk[:, 0, None], chunk[:, 1, None], network.lat[None, :], network.lon[None, :])
        choice = np.argmin(dist, axis=1)  # el primero de los empatados: la red va ordenada por id
        best[first : first + len(chunk)] = choice
        best_km[first : first + len(chunk)] = dist[np.arange(len(chunk)), choice]
    return best[inverse], best_km[inverse]


def station_groups(network: Network, reglas: Reglas) -> np.ndarray:
    """Grupo aislado de cada estación de la red: su posición en `reglas.grupos_aislados`, o -1 si no tiene.

    Un traslado solo se permite entre estaciones con el mismo valor.
    """
    group = np.full(network.size, -1, dtype=np.int64)
    position = {station: s for s, station in enumerate(network.station_ids)}
    for g, (_, stations) in enumerate(reglas.grupos_aislados):
        for station in stations:
            if station in position:
                group[position[station]] = g
    return group


def move_neighbors(network: Network, reglas: Reglas) -> list[np.ndarray]:
    """Por estación, con cuáles puede intercambiar bicis: dentro de la distancia máxima y en su mismo grupo.

    De la más cercana a la más lejana. Sin grupos aislados es `neighbors_within(distancia_maxima_km)`.
    """
    near = network.neighbors_within(reglas.distancia_maxima_km)
    group = station_groups(network, reglas)
    if (group < 0).all():
        return near
    return [other[group[other] == group[s]] for s, other in enumerate(near)]


# --------------------------------------------------------------------------- #
# Calendario
# --------------------------------------------------------------------------- #

def local_us(day: date, hhmm: str = "00:00") -> int:
    """Instante de una fecha y hora de Nueva York, en microsegundos desde 1970 (UTC)."""
    naive = pd.Timestamp(f"{day.isoformat()} {hhmm}")
    return int(naive.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").value // 1000)


def to_timestamp(instant_us: int) -> pd.Timestamp:
    """El instante como fecha con zona de Nueva York."""
    return pd.Timestamp(int(instant_us), unit="us", tz="UTC").tz_convert(TZ)


class Slot(NamedTuple):
    """Una intervención: cuándo, de qué día y para qué franja."""

    instant_us: int
    day_index: int
    franja_index: int


@dataclass(frozen=True)
class Calendar:
    """Días simulados, franjas de cada día e instantes de intervención."""

    days: tuple[date, ...]
    warmup_days: int                 # primeros días: se simulan y no cuentan
    start_us: int
    end_us: int
    day_start_us: tuple[int, ...]    # inicio de cada día y, al final, el fin del tramo
    slots: tuple[Slot, ...]          # intervenciones en orden
    window_start_us: np.ndarray      # [día, franja]: inicio de la franja
    window_end_us: np.ndarray        # [día, franja]: fin de la franja
    labels: tuple[str, ...]          # nombres de las franjas, NIGHT (si hay franja de noche) y OUTSIDE
    bin_edges_us: np.ndarray         # cortes de tiempo: días y bordes de franja
    bin_day: np.ndarray              # día de cada tramo de tiempo
    bin_label: np.ndarray            # etiqueta de cada tramo de tiempo

    @property
    def evaluated(self) -> np.ndarray:
        """Índices de los días que cuentan en las métricas."""
        return np.arange(self.warmup_days, len(self.days))


def _day_segments(franjas: tuple[Franja, ...], noche: tuple[str, str] | None) -> list[tuple[int, int, int]]:
    """Reparte las 24 horas en tramos (minuto de inicio, minuto de fin, etiqueta).

    La etiqueta es el índice de la franja de operación; si el minuto no cae en
    ninguna, la franja de noche (puede cruzar la medianoche); si tampoco, el
    resto. Donde la noche se encima con una franja de operación gana la franja.
    """
    n = len(franjas)
    night, outside = (n, n + 1) if noche else (None, n)
    cuts = {0, 24 * 60}
    for franja in franjas:
        cuts |= {franja.inicio_min, franja.fin_min}
    if noche:
        night_start, night_end = _minutes(noche[0]), _minutes(noche[1])
        if night_start == night_end:
            raise SimulacionError("La franja de noche debe tener inicio y fin distintos.")
        cuts |= {night_start, night_end}
    ordered = sorted(cuts)
    segments: list[tuple[int, int, int]] = []
    for low, high in zip(ordered, ordered[1:]):
        label = outside
        for f, franja in enumerate(franjas):
            if franja.inicio_min <= low < franja.fin_min:
                label = f
                break
        else:
            if noche and (
                night_start <= low < night_end if night_start < night_end else low >= night_start or low < night_end
            ):
                label = night
        if segments and segments[-1][2] == label:
            segments[-1] = (segments[-1][0], high, label)
        else:
            segments.append((low, high, label))
    return segments


def hour_labels(franjas: tuple[Franja, ...], noche: tuple[str, str] | None = None) -> tuple[str, ...]:
    """Franja del reporte de cada hora de reloj, de la 0 a la 23.

    Son las mismas etiquetas con las que se reporta el riesgo: el nombre de
    la franja de operación, NIGHT o OUTSIDE. Una hora lleva la etiqueta del
    minuto en que empieza: si un borde de franja no cae en hora exacta, la
    hora que lo contiene cuenta completa en la franja donde empieza.
    """
    names = tuple(f.nombre for f in franjas) + ((NIGHT,) if noche else ()) + (OUTSIDE,)
    segments = _day_segments(franjas, noche)
    return tuple(
        next(names[label] for low, high, label in segments if low <= hour * 60 < high) for hour in range(24)
    )


def build_calendar(
    first_day: date,
    last_day: date,
    franjas: tuple[Franja, ...],
    warmup_days: int = 0,
    noche: tuple[str, str] | None = None,
) -> Calendar:
    """Calendario del tramo, del primer día al último, los dos incluidos.

    `noche` es la franja de noche del reporte, ("HH:MM", "HH:MM"). Sin ella el
    tiempo fuera de las franjas de operación va todo a OUTSIDE.
    """
    n_days = (last_day - first_day).days + 1
    if n_days < 1:
        raise SimulacionError("El tramo simulado no tiene días.")
    if not 0 <= warmup_days < n_days:
        raise SimulacionError(
            f"dias_de_calentamiento ({warmup_days}) debe ser menor que los {n_days} días simulados: "
            "tiene que quedar al menos un día que cuente."
        )
    days = tuple(first_day + timedelta(days=i) for i in range(n_days))
    day_start = tuple(local_us(day) for day in days) + (local_us(last_day + timedelta(days=1)),)
    segments = _day_segments(franjas, noche)

    slots, edges, bin_day, bin_label = [], [day_start[0]], [], []
    window_start = np.zeros((n_days, len(franjas)), dtype=np.int64)
    window_end = np.zeros((n_days, len(franjas)), dtype=np.int64)
    for d, day in enumerate(days):
        cursor = day_start[d]
        for _, high, label in segments:
            until = day_start[d + 1] if high == 24 * 60 else local_us(day, f"{high // 60:02d}:{high % 60:02d}")
            if until > cursor:
                edges.append(until)
                bin_day.append(d)
                bin_label.append(label)
                cursor = until
        for f, franja in enumerate(franjas):
            window_start[d, f], window_end[d, f] = local_us(day, franja.inicio), local_us(day, franja.fin)
            slots.append(Slot(local_us(day, franja.momento_intervencion), d, f))
    slots.sort(key=lambda slot: (slot.instant_us, slot.franja_index))
    return Calendar(
        days=days,
        warmup_days=warmup_days,
        start_us=day_start[0],
        end_us=day_start[-1],
        day_start_us=day_start,
        slots=tuple(slots),
        window_start_us=window_start,
        window_end_us=window_end,
        labels=tuple(f.nombre for f in franjas) + ((NIGHT,) if noche else ()) + (OUTSIDE,),
        bin_edges_us=np.array(edges, dtype=np.int64),
        bin_day=np.array(bin_day, dtype=np.int64),
        bin_label=np.array(bin_label, dtype=np.int64),
    )


def valet_windows(network: Network, days: tuple, reglas: Reglas) -> dict:
    """Intervalos con valet de cada estación de la red: {índice: [(inicio_us, fin_us), ...]}, en orden.

    Un intervalo por cada día de `days` cuyo día de la semana está en `reglas.valet_dias` y por cada
    intervalo de la estación. Las estaciones del archivo que no están en la red se ignoran.
    """
    position = {station: s for s, station in enumerate(network.station_ids)}
    out: dict[int, list] = {}
    for station, intervals in reglas.valet_estaciones:
        if station not in position:
            continue
        windows = [
            (local_us(day, start), local_us(day, end))
            for day in days
            if day.weekday() in reglas.valet_dias
            for start, end in intervals
        ]
        if windows:
            out[position[station]] = sorted(windows)
    return out


def valet_at(windows: dict, instant_us: int, size: int) -> np.ndarray:
    """Qué estaciones están en un intervalo con valet [inicio, fin) en ese instante."""
    active = np.zeros(size, dtype=bool)
    for station, spans in windows.items():
        active[station] = any(start <= instant_us < end for start, end in spans)
    return active


def valet_hours(windows: dict, calendar: Calendar) -> np.ndarray:
    """Horas de estación con valet de cada día simulado: el tiempo que no puede contar como riesgo."""
    out = np.zeros(len(calendar.days), dtype="float64")
    edges = np.array(calendar.day_start_us, dtype=np.int64)
    for spans in windows.values():
        for start, end in spans:
            out += (np.clip(end, edges[:-1], edges[1:]) - np.clip(start, edges[:-1], edges[1:])) / US_PER_HOUR
    return out


def split_valet(intervals: list, windows: dict) -> tuple[list, list]:
    """Parte los intervalos en riesgo en (lo que cuenta, lo que cae en un intervalo con valet y no cuenta)."""
    if not windows:
        return intervals, []
    kept, removed = [], []
    for item in intervals:
        spans = windows.get(item[0])
        if spans is None:
            kept.append(item)
            continue
        s, cursor, end, state = item
        for low, high in spans:
            if high <= cursor:
                continue
            if low >= end:
                break
            if low > cursor:
                kept.append((s, cursor, low, state))
            removed.append((s, max(cursor, low), min(end, high), state))
            cursor = min(end, high)
            if cursor >= end:
                break
        if cursor < end:
            kept.append((s, cursor, end, state))
    return kept, removed


def station_day_risk(intervals: list, calendar: Calendar, size: int) -> np.ndarray:
    """Microsegundos en riesgo (de cualquier tipo) por [día, estación]."""
    edges = np.array(calendar.day_start_us, dtype=np.int64)
    low, high = edges[:-1], edges[1:]
    out = np.zeros((size, len(low)), dtype=np.int64)
    if intervals:
        data = np.array(intervals, dtype=np.int64)
        for first in range(0, len(data), _CHUNK):
            chunk = data[first : first + _CHUNK]
            overlap = np.clip(chunk[:, 2, None], low, high) - np.clip(chunk[:, 1, None], low, high)
            np.add.at(out, chunk[:, 0], overlap)
    return out.T.copy()


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #

TRIP_COLUMNS = ("ride_id", "started_at", "ended_at", "start_station_id", "end_station_id")
# Coordenadas crudas de cada punta del viaje. Si la tabla no las trae se toman como vacías.
POINT_COLUMNS = ("start_lat", "start_lng", "end_lat", "end_lng")


@dataclass(frozen=True)
class Events:
    """Salidas y llegadas del tramo, ya ordenadas. Listas de Python: el recorrido es más rápido."""

    time_us: list
    code: list
    station: list
    trip: list                 # fila del viaje en la tabla de entrada
    time_array: np.ndarray     # los mismos instantes, para ubicar las intervenciones
    n_trips: int
    counts: dict               # cuántos eventos de cada tipo, para el reporte

    def __len__(self) -> int:
        return len(self.time_us)


def _epoch_us(column: pd.Series, name: str) -> np.ndarray:
    if not isinstance(column.dtype, pd.DatetimeTZDtype):
        raise SimulacionError(f"{name} debe traer zona horaria (tipo recibido: {column.dtype}).")
    if column.isna().any():
        raise SimulacionError(f"{name} trae valores vacíos.")
    return pd.DatetimeIndex(column).as_unit("us").asi8


def _point(trips: pd.DataFrame, lat: str, lon: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordenadas de una punta y cuáles filas las traen completas."""
    if lat not in trips.columns or lon not in trips.columns:
        empty = np.full(len(trips), np.nan)
        return empty, empty, np.zeros(len(trips), dtype=bool)
    y = pd.to_numeric(trips[lat], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    x = pd.to_numeric(trips[lon], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return y, x, np.isfinite(y) & np.isfinite(x)


def _distance_summary(km: np.ndarray) -> dict:
    if not len(km):
        return {"viajes": 0, "mediana_km": None, "p90_km": None, "p99_km": None, "maximo_km": None}
    p50, p90, p99 = np.quantile(km, [0.5, 0.9, 0.99])
    return {
        "viajes": int(len(km)),
        "mediana_km": float(p50),
        "p90_km": float(p90),
        "p99_km": float(p99),
        "maximo_km": float(km.max()),
    }


def prepare_events(trips: pd.DataFrame, network: Network, start_us: int, end_us: int) -> Events:
    """Convierte los viajes en eventos de las estaciones simuladas dentro de [start_us, end_us).

    Solo cuentan los viajes que EMPIEZAN dentro del tramo. Cada uno saca una
    bici de una estación simulada y, si termina dentro del tramo, la devuelve
    a otra (o a la misma): así el total de bicis no cambia.

    - Estación de salida: la del viaje si se simula; si no, la simulada más
      cercana al punto de inicio; sin punto de inicio, el viaje se ignora.
    - Estación de llegada: la del viaje si se simula; si no, la simulada más
      cercana al punto de fin; sin punto de fin, la estación de salida.
    - Un viaje que termina después del tramo solo tiene salida. De su llegada
      no se mira ni la estación ni el punto: así ningún dato posterior al
      tramo simulado (la prueba final) entra en la cuenta.
    - Un viaje que empezó antes del tramo se ignora.
    """
    missing = [column for column in TRIP_COLUMNS if column not in trips.columns]
    if missing:
        raise SimulacionError(f"A la tabla de viajes le faltan columnas: {missing}.")
    if trips["ride_id"].isna().any() or trips["ride_id"].duplicated().any():
        raise SimulacionError("ride_id debe venir completo y sin repetidos: es el desempate del orden.")
    n = len(trips)
    started = _epoch_us(trips["started_at"], "started_at")
    ended = _epoch_us(trips["ended_at"], "ended_at")
    index = network.index
    origin = index.get_indexer(trips["start_station_id"].astype("object")).astype(np.int64)
    destination = index.get_indexer(trips["end_station_id"].astype("object")).astype(np.int64)
    no_start = trips["start_station_id"].isna().to_numpy()
    no_end = trips["end_station_id"].isna().to_numpy()

    starts_inside = (started >= start_us) & (started < end_us)
    ends_inside = ended < end_us  # junto con `starts_inside`: el viaje termina dentro del tramo
    if (ended[starts_inside] < started[starts_inside]).any():
        raise SimulacionError("Hay viajes que terminan antes de empezar: la limpieza debió quitarlos.")

    # Salida: de su estación o de la simulada más cercana al punto de inicio.
    start_lat, start_lon, start_point = _point(trips, "start_lat", "start_lng")
    moved_origin = starts_inside & (origin < 0) & start_point
    origin_km = np.zeros(0)
    if moved_origin.any():
        origin[moved_origin], origin_km = nearest_station(network, start_lat[moved_origin], start_lon[moved_origin])
    departs = starts_inside & (origin >= 0)
    ignored_start = starts_inside & ~departs

    # Llegada: a su estación, a la simulada más cercana al punto de fin o de vuelta a la de salida.
    arrives = departs & ends_inside
    end_lat, end_lon, end_point = _point(trips, "end_lat", "end_lng")
    outside_end = arrives & (destination < 0)
    moved_end = outside_end & end_point
    back_home = outside_end & ~end_point
    end_km = np.zeros(0)
    if moved_end.any():
        destination[moved_end], end_km = nearest_station(network, end_lat[moved_end], end_lon[moved_end])
    destination[back_home] = origin[back_home]
    open_trip = departs & ~ends_inside
    zero = arrives & (ended == started)

    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(trips["ride_id"].to_numpy(dtype=object), kind="stable")] = np.arange(n)
    rows = np.arange(n, dtype=np.int64)

    time = np.concatenate([ended[arrives], started[departs]])
    # Orden dentro de un mismo instante: llegadas, salidas y, al final, la
    # llegada de un viaje de duración cero (después de su propia salida).
    turn = np.concatenate([np.where(zero[arrives], 2, 0), np.ones(int(departs.sum()), dtype=np.int64)])
    code = np.concatenate(
        [
            np.full(int(arrives.sum()), ARRIVAL, dtype=np.int64),
            np.where(arrives[departs], DEPARTURE, DEPARTURE_OPEN),
        ]
    )
    station = np.concatenate([destination[arrives], origin[departs]])
    trip = np.concatenate([rows[arrives], rows[departs]])
    tiebreak = np.concatenate([rank[arrives], rank[departs]])
    order = np.lexsort((tiebreak, turn, time))
    time = time[order]
    # Los viajes tal como pasaron. Una salida que la simulación bloquea no saca la bici ni genera llegada:
    # eso se cuenta aparte, en cada corrida.
    counts = {
        "viajes_leidos": int(n),
        "salidas": int(departs.sum()),
        "llegadas": int(arrives.sum()),
        "salidas_por_origen": {
            "de_su_estacion": int((departs & ~moved_origin).sum()),
            "de_la_estacion_mas_cercana_al_punto_de_inicio": int(moved_origin.sum()),
            "de_ellas_sin_estacion_en_la_tabla": int((moved_origin & no_start).sum()),
            "de_ellas_con_estacion_que_no_se_simula": int((moved_origin & ~no_start).sum()),
        },
        "llegadas_por_destino": {
            "a_su_estacion": int((arrives & ~outside_end).sum()),
            "a_la_estacion_mas_cercana_al_punto_de_fin": int(moved_end.sum()),
            "de_vuelta_a_la_estacion_de_salida": int(back_home.sum()),
            "fuera_del_conjunto_sin_estacion_en_la_tabla": int((outside_end & no_end).sum()),
            "fuera_del_conjunto_con_estacion_que_no_se_simula": int((outside_end & ~no_end).sum()),
        },
        "ignorados": {
            "ya_iban_en_camino_al_empezar_el_tramo": int((started < start_us).sum()),
            "empiezan_fuera_del_conjunto_y_sin_punto_de_inicio": int(ignored_start.sum()),
            "empiezan_despues_del_tramo": int((started >= end_us).sum()),
        },
        "siguen_en_camino_al_terminar_el_tramo": int(open_trip.sum()),
        "distancia_a_la_estacion_mas_cercana": {
            "punto_de_inicio": _distance_summary(origin_km),
            "punto_de_fin": _distance_summary(end_km),
        },
    }
    return Events(
        time_us=time.tolist(),
        code=code[order].tolist(),
        station=station[order].tolist(),
        trip=trip[order].tolist(),
        time_array=time,
        n_trips=n,
        counts=counts,
    )


# --------------------------------------------------------------------------- #
# Lo que ve y lo que devuelve una política
# --------------------------------------------------------------------------- #

class Move(NamedTuple):
    """Un traslado: `bikes` bicis de la estación `origin` a la estación `destination` (índices de la red)."""

    origin: int
    destination: int
    bikes: int


@dataclass(frozen=True)
class InterventionContext:
    """Todo lo que una política recibe en una intervención. No hay viajes aquí.

    Los datos fijos de las estaciones, las reglas y la fuente de información
    de la política (perfil histórico o pronóstico) los tiene la política desde
    que se crea.
    """

    instant: pd.Timestamp     # momento de la intervención, hora de Nueva York
    day: date                 # fecha local
    franja: Franja            # franja que se prepara
    inventory: np.ndarray     # bicis por estación en este momento, en la corrida de esta política
    budget: int               # bicis que todavía se pueden mover en esta intervención
    valet: np.ndarray | None = None   # estaciones con valet en este instante: ni entregan ni reciben


class Policy(Protocol):
    name: str

    def decide(self, context: InterventionContext) -> list[Move]:
        """Devuelve los traslados de esta intervención."""


@dataclass(frozen=True)
class InterventionRecord:
    """Lo que pasó en una intervención."""

    day_index: int
    franja_index: int
    instant_us: int
    budget_available: int
    budget_used: int
    inventory_before: np.ndarray   # inventario que vio la política
    net: np.ndarray                # bicis que recibió (+) o entregó (-) cada estación
    moves: tuple[Move, ...]


@dataclass(frozen=True)
class RunResult:
    """Resultado de una corrida: una política con un presupuesto."""

    policy: str
    daily_budget: int
    initial_total: int
    final_inventory: np.ndarray
    blocked: np.ndarray            # por día: salidas con inventario cero (viajes bloqueados)
    diverted: np.ndarray           # por día: llegadas a estación llena
    no_room: np.ndarray            # por día: llegadas sin lugar en ninguna estación (la bici queda en tránsito)
    in_transit: np.ndarray         # bicis en un viaje al terminar cada día
    budget_available: np.ndarray   # [día, franja]
    budget_used: np.ndarray        # [día, franja]
    risk_us: np.ndarray            # [día, etiqueta, tipo de riesgo] en microsegundos
    interventions: tuple[InterventionRecord, ...]
    risk_station_us: np.ndarray | None = None   # [día, estación]: riesgo de cualquier tipo, en microsegundos
    valet_risk_us: np.ndarray | None = None     # [día]: tiempo bajo un umbral en intervalos con valet (no cuenta)

    @property
    def risk_hours(self) -> np.ndarray:
        """Horas de estación en riesgo: [día, etiqueta de franja, tipo de riesgo]."""
        return self.risk_us / US_PER_HOUR

    @property
    def final_docked(self) -> int:
        """Bicis en estaciones al final del tramo."""
        return int(self.final_inventory.sum())

    @property
    def final_in_transit(self) -> int:
        """Bicis en un viaje al final del tramo."""
        return int(self.in_transit[-1])

    @property
    def final_total(self) -> int:
        """Bicis en estaciones más bicis en tránsito al final: siempre el total inicial."""
        return self.final_docked + self.final_in_transit

    def digest(self) -> str:
        """Huella del resultado completo: dos corridas iguales dan la misma."""
        h = hashlib.sha256()
        h.update(f"{self.policy}|{self.daily_budget}|{self.initial_total}".encode())
        for array in (
            self.final_inventory, self.blocked, self.diverted, self.no_room, self.in_transit,
            self.budget_available, self.budget_used, self.risk_us,
        ):
            h.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
        for record in self.interventions:
            h.update(np.ascontiguousarray(record.inventory_before, dtype=np.int64).tobytes())
            h.update(np.array(record.moves, dtype=np.int64).tobytes())
        return h.hexdigest()


# --------------------------------------------------------------------------- #
# Riesgo por tramo de tiempo
# --------------------------------------------------------------------------- #

def bin_risk(intervals: list, calendar: Calendar) -> np.ndarray:
    """Reparte los intervalos en riesgo entre días y franjas.

    `intervals` trae (estación, inicio, fin, estado) con estado 1 = pocas
    bicis, 2 = pocos lugares, 3 = las dos cosas. Devuelve microsegundos en
    [día, etiqueta, tipo], con los tipos de KINDS.
    """
    edges = calendar.bin_edges_us
    low, high = edges[:-1], edges[1:]
    by_state = np.zeros((len(low), 4), dtype=np.int64)
    if intervals:
        data = np.array(intervals, dtype=np.int64)
        for first in range(0, len(data), _CHUNK):
            chunk = data[first : first + _CHUNK]
            overlap = np.clip(chunk[:, 2, None], low, high) - np.clip(chunk[:, 1, None], low, high)
            for state in (1, 2, 3):
                by_state[:, state] += overlap[chunk[:, 3] == state].sum(axis=0)
    by_kind = np.stack(
        [by_state[:, 1:].sum(axis=1), by_state[:, 1] + by_state[:, 3], by_state[:, 2] + by_state[:, 3]], axis=1
    )
    out = np.zeros((len(calendar.days), len(calendar.labels), len(KINDS)), dtype=np.int64)
    np.add.at(out, (calendar.bin_day, calendar.bin_label), by_kind)
    return out


# --------------------------------------------------------------------------- #
# Recorrido de eventos
# --------------------------------------------------------------------------- #

def _process(times, codes, stations, trips, inv, cap, hi, bmin, state, since, intervals, is_blocked, near, nearest):
    """Aplica un bloque de eventos seguidos. Devuelve (bloqueados, desviadas, sin lugar, cambio de bicis en tránsito).

    `state` es el riesgo de cada estación (0 = ninguno, 1 = pocas bicis,
    2 = pocos lugares, 3 = los dos) y `since` desde cuándo está así. Cada vez
    que una estación cambia de estado se cierra su intervalo en `intervals`.
    """
    blocked = diverted = no_room = transit = 0
    below = bmin - 1
    for t, code, s, trip in zip(times, codes, stations, trips):
        if code >= DEPARTURE:
            v = inv[s]
            if v:
                inv[s] = v - 1
                transit += 1
                top = hi[s]
                if v == bmin or v == top + 1:
                    old = state[s]
                    if old:
                        intervals.append((s, since[s], t, old))
                    state[s] = (1 if v - 1 < bmin else 0) | (2 if v - 1 > top else 0)
                    since[s] = t
            else:
                blocked += 1
                is_blocked[trip] = 1
        else:
            if is_blocked[trip]:
                continue  # el viaje se bloqueó al salir: no hay llegada
            v = inv[s]
            if v >= cap[s]:
                diverted += 1
                for j in near[s]:
                    if inv[j] < cap[j]:
                        break
                else:
                    for j in nearest[s].tolist():
                        if inv[j] < cap[j]:
                            break
                    else:
                        # Ninguna estación tiene lugar: la bici se queda en tránsito, dentro del total.
                        no_room += 1
                        continue
                s = j
                v = inv[s]
            inv[s] = v + 1
            transit -= 1
            top = hi[s]
            if v == below or v == top:
                old = state[s]
                if old:
                    intervals.append((s, since[s], t, old))
                state[s] = (1 if v + 1 < bmin else 0) | (2 if v + 1 > top else 0)
                since[s] = t
    return blocked, diverted, no_room, transit


def _as_index(value, size: int, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or not 0 <= int(value) < size:
        raise MoveError(f"Movimiento inválido: {what} {value!r} no es una estación de la red.")
    return int(value)


def _apply_moves(
    moves, inv, cap, network, reglas, available, state, since, intervals, instant, group=None, valet=None
) -> int:
    """Valida y aplica los traslados en orden. Devuelve las bicis movidas. Falla al primer límite roto.

    `group` es el grupo aislado de cada estación (-1 sin grupo) y `valet` qué estaciones están en un intervalo
    con valet en este instante.
    """
    used = 0
    n = network.size
    bmin = reglas.bicis_minimas
    for move in moves:
        if len(move) != 3:
            raise MoveError(f"Movimiento inválido: {move!r} (se espera origen, destino y bicis).")
        origin = _as_index(move[0], n, "origen")
        destination = _as_index(move[1], n, "destino")
        bikes = move[2]
        if isinstance(bikes, (bool, np.bool_)) or int(bikes) != bikes or bikes < 1:
            raise MoveError(f"Movimiento inválido: {bikes!r} bicis (debe ser un entero mayor que 0).")
        bikes = int(bikes)
        names = f"{network.station_ids[origin]} -> {network.station_ids[destination]}"
        if origin == destination:
            raise MoveError(f"Movimiento inválido: origen y destino son la misma estación ({names}).")
        if used + bikes > available:
            raise MoveError(
                f"Presupuesto: mover {bikes} bicis ({names}) pasa de las {available} disponibles en esta "
                f"intervención (ya se usaron {used})."
            )
        if group is not None and group[origin] != group[destination]:
            raise MoveError(
                f"Grupo aislado: {names} no están en el mismo grupo. Las estaciones de un grupo aislado solo "
                "intercambian bicis entre ellas."
            )
        if valet is not None and (valet[origin] or valet[destination]):
            raise MoveError(
                f"Valet: {names} toca una estación que está en un intervalo con valet en esta intervención; "
                "ahí no entrega ni recibe bicis."
            )
        distance = float(network.dist_km[origin, destination])
        if distance > reglas.distancia_maxima_km:
            raise MoveError(
                f"Distancia: {names} están a {distance:.3f} km y el máximo es {reglas.distancia_maxima_km} km."
            )
        if inv[origin] < bikes:
            raise MoveError(f"Disponibilidad: {names} pide {bikes} bicis y el origen tiene {inv[origin]}.")
        if inv[destination] + bikes > cap[destination]:
            raise MoveError(
                f"Capacidad: {names} lleva {bikes} bicis y al destino le quedan "
                f"{cap[destination] - inv[destination]} lugares."
            )
        for s, change in ((origin, -bikes), (destination, bikes)):
            inv[s] += change
            new = (1 if inv[s] < bmin else 0) | (2 if inv[s] > cap[s] - reglas.lugares_libres_minimos else 0)
            if new != state[s]:
                if state[s]:
                    intervals.append((s, since[s], instant, state[s]))
                state[s] = new
                since[s] = instant
        used += bikes
    return used


def _check_bounds(inv: list, cap: list, when: str) -> None:
    for s, (bikes, places) in enumerate(zip(inv, cap)):
        if not 0 <= bikes <= places:
            raise CheckError(f"Inventario fuera de [0, capacidad] {when}: estación {s} con {bikes} de {places}.")


def simulate(
    network: Network,
    events: Events,
    calendar: Calendar,
    reglas: Reglas,
    policy: Policy,
    budget: int,
    start_inventory: np.ndarray,
) -> RunResult:
    """Corre una política con un presupuesto diario (en bicis) sobre todo el tramo.

    Comprobaciones que hace siempre y que detienen la corrida si fallan:
    inventario dentro de [0, capacidad] (se revisa en cada intervención, en
    cada cambio de día y al final; entre revisiones solo cambia de una bici en
    una y nunca por debajo de 0 ni por encima de la capacidad), cada traslado
    dentro de sus cuatro límites, los traslados no crean ni destruyen bicis,
    el presupuesto de cada franja no se rebasa y el total de bicis (en
    estaciones más en tránsito) es el inicial en cada intervención, en cada
    cambio de día y al final.
    """
    n = network.size
    n_days, n_franjas = len(calendar.days), len(reglas.franjas)
    cap = network.capacity.tolist()
    inv = [int(value) for value in start_inventory]
    if len(inv) != n:
        raise SimulacionError("El inventario inicial no tiene una cifra por estación.")
    _check_bounds(inv, cap, "al inicio")
    bmin, lmin = reglas.bicis_minimas, reglas.lugares_libres_minimos
    hi = [places - lmin for places in cap]
    state = [(1 if v < bmin else 0) | (2 if v > top else 0) for v, top in zip(inv, hi)]
    since = [calendar.start_us] * n
    intervals: list = []
    is_blocked = bytearray(events.n_trips)
    near = network.nearest[:, :_NEAR].tolist()
    nearest = network.nearest
    initial_total = sum(inv)
    shares = split_budget(int(budget), reglas.franjas)
    group = station_groups(network, reglas)
    group = None if (group < 0).all() else group
    windows = valet_windows(network, calendar.days, reglas)

    blocked, diverted, no_room, in_transit = (np.zeros(n_days, dtype=np.int64) for _ in range(4))
    transit = 0  # bicis que van en un viaje en este momento

    def check_total(when: str) -> None:
        if sum(inv) + transit != initial_total or transit < 0:
            raise CheckError(
                f"El total de bicis cambió {when}: hay {sum(inv):,} en estaciones y {transit:,} en tránsito, "
                f"y el total inicial es {initial_total:,}."
            )

    available_by = np.zeros((n_days, n_franjas), dtype=np.int64)
    used_by = np.zeros((n_days, n_franjas), dtype=np.int64)
    records: list[InterventionRecord] = []

    # Puntos de control en orden: cambio de día (0), intervención (1) y fin del tramo (2).
    checkpoints = [(calendar.day_start_us[d], 0, d) for d in range(1, n_days)]
    checkpoints += [(slot.instant_us, 1, i) for i, slot in enumerate(calendar.slots)]
    checkpoints.sort()
    checkpoints.append((calendar.end_us, 2, 0))

    day, carry, k = 0, 0, 0
    for instant, kind, payload in checkpoints:
        k_end = int(np.searchsorted(events.time_array, instant, side="left"))
        if k_end > k:
            counts = _process(
                events.time_us[k:k_end], events.code[k:k_end], events.station[k:k_end], events.trip[k:k_end],
                inv, cap, hi, bmin, state, since, intervals, is_blocked, near, nearest,
            )
            for total, value in zip((blocked, diverted, no_room), counts):
                total[day] += value
            transit += counts[3]
            k = k_end
        if kind == 0:
            in_transit[day] = transit  # `day` todavía es el día que termina
            day, carry = payload, 0  # lo que sobró del día anterior no pasa
            _check_bounds(inv, cap, f"al empezar el día {calendar.days[day]}")
            check_total(f"al empezar el día {calendar.days[day]}")
        elif kind == 1:
            slot = calendar.slots[payload]
            if slot.day_index != day:
                raise CheckError("Una intervención quedó en un día que no es el suyo.")
            f = slot.franja_index
            check_total(f"antes de la intervención de {calendar.days[day]} {reglas.franjas[f].nombre}")
            available = shares[f] + carry
            before = np.array(inv, dtype=np.int64)
            before.setflags(write=False)
            valet = valet_at(windows, instant, n) if windows else None
            if valet is not None:
                valet.setflags(write=False)
            context = InterventionContext(
                instant=to_timestamp(instant),
                day=calendar.days[day],
                franja=reglas.franjas[f],
                inventory=before,
                budget=int(available),
                valet=valet,
            )
            moves = tuple(Move(*move) if len(move) == 3 else move for move in policy.decide(context))
            used = _apply_moves(
                moves, inv, cap, network, reglas, available, state, since, intervals, instant, group, valet
            )
            after = np.array(inv, dtype=np.int64)
            if int(after.sum()) != int(before.sum()):
                raise CheckError("Los traslados cambiaron el total de bicis.")
            if used > available:
                raise CheckError("Se usó más presupuesto del disponible en la franja.")
            _check_bounds(inv, cap, f"después de la intervención de {calendar.days[day]} {reglas.franjas[f].nombre}")
            carry = available - used
            available_by[day, f], used_by[day, f] = available, used
            net = after - before
            net.setflags(write=False)
            records.append(InterventionRecord(day, f, instant, int(available), int(used), before, net, moves))
    if k != len(events):
        raise CheckError("Quedaron eventos sin recorrer.")
    for s in range(n):
        if state[s]:
            intervals.append((s, since[s], calendar.end_us, state[s]))

    _check_bounds(inv, cap, "al final")
    check_total("al final del tramo")
    in_transit[n_days - 1] = transit
    final = np.array(inv, dtype=np.int64)
    # El tiempo de una estación dentro de sus intervalos con valet no cuenta como riesgo.
    intervals, in_valet = split_valet(intervals, windows)
    risk_us = bin_risk(intervals, calendar)
    risk_station_us = station_day_risk(intervals, calendar, n)
    if (risk_station_us.sum(axis=1) != risk_us[:, :, 0].sum(axis=1)).any():
        raise CheckError("El riesgo por estación no suma el riesgo por franja.")
    valet_risk_us = bin_risk(in_valet, calendar)[:, :, 0].sum(axis=1)
    lengths = np.zeros((n_days, len(calendar.labels)), dtype=np.int64)
    np.add.at(lengths, (calendar.bin_day, calendar.bin_label), np.diff(calendar.bin_edges_us))
    if (risk_us < 0).any() or (risk_us > n * lengths[:, :, None]).any():
        raise CheckError("El tiempo en riesgo no cabe en el tiempo simulado.")
    return RunResult(
        policy=policy.name,
        daily_budget=int(budget),
        initial_total=initial_total,
        final_inventory=final,
        blocked=blocked,
        diverted=diverted,
        no_room=no_room,
        in_transit=in_transit,
        budget_available=available_by,
        budget_used=used_by,
        risk_us=risk_us,
        interventions=tuple(records),
        risk_station_us=risk_station_us,
        valet_risk_us=valet_risk_us,
    )
