"""Datos SINTÉTICOS de muestra para construir el dashboard antes de tener datos reales.

Genera los seis archivos de contrato en data/sample con números INVENTADOS.
No son resultados: ningún número de aquí se puede citar en el pitch.

Primero se inventan los viajes (de qué estación a cuál, en qué hora). De esos
mismos viajes salen station_hour (salidas y llegadas por hora) y routes (viajes
entre pares de estaciones en todo el periodo), así que las dos tablas cuadran.

Para que nadie los confunda con datos reales, las estaciones tienen id
"SIM-001", "SIM-002", ... y nombres "Sintética 01 - ...", y la carpeta lleva un
LEEME.md que este mismo módulo escribe (así nunca queda desactualizado).

Todo es determinista: la semilla es fija y dos corridas dan tablas idénticas.

Uso:
    python scripts/make_sample_data.py [--salida CARPETA]

CARPETA no puede ser data/processed ni data/raw (ni estar dentro): ahí van solo
datos reales, y el dashboard mostraría los inventados como si fueran resultados.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.contracts import (
    CONTRACTS,
    POLICIES,
    PREDICTIONS,
    RECOMMENDATIONS,
    RISK_LEVELS,
    STATION_TYPES,
    ContractError,
    validate_file,
    write_contract,
)
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE, ROOT, TZ
from src.sim.operacion import load_operacion, presupuesto_en_bicis

SEED = 20251013

# 28 días en hora local de Nueva York. END no se incluye.
# El periodo contiene a propósito el fin del horario de verano.
START, END = "2025-10-13", "2025-11-10"
DST_DAY = pd.Timestamp("2025-11-02", tz=TZ)     # ese día tiene 25 horas
VAL_START = pd.Timestamp("2025-10-27", tz=TZ)   # antes de esto: historia del baseline
TEST_START = pd.Timestamp("2025-11-03", tz=TZ)  # val = 27 oct a 2 nov, test = 3 a 9 nov

DEFAULT_CAPACITY = 20        # capacidad que se supone cuando capacity viene vacío
MAX_TOTAL_BYTES = 1_500_000  # la carpeta va en Git: debe seguir siendo pequeña

# Estaciones que no cubren todo el periodo: (primer día con viajes, último día
# con viajes). None = desde el inicio o hasta el final. Fuera de esos días la
# estación no tiene viajes, y por eso tampoco filas: station_hour solo lleva
# filas entre el primer y el último viaje de cada estación.
# Al cambiarlas, cuidar dos cosas (el generador se detiene si no se cumplen):
# todas cubren el día de 25 horas, y las que empiezan tarde tienen una semana
# completa antes de VAL_START, que es la historia del baseline.
PARTIAL_STATIONS = {
    "SIM-015": ("2025-10-19", None),          # instalada a medio periodo
    "SIM-004": (None, "2025-11-04"),          # retirada antes del final
    "SIM-029": ("2025-10-17", "2025-11-06"),  # las dos cosas
}

# Zonas donde se colocan las estaciones: (tipo, cuántas, lat, lon, radio en grados).
# Son puntos aproximados de Manhattan (debajo de la calle 59) y Brooklyn cercano.
CLUSTERS = [
    ("oficinas", 7, 40.7540, -73.9840, 0.0050),     # Midtown
    ("oficinas", 5, 40.7085, -74.0100, 0.0030),     # Distrito Financiero
    ("residencial", 4, 40.7270, -73.9830, 0.0040),  # East Village
    ("residencial", 4, 40.7140, -73.9560, 0.0040),  # Williamsburg
    ("residencial", 4, 40.6730, -73.9790, 0.0040),  # Park Slope
    ("recreativa", 3, 40.7400, -74.0070, 0.0015),   # orilla del río Hudson
    ("recreativa", 2, 40.6995, -73.9940, 0.0020),   # Brooklyn Bridge Park
    ("recreativa", 2, 40.6720, -73.9700, 0.0025),   # Prospect Park
    ("mixta", 4, 40.7370, -73.9900, 0.0040),        # Union Square
    ("mixta", 3, 40.6920, -73.9850, 0.0040),        # Downtown Brooklyn
    ("mixta", 2, 40.7240, -74.0000, 0.0030),        # SoHo
]

# Forma del día por tipo de estación: (tipo, ¿fin de semana?) -> pesos de
# salidas y de llegadas. Cada grupo de pesos es (base, mañana, mediodía, tarde).
PROFILES = {
    # Oficinas: llegan en la mañana y se van en la tarde.
    ("oficinas", False): ((0.05, 0.25, 0.30, 1.00), (0.05, 1.00, 0.30, 0.25)),
    ("oficinas", True): ((0.03, 0.00, 0.25, 0.05), (0.03, 0.00, 0.25, 0.05)),
    # Residencial: al revés.
    ("residencial", False): ((0.05, 1.00, 0.20, 0.25), (0.05, 0.25, 0.20, 1.00)),
    ("residencial", True): ((0.05, 0.15, 0.55, 0.20), (0.05, 0.05, 0.50, 0.35)),
    # Recreativa: mediodía del fin de semana.
    ("recreativa", False): ((0.03, 0.05, 0.35, 0.25), (0.03, 0.05, 0.35, 0.25)),
    ("recreativa", True): ((0.05, 0.10, 1.20, 0.35), (0.05, 0.20, 1.20, 0.20)),
    # Mixta: más plana.
    ("mixta", False): ((0.10, 0.35, 0.50, 0.40), (0.10, 0.35, 0.50, 0.40)),
    ("mixta", True): ((0.10, 0.10, 0.60, 0.30), (0.10, 0.10, 0.60, 0.30)),
}

# Fracción de salidas hechas por miembros (baja un poco en fin de semana).
MEMBER_SHARE = {"oficinas": 0.88, "residencial": 0.85, "recreativa": 0.55, "mixta": 0.75}

# A qué estación va cada viaje: peso inventado de cada par (inicio, fin).
ROUTE_SCALE_KM = 2.0  # cada 2 km de distancia el peso del par cae a cerca de un tercio
ROUTE_MAX_KM = 6.0    # más lejos que esto no hay viajes: no todos los pares existen
COMMUTE_BOOST = 2.5   # entre una estación residencial y una de oficinas el peso se multiplica
# Peso de volver a la misma estación (viaje redondo), frente a 1.0 de una estación pegada.
ROUND_TRIP = {"oficinas": 0.03, "residencial": 0.05, "recreativa": 0.40, "mixta": 0.08}
EARTH_RADIUS_KM = 6371.0
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre")

# Texto de la tabla del LEEME: qué contiene cada archivo.
DESCRIPTIONS = {
    "stations": (
        "Estaciones inventadas en Manhattan y Brooklyn. "
        "`capacity` y `station_type` traen vacíos a propósito."
    ),
    "station_hour": (
        "Salidas y llegadas por estación y hora, 28 días. Tabla densa: una fila por estación y hora, "
        "también las que valen cero, solo entre el primer y el último viaje de cada estación."
    ),
    "predictions": (
        "Pronóstico por estación y hora de los últimos 14 días, modelos `baseline` y `lgbm`, "
        "para las horas que tienen fila en `station_hour`. "
        "Las filas `lgbm` son relleno simulado, no un modelo."
    ),
    "recommendations": (
        "Bicis a mover por estación, franja y política, para los 7 días de test. "
        "Números inventados."
    ),
    "policy_curve": (
        "Curva de riesgo contra presupuesto, por política. `budget` es el total de bicis movidas al día. "
        "Números inventados para acomodar la gráfica."
    ),
    "routes": (
        "Viajes entre pares de estaciones, sumados sobre los 28 días. "
        "Son viajes de usuarios de una estación a otra, NO rutas de camiones de rebalanceo."
    ),
}

LEEME = """# DATOS SINTÉTICOS: NO SON RESULTADOS

Los archivos de esta carpeta son **datos inventados** por un programa. Sirven
para una sola cosa: construir y probar el dashboard mientras llegan los datos
reales.

- **No son resultados** del proyecto. No vienen de viajes reales de Citi Bike.
- **Ningún número de esta carpeta se puede citar en el pitch**, ni en láminas
  ni en conclusiones.
- Las estaciones no existen. Sus id empiezan con `SIM-` y sus nombres con
  `Sintética`.
- En `predictions.parquet`, las filas con `model = "lgbm"` son un **relleno
  simulado**. No salen de ningún modelo entrenado.
- Las recomendaciones y la curva de `policy_curve.parquet` son números
  inventados para acomodar las gráficas. No miden nada.
- `routes.parquet` son **viajes de usuarios entre estaciones**, también
  inventados. **No son rutas de camiones de rebalanceo**: de qué estación a
  cuál mover bicis no está en ningún archivo.

## Qué contiene cada archivo

| Archivo | Filas | Qué contiene |
|---|---|---|
{file_rows}

Las columnas y sus tipos son los de los contratos de `src/contracts.py`. Los
archivos reales del flujo irán en `data/processed/` con los mismos nombres y
columnas.

## Detalles que el dashboard debe aguantar

- Periodo: del 13 de octubre al 9 de noviembre de 2025, hora local de Nueva
  York.
- El 2 de noviembre de 2025 termina el horario de verano: ese día tiene 25
  horas y la 01:00 aparece dos veces. Es a propósito.
- Estaciones sin `capacity`: {no_capacity}.
- Estaciones sin `station_type`: {no_type}.
- `station_hour.parquet` es densa: cada estación tiene una fila por cada hora
  entre su primer y su último viaje, también las horas en cero ({zero_rows}
  filas sin ningún viaje). Antes del primer viaje y después del último no hay
  filas. Por eso no todas las estaciones empiezan ni terminan a la misma
  hora, y la tabla no es una malla completa de estaciones por horas.
- Estaciones que no cubren todo el periodo:
  {partial}.
  Fuera de esos días no tienen filas en `station_hour`, ni pronóstico, ni
  recomendaciones. Las demás empiezan el 13 de octubre y terminan el 9 de
  noviembre.
- Cada viaje inventado llega en la misma hora en que sale. Por eso, en cada
  hora, las llegadas de todas las estaciones suman lo mismo que las salidas.
- `routes.parquet`: una fila por par (estación de inicio, estación de fin) con
  al menos un viaje en el periodo. Sale de los mismos viajes inventados que
  `station_hour` y cuadra exacto por los dos lados: para cada estación, la
  suma de `trips` de las rutas que empiezan en ella es igual a sus
  `departures` de todo el periodo, y la suma de las rutas que terminan en ella
  es igual a sus `arrivals`. Hay más viajes entre estaciones cercanas y entre
  zonas residenciales y de oficinas. Hay {round_trips} rutas de una estación a sí
  misma (viajes redondos). No todos los pares existen: tienen fila
  {routes} de los {pairs} pares posibles.
- `predictions.parquet`: `split = "val"` es del 27 de octubre al 2 de
  noviembre; `split = "test"` es del 3 al 9 de noviembre. Las filas
  `baseline` son el promedio de los primeros 14 días de estos mismos datos
  inventados. Las filas `lgbm` son relleno simulado.
- `recommendations.parquet`: una fila por día de test, franja de
  `config/operacion.yaml`, estación y política. Solo entran las estaciones
  que tienen fila en `station_hour` a la hora en que empieza la franja.
  - En estos datos inventados, las tres políticas empiezan cada franja con el
    mismo inventario: la `ocupacion_respaldo` de esa franja (según el tipo de
    la estación) por su capacidad, redondeado. Es una simplificación de la
    muestra. En la simulación real solo el primer día empieza con la
    ocupación de respaldo, y ese día es de calentamiento; después cada día
    empieza con las bicis que dejó el anterior.
  - Con `policy = "none"` no se mueve nada: `start_inventory_target` es ese
    inventario de respaldo.
  - En las otras dos, lo que reciben unas estaciones es lo que entregan
    otras. La regla histórica (`historical`) no usa `ocupacion_respaldo`: el
    inventario que busca se inventa aparte.
  - Unas pocas filas tienen `feasible` en falso. En ellas las tres políticas
    se quedan con el inventario de respaldo y `bikes_to_move` es 0.
  - Donde falta `capacity` se supuso {default_capacity}. Donde falta `station_type` se usa
    la estrategia del tipo que dice `tipo_si_falta` en
    `config/operacion.yaml` (hoy `{default_type}`).
  - Los traslados de la muestra no respetan ningún presupuesto ni el reparto
    por franja (`parte_del_presupuesto`): son inventados. Aquí se llevan
    entre {moved_min} y {moved_max} bicis al día, según el día y la política; el
    presupuesto de referencia serían {reference_budget}. Los reales salen de la
    simulación con el presupuesto de referencia.
  - Hay filas con los tres niveles de riesgo.
- `policy_curve.parquet`: una fila por política y presupuesto. `budget` es el
  total de bicis que se pueden mover en un día. `config/operacion.yaml` da
  los presupuestos en bicis por estación por día
  (`presupuestos_por_estacion`: {per_station}).
  Aquí cada uno se multiplica por las {stations} estaciones de la muestra y se
  redondea: {budgets}.
  Si dos valores dan el mismo total, queda una sola fila.
  El presupuesto de referencia (`presupuesto_de_referencia`: {reference}) es
  la fila con `budget` = {reference_budget}.

## Cómo regenerarlos

    python scripts/make_sample_data.py

El resultado es siempre el mismo (la semilla es fija). Hay que volver a
correrlo si en `config/operacion.yaml` cambian las franjas (cuántas son, sus
nombres o sus horas), `presupuestos_por_estacion`,
`presupuesto_de_referencia`, `tipo_si_falta` o la `ocupacion_respaldo`.
Cambiar solo `parte_del_presupuesto` no cambia la muestra.

No edites estos archivos a mano. Este LEEME lo escribe el mismo script.
"""


# --------------------------------------------------------------------------- #
# Piezas pequeñas
# --------------------------------------------------------------------------- #

def _bump(hour: np.ndarray, center: float, width: float) -> np.ndarray:
    """Campana centrada en la hora `center`."""
    return np.exp(-0.5 * ((hour - center) / width) ** 2)


def _hourly_shape(weights: tuple) -> np.ndarray:
    """Curva de 24 horas a partir de los pesos (base, mañana, mediodía, tarde)."""
    hour = np.arange(24)
    base, morning, midday, evening = weights
    peaks = morning * _bump(hour, 8.5, 1.5) + midday * _bump(hour, 13.5, 3.0) + evening * _bump(hour, 17.5, 1.7)
    return base + peaks


def _poisson_quantile(mean: np.ndarray, q: float) -> np.ndarray:
    """Cuantil `q` de una Poisson para cada media: el menor k con P(X <= k) >= q."""
    mean = np.asarray(mean, dtype=float)
    top = float(mean.max())
    k = np.arange(int(top + 10 * math.sqrt(top) + 10))
    log_factorial = np.concatenate([[0.0], np.cumsum(np.log(k[1:]))])
    # El mínimo evita log(0); con media 0 el cuantil sale 0, que es lo correcto.
    log_pmf = k * np.log(np.maximum(mean, 1e-12))[:, None] - mean[:, None] - log_factorial
    cdf = np.cumsum(np.exp(log_pmf), axis=1)
    return (cdf < q).sum(axis=1).astype(float)


def _balance(moves: np.ndarray) -> np.ndarray:
    """Recorta los traslados para que sumen cero: las bicis se conservan.

    Si se quiere llevar más de lo que se retira (o al revés), se le quita una
    bici a la vez al traslado más grande del lado que sobra.
    """
    moves = np.array(moves, dtype=int)
    excess = int(moves.sum())
    side = 1 if excess > 0 else -1
    for _ in range(abs(excess)):
        moves[np.argmax(moves * side)] -= side
    return moves


def _pair_weights(stations: pd.DataFrame, demand: pd.DataFrame) -> np.ndarray:
    """Peso inventado de cada par de estaciones (fila: inicio, columna: fin).

    Pesa más un par de estaciones cercanas y un par residencial-oficinas. Los
    pares a más de ROUTE_MAX_KM pesan cero: no hay viajes entre ellos. La
    diagonal es volver a la misma estación (viaje redondo).
    """
    lat = np.radians(stations["lat"].to_numpy(dtype=float))
    lon = np.radians(stations["lon"].to_numpy(dtype=float))
    # Distancia en línea recta. A esta escala basta tratar el mapa como plano.
    north = (lat[:, None] - lat[None, :]) * EARTH_RADIUS_KM
    east = (lon[:, None] - lon[None, :]) * EARTH_RADIUS_KM * math.cos(float(lat.mean()))
    km = np.hypot(north, east)

    weights = np.exp(-km / ROUTE_SCALE_KM)
    weights[km > ROUTE_MAX_KM] = 0.0
    kind = demand["profile"].to_numpy()
    home, office = kind == "residencial", kind == "oficinas"
    commute = (home[:, None] & office[None, :]) | (office[:, None] & home[None, :])
    weights[commute] *= COMMUTE_BOOST
    np.fill_diagonal(weights, [ROUND_TRIP[k] for k in kind])
    return weights


def _day_text(moment: pd.Timestamp) -> str:
    """Día del reloj de Nueva York en palabras, por ejemplo "19 de octubre"."""
    return f"{moment.day} de {MONTHS[moment.month - 1]}"


# --------------------------------------------------------------------------- #
# Una función por archivo
# --------------------------------------------------------------------------- #

def build_stations(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estaciones inventadas, agrupadas en el mapa según su tipo.

    Devuelve la tabla del contrato y, aparte, la demanda de cada estación (el
    tipo con el que se generan sus viajes y su tamaño). Lo segundo no se guarda.
    """
    points = []
    for kind, count, lat, lon, radius in CLUSTERS:
        for _ in range(count):
            points.append((kind, lat + rng.uniform(-radius, radius), lon + rng.uniform(-radius, radius)))
    profile = [point[0] for point in points]
    n = len(points)
    station_id = [f"SIM-{i:03d}" for i in range(1, n + 1)]

    # Huecos a propósito: 3 estaciones sin capacidad y 3 sin tipo (una de ellas
    # sin ninguno de los dos). El contrato lo permite y el dashboard debe aguantarlo.
    holes = rng.choice(n, size=5, replace=False).tolist()
    no_capacity, no_type = set(holes[:3]), set(holes[2:])

    capacity = pd.array(rng.integers(15, 61, size=n), dtype="Int64")
    capacity[sorted(no_capacity)] = pd.NA
    station_type = [None if i in no_type else kind for i, kind in enumerate(profile)]
    labels = ["Sin clasificar" if kind is None else kind.capitalize() for kind in station_type]

    stations = pd.DataFrame(
        {
            "station_id": station_id,
            "name": [f"Sintética {i:02d} - {label}" for i, label in enumerate(labels, start=1)],
            "lat": [round(point[1], 6) for point in points],
            "lon": [round(point[2], 6) for point in points],
            "capacity": capacity,
            "station_type": station_type,
        }
    )
    # scale: viajes por hora en el pico más alto de la estación.
    demand = pd.DataFrame({"station_id": station_id, "profile": profile, "scale": rng.uniform(4.0, 12.0, size=n)})
    return stations, demand


def build_station_hour(
    stations: pd.DataFrame, demand: pd.DataFrame, rng: np.random.Generator, rng_routes: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Viajes inventados y, con ellos, la tabla densa por estación y hora.

    Las salidas son conteos Poisson alrededor de un perfil diario. A cada viaje
    se le elige destino con `rng_routes`, y las llegadas son esos mismos viajes
    contados en la estación de destino, en la misma hora en que salen.

    Devuelve tres cosas:
    - La tabla del contrato: una fila por estación y hora, con ceros, solo
      entre el primer y el último viaje de cada estación.
    - Las tasas con las que se generó (misma cantidad de filas, mismo orden),
      que luego usa el pronóstico de relleno.
    - Los viajes de todo el periodo entre cada par de estaciones (fila: inicio,
      columna: fin), de donde sale la tabla de rutas.
    """
    hours = pd.date_range(START, END, freq="h", tz=TZ, inclusive="left")
    hour = np.asarray(hours.hour)
    weekend = np.asarray(hours.dayofweek >= 5)

    lam_dep = np.empty((len(demand), len(hours)))
    lam_arr = np.empty_like(lam_dep)
    for i, (profile, scale) in enumerate(zip(demand["profile"], demand["scale"])):
        for is_weekend in (False, True):
            dep_weights, arr_weights = PROFILES[profile, is_weekend]
            rows = weekend == is_weekend
            lam_dep[i, rows] = scale * _hourly_shape(dep_weights)[hour[rows]]
            lam_arr[i, rows] = scale * _hourly_shape(arr_weights)[hour[rows]]

    # Un factor por día, igual para todas las estaciones (imita días buenos y malos).
    day_index, days = pd.factorize(hours.normalize())
    day_factor = rng.lognormal(0.0, 0.15, size=len(days))[day_index]
    lam_dep *= day_factor
    lam_arr *= day_factor

    # Estaciones que no cubren todo el periodo: fuera de sus días no salen ni llegan viajes.
    station_ids = stations["station_id"].tolist()
    day = hours.tz_localize(None).normalize()
    active = np.ones(lam_dep.shape, dtype=bool)
    for station_id, (first_day, last_day) in PARTIAL_STATIONS.items():
        i = station_ids.index(station_id)
        if first_day is not None:
            active[i] &= np.asarray(day >= pd.Timestamp(first_day))
        if last_day is not None:
            active[i] &= np.asarray(day <= pd.Timestamp(last_day))

    departures = rng.poisson(lam_dep) * active
    lam_dep = lam_dep * active
    share = np.array([MEMBER_SHARE[p] for p in demand["profile"]])[:, None] - 0.10 * weekend[None, :]
    member = rng.binomial(departures, share)

    # Destino de cada viaje. La probabilidad de ir de la estación i a la j en una
    # hora es el peso del par por las llegadas que el perfil de j espera a esa hora.
    pull = lam_arr * active
    chance = _pair_weights(stations, demand)[:, None, :] * pull.T[None, :, :]  # (inicio, hora, fin)
    total = chance.sum(axis=2, keepdims=True)
    assert (total[departures > 0] > 0).all(), "hay viajes sin ningún destino posible"
    chance = np.divide(chance, total, out=np.zeros_like(chance), where=total > 0)
    trips = rng_routes.multinomial(departures, chance)  # (inicio, hora, fin)
    arrivals = trips.sum(axis=0).T                      # (fin, hora)
    flows = trips.sum(axis=1)                           # (inicio, fin)
    lam_arr = np.einsum("ih,ihj->jh", lam_dep, chance)  # llegadas esperadas con esos destinos

    grid = pd.MultiIndex.from_product([station_ids, hours], names=["station_id", "ts_hour"])
    frame = grid.to_frame(index=False)
    frame["departures"] = departures.ravel()
    frame["arrivals"] = arrivals.ravel()
    frame["net_flow"] = frame["arrivals"] - frame["departures"]
    frame["departures_member"] = member.ravel()
    frame["departures_casual"] = frame["departures"] - frame["departures_member"]
    rates = pd.DataFrame({"dep": lam_dep.ravel(), "arr": lam_arr.ravel()})

    # Tabla densa: de cada estación quedan solo las horas entre su primer y su
    # último viaje. Dentro de ese tramo se conservan las horas en cero.
    busy = (departures + arrivals) > 0
    assert busy.any(axis=1).all(), "hay estaciones sin ningún viaje"
    first = busy.argmax(axis=1)
    last = busy.shape[1] - 1 - busy[:, ::-1].argmax(axis=1)
    position = np.arange(len(hours))
    keep = ((position >= first[:, None]) & (position <= last[:, None])).ravel()
    frame, rates = frame[keep].reset_index(drop=True), rates[keep].reset_index(drop=True)

    # El día del cambio de horario debe traer 25 horas por estación.
    per_station = frame[frame["ts_hour"].dt.normalize() == DST_DAY].groupby("station_id").size()
    assert len(per_station) == len(stations) and (per_station == 25).all(), "el 2025-11-02 debe tener 25 horas"
    return frame, rates, flows


def build_routes(stations: pd.DataFrame, flows: np.ndarray) -> pd.DataFrame:
    """Viajes entre pares de estaciones en todo el periodo: una fila por par con al menos un viaje.

    Son viajes de usuarios de una estación a otra. NO son rutas de camiones de rebalanceo.
    """
    start, end = np.nonzero(flows)
    station_ids = stations["station_id"].to_numpy()
    return pd.DataFrame(
        {"start_station_id": station_ids[start], "end_station_id": station_ids[end], "trips": flows[start, end]}
    )


def build_predictions(station_hour: pd.DataFrame, rates: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Pronósticos de los últimos 14 días (val y test) para los dos modelos.

    - "baseline": promedio por estación x día de la semana x hora, calculado
      solo con los primeros 14 días.
    - "lgbm": NO es un modelo. Es un RELLENO SIMULADO: la tasa con la que se
      generaron los datos más ruido. No sale de ningún modelo entrenado.
    """
    history = station_hour.assign(
        dow=station_hour["ts_hour"].dt.dayofweek, hour=station_hour["ts_hour"].dt.hour
    )
    is_train = history["ts_hour"] < VAL_START
    target = history.loc[~is_train, ["station_id", "ts_hour", "dow", "hour"]].reset_index(drop=True)
    target["split"] = np.where(target["ts_hour"] < TEST_START, "val", "test")

    keys = ["station_id", "dow", "hour"]
    means = history[is_train].groupby(keys, as_index=False)[["departures", "arrivals"]].mean()
    baseline = target.merge(means, on=keys, how="left", validate="many_to_one")
    assert baseline[["departures", "arrivals"]].notna().all().all(), "falta historia para el baseline"

    # Relleno simulado (NO es salida de LightGBM). Dos decimales para que el archivo pese poco.
    noise = rng.lognormal(0.0, 0.20, size=(len(target), 2))
    placeholder = np.round(rates.loc[~is_train.to_numpy(), ["dep", "arr"]].to_numpy() * noise, 2)

    medians = {
        "baseline": (baseline["departures"].to_numpy(), baseline["arrivals"].to_numpy()),
        "lgbm": (placeholder[:, 0], placeholder[:, 1]),
    }
    parts = []
    for model, (dep_q50, arr_q50) in medians.items():
        part = target[["station_id", "ts_hour", "split"]].copy()
        part["model"] = model
        for prefix, q50 in (("dep", dep_q50), ("arr", arr_q50)):
            # Cuantiles de una Poisson con esa media. Se recortan contra q50 para
            # que nunca se crucen (con medias muy chicas el cuantil 90 es 0).
            part[f"{prefix}_q10"] = np.minimum(_poisson_quantile(q50, 0.10), q50)
            part[f"{prefix}_q50"] = q50
            part[f"{prefix}_q90"] = np.maximum(_poisson_quantile(q50, 0.90), q50)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[PREDICTIONS.columns]


def build_recommendations(
    stations: pd.DataFrame, station_hour: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Recomendaciones inventadas: día de test x franja x estación x política.

    `ocupacion_respaldo` de config/operacion.yaml se usa como quedó decidido:

    - Es el inventario inicial común de las tres políticas. Aquí, las tres
      empiezan cada franja con ocupacion_respaldo x capacidad, redondeado.
      Es una simplificación de la muestra: en la simulación real solo el
      primer día empieza así y después el inventario pasa de un día al otro.
    - Es el valor que se usa cuando no hay solución factible. Aquí, en las
      filas con feasible en falso las tres políticas se quedan con ese
      inventario y no mueven nada.
    - La regla histórica NO lo usa: el inventario que busca se inventa aparte.

    Una estación sin tipo usa la estrategia del tipo que dice `tipo_si_falta`.

    Los traslados NO respetan ningún presupuesto ni el reparto por franja: son
    inventados. Los reales salen de la simulación con el presupuesto de
    referencia.

    Una estación entra en una franja solo si tiene fila en station_hour a la
    hora en que la franja empieza (ver PARTIAL_STATIONS).
    """
    capacity = stations["capacity"].fillna(DEFAULT_CAPACITY).to_numpy(dtype=int)
    span = station_hour.groupby("station_id")["ts_hour"].agg(["min", "max"]).reindex(stations["station_id"])
    parts = []
    for day in pd.date_range(TEST_START.tz_localize(None), END, freq="D", inclusive="left"):
        for franja in cfg["franjas"]:
            window_start = pd.Timestamp(f"{day:%Y-%m-%d} {franja['inicio']}", tz=TZ)
            window_end = pd.Timestamp(f"{day:%Y-%m-%d} {franja['fin']}", tz=TZ)
            present = ((span["min"] <= window_start) & (window_start <= span["max"])).to_numpy()
            n, cap = int(present.sum()), capacity[present]

            # Ocupación de respaldo de la franja según el tipo de la estación.
            # Una estación sin tipo usa la del tipo que dice tipo_si_falta.
            rules = cfg["estrategia_por_tipo"]
            by_type = {kind: rule["ocupacion_respaldo"][franja["nombre"]] for kind, rule in rules.items()}
            kinds = stations.loc[present, "station_type"]
            occupancy = np.array([by_type.get(kind, by_type[cfg["tipo_si_falta"]]) for kind in kinds])
            # Inventario de respaldo: con él empiezan la franja las tres políticas.
            fallback = np.rint(cap * occupancy).astype(int)
            # Inventario que de verdad convendría ese día. Inventado: no sale del respaldo.
            ideal = np.rint(cap * rng.uniform(0.15, 0.85, size=n)).astype(int)
            # Inventario con el que cada política quiere empezar la franja. La regla
            # histórica y el modelo se acercan a lo que conviene, el modelo con menos error.
            wanted = {
                "none": fallback,
                "historical": np.clip(ideal + rng.integers(-6, 7, size=n), 0, cap),
                "model": np.clip(ideal + rng.integers(-2, 3, size=n), 0, cap),
            }
            # feasible es de la estación y la franja, no de la política: en unas pocas
            # ningún inventario inicial evita a la vez vaciarse y llenarse.
            feasible = rng.random(n) >= 0.06
            for policy in POLICIES:
                # Sin solución factible la estación se queda con el inventario de
                # respaldo. "none" queda en ceros en todas.
                moves = _balance(np.where(feasible, wanted[policy] - fallback, 0))
                target = fallback + moves
                assert ((target >= 0) & (target <= cap)).all(), "inventario fuera de la capacidad"
                gap = np.abs(ideal - target) / cap
                parts.append(
                    pd.DataFrame(
                        {
                            "station_id": stations.loc[present, "station_id"].to_numpy(),
                            "window_start": window_start,
                            "window_end": window_end,
                            "policy": policy,
                            "start_inventory_target": target,
                            "bikes_to_move": moves,
                            "risk_level": np.select([gap > 0.30, gap > 0.12], ["alto", "medio"], default="bajo"),
                            "feasible": feasible,
                        }
                    )
                )
    frame = pd.concat(parts, ignore_index=True)[RECOMMENDATIONS.columns]

    # Las bicis se conservan: en cada ventana y política los traslados suman cero.
    moved = frame.groupby(["window_start", "policy"])["bikes_to_move"].sum()
    assert (moved == 0).all(), "los traslados no suman cero"
    assert set(frame["risk_level"]) == set(RISK_LEVELS), "deben aparecer los tres niveles de riesgo"
    return frame


def build_policy_curve(cfg: dict, n_stations: int) -> pd.DataFrame:
    """Curva inventada de riesgo contra presupuesto. Solo sirve para acomodar la gráfica.

    `budget` es el total de bicis que se pueden mover en un día: cada valor de
    presupuestos_por_estacion (bicis por estación por día) por el número de
    estaciones de la muestra, redondeado al entero.

    Con pocas estaciones, dos valores de la lista pueden dar el mismo total
    (con 40 estaciones, 0 y 0.01 dan 0 bicis). La llave del contrato es
    política y presupuesto, así que de cada total queda una sola fila.
    """
    base = 420.0  # estaciones-hora en riesgo sin mover nada (número inventado)
    # (cuánto baja el riesgo como máximo, qué tan rápido baja). "none" no baja.
    # La rapidez va en bicis por estación por día, para que la forma de la
    # curva no dependa de cuántas estaciones tiene la muestra.
    shape = {"none": (0.0, 1.0), "historical": (0.40, 0.75), "model": (0.60, 0.50)}
    per_station: dict[int, float] = {}  # total de bicis al día -> bicis por estación por día
    for value in sorted(cfg["presupuestos_por_estacion"]):
        per_station.setdefault(presupuesto_en_bicis(value, n_stations), value)
    rows = []
    for policy in POLICIES:
        drop, speed = shape[policy]
        for budget, value in per_station.items():
            # Con presupuesto 0 las tres políticas dan exactamente `base`.
            risk = round(base * (1.0 - drop * (1.0 - math.exp(-value / speed))), 1)
            rows.append((policy, budget, risk, round(risk * 0.92, 1), round(risk * 1.08, 1)))
    frame = pd.DataFrame(rows, columns=["policy", "budget", "risk_station_hours", "ci_low", "ci_high"])
    assert not frame.duplicated(["policy", "budget"]).any(), "hay presupuestos repetidos en la curva"
    return frame


# --------------------------------------------------------------------------- #
# Armado, escritura y LEEME
# --------------------------------------------------------------------------- #

def build_sample(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Arma las seis tablas sintéticas. Cada una usa su propio generador de azar.

    El generador de "routes" elige el destino de cada viaje. De esos viajes
    salen las llegadas de station_hour y la tabla de rutas.
    """
    cfg = load_operacion() if cfg is None else cfg
    rng = {name: np.random.default_rng([SEED, i]) for i, name in enumerate(CONTRACTS)}

    stations, demand = build_stations(rng["stations"])
    assert set(demand["profile"]) == set(STATION_TYPES), "hay tipos de estación sin perfil"
    station_hour, rates, flows = build_station_hour(stations, demand, rng["station_hour"], rng["routes"])
    routes = build_routes(stations, flows)

    # Las rutas cuadran con station_hour por los dos lados, estación por estación.
    totals = station_hour.groupby("station_id")[["departures", "arrivals"]].sum()
    leaving = routes.groupby("start_station_id")["trips"].sum().reindex(totals.index, fill_value=0)
    arriving = routes.groupby("end_station_id")["trips"].sum().reindex(totals.index, fill_value=0)
    assert (leaving == totals["departures"]).all(), "las rutas no suman las salidas de station_hour"
    assert (arriving == totals["arrivals"]).all(), "las rutas no suman las llegadas de station_hour"
    return {
        "stations": stations,
        "station_hour": station_hour,
        "predictions": build_predictions(station_hour, rates, rng["predictions"]),
        "recommendations": build_recommendations(stations, station_hour, cfg, rng["recommendations"]),
        "policy_curve": build_policy_curve(cfg, len(stations)),
        "routes": routes,
    }


def _partial_text(station_hour: pd.DataFrame) -> str:
    """Estaciones que empiezan después del primer día o terminan antes del último, con sus fechas."""
    first_day, last_day = pd.Timestamp(START), pd.Timestamp(END) - pd.Timedelta(days=1)
    span = station_hour.groupby("station_id")["ts_hour"].agg(["min", "max"])
    notes = []
    for station_id, first, last in span.itertuples():
        starts_late = first.tz_localize(None).normalize() > first_day
        ends_early = last.tz_localize(None).normalize() < last_day
        if starts_late and ends_early:
            notes.append(f"{station_id} (del {_day_text(first)} al {_day_text(last)})")
        elif starts_late:
            notes.append(f"{station_id} (primer viaje el {_day_text(first)})")
        elif ends_early:
            notes.append(f"{station_id} (último viaje el {_day_text(last)})")
    return ", ".join(notes)


def _number_list(values) -> str:
    """Lista en palabras: "0, 4, 10 y 20"."""
    values = [str(value) for value in values]
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " y " + values[-1]


def write_leeme(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: dict) -> Path:
    """Escribe el LEEME.md de la carpeta con los conteos reales de cada tabla.

    `cfg` es la configuración de operación con la que se armaron las tablas.
    """
    stations, station_hour, routes = tables["stations"], tables["station_hour"], tables["routes"]
    budgets = sorted(tables["policy_curve"]["budget"].unique().tolist())
    reference_budget = presupuesto_en_bicis(cfg["presupuesto_de_referencia"], len(stations))
    assert reference_budget in budgets, "el presupuesto de referencia no está en la curva"
    # Bicis llevadas por día y política (cada bici llevada a una estación se retiró de otra).
    delivered = tables["recommendations"].query("bikes_to_move > 0")
    day = delivered["window_start"].dt.tz_localize(None).dt.normalize()
    moved = delivered.groupby([delivered["policy"], day])["bikes_to_move"].sum()
    file_rows = "\n".join(
        f"| `{CONTRACTS[name].filename}` | {len(table)} | {DESCRIPTIONS[name]} |" for name, table in tables.items()
    )
    text = LEEME.format(
        file_rows=file_rows,
        no_capacity=", ".join(stations.loc[stations["capacity"].isna(), "station_id"]),
        no_type=", ".join(stations.loc[stations["station_type"].isna(), "station_id"]),
        zero_rows=int(((station_hour["departures"] + station_hour["arrivals"]) == 0).sum()),
        partial=_partial_text(station_hour),
        round_trips=int((routes["start_station_id"] == routes["end_station_id"]).sum()),
        routes=len(routes),
        pairs=len(stations) ** 2,
        default_capacity=DEFAULT_CAPACITY,
        default_type=cfg["tipo_si_falta"],
        per_station=_number_list(cfg["presupuestos_por_estacion"]),
        stations=len(stations),
        budgets=_number_list(budgets),
        reference=cfg["presupuesto_de_referencia"],
        reference_budget=reference_budget,
        moved_min=int(moved.min()),
        moved_max=int(moved.max()),
    )
    path = out_dir / "LEEME.md"
    path.write_text(text, encoding="utf-8")
    return path


def check_output_folder(out_dir: str | Path) -> Path:
    """Devuelve la carpeta de salida. Lanza ValueError si es de datos reales.

    data/processed y data/raw son solo para datos reales. El dashboard decide
    si avisa "datos sintéticos" por la carpeta de donde lee: con la muestra en
    data/processed mostraría números inventados como si fueran resultados.
    """
    out_dir = Path(out_dir)
    target = out_dir.resolve()
    for real in (DATA_PROCESSED, DATA_RAW):
        for folder in (target, *target.parents):
            # samefile además de ==: en macOS y Windows "data/PROCESSED" es la misma carpeta.
            if folder == real.resolve() or (folder.exists() and real.exists() and folder.samefile(real)):
                shown = real.relative_to(ROOT) if real.is_relative_to(ROOT) else real
                raise ValueError(
                    f"No se escriben datos SINTÉTICOS en {target}. La carpeta {shown} es solo para datos "
                    "reales: el dashboard los mostraría como si fueran resultados. "
                    "Usa data/sample (corre el script sin --salida) u otra carpeta fuera de data/processed y data/raw."
                )
    return out_dir


def write_sample(out_dir: str | Path = DATA_SAMPLE) -> dict[str, Path]:
    """Genera, guarda y vuelve a validar los seis archivos. Imprime un resumen."""
    out_dir = check_output_folder(out_dir)  # antes de crear o escribir nada
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_operacion()
    tables = build_sample(cfg)
    paths = {}
    for name, table in tables.items():
        paths[name] = write_contract(table, name, out_dir / CONTRACTS[name].filename)
    leeme = write_leeme(tables, out_dir, cfg)

    # Segunda validación, ya sobre lo que quedó en disco.
    print(f"Datos SINTÉTICOS escritos en {out_dir} (no son resultados):")
    total = leeme.stat().st_size
    for name, path in paths.items():
        problems = validate_file(path, name)
        if problems:
            raise ContractError(f"[{name}] {path} no cumple el contrato:\n- " + "\n- ".join(problems))
        size = path.stat().st_size
        total += size
        rows = pq.read_metadata(path).num_rows
        print(f"  {path.name:<24}{rows:>8} filas {size / 1024:>8.1f} KB  cumple el contrato")
    print(f"  {leeme.name:<24}{'':>14} {leeme.stat().st_size / 1024:>8.1f} KB")
    print(f"  {'Total':<24}{'':>14} {total / 1024:>8.1f} KB")
    if total > MAX_TOTAL_BYTES:
        print(f"AVISO: la carpeta pesa más de {MAX_TOTAL_BYTES / 1e6:.1f} MB. Debe seguir siendo pequeña.")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera datos SINTÉTICOS de muestra para el dashboard. No son resultados."
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=DATA_SAMPLE,
        metavar="CARPETA",
        help="carpeta donde se guardan los archivos (por omisión: data/sample). "
        "No puede ser data/processed ni data/raw: ahí van solo datos reales",
    )
    args = parser.parse_args(argv)
    try:
        out_dir = check_output_folder(args.salida)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from None  # solo el mensaje, sin traza; código de salida 1
    write_sample(out_dir)
    return 0
