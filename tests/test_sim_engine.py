"""Pruebas del motor del simulador (src/sim/engine.py).

Redes de 2 a 6 estaciones (y una de 19, para la búsqueda de lugar más allá de
las vecinas cercanas) y viajes escritos a mano. Cada regla se compara con una
cuenta hecha a mano:

a) Reglas, reparto del presupuesto, red y calendario.
b) Eventos imposibles: una salida con inventario cero es un viaje bloqueado y
   no genera llegada; una llegada a estación llena va a la estación con lugar
   más cercana.
c) El total de bicis no cambia: un viaje que termina fuera del conjunto
   devuelve la bici a la estación simulada más cercana (o a la de salida), uno
   que empieza fuera sale de la más cercana, y lo que sigue en camino al final
   queda en tránsito.
d) Riesgo en tiempo exacto, por franja y por tipo, cruzando la medianoche.
e) Presupuesto: reparto por franja, lo no usado pasa a la franja siguiente del
   mismo día y nunca al día siguiente.
f) El motor rechaza un traslado que rompe cualquiera de sus cuatro límites.
g) Los traslados conservan el total y dos corridas iguales dan lo mismo.
h) Una política recibe solo el contexto de la intervención: ningún viaje.
i) El recorrido rápido da lo mismo que un simulador de referencia, lento y
   escrito aparte, con viajes al azar.

Todo corre sin red y sin tocar data/processed. Las reglas de operación de
estas pruebas están escritas aquí: no dependen de config/operacion.yaml.
"""

import copy
import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.contracts import STATION_TYPES
from src.paths import TZ
from src.sim.engine import (
    KINDS,
    NIGHT,
    OUTSIDE,
    US_PER_HOUR,
    InterventionContext,
    Move,
    MoveError,
    Network,
    Reglas,
    SimulacionError,
    build_calendar,
    build_network,
    daily_budget,
    haversine_km,
    initial_inventory,
    local_us,
    nearest_station,
    prepare_events,
    simulate,
    split_budget,
)
from src.sim.engine import (
    hour_labels, move_neighbors, station_groups, to_timestamp, valet_at, valet_hours, valet_windows,
)
from src.sim.policies import NoIntervention

DAY1, DAY2, DAY3 = date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)
STEP = 0.005  # grados de latitud entre estaciones vecinas: 0.556 km
STEP_KM = 0.005 * np.pi / 180 * 6371.0088
TOTAL, FEW_BIKES, FEW_DOCKS = (KINDS.index(kind) for kind in ("total", "pocas_bicis", "pocos_lugares"))


def make_cfg(**changes) -> dict:
    """Reglas de operación de prueba. Dos franjas: 06-12 (interviene 05) y 15-21 (interviene 14)."""
    cfg = {
        "estado": "provisional",
        "franjas": [
            {"nombre": "manana", "momento_intervencion": "05:00", "inicio": "06:00", "fin": "12:00",
             "parte_del_presupuesto": 0.5},
            {"nombre": "tarde", "momento_intervencion": "14:00", "inicio": "15:00", "fin": "21:00",
             "parte_del_presupuesto": 0.5},
        ],
        "umbral_riesgo": {"bicis_minimas": 2, "lugares_libres_minimos": 2},
        "presupuestos_por_estacion": [0, 0.5, 1.0],
        "presupuesto_de_referencia": 1.0,
        "distancia_maxima_km": 1.0,
        "estrategia_por_tipo": {
            kind: {"prioridad": 2, "ocupacion_respaldo": {"manana": 0.5, "tarde": 0.5}} for kind in STATION_TYPES
        },
    }
    cfg.update(copy.deepcopy(changes))
    return cfg


def make_reglas(**changes) -> Reglas:
    return Reglas.from_operacion(make_cfg(**changes))


def make_network(capacities: dict, types: dict | None = None) -> Network:
    """Estaciones en línea de sur a norte, en el orden dado, separadas 0.556 km."""
    types = types or {}
    stations = pd.DataFrame(
        {
            "station_id": list(capacities),
            "name": [f"Estación {name}" for name in capacities],
            "lat": [40.70 + STEP * i for i in range(len(capacities))],
            "lon": [-73.99] * len(capacities),
            "capacity": pd.array(list(capacities.values()), dtype="Int64"),
            "station_type": pd.Series([types.get(name) for name in capacities], dtype="object"),
        }
    )
    return build_network(stations)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ)


TRIP_FIELDS = ["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"]
POINT_FIELDS = ["start_lat", "start_lng", "end_lat", "end_lng"]
LON = -73.99


def lat_of(position: float) -> float:
    """Latitud de la estación en esa posición de la línea (0 = la primera). Admite puntos intermedios."""
    return 40.70 + STEP * position


def make_trips(rows: list[tuple]) -> pd.DataFrame:
    """Viajes: (ride_id, inicio, fin, estación de inicio, estación de fin[, punto de inicio, punto de fin]).

    None = sin estación. Un punto es la posición en la línea de estaciones (1.2 = un poco al norte de la
    segunda) o None si el viaje no trae coordenadas en esa punta.
    """
    frame = pd.DataFrame([row[:5] for row in rows], columns=TRIP_FIELDS)
    for column in ("started_at", "ended_at"):
        frame[column] = pd.DatetimeIndex([ts(value) for value in frame[column]]).as_unit("us")
    points = [(*row[5:7], None, None)[:2] for row in rows]
    for side, values in (("start", [point[0] for point in points]), ("end", [point[1] for point in points])):
        frame[f"{side}_lat"] = [np.nan if value is None else lat_of(value) for value in values]
        frame[f"{side}_lng"] = [np.nan if value is None else LON for value in values]
    return frame


def run(network, trips, start, *, days=(DAY1, DAY1), reglas=None, policy=None, budget=0, warmup=0, noche=None):
    reglas = reglas or make_reglas()
    calendar = build_calendar(days[0], days[1], reglas.franjas, warmup, noche)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    result = simulate(network, events, calendar, reglas, policy or NoIntervention(), budget, np.array(start))
    return result, calendar


class Scripted:
    """Política de prueba: devuelve los traslados escritos para cada fecha y franja, y guarda lo que vio."""

    name = "historical"

    def __init__(self, plan: dict | None = None):
        self.plan = plan or {}
        self.seen: list = []

    def decide(self, *args, **kwargs):
        assert len(args) == 1 and not kwargs, "la política debe recibir solo el contexto"
        (context,) = args
        self.seen.append(context)
        return self.plan.get((context.day, context.franja.nombre), [])


# --------------------------------------------------------------------------- #
# a) Reglas, reparto del presupuesto, red y calendario
# --------------------------------------------------------------------------- #

def test_las_reglas_salen_del_yaml_y_las_franjas_quedan_en_orden_de_intervencion():
    cfg = make_cfg()
    cfg["franjas"].reverse()  # el orden del archivo no importa
    cfg["umbral_riesgo"] = {"bicis_minimas": 3, "lugares_libres_minimos": 1}
    cfg["distancia_maxima_km"] = 2.5
    cfg["estrategia_por_tipo"]["oficinas"] = {"prioridad": 3, "ocupacion_respaldo": {"manana": 0.3, "tarde": 0.7}}
    reglas = Reglas.from_operacion(cfg)
    assert [franja.nombre for franja in reglas.franjas] == ["manana", "tarde"]
    assert [franja.share for franja in reglas.franjas] == [0.5, 0.5]
    assert not hasattr(reglas, "tipo_si_falta")
    assert (reglas.bicis_minimas, reglas.lugares_libres_minimos, reglas.distancia_maxima_km) == (3, 1, 2.5)
    assert reglas.prioridad["oficinas"] == 3 and reglas.ocupacion_respaldo["oficinas"]["tarde"] == 0.7
    assert reglas.presupuestos_por_estacion == (0.0, 0.5, 1.0) and reglas.presupuesto_de_referencia == 1.0
    # De la intervención (05:00) al fin de la franja (12:00): horas 5 a 11.
    assert reglas.franjas[0].hours == (5, 6, 7, 8, 9, 10, 11)
    assert reglas.franjas[1].hours == (14, 15, 16, 17, 18, 19, 20)


@pytest.mark.parametrize(
    "key", ["presupuestos_por_estacion", "presupuesto_de_referencia", "estrategia_por_tipo", "distancia_maxima_km"]
)
def test_si_falta_una_regla_el_simulador_lo_dice(key):
    cfg = make_cfg()
    del cfg[key]
    with pytest.raises(SimulacionError, match=key):
        Reglas.from_operacion(cfg)


def test_si_falta_la_parte_del_presupuesto_de_una_franja_el_simulador_lo_dice():
    cfg = make_cfg()
    del cfg["franjas"][1]["parte_del_presupuesto"]
    with pytest.raises(SimulacionError, match=r"franjas\[1\].parte_del_presupuesto"):
        Reglas.from_operacion(cfg)


@pytest.mark.parametrize(
    "daily, parts, expected",
    [
        (10, (0.5, 0.5), (5, 5)),
        (5, (0.5, 0.5), (3, 2)),       # la bici que sobra va a la franja que interviene primero
        (10, (0.6, 0.4), (6, 4)),
        (7, (0.25, 0.75), (2, 5)),     # 1.75 y 5.25: la fracción más grande es la de la primera
        (1, (0.3, 0.7), (0, 1)),
        (0, (0.5, 0.5), (0, 0)),
        (1178, (0.5, 0.5), (589, 589)),
    ],
)
def test_el_reparto_por_franja_es_entero_y_suma_el_presupuesto_del_dia(daily, parts, expected):
    cfg = make_cfg()
    for franja, part in zip(cfg["franjas"], parts):
        franja["parte_del_presupuesto"] = part
    shares = split_budget(daily, Reglas.from_operacion(cfg).franjas)
    assert shares == expected and sum(shares) == daily


def test_el_presupuesto_del_dia_es_por_estacion_por_el_numero_de_estaciones():
    assert daily_budget(0.5, 2300) == 1150
    assert daily_budget(0.1, 2356) == 236   # 235.6
    assert daily_budget(0, 2356) == 0


def test_la_red_ordena_por_id_y_mide_en_linea_recta():
    # Se entregan desordenadas: la red las ordena por id. A y B están a un paso; B y C, a dos.
    stations = pd.DataFrame(
        {
            "station_id": ["C", "A", "B"],
            "name": ["c", "a", "b"],
            "lat": [40.70 + 3 * STEP, 40.70, 40.70 + STEP],
            "lon": [-73.99] * 3,
            "capacity": pd.array([30, 10, 20], dtype="Int64"),
            "station_type": pd.Series(["oficinas", None, None], dtype="object"),
        }
    )
    network = build_network(stations)
    assert network.station_ids == ("A", "B", "C")
    assert network.capacity.tolist() == [10, 20, 30]
    # Una estación sin tipo se queda sin tipo: no toma el de ninguna otra.
    assert network.station_type == (None, None, "oficinas") and network.typed.tolist() == [False, False, True]
    # 0.005 grados de latitud son 0.556 km.
    assert STEP_KM == pytest.approx(0.55597, abs=1e-4)
    assert network.dist_km[0, 1] == pytest.approx(STEP_KM, rel=1e-9)
    assert network.dist_km[0, 2] == pytest.approx(3 * STEP_KM, rel=1e-9)
    assert (network.dist_km == network.dist_km.T).all() and (np.diag(network.dist_km) == 0).all()
    # Las demás, de la más cercana a la más lejana.
    assert network.nearest.tolist() == [[1, 2], [0, 2], [1, 0]]
    assert [near.tolist() for near in network.neighbors_within(0.6)] == [[1], [0], []]
    assert [near.tolist() for near in network.neighbors_within(1.2)] == [[1], [0, 2], [1]]
    # Un kilómetro medido con otra fórmula: un grado de longitud en el ecuador.
    assert float(haversine_km(0.0, 0.0, 0.0, 1.0)) == pytest.approx(111.195, abs=1e-3)


def test_dos_estaciones_a_la_misma_distancia_se_ordenan_por_id():
    # "M" queda justo en medio de "K" y "Z" (coordenadas exactas en binario): empate real.
    stations = pd.DataFrame(
        {
            "station_id": ["Z", "M", "K"],
            "name": ["z", "m", "k"],
            "lat": [40.5, 40.5, 40.5],
            "lon": [-74.0 + 0.0078125, -74.0, -74.0 - 0.0078125],
            "capacity": pd.array([10, 10, 10], dtype="Int64"),
            "station_type": pd.Series([None, None, None], dtype="object"),
        }
    )
    network = build_network(stations)
    assert network.station_ids == ("K", "M", "Z")
    assert network.dist_km[1, 0] == network.dist_km[1, 2]
    assert network.nearest[1].tolist() == [0, 2]   # K antes que Z


@pytest.mark.parametrize("capacity", [None, 0])
def test_la_red_no_acepta_estaciones_sin_capacidad(capacity):
    stations = pd.DataFrame(
        {"station_id": ["A", "B"], "name": ["a", "b"], "lat": [40.7, 40.71], "lon": [-73.99, -73.99],
         "capacity": pd.array([10, capacity], dtype="Int64"), "station_type": pd.Series([None, None], dtype="object")}
    )
    with pytest.raises(SimulacionError, match="capacidad"):
        build_network(stations)


def test_el_inventario_inicial_es_la_ocupacion_por_la_capacidad_redondeada():
    network = make_network({"A": 10, "B": 15, "C": 20, "D": 11})
    # Una ocupación por estación: 0.5 x 15 = 7.5 sube a 8; 0.3 x 11 = 3.3 baja a 3.
    assert initial_inventory(network, [0.5, 0.5, 0.3, 0.3]).tolist() == [5, 8, 6, 3]
    assert initial_inventory(network, [0.0, 1.0, 0.04, 0.96]).tolist() == [0, 15, 1, 11]
    # Un solo número vale para todas.
    assert initial_inventory(network, 0.4).tolist() == [4, 6, 8, 4]
    for bad in ([0.5, 0.5, 0.5, 1.2], [0.5, np.nan, 0.5, 0.5], -0.1):
        with pytest.raises(SimulacionError, match="entre 0 y 1"):
            initial_inventory(network, bad)


def test_la_estacion_mas_cercana_a_un_punto_y_el_empate_va_al_id_menor():
    network = make_network({"A": 10, "B": 10, "C": 10})
    index, km = nearest_station(network, [lat_of(0.1), lat_of(1.4), lat_of(9.0), lat_of(0.5)], [LON] * 4)
    assert index.tolist()[:3] == [0, 1, 2]
    assert km[0] == pytest.approx(0.1 * STEP_KM, rel=1e-6) and km[2] == pytest.approx(7 * STEP_KM, rel=1e-6)
    # Empate exacto entre K y Z (ver la prueba de la red): gana K, la de id menor.
    stations = pd.DataFrame(
        {"station_id": ["Z", "K"], "name": ["z", "k"], "lat": [40.5, 40.5],
         "lon": [-74.0 + 0.0078125, -74.0 - 0.0078125], "capacity": pd.array([10, 10], dtype="Int64"),
         "station_type": pd.Series([None, None], dtype="object")}
    )
    tied, _ = nearest_station(build_network(stations), [40.5], [-74.0])
    assert tied.tolist() == [0]
    empty, _ = nearest_station(network, [], [])
    assert len(empty) == 0


def test_el_calendario_reparte_cada_dia_en_franjas_y_fuera_de_franja():
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY2, reglas.franjas, warmup_days=1)
    assert calendar.days == (DAY1, DAY2) and calendar.evaluated.tolist() == [1]
    assert calendar.labels == ("manana", "tarde", OUTSIDE)
    assert calendar.start_us == local_us(DAY1) and calendar.end_us == local_us(DAY3)
    # Cinco tramos por día: 00-06 fuera, 06-12 mañana, 12-15 fuera, 15-21 tarde, 21-24 fuera.
    hours = np.diff(calendar.bin_edges_us) / US_PER_HOUR
    assert hours.tolist() == [6, 6, 3, 6, 3] * 2
    assert calendar.bin_label.tolist() == [2, 0, 2, 1, 2] * 2
    assert calendar.bin_day.tolist() == [0] * 5 + [1] * 5
    # Intervenciones a las 05:00 y a las 14:00 de cada día, hora de Nueva York (UTC-4 en septiembre).
    instants = [pd.Timestamp(slot.instant_us, unit="us", tz="UTC").tz_convert(TZ) for slot in calendar.slots]
    assert instants == [ts("2026-09-01 05:00"), ts("2026-09-01 14:00"), ts("2026-09-02 05:00"), ts("2026-09-02 14:00")]
    assert pd.Timestamp(calendar.slots[0].instant_us, unit="us") == pd.Timestamp("2026-09-01 09:00")  # UTC
    assert calendar.window_start_us[1, 1] == local_us(DAY2, "15:00")
    assert calendar.window_end_us[1, 1] == local_us(DAY2, "21:00")


def test_el_calendario_con_franja_de_noche_separa_noche_y_resto():
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY2, reglas.franjas, noche=("22:00", "06:00"))
    assert calendar.labels == ("manana", "tarde", NIGHT, OUTSIDE)
    hours = np.diff(calendar.bin_edges_us) / US_PER_HOUR
    # 00-06 noche, 06-12 mañana, 12-15 resto, 15-21 tarde, 21-22 resto, 22-24 noche.
    assert hours.tolist() == [6, 6, 3, 6, 1, 2] * 2
    assert calendar.bin_label.tolist() == [2, 0, 3, 1, 3, 2] * 2
    # Una noche que no cruza la medianoche, y una que se encima con la mañana: gana la franja de operación.
    early = build_calendar(DAY1, DAY1, reglas.franjas, noche=("00:00", "05:00"))
    assert (np.diff(early.bin_edges_us) / US_PER_HOUR).tolist() == [5, 1, 6, 3, 6, 3]
    assert early.bin_label.tolist() == [2, 3, 0, 3, 1, 3]
    overlap = build_calendar(DAY1, DAY1, reglas.franjas, noche=("20:00", "08:00"))
    assert (np.diff(overlap.bin_edges_us) / US_PER_HOUR).tolist() == [6, 6, 3, 6, 3]
    assert overlap.bin_label.tolist() == [2, 0, 3, 1, 2]
    with pytest.raises(SimulacionError, match="noche"):
        build_calendar(DAY1, DAY1, reglas.franjas, noche=("22:00", "22:00"))


def test_el_calentamiento_debe_dejar_al_menos_un_dia_que_cuente():
    with pytest.raises(SimulacionError, match="dias_de_calentamiento"):
        build_calendar(DAY1, DAY2, make_reglas().franjas, warmup_days=2)


# --------------------------------------------------------------------------- #
# b) Eventos imposibles
# --------------------------------------------------------------------------- #

def test_una_salida_con_inventario_cero_es_un_viaje_bloqueado_y_no_genera_llegada():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),   # A está en cero: bloqueado
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", "B", "A"),   # B tiene 5: sale y llega
            ("r3", "2026-09-01 10:00", "2026-09-01 10:05", "A", "B"),   # A ya tiene 1: sale y llega
        ]
    )
    result, _ = run(network, trips, [0, 5])
    assert result.blocked.tolist() == [1]
    # Sin la llegada de r1, B se queda en 5 hasta que sale r2. Si r1 hubiera llegado, B terminaría en 6.
    assert result.final_inventory.tolist() == [0, 5]
    assert result.diverted.tolist() == [0] and result.in_transit.tolist() == [0]
    assert result.final_total == result.initial_total == 5


def test_un_viaje_bloqueado_que_iba_hacia_fuera_no_devuelve_ninguna_bici():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", None, None, 1.0)])
    result, _ = run(network, trips, [0, 5])
    assert result.blocked.tolist() == [1] and result.in_transit.tolist() == [0]
    assert result.final_inventory.tolist() == [0, 5]


def test_una_llegada_a_estacion_llena_va_a_la_estacion_con_lugar_mas_cercana():
    # A y B llenas. La más cercana a A es B, que tampoco tiene lugar: la bici va a C.
    network = make_network({"A": 3, "B": 3, "C": 5, "D": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:20", "D", "A")])
    result, _ = run(network, trips, [3, 3, 1, 4])
    assert result.diverted.tolist() == [1] and result.no_room.tolist() == [0]
    assert result.final_inventory.tolist() == [3, 3, 2, 3]
    assert result.final_total == result.initial_total


def test_si_hay_lugar_en_la_vecina_mas_cercana_la_bici_se_queda_ahi():
    network = make_network({"A": 3, "B": 3, "C": 5, "D": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:20", "D", "A")])
    result, _ = run(network, trips, [3, 2, 1, 4])
    assert result.diverted.tolist() == [1]
    assert result.final_inventory.tolist() == [3, 3, 1, 3]


def test_una_llegada_desviada_busca_mas_alla_de_las_16_vecinas_mas_cercanas():
    # 19 estaciones en línea. S00 y sus 16 vecinas más cercanas (S01 a S16) están llenas; S17 tiene lugar.
    names = [f"S{i:02d}" for i in range(19)]
    network = make_network({name: 3 for name in names})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:20", "S18", "S00")])
    result, _ = run(network, trips, [3] * 17 + [1, 1])
    assert result.diverted.tolist() == [1] and result.no_room.tolist() == [0]
    assert result.final_inventory.tolist() == [3] * 17 + [2, 0]   # quedó en S17, no de vuelta en S18


def test_con_la_red_llena_la_bici_siempre_encuentra_lugar_el_que_dejo_al_salir():
    # Como el total no cambia, una bici en camino dejó un lugar libre en alguna estación: nunca se queda sin lugar.
    network = make_network({"A": 3, "B": 3})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:20", "A", "B"),   # B está llena: vuelve al lugar que dejó en A
            ("r2", "2026-09-01 09:00", "2026-09-01 09:00", "A", "A"),   # sale y vuelve en el mismo instante
        ]
    )
    result, _ = run(network, trips, [3, 3])
    assert result.diverted.tolist() == [1] and result.no_room.tolist() == [0]
    assert result.final_inventory.tolist() == [3, 3] and result.final_in_transit == 0
    assert result.final_total == result.initial_total == 6


# --------------------------------------------------------------------------- #
# c) El total de bicis no cambia
# --------------------------------------------------------------------------- #

def test_un_viaje_que_termina_fuera_devuelve_la_bici_a_la_estacion_mas_cercana_al_punto_de_fin():
    network = make_network({"A": 20, "B": 20, "C": 20})   # "X" no está en la red
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", None, 0.0, 1.8),   # sin estación de fin: cerca de C
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", "A", "X", 0.0, 1.2),    # estación que no se simula: cerca de B
            ("r3", "2026-09-01 10:00", "2026-09-01 10:10", "C", None, 2.0, None),  # sin punto de fin: vuelve a C
            ("r4", "2026-09-01 11:00", "2026-09-01 11:10", "B", "X"),              # sin punto de fin: vuelve a B
        ]
    )
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY1, reglas.franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    assert events.counts["salidas"] == 4 and events.counts["llegadas"] == 4
    assert events.counts["llegadas_por_destino"] == {
        "a_su_estacion": 0,
        "a_la_estacion_mas_cercana_al_punto_de_fin": 2,
        "de_vuelta_a_la_estacion_de_salida": 2,
        "fuera_del_conjunto_sin_estacion_en_la_tabla": 2,
        "fuera_del_conjunto_con_estacion_que_no_se_simula": 2,
    }
    far = events.counts["distancia_a_la_estacion_mas_cercana"]["punto_de_fin"]
    assert far["viajes"] == 2 and far["maximo_km"] == pytest.approx(0.2 * STEP_KM, rel=1e-6)
    result = simulate(network, events, calendar, reglas, NoIntervention(), 0, np.array([5, 5, 5]))
    # A: 5 -1 -1 = 3.  B: 5 +1 (r2) = 6, y r4 sale y vuelve.  C: 5 +1 (r1) = 6, y r3 sale y vuelve.
    assert result.final_inventory.tolist() == [3, 6, 6]
    assert result.final_total == result.initial_total == 15 and result.final_in_transit == 0


def test_la_bici_vuelve_a_la_hora_en_que_termino_el_viaje():
    # A con 2 bicis: sale una a las 08:00 (queda 1, en riesgo) y vuelve a A a las 09:30 porque no trae punto de fin.
    network = make_network({"A": 10, "B": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 09:30", "A", None)])
    result, calendar = run(network, trips, [2, 5])
    assert result.risk_hours[0, calendar.labels.index("manana"), FEW_BIKES] == 1.5
    assert result.final_inventory.tolist() == [2, 5] and result.in_transit.tolist() == [0]


def test_si_la_estacion_mas_cercana_al_punto_de_fin_esta_llena_la_bici_se_desvia():
    network = make_network({"A": 10, "B": 3, "C": 10, "D": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:10", "D", None, 3.0, 0.9)])   # cerca de B, llena
    result, _ = run(network, trips, [10, 3, 5, 5])
    assert result.diverted.tolist() == [1]
    assert result.final_inventory.tolist() == [10, 3, 6, 4]   # B y A llenas: la más cercana a B con lugar es C
    assert result.final_total == result.initial_total


def test_un_viaje_que_empieza_fuera_sale_de_la_estacion_mas_cercana_a_su_punto_de_inicio():
    network = make_network({"A": 20, "B": 20, "C": 20})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", None, "A", 1.9, 0.0),   # empieza cerca de C
            ("r2", "2026-09-01 08:20", "2026-09-01 08:30", "X", "A", 1.1, 0.0),    # empieza cerca de B, que está vacía
            ("r3", "2026-09-01 08:40", "2026-09-01 08:50", None, "A"),             # sin punto de inicio: se ignora
            ("r4", "2026-09-01 08:45", "2026-09-01 08:55", "X", "A", None, 0.0),   # igual
            ("r5", "2026-09-01 09:00", "2026-09-01 09:10", "X", "X", 1.8, 0.2),    # las dos puntas fuera: de C a A
        ]
    )
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY1, reglas.franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    assert events.counts["salidas"] == 3 and events.counts["llegadas"] == 3
    assert events.counts["salidas_por_origen"] == {
        "de_su_estacion": 0,
        "de_la_estacion_mas_cercana_al_punto_de_inicio": 3,
        "de_ellas_sin_estacion_en_la_tabla": 1,
        "de_ellas_con_estacion_que_no_se_simula": 2,
    }
    assert events.counts["ignorados"]["empiezan_fuera_del_conjunto_y_sin_punto_de_inicio"] == 2
    result = simulate(network, events, calendar, reglas, NoIntervention(), 0, np.array([5, 0, 5]))
    assert result.blocked.tolist() == [1]             # r2: B está vacía y el viaje no ocurre
    assert result.final_inventory.tolist() == [7, 0, 3]   # r1 y r5 llevan una bici de C a A
    assert result.final_total == result.initial_total == 10


def test_los_viajes_en_camino_al_empezar_se_ignoran_y_los_del_final_quedan_en_transito():
    network = make_network({"A": 20, "B": 20})
    trips = make_trips(
        [
            ("r1", "2026-08-31 23:50", "2026-09-01 00:10", "A", "B"),   # ya iba en camino: se ignora
            ("r2", "2026-09-01 23:55", "2026-09-02 00:05", "A", "B"),   # sigue en camino al terminar
            ("r3", "2026-09-01 23:50", "2026-09-02 00:00", "A", "B"),   # llega justo en el fin del tramo: igual
            ("r4", "2026-09-02 00:00", "2026-09-02 00:10", "A", "B"),   # empieza en el fin del tramo: fuera
            ("r5", "2026-09-01 12:00", "2026-09-01 12:10", "A", "B"),   # viaje normal
        ]
    )
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY1, reglas.franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    assert (events.counts["salidas"], events.counts["llegadas"]) == (3, 1)
    assert events.counts["ignorados"] == {
        "ya_iban_en_camino_al_empezar_el_tramo": 1,
        "empiezan_fuera_del_conjunto_y_sin_punto_de_inicio": 0,
        "empiezan_despues_del_tramo": 1,
    }
    assert events.counts["siguen_en_camino_al_terminar_el_tramo"] == 2
    result = simulate(network, events, calendar, reglas, NoIntervention(), 0, np.array([5, 5]))
    assert result.final_inventory.tolist() == [2, 6]
    assert (result.final_docked, result.final_in_transit) == (8, 2)
    assert result.final_total == result.initial_total == 10


def test_las_bicis_en_transito_se_cuentan_al_final_de_cada_dia():
    network = make_network({"A": 20, "B": 20})
    trips = make_trips(
        [
            ("r1", "2026-09-01 23:50", "2026-09-02 00:20", "A", "B"),   # cruza la medianoche
            ("r2", "2026-09-01 23:55", "2026-09-02 00:30", "A", "B"),
            ("r3", "2026-09-02 23:59", "2026-09-03 00:30", "B", "A"),   # sigue en camino al terminar el tramo
        ]
    )
    result, _ = run(network, trips, [5, 5], days=(DAY1, DAY2))
    assert result.in_transit.tolist() == [2, 1]
    assert result.final_inventory.tolist() == [3, 6] and result.final_total == result.initial_total


def test_de_la_punta_que_cae_despues_del_tramo_no_se_mira_ni_la_estacion_ni_el_punto():
    # Tres salidas de A que terminan después del tramo: hacia B, sin estación de fin y hacia fuera con punto.
    # Las tres dan lo mismo: de esa punta solo se sabe que cae fuera (ahí puede estar la prueba final).
    network = make_network({"A": 10, "B": 10})
    calendar = build_calendar(DAY1, DAY1, make_reglas().franjas)

    def events_of(end_station, end_point):
        trips = make_trips([("r1", "2026-09-01 23:50", "2026-09-02 00:10", "A", end_station, 0.0, end_point)])
        return prepare_events(trips, network, calendar.start_us, calendar.end_us)

    base = events_of("B", 1.0)
    for other in (events_of(None, None), events_of("X", 0.9), events_of("A", 0.0)):
        assert (other.time_us, other.code, other.station) == (base.time_us, base.code, base.station)
        assert other.counts == base.counts
    assert base.counts["llegadas"] == 0 and base.counts["siguen_en_camino_al_terminar_el_tramo"] == 1


def test_una_tabla_de_viajes_sin_coordenadas_se_acepta_y_nadie_se_reasigna():
    network = make_network({"A": 20, "B": 20})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", None, 0.0, 1.0),
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", None, "B", 0.0, 1.0),
        ]
    ).drop(columns=POINT_FIELDS)
    result, _ = run(network, trips, [5, 5])
    assert result.final_inventory.tolist() == [5, 5]   # r1 vuelve a A; r2 se ignora
    assert result.final_total == result.initial_total


# --------------------------------------------------------------------------- #
# Orden de los eventos
# --------------------------------------------------------------------------- #

def test_en_el_mismo_instante_va_primero_la_llegada_y_despues_la_salida():
    # A empieza en cero. A las 08:00 llega una bici y sale otra: la llegada va primero y la salida no se bloquea.
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-01 07:50", "2026-09-01 08:00", "B", "A"),
            ("r2", "2026-09-01 08:00", "2026-09-01 08:15", "A", "B"),
        ]
    )
    result, _ = run(network, trips, [0, 5])
    assert result.blocked.tolist() == [0]
    assert result.final_inventory.tolist() == [0, 5]


def test_en_el_mismo_instante_el_desempate_es_el_ride_id():
    # A tiene una sola bici y dos salidas a las 08:00: sale el ride_id menor y se bloquea el otro.
    network = make_network({"A": 10, "B": 10, "C": 10})
    trips = make_trips(
        [
            ("r2", "2026-09-01 08:00", "2026-09-01 08:10", "A", "C"),
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
        ]
    )
    result, _ = run(network, trips, [1, 0, 0])
    assert result.blocked.tolist() == [1]
    assert result.final_inventory.tolist() == [0, 1, 0]   # llegó r1 (a B); r2 (a C) se bloqueó
    # El orden de las filas de entrada no cambia nada.
    again, _ = run(network, trips.iloc[::-1].reset_index(drop=True), [1, 0, 0])
    assert again.digest() == result.digest()


def test_la_llegada_de_un_viaje_de_duracion_cero_va_despues_de_su_salida():
    # Viaje que sale y llega en el mismo instante, de una estación en cero: se bloquea y no hay llegada.
    network = make_network({"A": 10, "B": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:00", "A", "B")])
    result, _ = run(network, trips, [0, 5])
    assert result.blocked.tolist() == [1] and result.final_inventory.tolist() == [0, 5]
    # Con una bici en A, sale y llega.
    result, _ = run(network, trips, [1, 5])
    assert result.blocked.tolist() == [0] and result.final_inventory.tolist() == [0, 6]


def test_los_viajes_deben_traer_zona_horaria_y_ride_id_unico():
    network = make_network({"A": 10, "B": 10})
    calendar = build_calendar(DAY1, DAY1, make_reglas().franjas)
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B")])
    naive = trips.assign(started_at=trips["started_at"].dt.tz_localize(None))
    with pytest.raises(SimulacionError, match="zona horaria"):
        prepare_events(naive, network, calendar.start_us, calendar.end_us)
    with pytest.raises(SimulacionError, match="ride_id"):
        prepare_events(pd.concat([trips, trips]), network, calendar.start_us, calendar.end_us)
    backwards = make_trips([("r1", "2026-09-01 08:10", "2026-09-01 08:00", "A", "B")])
    with pytest.raises(SimulacionError, match="terminan antes de empezar"):
        prepare_events(backwards, network, calendar.start_us, calendar.end_us)


# --------------------------------------------------------------------------- #
# d) Riesgo en tiempo exacto
# --------------------------------------------------------------------------- #

def test_el_riesgo_se_mide_en_horas_exactas_por_franja_y_tipo_cruzando_la_medianoche():
    """Estación A: 10 lugares, empieza con 3 bicis. Umbral: menos de 2 bicis o menos de 2 lugares.

    Día 1
      07:30 sale una   -> 2 bicis (no es riesgo: no son MENOS de 2)
      08:00 sale otra  -> 1 bici  (riesgo por pocas bicis)
      13:30 llega una  -> 2 bicis (fin del riesgo): 08:00 a 13:30 = 4 h de mañana + 1.5 h fuera
      22:00 sale una   -> 1 bici  (riesgo): 22:00 a 24:00 = 2 h fuera
    Día 2
      06:30 llega una  -> 2 bicis: 00:00 a 06:30 = 6 h fuera + 0.5 h de mañana
      16:00 llegan 7   -> 9 bicis, 1 lugar (riesgo por pocos lugares)
      20:00 sale una   -> 8 bicis, 2 lugares (fin): 16:00 a 20:00 = 4 h de tarde
    Los viajes van y vienen de B, una estación grande que nunca está en riesgo.
    """
    network = make_network({"A": 10, "B": 100})
    rows = [
        ("d1", "2026-09-01 07:30", "2026-09-01 07:40", "A", "B"),
        ("d2", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
        ("a1", "2026-09-01 13:20", "2026-09-01 13:30", "B", "A"),
        ("d3", "2026-09-01 22:00", "2026-09-01 22:10", "A", "B"),
        ("a2", "2026-09-02 06:20", "2026-09-02 06:30", "B", "A"),
        ("d4", "2026-09-02 20:00", "2026-09-02 20:10", "A", "B"),
    ]
    rows += [(f"b{i}", "2026-09-02 15:50", "2026-09-02 16:00", "B", "A") for i in range(7)]
    result, calendar = run(network, make_trips(rows), [3, 50], days=(DAY1, DAY2))
    hours = result.risk_hours
    manana, tarde, fuera = (calendar.labels.index(label) for label in ("manana", "tarde", OUTSIDE))
    assert hours.shape == (2, 3, 3)
    # Día 1: todo es por pocas bicis.
    assert hours[0, manana].tolist() == [4.0, 4.0, 0.0]
    assert hours[0, tarde].tolist() == [0.0, 0.0, 0.0]
    assert hours[0, fuera].tolist() == [3.5, 3.5, 0.0]
    # Día 2: la madrugada sigue en riesgo desde el día anterior; la tarde es por pocos lugares.
    assert hours[1, fuera].tolist() == [6.0, 6.0, 0.0]
    assert hours[1, manana].tolist() == [0.5, 0.5, 0.0]
    assert hours[1, tarde].tolist() == [4.0, 0.0, 4.0]
    assert hours[:, :, TOTAL].sum() == 18.0
    assert result.final_inventory.tolist() == [8, 45]

    # Con franja de noche (22:00 a 06:00) lo de "fuera" se parte en noche y resto; el total no cambia.
    night, _ = run(network, make_trips(rows), [3, 50], days=(DAY1, DAY2), noche=("22:00", "06:00"))
    labels = ("manana", "tarde", NIGHT, OUTSIDE)
    by_label = {label: night.risk_hours[:, i, TOTAL].tolist() for i, label in enumerate(labels)}
    assert by_label == {"manana": [4.0, 0.5], "tarde": [0.0, 4.0], NIGHT: [2.0, 6.0], OUTSIDE: [1.5, 0.0]}
    assert night.risk_hours[:, :, TOTAL].sum() == 18.0


def test_una_estacion_bajo_el_umbral_todo_el_dia_suma_24_horas_y_las_franjas_suman_el_dia():
    # B: 10 lugares y 1 bici, sin viajes. A no está en riesgo.
    network = make_network({"A": 10, "B": 10})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])  # fuera del tramo
    result, calendar = run(network, trips, [5, 1])
    hours = result.risk_hours[0]
    assert hours[:, TOTAL].tolist() == [6.0, 6.0, 12.0]          # mañana, tarde y fuera de franja
    assert hours[:, FEW_BIKES].sum() == 24.0 and hours[:, FEW_DOCKS].sum() == 0.0


def test_una_estacion_puede_estar_en_los_dos_riesgos_a_la_vez_y_el_total_la_cuenta_una_vez():
    # 2 lugares y 1 bici: menos de 2 bicis y menos de 2 lugares libres al mismo tiempo.
    network = make_network({"A": 2, "B": 10})
    result, _ = run(network, make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")]), [1, 5])
    hours = result.risk_hours[0].sum(axis=0)
    assert hours.tolist() == [24.0, 24.0, 24.0]


def test_un_traslado_cambia_el_riesgo_desde_el_instante_de_la_intervencion():
    # A empieza con 1 bici (riesgo). A las 05:00 recibe 3 de B: deja de estar en riesgo. Son 5 h fuera de franja.
    network = make_network({"A": 10, "B": 10})
    policy = Scripted({(DAY1, "manana"): [Move(1, 0, 3)]})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    result, calendar = run(network, trips, [1, 5], policy=policy, budget=6)
    assert result.risk_hours[0, :, TOTAL].tolist() == [0.0, 0.0, 5.0]
    assert result.final_inventory.tolist() == [4, 2]


def test_un_traslado_que_deja_una_estacion_bajo_el_umbral_cuenta_el_riesgo_desde_la_intervencion():
    # A (3 bicis) y B (7 de 10) empiezan sin riesgo. A las 05:00 A da 2 a B: A queda con 1 bici y B con 1 lugar.
    # Las dos, en riesgo de 05:00 a 24:00 = 19 h: 6 de mañana, 6 de tarde y 7 fuera de franja. Nada antes.
    network = make_network({"A": 10, "B": 10})
    policy = Scripted({(DAY1, "manana"): [Move(0, 1, 2)]})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    result, _ = run(network, trips, [3, 7], policy=policy, budget=4)
    assert result.final_inventory.tolist() == [1, 9]
    assert result.risk_hours[0, :, TOTAL].tolist() == [12.0, 12.0, 14.0]
    assert result.risk_hours[0, :, FEW_BIKES].tolist() == [6.0, 6.0, 7.0]
    assert result.risk_hours[0, :, FEW_DOCKS].tolist() == [6.0, 6.0, 7.0]


# --------------------------------------------------------------------------- #
# El inventario pasa de un día al siguiente
# --------------------------------------------------------------------------- #

def test_el_inventario_pasa_de_un_dia_al_siguiente_sin_reinicios():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
            ("r2", "2026-09-01 09:00", "2026-09-01 09:10", "A", "B"),
            ("r3", "2026-09-02 08:00", "2026-09-02 08:10", "A", "B"),
        ]
    )
    policy = Scripted()
    result, _ = run(network, trips, [5, 5], days=(DAY1, DAY2), policy=policy)
    # Lo que ve la política el día 2 a las 05:00 es lo que dejó el día 1, no el inventario inicial.
    seen = {(context.day, context.franja.nombre): context.inventory.tolist() for context in policy.seen}
    assert seen == {
        (DAY1, "manana"): [5, 5],
        (DAY1, "tarde"): [3, 7],
        (DAY2, "manana"): [3, 7],
        (DAY2, "tarde"): [2, 8],
    }
    assert result.final_inventory.tolist() == [2, 8]


def test_la_intervencion_ve_el_inventario_antes_de_los_viajes_de_ese_mismo_instante():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-01 04:59:59", "2026-09-01 05:10", "A", "B"),   # un segundo antes: ya se ve
            ("r2", "2026-09-01 05:00:00", "2026-09-01 05:10", "A", "B"),   # a las 05:00 en punto: todavía no
        ]
    )
    policy = Scripted()
    run(network, trips, [5, 5], policy=policy)
    assert policy.seen[0].instant == ts("2026-09-01 05:00") and policy.seen[0].inventory.tolist() == [4, 5]


# --------------------------------------------------------------------------- #
# e) Presupuesto
# --------------------------------------------------------------------------- #

def test_lo_no_usado_pasa_a_la_franja_siguiente_del_mismo_dia_y_nunca_al_dia_siguiente():
    """Presupuesto de 5 bicis al día, mitad y mitad: 3 para la mañana y 2 para la tarde.

    Día 1: la mañana usa 1 (sobran 2) -> la tarde dispone de 2 + 2 = 4 y usa 1. Sobran 3 y se pierden.
    Día 2: la mañana dispone de 3 (no de 6) y no usa nada -> la tarde dispone de 2 + 3 = 5 y usa 2.
    """
    network = make_network({"A": 20, "B": 20})
    policy = Scripted(
        {
            (DAY1, "manana"): [Move(0, 1, 1)],
            (DAY1, "tarde"): [Move(0, 1, 1)],
            (DAY2, "tarde"): [Move(1, 0, 2)],
        }
    )
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    result, _ = run(network, trips, [10, 10], days=(DAY1, DAY2), policy=policy, budget=5)
    assert result.budget_available.tolist() == [[3, 4], [3, 5]]
    assert result.budget_used.tolist() == [[1, 1], [0, 2]]
    assert [context.budget for context in policy.seen] == [3, 4, 3, 5]
    assert [(r.day_index, r.franja_index, r.budget_available, r.budget_used) for r in result.interventions] == [
        (0, 0, 3, 1), (0, 1, 4, 1), (1, 0, 3, 0), (1, 1, 5, 2),
    ]
    assert result.final_inventory.tolist() == [10, 10]


def test_el_motor_registra_lo_que_recibio_y_entrego_cada_estacion():
    network = make_network({"A": 20, "B": 20, "C": 20})
    policy = Scripted({(DAY1, "manana"): [Move(0, 1, 2), Move(2, 1, 1)]})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    result, _ = run(network, trips, [10, 10, 10], policy=policy, budget=10)
    first, second = result.interventions
    assert first.inventory_before.tolist() == [10, 10, 10] and first.net.tolist() == [-2, 3, -1]
    assert first.moves == (Move(0, 1, 2), Move(2, 1, 1))
    assert second.inventory_before.tolist() == [8, 13, 9] and second.net.tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- #
# f) Límites de un traslado
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "moves, message",
    [
        ([Move(0, 1, 4)], "Disponibilidad"),              # A tiene 3 bicis
        ([Move(2, 1, 3)], "Capacidad"),                   # a B le quedan 2 lugares
        ([Move(0, 3, 1)], "Distancia"),                   # A y D están a 1.67 km y el máximo es 1
        ([Move(2, 3, 5)], "Presupuesto"),                 # la mañana dispone de 4
        ([Move(2, 3, 3), Move(2, 3, 2)], "Presupuesto"),  # 3 + 2 pasa de 4
        ([Move(2, 2, 1)], "misma estación"),
        ([Move(2, 3, 0)], "entero mayor que 0"),
        ([Move(2, 3, -1)], "entero mayor que 0"),
        ([Move(2, 3, 1.5)], "entero mayor que 0"),
        ([Move(2, 9, 1)], "no es una estación"),
        ([(2, 3)], "origen, destino y bicis"),
    ],
)
def test_el_motor_rechaza_un_traslado_que_rompe_un_limite(moves, message):
    network = make_network({"A": 10, "B": 10, "C": 10, "D": 10})
    policy = Scripted({(DAY1, "manana"): moves})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    with pytest.raises(MoveError, match=message):
        run(network, trips, [3, 8, 6, 2], policy=policy, budget=8)   # 4 por franja


def test_un_traslado_justo_en_cada_limite_si_se_acepta():
    network = make_network({"A": 10, "B": 10, "C": 10, "D": 10})
    # A da sus 3 bicis; B recibe hasta llenarse; el total es justo el presupuesto de la franja (4).
    policy = Scripted({(DAY1, "manana"): [Move(0, 1, 2), Move(0, 1, 1), Move(2, 3, 1)]})
    trips = make_trips([("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")])
    result, _ = run(network, trips, [3, 7, 6, 2], policy=policy, budget=8)
    assert result.final_inventory.tolist() == [0, 10, 5, 3]
    assert result.budget_used.tolist() == [[4, 0]]


# --------------------------------------------------------------------------- #
# g) Conservación y resultado repetible
# --------------------------------------------------------------------------- #

def test_los_traslados_conservan_el_total_de_bicis():
    network = make_network({"A": 20, "B": 20, "C": 20})
    policy = Scripted({(DAY1, "manana"): [Move(0, 1, 3), Move(1, 2, 2)], (DAY1, "tarde"): [Move(2, 1, 4)]})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "B"),
            ("r2", "2026-09-01 16:00", "2026-09-01 16:10", "B", "C"),
        ]
    )
    result, _ = run(network, trips, [10, 10, 10], policy=policy, budget=20)
    assert result.final_total == result.initial_total == 30
    for record in result.interventions:
        assert int(record.net.sum()) == 0
    # 05:00: [10,10,10] -> A da 3 a B -> [7,13,10] -> B da 2 a C -> [7,11,12].  08:00 r1 -> [6,12,12].
    # 14:00: C da 4 a B -> [6,16,8].  16:00 r2 -> [6,15,9].
    assert result.final_inventory.tolist() == [6, 15, 9]


def test_none_no_mueve_nada_y_dos_corridas_iguales_dan_lo_mismo():
    network, trips, start = random_case(seed=3)
    first, _ = run(network, trips, start, days=(DAY1, DAY2))
    second, _ = run(network, trips, start, days=(DAY1, DAY2))
    assert first.digest() == second.digest()
    assert int(first.budget_used.sum()) == 0 and all(not record.moves for record in first.interventions)
    # Con otros viajes la huella cambia: la comparación sí distingue.
    other, _ = run(network, trips.iloc[:-50], start, days=(DAY1, DAY2))
    assert other.digest() != first.digest()


# --------------------------------------------------------------------------- #
# h) Lo que recibe una política
# --------------------------------------------------------------------------- #

def test_una_politica_recibe_solo_el_contexto_y_el_contexto_no_trae_viajes():
    fields = {field.name for field in dataclasses.fields(InterventionContext)}
    assert fields == {"instant", "day", "franja", "inventory", "budget", "valet"}
    # Los datos fijos de la red tampoco traen nada de los viajes.
    assert {field.name for field in dataclasses.fields(Network)} == {
        "station_ids", "capacity", "lat", "lon", "station_type", "typed", "dist_km", "nearest",
    }
    network, trips, start = random_case(seed=5)
    policy = Scripted()   # falla si recibe algo más que el contexto
    run(network, trips, start, days=(DAY1, DAY2), policy=policy, budget=4)
    assert len(policy.seen) == 4
    for context in policy.seen:
        assert isinstance(context.instant, pd.Timestamp) and str(context.instant.tz) == TZ
        assert isinstance(context.inventory, np.ndarray) and context.inventory.shape == (network.size,)
        assert not context.inventory.flags.writeable   # la política no puede cambiar el inventario
        assert isinstance(context.budget, int)


# --------------------------------------------------------------------------- #
# i) Contra un simulador de referencia
# --------------------------------------------------------------------------- #

def random_case(seed: int, n_trips: int = 400):
    """Seis estaciones chicas y viajes al azar de dos días, con muchos instantes repetidos."""
    rng = np.random.default_rng(seed)
    names = ["A", "B", "C", "D", "E", "F"]
    stations = pd.DataFrame(
        {
            "station_id": names,
            "name": names,
            "lat": 40.70 + rng.random(6) * 0.02,
            "lon": -73.99 + rng.random(6) * 0.02,
            "capacity": pd.array(rng.integers(3, 8, 6), dtype="Int64"),
            "station_type": pd.Series([None] * 6, dtype="object"),
        }
    )
    network = build_network(stations)
    options = np.array(names + ["X", None], dtype=object)   # "X" no está en la red; None = sin estación
    minutes = rng.integers(-60, 2 * 24 * 60 + 60, n_trips)   # también antes y después del tramo
    started = ts("2026-09-01") + pd.to_timedelta(minutes, unit="min")
    duration = rng.choice([0, 0, 5, 10, 15, 45], n_trips)   # hay viajes de duración cero

    def points() -> tuple[np.ndarray, np.ndarray]:
        """Un punto al azar por viaje; uno de cada cuatro no lo trae."""
        missing = rng.random(n_trips) < 0.25
        lat = np.where(missing, np.nan, 40.70 + rng.random(n_trips) * 0.02)
        return lat, np.where(missing, np.nan, -73.99 + rng.random(n_trips) * 0.02)

    start_lat, start_lng = points()
    end_lat, end_lng = points()
    trips = pd.DataFrame(
        {
            "ride_id": [f"r{i:04d}" for i in rng.permutation(n_trips)],
            "started_at": pd.DatetimeIndex(started).as_unit("us"),
            "ended_at": pd.DatetimeIndex(started + pd.to_timedelta(duration, unit="min")).as_unit("us"),
            "start_station_id": options[rng.integers(0, len(options), n_trips)],
            "end_station_id": options[rng.integers(0, len(options), n_trips)],
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
        }
    )
    start = (network.capacity // 2).tolist()
    return network, trips, start


def reference(network, trips, start, calendar, reglas):
    """Simulador de referencia: lento y directo, sin atajos. Sin intervenciones."""
    ids = list(network.station_ids)
    capacity = dict(zip(ids, network.capacity.tolist()))
    inventory = dict(zip(ids, start))
    t0, t1 = calendar.start_us, calendar.end_us

    def closest(lat, lon):
        """La estación más cercana al punto, o None si el viaje no lo trae. En un empate, la de id menor."""
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None
        return min(ids, key=lambda name: (float(haversine_km(lat, lon, network.lat[ids.index(name)],
                                                             network.lon[ids.index(name)])), name))

    events, ignored, open_trips = [], 0, 0
    for row in trips.itertuples():
        begin, end = row.started_at.value // 1000, row.ended_at.value // 1000
        if not t0 <= begin < t1:
            continue   # ya iba en camino al empezar, o empieza después
        origin = row.start_station_id if row.start_station_id in capacity else closest(row.start_lat, row.start_lng)
        if origin is None:
            ignored += 1
            continue
        events.append((begin, 1, row.ride_id, "sale", origin))
        if end < t1:
            target = row.end_station_id if row.end_station_id in capacity else closest(row.end_lat, row.end_lng)
            events.append((end, 2 if begin == end else 0, row.ride_id, "llega", target or origin))
        else:
            open_trips += 1
    events.sort(key=lambda event: event[:3])

    def day_of(instant):
        return max(d for d in range(len(calendar.days)) if calendar.day_start_us[d] <= instant)

    counts = {name: [0] * len(calendar.days) for name in ("blocked", "diverted", "no_room")}
    transit_at = {}   # día -> bicis en camino después de su último evento
    transit = 0
    blocked_rides = set()
    history = {name: [(t0, inventory[name])] for name in ids}   # (desde cuándo, bicis)
    for instant, _, ride, what, station in events:
        day = day_of(instant)
        if what == "sale":
            if inventory[station] == 0:
                counts["blocked"][day] += 1
                blocked_rides.add(ride)
                continue
            inventory[station] -= 1
            transit += 1
            history[station].append((instant, inventory[station]))
        else:
            if ride in blocked_rides:
                continue
            if inventory[station] >= capacity[station]:
                counts["diverted"][day] += 1
                here = ids.index(station)
                by_distance = sorted(
                    (name for name in ids if name != station),
                    key=lambda name: (network.dist_km[here, ids.index(name)], name),
                )
                station = next(name for name in by_distance if inventory[name] < capacity[name])
            inventory[station] += 1
            transit -= 1
            history[station].append((instant, inventory[station]))
        transit_at[day] = transit
        assert sum(inventory.values()) + transit == sum(start)   # el total no cambia en ningún evento
    counts["in_transit"], last = [], 0
    for day in range(len(calendar.days)):
        last = transit_at.get(day, last)
        counts["in_transit"].append(last)

    # Tiempo en riesgo: se recorre cada tramo constante de cada estación y se reparte entre los cortes.
    risk = np.zeros((len(calendar.days), len(calendar.labels), 3), dtype=np.int64)
    edges = calendar.bin_edges_us.tolist()
    for name in ids:
        steps = history[name] + [(t1, None)]
        for (since, bikes), (until, _) in zip(steps, steps[1:]):
            few_bikes = bikes < reglas.bicis_minimas
            few_docks = capacity[name] - bikes < reglas.lugares_libres_minimos
            for b in range(len(edges) - 1):
                overlap = min(until, edges[b + 1]) - max(since, edges[b])
                if overlap <= 0:
                    continue
                cell = risk[calendar.bin_day[b], calendar.bin_label[b]]
                cell[TOTAL] += overlap * (few_bikes or few_docks)
                cell[FEW_BIKES] += overlap * few_bikes
                cell[FEW_DOCKS] += overlap * few_docks
    return [inventory[name] for name in ids], counts, risk, {"ignored": ignored, "open": open_trips}


@pytest.mark.parametrize("seed", range(8))
def test_el_recorrido_da_lo_mismo_que_el_simulador_de_referencia(seed):
    network, trips, start = random_case(seed)
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY2, reglas.franjas, noche=("22:00", "06:00"))
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    result = simulate(network, events, calendar, reglas, NoIntervention(), 0, np.array(start))
    final, counts, risk, extra = reference(network, trips, start, calendar, reglas)
    assert result.final_inventory.tolist() == final
    for name, expected in counts.items():
        assert getattr(result, name).tolist() == expected, name
    assert (result.risk_us == risk).all()
    assert events.counts["ignorados"]["empiezan_fuera_del_conjunto_y_sin_punto_de_inicio"] == extra["ignored"]
    assert events.counts["siguen_en_camino_al_terminar_el_tramo"] == extra["open"]
    # El caso al azar sí ejercita las reglas: hay bloqueados, desviadas, reasignados e ignorados.
    assert sum(counts["blocked"]) > 0 and sum(counts["diverted"]) > 0 and extra["ignored"] > 0
    assert events.counts["salidas_por_origen"]["de_la_estacion_mas_cercana_al_punto_de_inicio"] > 0
    assert events.counts["llegadas_por_destino"]["a_la_estacion_mas_cercana_al_punto_de_fin"] > 0
    assert events.counts["llegadas_por_destino"]["de_vuelta_a_la_estacion_de_salida"] > 0
    assert (0 <= result.final_inventory).all() and (result.final_inventory <= network.capacity).all()
    # La identidad: en estaciones más en tránsito, siempre el total inicial.
    assert result.final_docked + result.final_in_transit == result.initial_total == sum(start)


# --------------------------------------------------------------------------- #
# Franjas que se tocan, estaciones con valet y grupos aislados
# --------------------------------------------------------------------------- #

TOUCHING = [
    {"nombre": "manana", "momento_intervencion": "05:00", "inicio": "06:00", "fin": "14:00",
     "parte_del_presupuesto": 0.6},
    {"nombre": "tarde", "momento_intervencion": "14:00", "inicio": "14:00", "fin": "22:00",
     "parte_del_presupuesto": 0.4},
]
IDLE_TRIP = [("r1", "2026-09-05 08:00", "2026-09-05 08:10", "A", "B")]   # fuera de los días simulados
HOUR = 3_600_000_000


def test_dos_franjas_que_se_tocan_y_una_que_empieza_en_su_intervencion_se_simulan():
    """manana 06-14 y tarde 14-22 con intervención a las 14:00: el minuto 14:00 es de la tarde."""
    reglas = make_reglas(franjas=TOUCHING)
    assert [f.nombre for f in reglas.franjas] == ["manana", "tarde"]
    assert reglas.franjas[0].hours == tuple(range(5, 14)) and reglas.franjas[1].hours == tuple(range(14, 22))
    labels = hour_labels(reglas.franjas, ("22:00", "06:00"))
    assert labels[13] == "manana" and labels[14] == "tarde" and labels[21] == "tarde" and labels[22] == "noche"
    network = make_network({"A": 10, "B": 10})
    policy = Scripted({(DAY1, "tarde"): [Move(1, 0, 3)]})
    result, calendar = run(
        network, make_trips(IDLE_TRIP), [0, 5], reglas=reglas, policy=policy, budget=10, noche=("22:00", "06:00")
    )
    assert [to_timestamp(slot.instant_us).hour for slot in calendar.slots] == [5, 14]
    assert result.budget_available.tolist() == [[6, 10]] and result.budget_used.tolist() == [[0, 3]]
    # A está vacía hasta las 14:00 en punto: 6 horas de noche (00-06), 8 de la mañana y 0 de la tarde.
    hours = dict(zip(calendar.labels, result.risk_hours[0, :, TOTAL]))
    assert hours == {"manana": 8.0, "tarde": 0.0, "noche": 6.0, "resto": 0.0}


def valet_reglas(**changes) -> Reglas:
    """A tiene valet los martes de 07:00 a 11:00 (el 1 de septiembre de 2026 es martes)."""
    return make_reglas(
        excepciones_valet={"dias": ["martes"], "estaciones": {"A": [["07:00", "11:00"]], "Z": [["07:00", "11:00"]]}},
        **changes,
    )


def test_el_tiempo_con_valet_no_cuenta_como_riesgo_y_solo_en_sus_dias():
    network = make_network({"A": 10, "B": 10})
    reglas = valet_reglas()
    assert reglas.valet_dias == (1,) and dict(reglas.valet_estaciones)["A"] == (("07:00", "11:00"),)
    result, calendar = run(network, make_trips(IDLE_TRIP), [0, 5], days=(DAY1, DAY2), reglas=reglas)
    plain, _ = run(network, make_trips(IDLE_TRIP), [0, 5], days=(DAY1, DAY2))
    # A está vacía los dos días. El martes se le quitan sus 4 horas con valet; el miércoles, nada.
    assert plain.risk_hours[:, :, TOTAL].sum(axis=1).tolist() == [24.0, 24.0]
    assert result.risk_hours[:, :, TOTAL].sum(axis=1).tolist() == [20.0, 24.0]
    assert (result.risk_station_us / HOUR).tolist() == [[20.0, 0.0], [24.0, 0.0]]
    assert (result.valet_risk_us / HOUR).tolist() == [4.0, 0.0] and plain.valet_risk_us.tolist() == [0, 0]
    # Las 4 horas salen de la franja de la mañana (06 a 12), que pasa de 6 a 2.
    assert result.risk_hours[0, calendar.labels.index("manana"), TOTAL] == 2.0
    # Horas de estación con valet: la estación Z del archivo no está en la red y no cuenta.
    windows = valet_windows(network, calendar.days, reglas)
    assert list(windows) == [0] and valet_hours(windows, calendar).tolist() == [4.0, 0.0]


def test_una_estacion_con_valet_en_el_instante_de_la_intervencion_no_entrega_ni_recibe():
    """Con la intervención a las 08:00, dentro del valet de A: el motor rechaza el traslado y la política lo ve."""
    franjas = [dict(TOUCHING[0], momento_intervencion="08:00", inicio="08:00"), TOUCHING[1]]
    reglas = valet_reglas(franjas=franjas)
    network = make_network({"A": 10, "B": 10})
    for moves in ([Move(1, 0, 2)], [Move(0, 1, 1)]):
        with pytest.raises(MoveError, match="Valet"):
            run(network, make_trips(IDLE_TRIP), [3, 5], reglas=reglas, policy=Scripted({(DAY1, "manana"): moves}),
                budget=10)
    # El miércoles no hay valet, y el martes a las 14:00 tampoco: ahí el mismo traslado se acepta.
    policy = Scripted({(DAY1, "tarde"): [Move(1, 0, 2)], (DAY2, "manana"): [Move(1, 0, 1)]})
    result, calendar = run(network, make_trips(IDLE_TRIP), [3, 5], days=(DAY1, DAY2), reglas=reglas, policy=policy,
                           budget=10)
    assert result.final_inventory.tolist() == [6, 2]
    assert [None if c.valet is None else c.valet.tolist() for c in policy.seen] == [
        [True, False], [False, False], [False, False], [False, False],
    ]
    windows = valet_windows(network, calendar.days, reglas)
    assert valet_at(windows, calendar.slots[0].instant_us, 2).tolist() == [True, False]
    # El fin del intervalo queda fuera: a las 11:00 en punto ya no hay valet.
    assert not valet_at(windows, local_us(DAY1, "11:00"), 2).any()
    assert valet_at(windows, local_us(DAY1, "07:00"), 2).tolist() == [True, False]


def test_un_grupo_aislado_solo_intercambia_bicis_entre_sus_estaciones():
    network = make_network({"A": 10, "B": 10, "C": 10, "D": 10})
    reglas = make_reglas(grupos_aislados={"isla": ["A", "B", "no_esta_en_la_red"]})
    assert station_groups(network, reglas).tolist() == [0, 0, -1, -1]
    # Sin grupos B tiene de vecinas a A y C; con el grupo, solo a A. C pierde a B y se queda con D.
    assert [sorted(near.tolist()) for near in move_neighbors(network, make_reglas())] == [[1], [0, 2], [1, 3], [2]]
    assert [near.tolist() for near in move_neighbors(network, reglas)] == [[1], [0], [3], [2]]
    for moves in ([Move(1, 2, 1)], [Move(2, 1, 1)]):   # del grupo hacia fuera y de fuera hacia el grupo
        with pytest.raises(MoveError, match="Grupo aislado"):
            run(network, make_trips(IDLE_TRIP), [5, 5, 5, 5], reglas=reglas,
                policy=Scripted({(DAY1, "manana"): moves}), budget=8)
    policy = Scripted({(DAY1, "manana"): [Move(0, 1, 2), Move(2, 3, 1)]})   # dentro del grupo y entre dos de fuera
    result, _ = run(network, make_trips(IDLE_TRIP), [5, 5, 5, 5], reglas=reglas, policy=policy, budget=8)
    assert result.final_inventory.tolist() == [3, 7, 4, 6]


def test_las_llegadas_desviadas_no_miran_el_grupo_aislado():
    """Regla de siempre: una llegada a una estación llena va a la más cercana con lugar, sea o no de su grupo."""
    network = make_network({"A": 5, "B": 5, "C": 5})
    reglas = make_reglas(grupos_aislados={"isla": ["A", "B"]})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", "C")])
    result, _ = run(network, trips, [3, 0, 5], reglas=reglas)
    assert result.diverted.tolist() == [1] and result.final_inventory.tolist() == [2, 1, 5]


def test_una_estacion_no_puede_estar_en_dos_grupos():
    with pytest.raises(SimulacionError, match="más de un grupo"):
        make_reglas(grupos_aislados={"uno": ["A"], "dos": ["A"]})
