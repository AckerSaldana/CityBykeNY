"""Pruebas del motor del simulador (src/sim/engine.py).

Redes de 2 a 6 estaciones (y una de 19, para la búsqueda de lugar más allá de
las vecinas cercanas) y viajes escritos a mano. Cada regla se compara con una
cuenta hecha a mano:

a) Reglas, reparto del presupuesto, red y calendario.
b) Eventos imposibles: una salida con inventario cero es un viaje bloqueado y
   no genera llegada; una llegada a estación llena va a la estación con lugar
   más cercana.
c) Bicis que entran y salen del conjunto simulado, y el neto.
d) Riesgo en tiempo exacto, por franja y por tipo, cruzando la medianoche.
e) Presupuesto: reparto por franja, lo no usado pasa a la franja siguiente del
   mismo día y nunca al día siguiente.
f) El motor rechaza un traslado que rompe cualquiera de sus cuatro límites.
g) El total de bicis se conserva y dos corridas iguales dan lo mismo.
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
    prepare_events,
    simulate,
    split_budget,
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
        "tipo_si_falta": "mixta",
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
    return build_network(stations, "mixta")


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TZ)


def make_trips(rows: list[tuple]) -> pd.DataFrame:
    """Viajes: (ride_id, inicio, fin, estación de inicio, estación de fin). None = sin estación."""
    frame = pd.DataFrame(rows, columns=["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"])
    for column in ("started_at", "ended_at"):
        frame[column] = pd.DatetimeIndex([ts(value) for value in frame[column]]).as_unit("us")
    return frame


def run(network, trips, start, *, days=(DAY1, DAY1), reglas=None, policy=None, budget=0, warmup=0):
    reglas = reglas or make_reglas()
    calendar = build_calendar(days[0], days[1], reglas.franjas, warmup)
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
    assert reglas.first_franja.nombre == "manana"
    assert (reglas.bicis_minimas, reglas.lugares_libres_minimos, reglas.distancia_maxima_km) == (3, 1, 2.5)
    assert reglas.prioridad["oficinas"] == 3 and reglas.ocupacion_respaldo["oficinas"]["tarde"] == 0.7
    assert reglas.presupuestos_por_estacion == (0.0, 0.5, 1.0) and reglas.presupuesto_de_referencia == 1.0
    # De la intervención (05:00) al fin de la franja (12:00): horas 5 a 11.
    assert reglas.franjas[0].hours == (5, 6, 7, 8, 9, 10, 11)
    assert reglas.franjas[1].hours == (14, 15, 16, 17, 18, 19, 20)


@pytest.mark.parametrize(
    "key", ["presupuestos_por_estacion", "presupuesto_de_referencia", "tipo_si_falta", "distancia_maxima_km"]
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
    network = build_network(stations, "mixta")
    assert network.station_ids == ("A", "B", "C")
    assert network.capacity.tolist() == [10, 20, 30]
    assert network.station_type == ("mixta", "mixta", "oficinas") and network.typed.tolist() == [False, False, True]
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
    network = build_network(stations, "mixta")
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
        build_network(stations, "mixta")


def test_el_inventario_inicial_es_la_ocupacion_de_respaldo_de_la_primera_franja():
    cfg = make_cfg()
    cfg["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"] = {"manana": 0.3, "tarde": 0.7}
    cfg["estrategia_por_tipo"]["mixta"]["ocupacion_respaldo"] = {"manana": 0.5, "tarde": 0.9}
    network = make_network({"A": 10, "B": 15, "C": 20, "D": 11}, types={"C": "oficinas", "D": "oficinas"})
    # A y B no tienen tipo: usan "mixta" (0.5 en la mañana). 7.5 sube a 8. C y D: 0.3 x 20 = 6 y 0.3 x 11 = 3.3.
    assert initial_inventory(network, Reglas.from_operacion(cfg)).tolist() == [5, 8, 6, 3]


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
    assert result.diverted.tolist() == [0] and result.entered.tolist() == [0] and result.left.tolist() == [0]
    assert result.final_total == result.initial_total == 5


def test_un_viaje_bloqueado_que_iba_hacia_fuera_no_cuenta_como_bici_que_sale():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:10", "A", None)])
    result, _ = run(network, trips, [0, 5])
    assert result.blocked.tolist() == [1] and result.left.tolist() == [0]
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
    trips = make_trips([("r1", "2026-09-01 08:00", "2026-09-01 08:20", None, "S00")])
    result, _ = run(network, trips, [3] * 17 + [1, 1])
    assert result.diverted.tolist() == [1] and result.no_room.tolist() == [0]
    assert result.entered.tolist() == [1]
    assert result.final_inventory.tolist() == [3] * 17 + [2, 1]   # quedó en S17, no en S18


def test_si_ninguna_estacion_tiene_lugar_la_bici_queda_fuera_y_se_cuenta():
    network = make_network({"A": 3, "B": 3})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:20", None, "A"),   # viene de fuera y no cabe: no entra
            ("r2", "2026-09-01 09:00", "2026-09-01 09:00", "A", "A"),    # sale y vuelve en el mismo instante
        ]
    )
    result, _ = run(network, trips, [3, 3])
    assert result.diverted.tolist() == [1] and result.no_room.tolist() == [1]
    assert result.entered.tolist() == [0] and result.left.tolist() == [0]
    assert result.final_inventory.tolist() == [3, 3]


# --------------------------------------------------------------------------- #
# c) Bicis que entran y salen del conjunto
# --------------------------------------------------------------------------- #

def test_entran_y_salen_bicis_del_conjunto_y_se_reporta_el_neto():
    network = make_network({"A": 20, "B": 20})   # "X" no está en la red
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", "X", "A"),    # entra: viene de fuera del conjunto
            ("r2", "2026-09-01 08:20", "2026-09-01 08:30", None, "B"),   # entra: sin estación de inicio
            ("r3", "2026-08-31 23:50", "2026-09-01 00:10", "A", "B"),    # entra: salió antes del tramo
            ("r4", "2026-09-01 10:00", "2026-09-01 10:10", "A", None),   # sale: sin estación de fin
            ("r5", "2026-09-01 11:00", "2026-09-01 11:10", "B", "X"),    # sale: va fuera del conjunto
            ("r6", "2026-09-01 23:55", "2026-09-02 00:05", "A", "B"),    # sale: llega después del tramo
            ("r7", "2026-09-01 12:00", "2026-09-01 12:10", "A", "B"),    # viaje interno
            ("r8", "2026-09-01 13:00", "2026-09-01 13:10", "X", "X"),    # no toca el conjunto
            ("r9", "2026-09-02 08:00", "2026-09-02 08:10", "A", "B"),    # después del tramo: no cuenta
        ]
    )
    reglas = make_reglas()
    calendar = build_calendar(DAY1, DAY1, reglas.franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    assert events.counts == {
        "viajes_leidos": 9,
        "salidas": 4,
        "llegadas": 4,
        "viajes_internos": 1,
        "llegadas_que_entran": 3,
        "salidas_que_salen": 3,
        "viajes_sin_ningun_evento": 2,
        # Por qué entra cada una: r2 no trae estación de inicio, r1 viene de "X" y r3 salió antes del tramo.
        "llegadas_que_entran_por_motivo": {
            "sin_estacion_de_inicio": 1,
            "de_una_estacion_fuera_del_conjunto": 1,
            "salieron_antes_del_tramo": 1,
        },
        # Por qué sale cada una: r4 no trae estación de fin, r5 va a "X" y r6 llega después del tramo.
        "salidas_que_salen_por_motivo": {
            "sin_estacion_de_fin": 1,
            "a_una_estacion_fuera_del_conjunto": 1,
            "terminan_despues_del_tramo": 1,
        },
    }
    result = simulate(network, events, calendar, reglas, NoIntervention(), 0, np.array([5, 5]))
    assert result.entered.tolist() == [3] and result.left.tolist() == [3]
    assert int(result.entered.sum() - result.left.sum()) == 0
    # A: 5 +1 (r1) -1 (r4) -1 (r6) -1 (r7) = 3.  B: 5 +1 (r2) +1 (r3) -1 (r5) +1 (r7) = 7.
    assert result.final_inventory.tolist() == [3, 7]
    assert result.final_total == result.initial_total + 3 - 3


def test_un_viaje_que_empieza_justo_al_terminar_el_tramo_no_genera_eventos():
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-02 00:00", "2026-09-02 00:10", "A", "B"),   # empieza en el fin del tramo: fuera
            ("r2", "2026-09-01 23:50", "2026-09-02 00:00", "A", "B"),   # llega en el fin: sale del conjunto
        ]
    )
    calendar = build_calendar(DAY1, DAY1, make_reglas().franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    assert events.counts["salidas"] == 1 and events.counts["llegadas"] == 0
    assert events.counts["salidas_que_salen"] == 1 and events.counts["viajes_sin_ningun_evento"] == 1
    assert events.counts["salidas_que_salen_por_motivo"]["terminan_despues_del_tramo"] == 1
    result, _ = run(network, trips, [5, 5])
    assert result.left.tolist() == [1] and result.final_inventory.tolist() == [4, 5]


def test_de_la_punta_que_cae_fuera_del_tramo_no_se_mira_la_estacion():
    # Dos salidas de A que terminan después del tramo: una hacia B y otra sin estación de fin. Las dos cuentan
    # como "terminan después del tramo": de esa punta solo se sabe que cae fuera. Lo mismo con dos llegadas a B
    # de viajes que salieron antes del tramo, uno de A y otro sin estación de inicio.
    network = make_network({"A": 10, "B": 10})
    trips = make_trips(
        [
            ("r1", "2026-09-01 23:50", "2026-09-02 00:10", "A", "B"),
            ("r2", "2026-09-01 23:55", "2026-09-02 00:15", "A", None),
            ("r3", "2026-08-31 23:50", "2026-09-01 00:10", "A", "B"),
            ("r4", "2026-08-31 23:55", "2026-09-01 00:15", None, "B"),
        ]
    )
    calendar = build_calendar(DAY1, DAY1, make_reglas().franjas)
    counts = prepare_events(trips, network, calendar.start_us, calendar.end_us).counts
    assert (counts["salidas_que_salen"], counts["llegadas_que_entran"]) == (2, 2)
    assert counts["salidas_que_salen_por_motivo"] == {
        "terminan_despues_del_tramo": 2, "sin_estacion_de_fin": 0, "a_una_estacion_fuera_del_conjunto": 0,
    }
    assert counts["llegadas_que_entran_por_motivo"] == {
        "salieron_antes_del_tramo": 2, "sin_estacion_de_inicio": 0, "de_una_estacion_fuera_del_conjunto": 0,
    }


def test_el_neto_cambia_el_total_de_bicis():
    network = make_network({"A": 20, "B": 20})
    trips = make_trips(
        [
            ("r1", "2026-09-01 08:00", "2026-09-01 08:10", None, "A"),
            ("r2", "2026-09-01 08:20", "2026-09-01 08:30", None, "A"),
            ("r3", "2026-09-01 09:00", "2026-09-01 09:10", "B", None),
        ]
    )
    result, _ = run(network, trips, [5, 5])
    assert (int(result.entered.sum()), int(result.left.sum())) == (2, 1)
    assert result.final_total == 11 == result.initial_total + 2 - 1


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
    Las llegadas vienen de fuera y las salidas van hacia fuera: no dependen de otra estación.
    """
    network = make_network({"A": 10, "B": 10})
    rows = [
        ("d1", "2026-09-01 07:30", "2026-09-01 07:40", "A", None),
        ("d2", "2026-09-01 08:00", "2026-09-01 08:10", "A", None),
        ("a1", "2026-09-01 13:20", "2026-09-01 13:30", None, "A"),
        ("d3", "2026-09-01 22:00", "2026-09-01 22:10", "A", None),
        ("a2", "2026-09-02 06:20", "2026-09-02 06:30", None, "A"),
        ("d4", "2026-09-02 20:00", "2026-09-02 20:10", "A", None),
    ]
    rows += [(f"b{i}", "2026-09-02 15:50", "2026-09-02 16:00", None, "A") for i in range(7)]
    result, calendar = run(network, make_trips(rows), [3, 5], days=(DAY1, DAY2))
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
    assert result.final_inventory.tolist() == [8, 5]


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
    assert fields == {"instant", "day", "franja", "inventory", "budget"}
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
    network = build_network(stations, "mixta")
    options = np.array(names + ["X", None], dtype=object)   # "X" no está en la red; None = sin estación
    minutes = rng.integers(-60, 2 * 24 * 60 + 60, n_trips)   # también antes y después del tramo
    started = ts("2026-09-01") + pd.to_timedelta(minutes, unit="min")
    duration = rng.choice([0, 0, 5, 10, 15, 45], n_trips)   # hay viajes de duración cero
    trips = pd.DataFrame(
        {
            "ride_id": [f"r{i:04d}" for i in rng.permutation(n_trips)],
            "started_at": pd.DatetimeIndex(started).as_unit("us"),
            "ended_at": pd.DatetimeIndex(started + pd.to_timedelta(duration, unit="min")).as_unit("us"),
            "start_station_id": options[rng.integers(0, len(options), n_trips)],
            "end_station_id": options[rng.integers(0, len(options), n_trips)],
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
    events = []
    for row in trips.itertuples():
        begin, end = row.started_at.value // 1000, row.ended_at.value // 1000
        departs = row.start_station_id in capacity and t0 <= begin < t1
        arrives = row.end_station_id in capacity and t0 <= end < t1
        if departs:
            events.append((begin, 1, row.ride_id, "sale", row.start_station_id, arrives))
        if arrives:
            turn = 2 if (departs and begin == end) else 0
            events.append((end, turn, row.ride_id, "llega", row.end_station_id, departs))
    events.sort(key=lambda event: event[:3])

    def day_of(instant):
        return max(d for d in range(len(calendar.days)) if calendar.day_start_us[d] <= instant)

    counts = {name: [0] * len(calendar.days) for name in ("blocked", "diverted", "no_room", "entered", "left")}
    blocked_rides = set()
    history = {name: [(t0, inventory[name])] for name in ids}   # (desde cuándo, bicis)
    for instant, _, ride, what, station, other_side in events:
        day = day_of(instant)
        if what == "sale":
            if inventory[station] == 0:
                counts["blocked"][day] += 1
                blocked_rides.add(ride)
                continue
            inventory[station] -= 1
            history[station].append((instant, inventory[station]))
            if not other_side:
                counts["left"][day] += 1
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
                with_room = [name for name in by_distance if inventory[name] < capacity[name]]
                if not with_room:
                    counts["no_room"][day] += 1
                    if other_side:
                        counts["left"][day] += 1
                    continue
                station = with_room[0]
            inventory[station] += 1
            history[station].append((instant, inventory[station]))
            if not other_side:
                counts["entered"][day] += 1

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
    return [inventory[name] for name in ids], counts, risk


@pytest.mark.parametrize("seed", range(8))
def test_el_recorrido_da_lo_mismo_que_el_simulador_de_referencia(seed):
    network, trips, start = random_case(seed)
    reglas = make_reglas()
    result, calendar = run(network, trips, start, days=(DAY1, DAY2), reglas=reglas)
    final, counts, risk = reference(network, trips, start, calendar, reglas)
    assert result.final_inventory.tolist() == final
    for name, expected in counts.items():
        assert getattr(result, name).tolist() == expected, name
    assert (result.risk_us == risk).all()
    # El caso al azar sí ejercita las reglas: hay bloqueados, desviadas, entradas y salidas.
    assert sum(counts["blocked"]) > 0 and sum(counts["diverted"]) > 0
    assert sum(counts["entered"]) > 0 and sum(counts["left"]) > 0
    assert (0 <= result.final_inventory).all() and (result.final_inventory <= network.capacity).all()
    assert result.final_total == result.initial_total + sum(counts["entered"]) - sum(counts["left"])
