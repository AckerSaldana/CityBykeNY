"""Pruebas de las políticas del simulador (src/sim/policies.py).

Con estaciones, historia y pronósticos escritos a mano:

a) Regla histórica: el rango seguro (lo, hi) sale del flujo neto medio de la
   historia, por tipo de día; sin solución usa el punto medio y nunca la
   ocupación de respaldo; una estación sin información no se toca.
b) Modelo: el rango sale de la envolvente del pronóstico; sin solución usa la
   ocupación de respaldo.
c) De inventarios deseados a traslados: distancia máxima, orden de prioridad,
   primero la necesidad contraria y después quien presta o recibe sin salir de
   su rango, y alto al acabarse el presupuesto.
d) Sin información del futuro: cambiar o borrar viajes posteriores a una
   intervención no cambia lo que se decidió en ella, y lo que recibe una
   política no trae viajes.

Todo corre sin red y sin tocar data/processed. Las reglas de operación de
estas pruebas están escritas aquí: no dependen de config/operacion.yaml.
"""

import copy
import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.contracts import QUANTILE_COLUMNS, STATION_TYPES
from src.paths import TZ
from src.sim.engine import (
    InterventionContext,
    Move,
    Reglas,
    SimulacionError,
    build_calendar,
    build_network,
    prepare_events,
    simulate,
)
from src.sim.policies import (
    ForecastInfo,
    HistoricalPolicy,
    HistoricalProfile,
    ModelPolicy,
    NoIntervention,
    day_type,
    integer_range,
    match_moves,
)

STEP = 0.005  # grados de latitud entre estaciones vecinas: 0.556 km
MANANA = (5, 6, 7, 8, 9, 10, 11)   # horas de la intervención (05:00) al fin de la franja (12:00)
# El 1 de septiembre de 2026 es martes.
THURSDAY, FRIDAY, SATURDAY = date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 29)
WEDNESDAY, NEXT_SATURDAY = date(2026, 9, 2), date(2026, 9, 5)
START = pd.Timestamp("2026-09-01", tz=TZ)   # primer instante del tramo simulado


def make_cfg(**changes) -> dict:
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
            kind: {"prioridad": 2, "ocupacion_respaldo": {"manana": 0.3, "tarde": 0.6}} for kind in STATION_TYPES
        },
    }
    cfg.update(copy.deepcopy(changes))
    return cfg


def make_reglas(**changes) -> Reglas:
    return Reglas.from_operacion(make_cfg(**changes))


def make_network(capacities: dict, types: dict | None = None):
    """Estaciones en línea de sur a norte, en el orden dado, separadas 0.556 km."""
    types = types or {}
    stations = pd.DataFrame(
        {
            "station_id": list(capacities),
            "name": list(capacities),
            "lat": [40.70 + STEP * i for i in range(len(capacities))],
            "lon": [-73.99] * len(capacities),
            "capacity": pd.array(list(capacities.values()), dtype="Int64"),
            "station_type": pd.Series([types.get(name) for name in capacities], dtype="object"),
        }
    )
    return build_network(stations, "mixta")


def hourly(station: str, day: date, values: dict) -> list[tuple]:
    """Filas (estación, hora, valor) de un día: `values` va de hora de reloj a valor."""
    return [(station, pd.Timestamp(f"{day} {hour:02d}:00", tz=TZ), value) for hour, value in values.items()]


def history_table(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["station_id", "ts_hour", "net_flow"])
    frame["ts_hour"] = pd.DatetimeIndex(frame["ts_hour"]).as_unit("us")
    return frame


def forecast_table(rows: list[tuple]) -> pd.DataFrame:
    """Filas (estación, hora, (dep_q10, dep_q50, dep_q90, arr_q10, arr_q50, arr_q90))."""
    frame = pd.DataFrame(
        [(station, hour, "baseline", "val", *quantiles) for station, hour, quantiles in rows],
        columns=["station_id", "ts_hour", "model", "split", *QUANTILE_COLUMNS],
    )
    frame["ts_hour"] = pd.DatetimeIndex(frame["ts_hour"]).as_unit("us")
    return frame


def context(day: date, franja, inventory, budget: int = 1000) -> InterventionContext:
    return InterventionContext(
        instant=pd.Timestamp(f"{day} {franja.momento_intervencion}", tz=TZ),
        day=day,
        franja=franja,
        inventory=np.array(inventory, dtype=np.int64),
        budget=budget,
    )


# --------------------------------------------------------------------------- #
# a) Regla histórica
# --------------------------------------------------------------------------- #

def historical_case():
    """Cuatro estaciones y su historia.

    A (20 lugares). Flujo neto de la mañana (horas 5 a 11):
        jueves   [0, -2, -4, -6, 0, 2, 2]
        viernes  [0, -4, -4, -2, 0, 2, 4]
        media    [0, -3, -4, -4, 0, 2, 3]  -> acumulado [0, -3, -7, -11, -11, -9, -6]
      Baja hasta -11 y nunca sube de 0:  lo = 2 + 11 = 13,  hi = 20 - 2 - 0 = 18.
      Sábado: +1 cada hora -> acumulado 1..7:  lo = 2,  hi = 20 - 2 - 7 = 11.
    B (10 lugares). Laborable [-6, 0, 0, 7, 0, 0, 0] -> acumulado [-6, -6, -6, 1, 1, 1, 1]
      lo = 2 + 6 = 8,  hi = 10 - 2 - 1 = 7:  sin solución. Punto medio 7.5 -> 8.
    C (10 lugares): sin ninguna fila en la historia.
    D (10 lugares): solo tiene las horas 9 a 11 del viernes (empezó a operar ese día).
    """
    network = make_network({"A": 20, "B": 10, "C": 10, "D": 10})
    rows = []
    rows += hourly("A", THURSDAY, dict(zip(MANANA, [0, -2, -4, -6, 0, 2, 2])))
    rows += hourly("A", FRIDAY, dict(zip(MANANA, [0, -4, -4, -2, 0, 2, 4])))
    rows += hourly("A", SATURDAY, dict(zip(MANANA, [1] * 7)))
    # Horas fuera de la franja: no deben contar.
    rows += hourly("A", THURSDAY, {4: -50, 12: 50, 13: -50})
    for day in (THURSDAY, FRIDAY):
        rows += hourly("B", day, dict(zip(MANANA, [-6, 0, 0, 7, 0, 0, 0])))
    rows += hourly("D", FRIDAY, {9: 1, 10: 1, 11: 1})
    reglas = make_reglas()
    profile = HistoricalProfile.from_station_hour(history_table(rows), network, reglas, before=START)
    return network, reglas, profile


def test_la_regla_historica_calcula_el_rango_seguro_con_el_flujo_neto_medio():
    network, reglas, profile = historical_case()
    policy = HistoricalPolicy(network, reglas, profile)
    manana = reglas.franjas[0]
    targets = policy.targets(context(WEDNESDAY, manana, [10, 5, 5, 5]))
    assert (targets.low[0], targets.high[0]) == (13.0, 18.0)
    assert bool(targets.feasible[0]) and targets.has_info.tolist() == [True, True, False, False]
    # El inventario deseado es el actual llevado al rango [13, 18].
    for current, desired in ((10, 13), (13, 13), (15, 15), (18, 18), (19, 18), (0, 13), (20, 18)):
        assert policy.targets(context(WEDNESDAY, manana, [current, 5, 5, 5])).desired[0] == desired


def test_la_regla_historica_usa_el_perfil_del_tipo_de_dia():
    network, reglas, profile = historical_case()
    assert day_type(WEDNESDAY) == "laborable" and day_type(NEXT_SATURDAY) == "fin de semana"
    assert day_type(date(2026, 9, 7)) == "laborable"   # Labor Day es lunes: cuenta como laborable
    policy = HistoricalPolicy(network, reglas, profile)
    targets = policy.targets(context(NEXT_SATURDAY, reglas.franjas[0], [15, 5, 5, 5]))
    # Sábado: sube 7 -> rango [2, 11]. Con 15 bicis el deseado es 11.
    assert (targets.low[0], targets.high[0], targets.desired[0]) == (2.0, 11.0, 11)
    # B no tiene filas de fin de semana: ese día no tiene información y no se toca.
    assert targets.has_info.tolist() == [True, False, False, False]
    assert targets.desired.tolist() == [11, 5, 5, 5]


def test_sin_solucion_la_regla_historica_usa_el_punto_medio_y_no_la_ocupacion_de_respaldo():
    network, reglas, profile = historical_case()
    policy = HistoricalPolicy(network, reglas, profile)
    manana = reglas.franjas[0]
    for current in (0, 3, 8, 10):
        targets = policy.targets(context(WEDNESDAY, manana, [15, current, 5, 5]))
        assert (targets.low[1], targets.high[1]) == (8.0, 7.0) and not targets.feasible[1]
        assert targets.desired[1] == 8          # punto medio de 8 y 7, redondeado: 7.5 -> 8
    # La ocupación de respaldo de la mañana daría 0.3 x 10 = 3: la regla histórica no la usa.
    assert reglas.ocupacion_respaldo["mixta"]["manana"] * 10 == 3.0
    # Aunque cambie la ocupación de respaldo, la regla histórica decide igual.
    other = make_cfg()
    for kind in STATION_TYPES:
        other["estrategia_por_tipo"][kind]["ocupacion_respaldo"] = {"manana": 0.9, "tarde": 0.1}
    same = HistoricalPolicy(network, Reglas.from_operacion(other), profile)
    assert same.targets(context(WEDNESDAY, manana, [10, 3, 5, 5])).desired.tolist() == [13, 8, 5, 5]


def test_el_punto_medio_se_recorta_a_la_capacidad():
    # E (6 lugares): baja 9 -> lo = 11; sube 0 -> hi = 4. Punto medio 7.5 -> 8, recortado a 6.
    network = make_network({"E": 6, "F": 10})
    rows = hourly("E", THURSDAY, dict(zip(MANANA, [-9, 0, 0, 0, 0, 0, 0])))
    reglas = make_reglas()
    profile = HistoricalProfile.from_station_hour(history_table(rows), network, reglas, before=START)
    targets = HistoricalPolicy(network, reglas, profile).targets(context(WEDNESDAY, reglas.franjas[0], [2, 5]))
    assert (targets.low[0], targets.high[0], targets.desired[0]) == (11.0, 4.0, 6)


def test_una_estacion_sin_informacion_no_pide_ni_presta():
    network, reglas, profile = historical_case()
    assert profile.with_history.tolist() == [True, True, False, True]
    policy = HistoricalPolicy(network, reglas, profile)
    manana = reglas.franjas[0]
    # C no tiene historia. D tiene filas, pero no de todas las horas de la franja.
    targets = policy.targets(context(WEDNESDAY, manana, [15, 8, 0, 10]))
    assert targets.has_info.tolist() == [True, True, False, False]
    assert targets.feasible.tolist() == [True, False, True, True]
    assert targets.desired.tolist() == [15, 8, 0, 10]
    assert targets.low_int.tolist()[2:] == [0, 10] and targets.high_int.tolist()[2:] == [0, 10]
    # A necesita 13 y tiene 3. C y D están al lado con bicis de sobra, pero no prestan: no hay traslados
    # desde ellas. B (sin solución, quiere 8 y tiene 10) sí entrega sus 2.
    wide = HistoricalPolicy(network, make_reglas(distancia_maxima_km=5.0), profile)
    moves = wide.decide(context(WEDNESDAY, manana, [3, 10, 10, 10]))
    assert moves == [Move(1, 0, 2)]


def test_la_franja_de_la_tarde_usa_sus_propias_horas():
    # Horas 14 a 20. La mañana trae un flujo enorme que no debe influir.
    network = make_network({"A": 20, "B": 10})
    rows = hourly("A", THURSDAY, {hour: -1 for hour in range(14, 21)})
    rows += hourly("A", THURSDAY, {hour: 40 for hour in MANANA})
    reglas = make_reglas()
    profile = HistoricalProfile.from_station_hour(history_table(rows), network, reglas, before=START)
    targets = HistoricalPolicy(network, reglas, profile).targets(context(WEDNESDAY, reglas.franjas[1], [5, 5]))
    assert (targets.low[0], targets.high[0], targets.desired[0]) == (9.0, 18.0, 9)   # baja 7: lo = 2 + 7


def test_la_regla_historica_no_acepta_filas_del_tramo_simulado():
    network = make_network({"A": 20, "B": 10})
    rows = hourly("A", THURSDAY, {8: -1}) + hourly("A", date(2026, 9, 1), {0: -1})   # la primera hora del tramo
    with pytest.raises(SimulacionError, match="tramo simulado o posteriores"):
        HistoricalProfile.from_station_hour(history_table(rows), network, make_reglas(), before=START)


def test_un_rango_sin_ningun_entero_dentro_usa_el_punto_medio():
    # lo = 4.2 y hi = 4.8: hay solución, pero ningún inventario entero cabe. Punto medio 4.5 -> 5.
    low, high = np.array([4.2, 4.2, 3.0]), np.array([4.8, 6.9, 3.0])
    has_info = np.array([True, True, True])
    fallback = np.array([5, 99, 99])
    low_int, high_int = integer_range([1, 1, 9], low, high, has_info, fallback, np.array([10, 10, 10]))
    assert low_int.tolist() == [5, 5, 3] and high_int.tolist() == [5, 6, 3]


# --------------------------------------------------------------------------- #
# b) Modelo
# --------------------------------------------------------------------------- #

def model_case(**cfg_changes):
    """Pronóstico del miércoles 2 de septiembre, franja de la mañana (horas 5 a 11).

    A (20 lugares). Solo dos horas con viajes:
        06:00  salidas (2, 5, 9)  llegadas (0, 1, 4)
               centro m = 1 - 5 = -4
               hacia abajo d² = (1-0)² + (9-5)² = 17;  hacia arriba u² = (4-1)² + (5-2)² = 18
        07:00  salidas (1, 3, 6)  llegadas (0, 0, 0)
               m = -3;  d² = 0 + (6-3)² = 9;  u² = 0 + (3-1)² = 4
      Acumulado:  hasta 06:00  centro -4, inferior -4 - raíz(17) = -8.123, superior -4 + raíz(18) = 0.243
                  hasta 07:00  centro -7, inferior -7 - raíz(26) = -12.099, superior -7 + raíz(22) = -2.310
      lo = 2 + 12.099 = 14.099   hi = 20 - 2 - 0.243 = 17.757   -> enteros 15 a 17
    B (10 lugares). Sin solución:
        05:00  salidas (0, 6, 6)  -> m = -6, d² = 0, u² = 36
        08:00  llegadas (7, 7, 7) -> m = +7, d² = 0, u² = 0
      inferior mínima -6, superior máxima 1 + 6 = 7:  lo = 8,  hi = 10 - 2 - 7 = 1
    C (10 lugares): sin pronóstico.  D (10 lugares): con pronóstico pero sin historia.
    """
    network = make_network({"A": 20, "B": 10, "C": 10, "D": 10})
    zero = (0.0,) * 6
    rows = []
    a = {hour: zero for hour in MANANA} | {6: (2, 5, 9, 0, 1, 4), 7: (1, 3, 6, 0, 0, 0)}
    b = {hour: zero for hour in MANANA} | {5: (0, 6, 6, 0, 0, 0), 8: (0, 0, 0, 7, 7, 7)}
    d = {hour: (0, 1, 2, 0, 1, 2) for hour in MANANA}
    for station, values in (("A", a), ("B", b), ("D", d)):
        rows += hourly(station, WEDNESDAY, values)
    rows += hourly("A", WEDNESDAY, {4: (0, 90, 90, 0, 0, 0), 12: (0, 0, 0, 0, 90, 90)})   # fuera de la franja
    reglas = make_reglas(**cfg_changes)
    with_history = np.array([True, True, True, False])
    days = (date(2026, 9, 1), WEDNESDAY)
    forecast = ForecastInfo.from_predictions(forecast_table(rows), network, reglas, days, with_history)
    return network, reglas, forecast


def test_el_modelo_calcula_el_rango_seguro_con_la_envolvente_del_pronostico():
    network, reglas, forecast = model_case()
    window = forecast.window(WEDNESDAY, "manana")
    assert window.center_dip[0] == -7.0 and window.center_rise[0] == 0.0
    assert window.lower_dip[0] == pytest.approx(-7 - np.sqrt(26))
    assert window.upper_rise[0] == pytest.approx(-4 + np.sqrt(18))
    policy = ModelPolicy(network, reglas, forecast)
    manana = reglas.franjas[0]
    targets = policy.targets(context(WEDNESDAY, manana, [10, 5, 5, 5]))
    assert targets.low[0] == pytest.approx(14.0990, abs=1e-4)
    assert targets.high[0] == pytest.approx(17.7574, abs=1e-4)
    assert bool(targets.feasible[0]) and (targets.low_int[0], targets.high_int[0]) == (15, 17)
    for current, desired in ((10, 15), (15, 15), (16, 16), (17, 17), (19, 17), (0, 15)):
        assert policy.targets(context(WEDNESDAY, manana, [current, 5, 5, 5])).desired[0] == desired


def test_sin_solucion_el_modelo_usa_la_ocupacion_de_respaldo_de_la_franja():
    network, reglas, forecast = model_case()
    policy = ModelPolicy(network, reglas, forecast)
    manana = reglas.franjas[0]
    for current in (0, 3, 9):
        targets = policy.targets(context(WEDNESDAY, manana, [15, current, 5, 5]))
        assert (targets.low[1], targets.high[1]) == (8.0, 1.0) and not targets.feasible[1]
        assert targets.desired[1] == 3          # 0.3 x 10 lugares. El punto medio sería 4.5 -> 5.
    # Con otra ocupación de respaldo cambia el deseado; y depende del tipo de la estación.
    cfg = make_cfg()
    cfg["estrategia_por_tipo"]["mixta"]["ocupacion_respaldo"] = {"manana": 0.7, "tarde": 0.1}
    cfg["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"] = {"manana": 0.45, "tarde": 0.1}
    other = Reglas.from_operacion(cfg)
    assert ModelPolicy(network, other, forecast).targets(context(WEDNESDAY, manana, [15, 3, 5, 5])).desired[1] == 7
    typed = make_network({"A": 20, "B": 10, "C": 10, "D": 10}, types={"B": "oficinas"})
    # 0.45 x 10 = 4.5 -> 5 (un medio sube).
    assert ModelPolicy(typed, other, forecast).targets(context(WEDNESDAY, manana, [15, 3, 5, 5])).desired[1] == 5


def test_el_modelo_no_toca_una_estacion_sin_pronostico_o_sin_historia():
    network, reglas, forecast = model_case()
    window = forecast.window(WEDNESDAY, "manana")
    assert window.has_info.tolist() == [True, True, False, False]
    targets = ModelPolicy(network, reglas, forecast).targets(context(WEDNESDAY, reglas.franjas[0], [16, 3, 0, 10]))
    assert targets.desired.tolist() == [16, 3, 0, 10]
    assert targets.feasible.tolist() == [True, False, True, True]
    # El martes 1 no hay ninguna fila de pronóstico: nadie tiene información.
    assert not forecast.window(date(2026, 9, 1), "manana").has_info.any()
    # Si falta el pronóstico de una sola hora de la franja, la estación queda sin información.
    rows = hourly("A", WEDNESDAY, {hour: (0.0,) * 6 for hour in MANANA if hour != 9})
    partial = ForecastInfo.from_predictions(
        forecast_table(rows), network, reglas, (WEDNESDAY,), np.ones(4, dtype=bool)
    )
    assert not partial.window(WEDNESDAY, "manana").has_info[0]
    with pytest.raises(SimulacionError, match="No hay pronóstico preparado"):
        forecast.window(date(2026, 9, 9), "manana")


# --------------------------------------------------------------------------- #
# c) De inventarios deseados a traslados
# --------------------------------------------------------------------------- #

def fixed(values) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def test_solo_se_buscan_contrapartes_dentro_de_la_distancia_maxima():
    # A, B, C, D en línea, a 0.556 km cada paso. A necesita 3 (tiene 2, quiere 5). D, a 1.67 km, tiene 5 de
    # sobra. B puede prestar hasta 4 sin salir de su rango [4, 8] y no puede recibir. C no presta ni recibe.
    network = make_network({"A": 20, "B": 20, "C": 20, "D": 20})
    inventory = fixed([2, 8, 10, 15])
    low, high = fixed([5, 4, 10, 10]), fixed([5, 8, 10, 10])
    priority = fixed([2, 2, 2, 2])
    # Con 1.2 km D no alcanza a A (ni tiene a quién darle): B presta las 3.
    near = network.neighbors_within(1.2)
    assert match_moves(inventory, low, high, priority, near, budget=100) == [Move(1, 0, 3)]
    # Con 2 km D sí alcanza a A y B ya no hace falta.
    far = network.neighbors_within(2.0)
    assert match_moves(inventory, low, high, priority, far, budget=100) == [Move(3, 0, 3)]
    # Con 0.5 km nadie alcanza a nadie.
    assert match_moves(inventory, low, high, priority, network.neighbors_within(0.5), budget=100) == []


def test_se_atiende_por_prioridad_despues_por_diferencia_y_despues_por_id():
    # B, en medio, tiene bicis de sobra. A y C necesitan bicis y solo B les queda cerca. Todas quieren 6.
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(0.6)
    low = high = fixed([6, 6, 6])
    same = fixed([2, 2, 2])
    # Prioridad: C (prioridad 3) va antes que A (prioridad 1) aunque a A le falte más.
    # C toma 4 de las 6 que le sobran a B; después se atiende B (prioridad 2), que da las 2 restantes a A.
    assert match_moves(fixed([1, 12, 2]), low, high, fixed([1, 2, 3]), near, 100) == [Move(1, 2, 4), Move(1, 0, 2)]
    # Misma prioridad: primero la diferencia más grande. A B le sobran 3.
    assert match_moves(fixed([1, 9, 2]), low, high, same, near, 100) == [Move(1, 0, 3)]   # a A le faltan 5
    assert match_moves(fixed([2, 9, 1]), low, high, same, near, 100) == [Move(1, 2, 3)]   # a C le faltan 5
    # Misma prioridad y misma diferencia (4 y 4): primero el id menor.
    assert match_moves(fixed([2, 9, 2]), low, high, same, near, 100) == [Move(1, 0, 3)]


def test_primero_la_mas_cercana_entre_las_que_tienen_la_necesidad_contraria():
    # A necesita 6. B (cerca) tiene 1 de sobra y C (más lejos) tiene 5.
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    inventory, low, high = fixed([0, 7, 11]), fixed([6, 6, 6]), fixed([6, 6, 6])
    assert match_moves(inventory, low, high, fixed([2, 2, 2]), near, budget=100) == [Move(1, 0, 1), Move(2, 0, 5)]


def test_quien_presta_o_recibe_no_sale_de_su_rango_seguro():
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    priority = fixed([2, 2, 2])
    # A necesita 6. B (8 bicis, rango 5 a 9) puede prestar 3. C (7, rango 7 a 10) no puede prestar ninguna.
    moves = match_moves(fixed([0, 8, 7]), fixed([6, 5, 7]), fixed([9, 9, 10]), priority, near, budget=100)
    assert moves == [Move(1, 0, 3)]
    # A le sobran 4 (10 bicis, rango 2 a 6). B (5, rango 3 a 7) recibe 2 y C (9, rango 4 a 10) recibe 1.
    # La cuarta no tiene a dónde ir y se queda en A.
    moves = match_moves(fixed([10, 5, 9]), fixed([2, 3, 4]), fixed([6, 7, 10]), priority, near, budget=100)
    assert moves == [Move(0, 1, 2), Move(0, 2, 1)]


def test_se_detiene_cuando_se_acaba_el_presupuesto():
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    inventory, low, high = fixed([0, 12, 12]), fixed([8, 6, 6]), fixed([8, 6, 6])
    priority = fixed([2, 2, 2])
    assert match_moves(inventory, low, high, priority, near, budget=100) == [Move(1, 0, 6), Move(2, 0, 2)]
    assert match_moves(inventory, low, high, priority, near, budget=7) == [Move(1, 0, 6), Move(2, 0, 1)]
    assert match_moves(inventory, low, high, priority, near, budget=4) == [Move(1, 0, 4)]
    assert match_moves(inventory, low, high, priority, near, budget=0) == []


def test_con_presupuesto_cero_las_politicas_no_mueven_nada():
    network, reglas, profile = historical_case()
    policy = HistoricalPolicy(network, make_reglas(distancia_maxima_km=5.0), profile)
    manana = reglas.franjas[0]
    assert policy.decide(context(WEDNESDAY, manana, [3, 10, 10, 10], budget=0)) == []
    assert policy.decide(context(WEDNESDAY, manana, [3, 10, 10, 10], budget=1)) == [Move(1, 0, 1)]
    assert NoIntervention().decide(context(WEDNESDAY, manana, [3, 10, 10, 10], budget=50)) == []


def test_la_prioridad_sale_del_tipo_de_la_estacion():
    cfg = make_cfg(distancia_maxima_km=0.6)
    cfg["estrategia_por_tipo"]["oficinas"]["prioridad"] = 3
    cfg["estrategia_por_tipo"]["recreativa"]["prioridad"] = 1
    network = make_network({"A": 20, "B": 20, "C": 20}, types={"A": "recreativa", "C": "oficinas"})
    rows = []
    for station, flow in (("A", -3), ("B", 3), ("C", -3)):   # A y C pierden 3 en la mañana; B gana 3
        rows += hourly(station, THURSDAY, dict(zip(MANANA, [flow, 0, 0, 0, 0, 0, 0])))
    reglas = Reglas.from_operacion(cfg)
    profile = HistoricalProfile.from_station_hour(history_table(rows), network, reglas, before=START)
    policy = HistoricalPolicy(network, reglas, profile)
    assert policy.priority.tolist() == [1, 2, 3]
    # A y C necesitan 5 (lo = 2 + 3) y tienen 4. B (rango 2 a 15) tiene 16: le sobra 1, que va a C por prioridad.
    # Después B presta sin salir de su rango: C ya está completa y A recibe 1.
    moves = policy.decide(context(WEDNESDAY, reglas.franjas[0], [4, 16, 4]))
    assert moves == [Move(1, 2, 1), Move(1, 0, 1)]


# --------------------------------------------------------------------------- #
# d) Sin información del futuro
# --------------------------------------------------------------------------- #

def future_case(seed: int = 11):
    """Seis estaciones, viajes al azar de tres días, y las dos políticas con información al azar."""
    rng = np.random.default_rng(seed)
    names = ["A", "B", "C", "D", "E", "F"]
    stations = pd.DataFrame(
        {
            "station_id": names,
            "name": names,
            "lat": 40.70 + rng.random(6) * 0.01,
            "lon": -73.99 + rng.random(6) * 0.01,
            "capacity": pd.array(rng.integers(8, 16, 6), dtype="Int64"),
            "station_type": pd.Series([None] * 6, dtype="object"),
        }
    )
    reglas = make_reglas(distancia_maxima_km=3.0)
    network = build_network(stations, "mixta")
    n_trips = 900
    started = START + pd.to_timedelta(rng.integers(0, 3 * 24 * 3600, n_trips), unit="s")
    ended = started + pd.to_timedelta(rng.integers(60, 1800, n_trips), unit="s")
    trips = pd.DataFrame(
        {
            "ride_id": [f"r{i:04d}" for i in range(n_trips)],
            "started_at": pd.DatetimeIndex(started).as_unit("us"),
            "ended_at": pd.DatetimeIndex(ended).as_unit("us"),
            "start_station_id": rng.choice(names, n_trips, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]),
            "end_station_id": rng.choice(names, n_trips, p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.3]),
        }
    )
    days = tuple(date(2026, 9, d) for d in (1, 2, 3))
    history, predictions = [], []
    for station in names:
        for day in (THURSDAY, FRIDAY, SATURDAY):
            history += hourly(station, day, {hour: int(rng.integers(-3, 4)) for hour in range(24)})
        for day in days:
            for hour in range(24):
                dep, arr = rng.random() * 3, rng.random() * 3
                predictions += hourly(station, day, {hour: (dep * 0.5, dep, dep * 1.5, arr * 0.5, arr, arr * 1.5)})
    profile = HistoricalProfile.from_station_hour(history_table(history), network, reglas, before=START)
    forecast = ForecastInfo.from_predictions(
        forecast_table(predictions), network, reglas, days, profile.with_history
    )
    policies = {
        "historical": HistoricalPolicy(network, reglas, profile),
        "model": ModelPolicy(network, reglas, forecast),
    }
    return network, reglas, trips, days, policies


def decisions(network, reglas, trips, days, policy, budget=12):
    calendar = build_calendar(days[0], days[-1], reglas.franjas)
    events = prepare_events(trips, network, calendar.start_us, calendar.end_us)
    start = network.capacity // 2
    result = simulate(network, events, calendar, reglas, policy, budget, start)
    return result, calendar


@pytest.mark.parametrize("name", ["historical", "model"])
@pytest.mark.parametrize("cut", [1, 2, 4])
def test_cambiar_o_borrar_viajes_posteriores_no_cambia_lo_decidido_antes(name, cut):
    network, reglas, trips, days, policies = future_case()
    policy = policies[name]
    base, calendar = decisions(network, reglas, trips, days, policy)
    assert sum(len(record.moves) for record in base.interventions) > 0   # la política sí mueve bicis
    instant = pd.Timestamp(calendar.slots[cut].instant_us, unit="us", tz="UTC")
    later = trips["started_at"] >= instant

    # 1) Se borran todos los viajes que empiezan en ese instante o después.
    deleted, _ = decisions(network, reglas, trips[~later].reset_index(drop=True), days, policy)
    # 2) Se cambian: otro destino, otra duración y además viajes nuevos.
    changed = trips.copy()
    changed.loc[later, "end_station_id"] = "A"
    changed.loc[later, "ended_at"] = changed.loc[later, "started_at"] + pd.Timedelta(minutes=1)
    extra = changed[later].head(40).assign(
        ride_id=lambda frame: "x" + frame["ride_id"], start_station_id="F", end_station_id="B"
    )
    altered, _ = decisions(network, reglas, pd.concat([changed, extra], ignore_index=True), days, policy)

    for other in (deleted, altered):
        for before, after in zip(base.interventions[: cut + 1], other.interventions[: cut + 1]):
            assert after.moves == before.moves
            assert (after.inventory_before == before.inventory_before).all()
            assert (after.net == before.net).all()
    # La prueba distingue: después del corte el inventario que ven las políticas sí cambia.
    assert any(
        (after.inventory_before != before.inventory_before).any()
        for before, after in zip(base.interventions[cut + 1 :], altered.interventions[cut + 1 :])
    )


def test_las_politicas_no_reciben_viajes_ni_al_crearse_ni_al_decidir():
    assert list(inspect.signature(HistoricalPolicy.__init__).parameters) == [
        "self", "network", "reglas", "profile", "neighbors",
    ]
    assert list(inspect.signature(ModelPolicy.__init__).parameters) == [
        "self", "network", "reglas", "forecast", "neighbors",
    ]
    for policy in (HistoricalPolicy, ModelPolicy, NoIntervention):
        assert list(inspect.signature(policy.decide).parameters) == ["self", "context"]
    # Las fuentes de información se arman sin viajes: con station_hour anterior al tramo o con predictions.
    assert "trips" not in inspect.signature(HistoricalProfile.from_station_hour).parameters
    assert "trips" not in inspect.signature(ForecastInfo.from_predictions).parameters


def test_la_misma_intervencion_da_siempre_la_misma_decision():
    network, reglas, trips, days, policies = future_case()
    for policy in policies.values():
        first, _ = decisions(network, reglas, trips, days, policy)
        second, _ = decisions(network, reglas, trips, days, policy)
        assert first.digest() == second.digest()
        # Y los traslados que salen de las políticas pasan los cuatro límites del motor (si no, habría fallado).
        assert int(first.budget_used.sum()) > 0
        assert (first.budget_used <= first.budget_available).all()
