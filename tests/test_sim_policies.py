"""Pruebas de las políticas del simulador (src/sim/policies.py).

Con estaciones y pronósticos escritos a mano:

a) Una sola maquinaria: la regla histórica y el modelo son `ForecastPolicy` y
   con el mismo pronóstico deciden lo mismo. Lo único que las distingue es el
   pronóstico que reciben.
b) Banda de incertidumbre: el rango seguro (lo, hi) sale de la trayectoria
   central y de la dispersión por el multiplicador (0, 0.5 o 1). Dentro del
   rango no se mueve nada; fuera, la estación va solo al borde más cercano.
c) Franja sin solución: una estación con tipo va a `ocupacion_respaldo` de su
   tipo, recortada a [bicis_minimas, capacidad - lugares_libres_minimos]; una
   estación sin tipo no se toca.
d) Reparto: prioridad, después la diferencia MÁS PEQUEÑA y después el id;
   distancia máxima; solo entre estaciones con necesidades contrarias (la que
   está dentro de su rango no presta ni recibe); alto al acabarse el
   presupuesto.
e) Sin información del futuro: cambiar o borrar viajes posteriores a una
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
from src.sim import policies as policies_module
from src.sim.policies import (
    UNTYPED_PRIORITY,
    ForecastInfo,
    ForecastPolicy,
    NoIntervention,
    accepted_range,
    match_moves,
    respaldo_inventory,
    stations_with_history,
)

STEP = 0.005  # grados de latitud entre estaciones vecinas: 0.556 km
MANANA = (5, 6, 7, 8, 9, 10, 11)   # horas de la intervención (05:00) al fin de la franja (12:00)
TARDE = (14, 15, 16, 17, 18, 19, 20)
# El 1 de septiembre de 2026 es martes.
THURSDAY = date(2026, 8, 27)
WEDNESDAY = date(2026, 9, 2)
START = pd.Timestamp("2026-09-01", tz=TZ)   # primer instante del tramo simulado
ZERO = (0.0,) * 6


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
    return build_network(stations)


def hourly(station: str, day: date, values: dict) -> list[tuple]:
    """Filas (estación, hora, valor) de un día: `values` va de hora de reloj a valor."""
    return [(station, pd.Timestamp(f"{day} {hour:02d}:00", tz=TZ), value) for hour, value in values.items()]


def forecast_table(rows: list[tuple], model: str = "baseline") -> pd.DataFrame:
    """Filas (estación, hora, (dep_q10, dep_q50, dep_q90, arr_q10, arr_q50, arr_q90))."""
    frame = pd.DataFrame(
        [(station, hour, model, "val", *quantiles) for station, hour, quantiles in rows],
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


def fixed(values) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def case(types: dict | None = None, **cfg_changes):
    """Pronóstico del miércoles 2 de septiembre, franja de la mañana (horas 5 a 11).

    A (20 lugares). Solo dos horas con viajes:
        06:00  salidas (2, 5, 9)  llegadas (0, 1, 4)
               centro m = 1 - 5 = -4
               hacia abajo d² = (1-0)² + (9-5)² = 17;  hacia arriba u² = (4-1)² + (5-2)² = 18
        07:00  salidas (1, 3, 6)  llegadas (0, 0, 0)
               m = -3;  d² = 0 + (6-3)² = 9;  u² = 0 + (3-1)² = 4
      Acumulado:  hasta 06:00  centro -4, dispersión abajo raíz(17) = 4.123, arriba raíz(18) = 4.243
                  hasta 07:00  centro -7, dispersión abajo raíz(26) = 5.099, arriba raíz(22) = 4.690
      Banda 1:    inferior mínima -7 - 5.099 = -12.099;  superior máxima -4 + 4.243 = 0.243
                  lo = 2 + 12.099 = 14.099   hi = 20 - 2 - 0.243 = 17.757   -> enteros 15 a 17
      Banda 0.5:  inferior -7 - 2.550 = -9.550;  superior máxima 0 (nunca pasa del inicio)
                  lo = 11.550   hi = 18   -> enteros 12 a 18
      Banda 0:    lo = 2 + 7 = 9   hi = 18   -> enteros 9 a 18
    B (10 lugares). Con banda 1 no tiene solución:
        05:00  salidas (0, 6, 6)  -> m = -6, d² = 0, u² = 36
        08:00  llegadas (7, 7, 7) -> m = +7, d² = 0, u² = 0
      Banda 1:  inferior mínima -6, superior máxima 1 + 6 = 7:  lo = 8,  hi = 10 - 2 - 7 = 1
      Banda 0:  baja 6 y sube 1:  lo = 8,  hi = 7  -> tampoco
    C (10 lugares): sin pronóstico.  D (10 lugares): con pronóstico pero sin historia.
    """
    network = make_network({"A": 20, "B": 10, "C": 10, "D": 10}, types)
    rows = []
    a = {hour: ZERO for hour in MANANA} | {6: (2, 5, 9, 0, 1, 4), 7: (1, 3, 6, 0, 0, 0)}
    b = {hour: ZERO for hour in MANANA} | {5: (0, 6, 6, 0, 0, 0), 8: (0, 0, 0, 7, 7, 7)}
    d = {hour: (0, 1, 2, 0, 1, 2) for hour in MANANA}
    for station, values in (("A", a), ("B", b), ("D", d)):
        rows += hourly(station, WEDNESDAY, values)
    rows += hourly("A", WEDNESDAY, {4: (0, 90, 90, 0, 0, 0), 12: (0, 0, 0, 0, 90, 90)})   # fuera de la franja
    reglas = make_reglas(**cfg_changes)
    with_history = np.array([True, True, True, False])
    days = (date(2026, 9, 1), WEDNESDAY)
    forecast = ForecastInfo.from_predictions(forecast_table(rows), network, reglas, days, with_history)
    return network, reglas, forecast


# --------------------------------------------------------------------------- #
# a) Una sola maquinaria
# --------------------------------------------------------------------------- #

def test_la_regla_historica_y_el_modelo_son_la_misma_clase_y_con_el_mismo_pronostico_deciden_igual():
    network, _, forecast = case(types={"B": "mixta"})   # B, sin solución y con tipo, va a su respaldo (3)
    reglas = make_reglas(distancia_maxima_km=5.0)
    historical = ForecastPolicy("historical", network, reglas, forecast, band=0.5)
    model = ForecastPolicy("model", network, reglas, forecast, band=0.5)
    assert type(historical) is type(model) and (historical.name, model.name) == ("historical", "model")
    manana = reglas.franjas[0]
    for inventory in ([3, 9, 5, 5], [19, 2, 2, 2], [12, 5, 5, 5]):
        ctx = context(WEDNESDAY, manana, inventory)
        assert historical.decide(ctx) == model.decide(ctx)
        one, other = historical.targets(ctx), model.targets(ctx)
        assert (one.low_int == other.low_int).all() and (one.high_int == other.high_int).all()
    assert historical.decide(context(WEDNESDAY, manana, [3, 9, 5, 5])) == [Move(1, 0, 6)]
    # Ya no existe el camino aparte de la regla histórica (perfil por tipo de día, punto medio).
    for old in ("HistoricalPolicy", "HistoricalProfile", "ModelPolicy", "day_type", "integer_range"):
        assert not hasattr(policies_module, old), old


def test_lo_unico_que_cambia_entre_politicas_es_el_pronostico():
    network, reglas, forecast = case()
    rows = hourly("A", WEDNESDAY, {hour: ZERO for hour in MANANA} | {6: (0, 1, 2, 0, 0, 0)})   # A solo pierde 1
    other = ForecastInfo.from_predictions(
        forecast_table(rows, "lgbm"), network, reglas, (date(2026, 9, 1), WEDNESDAY), np.ones(4, dtype=bool)
    )
    manana = reglas.franjas[0]
    ctx = context(WEDNESDAY, manana, [5, 5, 5, 5])
    assert ForecastPolicy("historical", network, reglas, forecast, 0.0).targets(ctx).desired[0] == 9
    assert ForecastPolicy("model", network, reglas, other, 0.0).targets(ctx).desired[0] == 5   # lo = 2 + 1 = 3
    # Un pronóstico trae filas de un solo modelo.
    mixed = pd.concat([forecast_table(rows, "baseline"), forecast_table(rows, "lgbm")], ignore_index=True)
    with pytest.raises(SimulacionError, match="un solo modelo"):
        ForecastInfo.from_predictions(mixed, network, reglas, (WEDNESDAY,), np.ones(4, dtype=bool))


# --------------------------------------------------------------------------- #
# b) Banda de incertidumbre
# --------------------------------------------------------------------------- #

def test_la_banda_multiplica_la_dispersion_del_pronostico():
    network, reglas, forecast = case()
    full = forecast.window(WEDNESDAY, "manana", 1.0)
    assert full.center_dip[0] == -7.0 and full.center_rise[0] == 0.0
    assert full.lower_dip[0] == pytest.approx(-7 - np.sqrt(26))
    assert full.upper_rise[0] == pytest.approx(-4 + np.sqrt(18))
    half = forecast.window(WEDNESDAY, "manana", 0.5)
    assert half.lower_dip[0] == pytest.approx(-7 - 0.5 * np.sqrt(26)) and half.upper_rise[0] == 0.0
    none = forecast.window(WEDNESDAY, "manana", 0.0)
    assert (none.lower_dip[0], none.upper_rise[0]) == (none.center_dip[0], none.center_rise[0]) == (-7.0, 0.0)
    # La trayectoria central no depende de la banda.
    assert half.center_dip[0] == full.center_dip[0] and (half.has_info == full.has_info).all()


@pytest.mark.parametrize(
    "band, low, high, accepted",
    [(1.0, 14.0990, 17.7574, (15, 17)), (0.5, 11.5495, 18.0, (12, 18)), (0.0, 9.0, 18.0, (9, 18))],
)
def test_el_rango_aceptado_sale_de_la_banda(band, low, high, accepted):
    network, reglas, forecast = case()
    policy = ForecastPolicy("historical", network, reglas, forecast, band)
    targets = policy.targets(context(WEDNESDAY, reglas.franjas[0], [10, 5, 5, 5]))
    assert targets.low[0] == pytest.approx(low, abs=1e-4) and targets.high[0] == pytest.approx(high, abs=1e-4)
    assert bool(targets.feasible[0]) and (targets.low_int[0], targets.high_int[0]) == accepted


def test_dentro_del_rango_no_se_mueve_nada_y_fuera_se_va_solo_al_borde_mas_cercano():
    network, reglas, forecast = case()
    policy = ForecastPolicy("historical", network, reglas, forecast, 1.0)   # A acepta de 15 a 17
    manana = reglas.franjas[0]
    for current, desired in ((10, 15), (0, 15), (14, 15), (15, 15), (16, 16), (17, 17), (18, 17), (20, 17)):
        assert policy.targets(context(WEDNESDAY, manana, [current, 5, 5, 5])).desired[0] == desired
    # Si todas las estaciones están dentro de su rango no se mueve nada, aunque haya presupuesto.
    typed, _, _ = case(types={"B": "mixta"})   # B, sin solución y con tipo, acepta solo su respaldo: 3
    wide = ForecastPolicy("historical", typed, make_reglas(distancia_maxima_km=5.0), forecast, 1.0)
    for inside in (15, 16, 17):
        assert wide.decide(context(WEDNESDAY, manana, [inside, 3, 5, 5])) == []
    # Fuera del rango recibe justo lo que le falta para llegar al borde, no más: a A le faltan 2 y a B le sobran 2.
    assert wide.decide(context(WEDNESDAY, manana, [13, 5, 5, 5])) == [Move(1, 0, 2)]
    # Aunque a B le sobren 6, A solo toma las 2 que la llevan al borde (15). Las otras 4 se quedan en B:
    # ya nadie las necesita, y una estación dentro de su rango no recibe.
    assert wide.decide(context(WEDNESDAY, manana, [13, 9, 5, 5])) == [Move(1, 0, 2)]
    # Y si a A le sobran (19, máximo 17), entrega solo 2 aunque a B le falten más.
    assert wide.decide(context(WEDNESDAY, manana, [19, 0, 5, 5])) == [Move(0, 1, 2)]


def test_un_rango_sin_ningun_entero_dentro_usa_el_punto_medio():
    # lo = 4.2 y hi = 4.8: hay solución, pero ningún entero cabe. Se usa el punto medio redondeado.
    low_int, high_int, feasible = accepted_range(
        fixed([9]), np.array([4.2]), np.array([4.8]), np.array([True]), fixed([-1]), fixed([10])
    )
    assert (low_int.tolist(), high_int.tolist(), feasible.tolist()) == ([5], [5], [True])


def test_una_estacion_sin_pronostico_o_sin_historia_no_pide_ni_presta():
    network, reglas, forecast = case()
    window = forecast.window(WEDNESDAY, "manana", 1.0)
    assert window.has_info.tolist() == [True, True, False, False]
    policy = ForecastPolicy("historical", network, make_reglas(distancia_maxima_km=5.0), forecast, 1.0)
    manana = reglas.franjas[0]
    targets = policy.targets(context(WEDNESDAY, manana, [16, 3, 0, 10]))
    assert targets.desired.tolist() == [16, 3, 0, 10]
    assert targets.feasible.tolist() == [True, False, True, True]
    # A necesita bicis y C y D tienen de sobra, pero no tienen información: no prestan.
    assert policy.decide(context(WEDNESDAY, manana, [10, 3, 10, 10])) == []
    # El martes 1 no hay ninguna fila de pronóstico: nadie tiene información.
    assert not forecast.window(date(2026, 9, 1), "manana", 1.0).has_info.any()
    # Si falta el pronóstico de una sola hora de la franja, la estación queda sin información.
    rows = hourly("A", WEDNESDAY, {hour: ZERO for hour in MANANA if hour != 9})
    partial = ForecastInfo.from_predictions(forecast_table(rows), network, reglas, (WEDNESDAY,), np.ones(4, dtype=bool))
    assert not partial.window(WEDNESDAY, "manana", 1.0).has_info[0]
    with pytest.raises(SimulacionError, match="No hay pronóstico preparado"):
        forecast.window(date(2026, 9, 9), "manana", 1.0)


def test_la_franja_de_la_tarde_usa_sus_propias_horas():
    network, reglas, _ = case()
    rows = hourly("A", WEDNESDAY, {hour: ZERO for hour in TARDE} | {16: (0, 0, 0, 3, 3, 3), 13: (0, 50, 50, 0, 0, 0)})
    forecast = ForecastInfo.from_predictions(forecast_table(rows), network, reglas, (WEDNESDAY,), np.ones(4, dtype=bool))
    tarde = forecast.window(WEDNESDAY, "tarde", 0.0)
    assert tarde.has_info[0] and tarde.center_rise[0] == 3.0 and tarde.center_dip[0] == 0.0   # la hora 13 no cuenta
    assert not forecast.window(WEDNESDAY, "manana", 0.0).has_info[0]


def test_que_estaciones_tienen_historia_y_la_historia_no_puede_traer_filas_del_tramo():
    network = make_network({"A": 20, "B": 10, "C": 10})
    rows = [("A", pd.Timestamp("2026-08-27 08:00", tz=TZ)), ("C", pd.Timestamp("2026-08-31 23:00", tz=TZ)),
            ("Z", pd.Timestamp("2026-08-27 08:00", tz=TZ))]   # Z no está en la red
    history = pd.DataFrame(rows, columns=["station_id", "ts_hour"])
    assert stations_with_history(history, network, before=START).tolist() == [True, False, True]
    leak = pd.concat([history, pd.DataFrame({"station_id": ["B"], "ts_hour": [START]})], ignore_index=True)
    with pytest.raises(SimulacionError, match="filas del tramo simulado"):
        stations_with_history(leak, network, before=START)


# --------------------------------------------------------------------------- #
# c) Franja sin solución
# --------------------------------------------------------------------------- #

def test_sin_solucion_una_estacion_con_tipo_va_a_la_ocupacion_de_respaldo_de_su_tipo():
    cfg = make_cfg()
    cfg["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"] = {"manana": 0.45, "tarde": 0.1}
    cfg["estrategia_por_tipo"]["residencial"]["ocupacion_respaldo"] = {"manana": 0.7, "tarde": 0.1}
    reglas = Reglas.from_operacion(cfg)
    manana = reglas.franjas[0]
    for kind, expected in (("oficinas", 5), ("residencial", 7), ("mixta", 3)):   # 0.45 x 10 = 4.5 sube a 5
        network, _, forecast = case(types={"B": kind})
        policy = ForecastPolicy("historical", network, reglas, forecast, 1.0)
        for current in (0, 3, 9):
            targets = policy.targets(context(WEDNESDAY, manana, [15, current, 5, 5]))
            assert (targets.low[1], targets.high[1]) == (8.0, 1.0) and not targets.feasible[1]
            assert (targets.low_int[1], targets.high_int[1]) == (expected, expected)
            assert targets.desired[1] == expected


def test_sin_solucion_una_estacion_sin_tipo_no_se_toca():
    network, _, forecast = case()   # ninguna estación tiene tipo
    reglas = make_reglas(distancia_maxima_km=5.0)
    policy = ForecastPolicy("historical", network, reglas, forecast, 1.0)
    manana = reglas.franjas[0]
    for current in (0, 3, 9):
        targets = policy.targets(context(WEDNESDAY, manana, [15, current, 5, 5]))
        assert not targets.feasible[1]
        assert (targets.low_int[1], targets.high_int[1], targets.desired[1]) == (current, current, current)
    # B está vacía (0) o llena (10) y sin solución: ni recibe ni entrega, aunque A necesite o le sobren bicis.
    for inventory in ([10, 10, 5, 5], [20, 0, 5, 5]):
        moves = policy.decide(context(WEDNESDAY, manana, inventory))
        assert all(1 not in (move.origin, move.destination) for move in moves)
    # La misma estación CON tipo sí se mueve hacia su respaldo (3): entrega 7 de sus 10.
    typed, _, _ = case(types={"B": "mixta"})
    moves = ForecastPolicy("historical", typed, reglas, forecast, 1.0).decide(
        context(WEDNESDAY, manana, [10, 10, 5, 5])
    )
    assert sum(move.bikes for move in moves if move.origin == 1) > 0


def test_el_respaldo_se_recorta_para_no_dejar_la_estacion_bajo_un_umbral():
    cfg = make_cfg()
    cfg["estrategia_por_tipo"]["oficinas"]["ocupacion_respaldo"] = {"manana": 0.0, "tarde": 1.0}
    cfg["estrategia_por_tipo"]["mixta"]["ocupacion_respaldo"] = {"manana": 0.5, "tarde": 0.5}
    reglas = Reglas.from_operacion(cfg)   # bicis_minimas 2, lugares_libres_minimos 2
    network = make_network(
        {"A": 10, "B": 10, "C": 10, "D": 3}, types={"A": "oficinas", "B": "mixta", "D": "oficinas"}
    )
    # A: 0 % daría 0, sube a 2; 100 % daría 10, baja a 8. B: 5 en las dos. C no tiene tipo: -1.
    # D (3 lugares): el rango [2, 1] no existe; manda bicis_minimas (2), sin pasar de la capacidad.
    assert respaldo_inventory(network, reglas, "manana").tolist() == [2, 5, -1, 2]
    assert respaldo_inventory(network, reglas, "tarde").tolist() == [8, 5, -1, 2]
    # Con los umbrales en 1 el recorte cambia con ellos: no hay números escritos en el código.
    loose = Reglas.from_operacion({**cfg, "umbral_riesgo": {"bicis_minimas": 1, "lugares_libres_minimos": 3}})
    assert respaldo_inventory(network, loose, "manana").tolist() == [1, 5, -1, 1]
    assert respaldo_inventory(network, loose, "tarde").tolist() == [7, 5, -1, 1]


def test_el_rango_aceptado_a_mano_en_los_cuatro_casos():
    inventory, capacity = fixed([4, 4, 4, 4]), fixed([10, 10, 10, 10])
    low, high = np.array([6.0, 8.0, 8.0, 0.0]), np.array([7.5, 1.0, 1.0, 0.0])
    has_info = np.array([True, True, True, False])
    respaldo = fixed([3, 3, -1, 3])
    low_int, high_int, feasible = accepted_range(inventory, low, high, has_info, respaldo, capacity)
    # con solución; sin solución y con tipo; sin solución y sin tipo; sin información
    assert low_int.tolist() == [6, 3, 4, 4] and high_int.tolist() == [7, 3, 4, 4]
    assert feasible.tolist() == [True, False, False, True]


# --------------------------------------------------------------------------- #
# d) Reparto del presupuesto
# --------------------------------------------------------------------------- #

def test_solo_se_buscan_contrapartes_dentro_de_la_distancia_maxima():
    # A, B, C, D en línea, a 0.556 km cada paso. A necesita 3 (tiene 2, quiere 5). D, a 1.67 km, tiene 5 de
    # sobra. B (8, rango 4 a 8) y C (10) están dentro de su rango: no prestan ni reciben.
    network = make_network({"A": 20, "B": 20, "C": 20, "D": 20})
    inventory = fixed([2, 8, 10, 15])
    low, high = fixed([5, 4, 10, 10]), fixed([5, 8, 10, 10])
    priority = fixed([2, 2, 2, 2])
    # Con 1.2 km D no alcanza a A, y B, que está dentro de su rango, no presta: no se mueve nada.
    near = network.neighbors_within(1.2)
    assert match_moves(inventory, low, high, priority, near, budget=100) == []
    # Con 2 km D sí alcanza a A.
    far = network.neighbors_within(2.0)
    assert match_moves(inventory, low, high, priority, far, budget=100) == [Move(3, 0, 3)]
    # Con 0.5 km nadie alcanza a nadie.
    assert match_moves(inventory, low, high, priority, network.neighbors_within(0.5), budget=100) == []


def test_se_atiende_por_prioridad_despues_por_la_diferencia_mas_pequena_y_despues_por_id():
    # B, en medio, tiene bicis de sobra. A y C necesitan bicis y solo B les queda cerca. Todas quieren 6.
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(0.6)
    low = high = fixed([6, 6, 6])
    same = fixed([2, 2, 2])
    # Prioridad: C (prioridad 3) va antes que A (prioridad 1) aunque a A le falte menos.
    # C toma 4 de las 6 que le sobran a B; después se atiende B (prioridad 2), que da las 2 restantes a A.
    assert match_moves(fixed([4, 12, 2]), low, high, fixed([1, 2, 3]), near, 100) == [Move(1, 2, 4), Move(1, 0, 2)]
    # Misma prioridad: primero la diferencia MÁS PEQUEÑA. A B le sobran 3 (diferencia 3).
    # A le falta 1 y a C le faltan 5: va primero A; después B (3) reparte las 2 que le quedan a C.
    assert match_moves(fixed([5, 9, 1]), low, high, same, near, 100) == [Move(1, 0, 1), Move(1, 2, 2)]
    # Al revés: a C le falta 1 y a A le faltan 5.
    assert match_moves(fixed([1, 9, 5]), low, high, same, near, 100) == [Move(1, 2, 1), Move(1, 0, 2)]
    # Misma prioridad y misma diferencia (a A y a C les faltan 4; a B le sobran 6): primero el id menor.
    assert match_moves(fixed([2, 12, 2]), low, high, same, near, 1) == [Move(1, 0, 1)]
    # Con poco presupuesto se nota el orden: una sola bici va a la estación a la que le falta menos.
    assert match_moves(fixed([1, 12, 5]), low, high, same, near, 1) == [Move(1, 2, 1)]
    assert match_moves(fixed([5, 12, 1]), low, high, same, near, 1) == [Move(1, 0, 1)]


def test_con_poco_presupuesto_la_diferencia_mas_pequena_atiende_a_mas_estaciones():
    # Cinco estaciones en línea; la del centro (C) tiene 12 bicis de sobra. A dos les falta 1 y a dos les faltan 5.
    network = make_network({"A": 20, "B": 20, "C": 30, "D": 20, "E": 20})
    near = network.neighbors_within(1.2)   # C alcanza a todas
    inventory = fixed([1, 5, 18, 5, 1])
    low, high = fixed([6, 6, 6, 6, 6]), fixed([6, 6, 6, 6, 6])
    moves = match_moves(inventory, low, high, fixed([2] * 5), near, budget=2)
    # Las 2 bicis completan a B y a D (les faltaba 1), no se quedan en A (que necesitaba 5).
    assert moves == [Move(2, 1, 1), Move(2, 3, 1)]


def test_primero_la_mas_cercana_entre_las_que_tienen_la_necesidad_contraria():
    # A necesita 6. B (cerca) tiene 1 de sobra y C (más lejos) tiene 5. B va primero por tener la diferencia
    # más pequeña y da su bici a A; después C (5) le da el resto.
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    inventory, low, high = fixed([0, 7, 11]), fixed([6, 6, 6]), fixed([6, 6, 6])
    assert match_moves(inventory, low, high, fixed([2, 2, 2]), near, budget=100) == [Move(1, 0, 1), Move(2, 0, 5)]


def test_una_estacion_dentro_de_su_rango_no_presta_ni_recibe_y_nadie_pasa_de_su_borde():
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    priority = fixed([2, 2, 2])
    # A necesita 6. B (8 bicis, rango 5 a 9) y C (7, rango 7 a 10) están dentro de su rango: no prestan.
    assert match_moves(fixed([0, 8, 7]), fixed([6, 5, 7]), fixed([9, 9, 10]), priority, near, budget=100) == []
    # A le sobran 4 (10 bicis, rango 2 a 6). B (5, rango 3 a 7) y C (9, rango 4 a 10) no reciben.
    assert match_moves(fixed([10, 5, 9]), fixed([2, 3, 4]), fixed([6, 7, 10]), priority, near, budget=100) == []
    # A necesita 6 y a B le sobran 3 (12 bicis, rango 5 a 9): B baja solo hasta su borde (9), no hasta 5.
    moves = match_moves(fixed([0, 12, 7]), fixed([6, 5, 7]), fixed([9, 9, 10]), priority, near, budget=100)
    assert moves == [Move(1, 0, 3)]
    # A le sobran 4 y a C le falta 1 (3 bicis, rango 4 a 10): C sube solo hasta su borde (4).
    moves = match_moves(fixed([10, 5, 3]), fixed([2, 3, 4]), fixed([6, 7, 10]), priority, near, budget=100)
    assert moves == [Move(0, 2, 1)]


def test_se_detiene_cuando_se_acaba_el_presupuesto():
    network = make_network({"A": 20, "B": 20, "C": 20})
    near = network.neighbors_within(1.2)
    inventory, low, high = fixed([0, 12, 12]), fixed([8, 6, 6]), fixed([8, 6, 6])
    priority = fixed([2, 2, 2])
    # B y C (les sobran 6) van antes que A (le faltan 8); B le da 6 y C las 2 que faltan.
    assert match_moves(inventory, low, high, priority, near, budget=100) == [Move(1, 0, 6), Move(2, 0, 2)]
    assert match_moves(inventory, low, high, priority, near, budget=7) == [Move(1, 0, 6), Move(2, 0, 1)]
    assert match_moves(inventory, low, high, priority, near, budget=4) == [Move(1, 0, 4)]
    assert match_moves(inventory, low, high, priority, near, budget=0) == []


def test_con_presupuesto_cero_las_politicas_no_mueven_nada():
    network, _, forecast = case(types={"B": "mixta"})
    reglas = make_reglas(distancia_maxima_km=5.0)
    policy = ForecastPolicy("historical", network, reglas, forecast, 0.0)   # A acepta de 9 a 18; B, solo 3
    manana = reglas.franjas[0]
    assert policy.decide(context(WEDNESDAY, manana, [3, 9, 5, 5], budget=0)) == []
    assert policy.decide(context(WEDNESDAY, manana, [3, 9, 5, 5], budget=2)) == [Move(1, 0, 2)]
    assert NoIntervention().decide(context(WEDNESDAY, manana, [3, 9, 5, 5], budget=50)) == []


def test_la_prioridad_sale_del_tipo_y_una_estacion_sin_tipo_va_al_final():
    cfg = make_cfg(distancia_maxima_km=0.6)
    cfg["estrategia_por_tipo"]["oficinas"]["prioridad"] = 3
    cfg["estrategia_por_tipo"]["recreativa"]["prioridad"] = 1
    reglas = Reglas.from_operacion(cfg)
    network = make_network({"A": 20, "B": 20, "C": 20}, types={"A": "recreativa", "C": "oficinas"})
    rows = []
    for station, flow in (("A", (3, 3, 3, 0, 0, 0)), ("B", (0, 0, 0, 3, 3, 3)), ("C", (3, 3, 3, 0, 0, 0))):
        rows += hourly(station, WEDNESDAY, {hour: ZERO for hour in MANANA} | {5: flow})   # A y C pierden 3; B gana 3
    forecast = ForecastInfo.from_predictions(forecast_table(rows), network, reglas, (WEDNESDAY,), np.ones(3, dtype=bool))
    policy = ForecastPolicy("historical", network, reglas, forecast, 0.0)
    assert UNTYPED_PRIORITY == 0
    assert policy.priority.tolist() == [1, UNTYPED_PRIORITY, 3]   # B no tiene tipo
    # A y C necesitan 5 (lo = 2 + 3) y tienen 4. B (rango 2 a 15) tiene 16: le sobra 1. Va primero C (prioridad 3)
    # y se lleva esa bici. A (prioridad 1) se queda sin ella: B ya está dentro de su rango y no presta.
    assert policy.decide(context(WEDNESDAY, reglas.franjas[0], [4, 16, 4])) == [Move(1, 2, 1)]
    # Si a B le sobran 2, la segunda sí llega a A, después de C.
    assert policy.decide(context(WEDNESDAY, reglas.franjas[0], [4, 17, 4])) == [Move(1, 2, 1), Move(1, 0, 1)]
    # Con una sola bici de presupuesto se la lleva C, la de más prioridad.
    assert policy.decide(context(WEDNESDAY, reglas.franjas[0], [4, 17, 4], budget=1)) == [Move(1, 2, 1)]


# --------------------------------------------------------------------------- #
# e) Sin información del futuro
# --------------------------------------------------------------------------- #

def future_case(seed: int = 11):
    """Seis estaciones, viajes al azar de tres días, y las dos políticas con pronósticos al azar distintos."""
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
    network = build_network(stations)
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
    policies = {}
    for name, model in (("historical", "baseline"), ("model", "lgbm")):
        predictions = []
        for station in names:
            for day in days:
                for hour in range(24):
                    dep, arr = rng.random() * 3, rng.random() * 3
                    predictions += hourly(station, day, {hour: (dep * 0.5, dep, dep * 1.5, arr * 0.5, arr, arr * 1.5)})
        forecast = ForecastInfo.from_predictions(
            forecast_table(predictions, model), network, reglas, days, np.ones(6, dtype=bool)
        )
        policies[name] = ForecastPolicy(name, network, reglas, forecast, 0.5)
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
    assert list(inspect.signature(ForecastPolicy.__init__).parameters) == [
        "self", "name", "network", "reglas", "forecast", "band", "neighbors",
    ]
    for policy in (ForecastPolicy, NoIntervention):
        assert list(inspect.signature(policy.decide).parameters) == ["self", "context"]
    # El pronóstico se arma sin viajes: con predictions y con qué estaciones tienen historia.
    assert "trips" not in inspect.signature(ForecastInfo.from_predictions).parameters
    assert "trips" not in inspect.signature(stations_with_history).parameters


def test_la_misma_intervencion_da_siempre_la_misma_decision():
    network, reglas, trips, days, policies = future_case()
    digests = set()
    for policy in policies.values():
        first, _ = decisions(network, reglas, trips, days, policy)
        second, _ = decisions(network, reglas, trips, days, policy)
        assert first.digest() == second.digest()
        digests.add(first.digest())
        # Y los traslados que salen de las políticas pasan los cuatro límites del motor (si no, habría fallado).
        assert int(first.budget_used.sum()) > 0
        assert (first.budget_used <= first.budget_available).all()
        assert first.final_total == first.initial_total   # el total de bicis no cambia
    assert len(digests) == 2   # con pronósticos distintos, las dos políticas no deciden lo mismo
