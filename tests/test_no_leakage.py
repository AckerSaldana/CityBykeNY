"""Pruebas contra la fuga de información en las variables (features).

Regla (decisión 5 de CLAUDE.md): las variables de una fila cuyo día local de
Nueva York es D solo pueden depender de filas de station_hour de días
anteriores a D. La explicación completa está en src/features/leakage.py.

Qué revisan:

a) Todo constructor registrado en src/features pasa la revisión de fuga con
   los datos de muestra, en varios días de corte. No hay que anotarlo aquí:
   basta con registrarlo con @register.
b) La revisión sí muerde: constructores tramposos, escritos aquí a propósito,
   salen marcados. Unos leen conteos del mismo día o del futuro; otros no leen
   ningún conteo y solo miran qué filas existen (cuántas hay hoy, si hay una
   en la hora siguiente). Se prueban con la muestra completa y con una versión
   sin las horas vacías.
c) El constructor de referencia calcula bien sus valores en una tabla hecha a
   mano, también junto a los días de 25 y de 23 horas.
d) Todo constructor devuelve una fila por cada fila recibida, sin llaves
   repetidas.
e) Un constructor que pierde o duplica filas se rechaza con un mensaje claro.

Si falla (a) con un constructor tuyo, el mensaje dice qué variables usan datos
del mismo día o del futuro. Córrela sola con:  pytest tests/test_no_leakage.py

Ojo: un rezago por posición (`shift(24)`) también sale marcado, porque depende
de cuántas filas hay. Los rezagos se buscan por fecha y hora local, como en
src/features/lags.py.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from src.contracts import read_contract
from src.features import registry
from src.features.lags import FEATURE_COLUMNS, lags_dias_previos
from src.features.leakage import (
    VALUE_COLUMNS,
    BuilderOutputError,
    check_builder_output,
    check_no_leakage,
    drop_rows_from_date,
    local_day,
    perturb_from_date,
)
from src.features.registry import BUILDERS, KEY_COLUMNS, load_builders
from src.paths import DATA_SAMPLE, TZ

# Importa todos los archivos de src/features: así quedan registrados todos los constructores.
BUILDER_NAMES = sorted(load_builders())

# Días de corte sobre la muestra (13 de octubre a 9 de noviembre de 2025):
# primer día, días intermedios, el día de 25 horas (2 de noviembre), el
# siguiente y el último.
CUTOFFS = [
    "2025-10-13",
    "2025-10-14",
    "2025-10-20",
    "2025-10-27",
    "2025-11-01",
    "2025-11-02",
    "2025-11-03",
    "2025-11-09",
]


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    """station_hour de data/sample (sintético). No se modifica: cada prueba trabaja sobre copias."""
    path = DATA_SAMPLE / "station_hour.parquet"
    if not path.exists():
        pytest.fail(f"Falta {path}. Para regenerarlo: python scripts/make_sample_data.py", pytrace=False)
    return read_contract("station_hour", path)


@pytest.fixture(scope="module")
def sparse(sample) -> pd.DataFrame:
    """La muestra sin las horas en que no hubo viajes.

    Así sale station_hour de un GROUP BY sobre los viajes si no se rellenan las
    horas vacías (decisión abierta en CLAUDE.md). La muestra sí las trae.
    """
    table = sample[(sample["departures"] + sample["arrivals"]) > 0].reset_index(drop=True)
    assert 0 < len(table) < len(sample)
    return table


def _grid(start: str, end: str, stations: tuple[str, ...] = ("A",)) -> pd.DataFrame:
    """Tabla hecha a mano: todas las horas locales desde `start` hasta antes de `end`.

    departures cuenta las horas desde el inicio (0, 1, 2, ...), así cada valor
    dice de qué hora salió. La segunda estación empieza en 100000 para notar si
    se mezclan. arrivals es otro número distinto: el triple más uno.
    """
    hours = pd.date_range(start, end, freq="h", tz=TZ, inclusive="left")
    parts = []
    for position, station in enumerate(stations):
        departures = 100_000 * position + np.arange(len(hours))
        arrivals = 3 * departures + 1
        parts.append(
            pd.DataFrame(
                {
                    "station_id": station,
                    "ts_hour": hours,
                    "departures": departures,
                    "arrivals": arrivals,
                    "net_flow": arrivals - departures,
                    "departures_member": departures,
                    "departures_casual": 0,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _at(table: pd.DataFrame, moment: str, column: str, station: str = "A"):
    """Valor de `column` en la fila de esa estación y ese instante.

    El instante se escribe con su desfase, por ejemplo "2025-11-02 01:00-04:00"
    (horario de verano) o "2025-11-02 01:00-05:00" (horario de invierno).
    """
    rows = table[(table["station_id"] == station) & (table["ts_hour"] == pd.Timestamp(moment))]
    assert len(rows) == 1, f"se esperaba una fila para {station} {moment} y hay {len(rows)}"
    return rows[column].iloc[0]


def _fail_if_leaking(name: str, table: pd.DataFrame, cutoff: str) -> None:
    leaking = check_no_leakage(BUILDERS[name], table, cutoff)
    if leaking:
        pytest.fail(
            f"El constructor '{name}' tiene FUGA de información en: {leaking}.\n"
            f"Se hicieron dos experimentos desde el {cutoff}: alterar los conteos y quitar filas. En alguno "
            "de los dos, esas variables cambiaron en filas de ese día o de días anteriores.\n"
            "Eso quiere decir que usan datos del mismo día o del futuro, o que dependen de cuántas filas hay "
            "(por ejemplo, un rezago por posición). Solo se permiten datos de días anteriores (decisión 5 de "
            "CLAUDE.md). La explicación completa está en src/features/leakage.py.",
            pytrace=False,
        )


# --------------------------------------------------------------------------- #
# a) Ningún constructor registrado tiene fuga
# --------------------------------------------------------------------------- #

def test_el_constructor_de_referencia_esta_registrado():
    assert "lags_dias_previos" in BUILDER_NAMES
    assert BUILDERS["lags_dias_previos"] is lags_dias_previos


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("name", BUILDER_NAMES)
def test_constructor_registrado_no_tiene_fuga(name, cutoff, sample):
    _fail_if_leaking(name, sample, cutoff)


@pytest.mark.parametrize("cutoff", ["2026-03-08", "2026-03-09"])
@pytest.mark.parametrize("name", BUILDER_NAMES)
def test_constructor_registrado_no_tiene_fuga_junto_al_dia_de_23_horas(name, cutoff):
    """La muestra no trae el cambio de horario de marzo: se revisa con una tabla hecha a mano."""
    _fail_if_leaking(name, _grid("2026-02-27", "2026-03-12", stations=("A", "B")), cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_el_constructor_de_referencia_sigue_limpio_en_una_tabla_sin_horas_vacias(cutoff, sparse):
    """La revisión no da falsas alarmas cuando faltan filas: los rezagos por fecha no dependen de ellas."""
    assert check_no_leakage(lags_dias_previos, sparse, cutoff) == []


# --------------------------------------------------------------------------- #
# b) La revisión sí detecta fugas
# --------------------------------------------------------------------------- #
# Los constructores de esta sección están MAL a propósito. No se registran.

def _leaky_one_hour_lag(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: salidas de la hora anterior. A las 09:00 usa lo que pasó a las 08:00 de ese mismo día."""
    ordered = station_hour.sort_values(KEY_COLUMNS)
    out = ordered[KEY_COLUMNS].copy()
    out["dep_lag_1h"] = ordered.groupby("station_id")["departures"].shift(1)
    return out


def _leaky_centered_mean(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: promedio móvil centrado. Junta las dos horas anteriores con las dos siguientes."""
    ordered = station_hour.sort_values(KEY_COLUMNS)
    out = ordered[KEY_COLUMNS].copy()
    out["dep_media_centrada"] = ordered.groupby("station_id")["departures"].transform(
        lambda series: series.rolling(5, center=True, min_periods=1).mean()
    )
    return out


def _leaky_station_mean(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: promedio de la estación sobre TODO el periodo, que incluye el futuro."""
    out = station_hour[KEY_COLUMNS].copy()
    out["dep_media_estacion"] = station_hour.groupby("station_id")["departures"].transform("mean")
    return out


LEAKY_BUILDERS = [
    pytest.param(_leaky_one_hour_lag, "dep_lag_1h", id="rezago de una hora"),
    pytest.param(_leaky_centered_mean, "dep_media_centrada", id="promedio móvil centrado"),
    pytest.param(_leaky_station_mean, "dep_media_estacion", id="promedio de todo el periodo"),
]


@pytest.mark.parametrize("cutoff", ["2025-10-14", "2025-11-02", "2025-11-03", "2025-11-09"])
@pytest.mark.parametrize("builder, leaking_column", LEAKY_BUILDERS)
def test_la_revision_detecta_un_constructor_con_fuga(builder, leaking_column, cutoff, sample):
    assert check_no_leakage(builder, sample, cutoff) == [leaking_column]


# --- Fugas que no leen ningún conteo: solo miran qué filas existen ---

def _leaky_rows_today(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: cuántas filas tiene la estación ese mismo día.

    Si la tabla no trae las horas sin viajes, eso dice cuántas horas con
    actividad tuvo HOY la estación, que todavía no se sabe al decidir.
    """
    out = station_hour[KEY_COLUMNS].copy()
    day = local_day(station_hour["ts_hour"])
    out["filas_hoy"] = station_hour.groupby([station_hour["station_id"], day])["ts_hour"].transform("count").to_numpy()
    return out


def _leaky_next_hour_exists(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: si la estación tiene fila en la hora siguiente. Mira el futuro sin leer ningún conteo."""
    out = station_hour[KEY_COLUMNS].copy()
    rows = pd.MultiIndex.from_arrays([station_hour["station_id"], station_hour["ts_hour"]])
    following = pd.MultiIndex.from_arrays([station_hour["station_id"], station_hour["ts_hour"] + pd.Timedelta(hours=1)])
    out["hay_fila_siguiente"] = following.isin(rows)
    return out


def _leaky_hours_to_next_row(station_hour: pd.DataFrame) -> pd.DataFrame:
    """FUGA: horas que faltan para la siguiente fila de la estación."""
    ordered = station_hour.sort_values(KEY_COLUMNS)
    out = ordered[KEY_COLUMNS].copy()
    following = ordered.groupby("station_id")["ts_hour"].shift(-1)
    out["horas_a_la_siguiente"] = (following - ordered["ts_hour"]).dt.total_seconds() / 3600
    return out


ROW_LEAKS = [
    pytest.param(_leaky_rows_today, "filas_hoy", id="filas del mismo día"),
    pytest.param(_leaky_next_hour_exists, "hay_fila_siguiente", id="existe la fila siguiente"),
    pytest.param(_leaky_hours_to_next_row, "horas_a_la_siguiente", id="horas hasta la siguiente fila"),
]


@pytest.mark.parametrize("cutoff", ["2025-10-14", "2025-11-02", "2025-11-03", "2025-11-09"])
@pytest.mark.parametrize("table_name", ["sample", "sparse"], ids=["tabla completa", "sin horas vacías"])
@pytest.mark.parametrize("builder, leaking_column", ROW_LEAKS)
def test_la_revision_detecta_fugas_por_las_filas_que_existen(builder, leaking_column, table_name, cutoff, request):
    table = request.getfixturevalue(table_name)
    assert check_no_leakage(builder, table, cutoff) == [leaking_column]


@pytest.mark.parametrize("builder, leaking_column", ROW_LEAKS)
def test_alterar_solo_los_conteos_no_basta_para_ver_esas_fugas(builder, leaking_column, sparse):
    """Por eso la revisión también quita filas: con los conteos alterados estas variables salen idénticas."""
    with_original = builder(sparse.copy())
    with_altered = builder(perturb_from_date(sparse, "2025-11-02"))
    pd.testing.assert_frame_equal(with_original, with_altered)
    assert with_original[leaking_column].nunique() > 1  # y en esta tabla sí llevan información


def test_quitar_filas_deja_intacto_lo_anterior_al_corte_y_quita_cerca_de_la_mitad_despues(sample):
    cutoff = pd.Timestamp("2025-11-02")
    thinned = drop_rows_from_date(sample, cutoff)

    def before(table: pd.DataFrame) -> pd.DataFrame:
        return table[local_day(table["ts_hour"]) < cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(before(thinned), before(sample))  # antes del corte no falta ni cambia nada
    from_cutoff = len(sample) - len(before(sample))
    kept = len(thinned) - len(before(thinned))
    assert 0.4 * from_cutoff < kept < 0.6 * from_cutoff
    # Las filas que quedan son filas de la tabla original, con sus mismos valores.
    assert len(thinned.merge(sample, how="inner", on=list(sample.columns))) == len(thinned)
    assert not thinned.duplicated(KEY_COLUMNS).any()
    # En cada estación quedan filas del propio día de corte: ahí es donde se nota una fuga del mismo día.
    on_cutoff = thinned[local_day(thinned["ts_hour"]) == cutoff]
    assert on_cutoff["station_id"].nunique() == sample["station_id"].nunique()
    assert thinned.equals(drop_rows_from_date(sample, "2025-11-02"))  # siempre las mismas filas


def test_los_nombres_de_las_variables_no_pueden_depender_de_los_datos(sample):
    def named_by_values(station_hour: pd.DataFrame) -> pd.DataFrame:
        return station_hour[KEY_COLUMNS].assign(**{f"max_{station_hour['departures'].max()}": 0.0})

    def named_by_rows(station_hour: pd.DataFrame) -> pd.DataFrame:
        return station_hour[KEY_COLUMNS].assign(**{f"filas_{len(station_hour)}": 0.0})

    for builder in (named_by_values, named_by_rows):
        with pytest.raises(BuilderOutputError, match="columnas distintas al cambiar los datos"):
            check_no_leakage(builder, sample, "2025-11-02")


def test_un_constructor_que_modifica_su_entrada_no_ensucia_la_tabla_de_quien_llama(sample):
    def dirty(station_hour: pd.DataFrame) -> pd.DataFrame:
        station_hour["departures"] = 0  # modifica la tabla que recibió
        return lags_dias_previos(station_hour)

    table = sample.copy()
    check_no_leakage(dirty, table, "2025-11-02")
    pd.testing.assert_frame_equal(table, sample)


def test_la_revision_senala_solo_las_variables_con_fuga(sample):
    def mixed(station_hour: pd.DataFrame) -> pd.DataFrame:
        out = lags_dias_previos(station_hour)
        out["arr_misma_hora"] = station_hour["arrivals"].to_numpy()  # FUGA: el dato de la propia hora
        return out

    assert check_no_leakage(mixed, sample, "2025-11-02") == ["arr_misma_hora"]


def test_un_vacio_que_pasa_a_tener_valor_tambien_es_fuga(sample):
    """La fuga no siempre cambia un número por otro: aquí cambia un vacío (NaN) por un valor."""

    def leaky_peak(station_hour: pd.DataFrame) -> pd.DataFrame:
        out = station_hour[KEY_COLUMNS].copy()
        # FUGA: salidas de la propia hora, solo cuando son 500 o más. En la muestra nunca lo son.
        out["dep_pico"] = station_hour["departures"].where(station_hour["departures"] >= 500).to_numpy()
        return out

    assert leaky_peak(sample)["dep_pico"].isna().all()
    assert check_no_leakage(leaky_peak, sample, "2025-11-02") == ["dep_pico"]


def test_el_orden_de_las_filas_de_salida_no_cambia_el_resultado(sample):
    """La comparación se hace por llave: un constructor puede devolver las filas en otro orden."""
    assert check_no_leakage(lambda table: lags_dias_previos(table).iloc[::-1], sample, "2025-11-02") == []
    assert check_no_leakage(lambda table: _leaky_one_hour_lag(table).iloc[::-1], sample, "2025-11-02") == ["dep_lag_1h"]


def test_la_alteracion_cambia_todos_los_conteos_desde_el_corte_y_nada_antes(sample):
    altered = perturb_from_date(sample, "2025-11-02")
    from_cutoff = (local_day(sample["ts_hour"]) >= pd.Timestamp("2025-11-02")).to_numpy()
    assert from_cutoff.any() and not from_cutoff.all()

    assert altered[KEY_COLUMNS].equals(sample[KEY_COLUMNS])
    assert altered[~from_cutoff].equals(sample[~from_cutoff])
    change = (altered.loc[from_cutoff, VALUE_COLUMNS] - sample.loc[from_cutoff, VALUE_COLUMNS]).abs()
    assert (change >= 1000).all().all()  # las cinco columnas, en todas las filas, y por mucho
    assert altered.equals(perturb_from_date(sample, "2025-11-02"))  # siempre la misma alteración


@pytest.mark.parametrize("cutoff", ["2025-10-12", "2025-11-10"])
def test_un_dia_de_corte_fuera_de_los_datos_se_rechaza(cutoff, sample):
    """Con un corte fuera del periodo no se alteraría o no se compararía nada: pasaría sin probar."""
    with pytest.raises(ValueError, match="fuera de los datos"):
        check_no_leakage(lags_dias_previos, sample, cutoff)


# --------------------------------------------------------------------------- #
# c) Valores del constructor de referencia, a mano
# --------------------------------------------------------------------------- #
# Desfases: -04:00 es horario de verano y -05:00 es horario de invierno.

def test_referencia_sin_historia_queda_vacio_y_el_promedio_usa_lo_que_hay():
    table = _grid("2025-10-01", "2025-10-12")
    features = lags_dias_previos(table)

    # Primer día: no hay nada antes.
    first_day = features[local_day(features["ts_hour"]) == pd.Timestamp("2025-10-01")]
    assert len(first_day) == 24 and first_day[FEATURE_COLUMNS].isna().all().all()

    # Segundo día: hay un día de historia. El promedio de 7 días es ese único valor.
    assert _at(table, "2025-10-01 10:00-04:00", "departures") == 10
    assert _at(table, "2025-10-01 10:00-04:00", "arrivals") == 31
    assert _at(features, "2025-10-02 10:00-04:00", "dep_lag_1d") == 10
    assert _at(features, "2025-10-02 10:00-04:00", "arr_lag_1d") == 31
    assert pd.isna(_at(features, "2025-10-02 10:00-04:00", "dep_lag_7d"))
    assert _at(features, "2025-10-02 10:00-04:00", "dep_media_7d") == 10

    # Día 9: ya hay 7 días completos antes. El día 1 queda fuera del promedio.
    assert _at(features, "2025-10-09 10:00-04:00", "dep_lag_1d") == 7 * 24 + 10
    assert _at(features, "2025-10-09 10:00-04:00", "dep_lag_7d") == 1 * 24 + 10
    week = [day * 24 + 10 for day in range(1, 8)]  # las 10:00 de los días 2 al 8
    assert _at(features, "2025-10-09 10:00-04:00", "dep_media_7d") == pytest.approx(sum(week) / 7)
    assert _at(features, "2025-10-09 10:00-04:00", "arr_media_7d") == pytest.approx(sum(3 * v + 1 for v in week) / 7)


def test_referencia_en_el_dia_de_25_horas_y_el_dia_siguiente():
    table = _grid("2025-10-24", "2025-11-12")
    features = lags_dias_previos(table)
    assert len(features) == len(table)
    assert (local_day(table["ts_hour"]) == pd.Timestamp("2025-11-02")).sum() == 25

    def departures(moment: str) -> int:
        return _at(table, moment, "departures")

    # El 2 de noviembre a las 05:00 ya es horario de invierno. "Un día antes" son
    # las 05:00 del 1 de noviembre, que quedan 25 horas atrás (no 24).
    assert _at(features, "2025-11-02 05:00-05:00", "dep_lag_1d") == departures("2025-11-01 05:00-04:00")
    assert departures("2025-11-02 05:00-05:00") - departures("2025-11-01 05:00-04:00") == 25
    assert _at(features, "2025-11-02 05:00-05:00", "arr_lag_1d") == _at(table, "2025-11-01 05:00-04:00", "arrivals")
    assert _at(features, "2025-11-02 05:00-05:00", "dep_lag_7d") == departures("2025-10-26 05:00-04:00")

    # La 01:00 aparece dos veces ese día. Las dos filas reciben el mismo rezago.
    for repeated in ("2025-11-02 01:00-04:00", "2025-11-02 01:00-05:00"):
        assert _at(features, repeated, "dep_lag_1d") == departures("2025-11-01 01:00-04:00")

    # Día siguiente, a la 01:00: el origen son DOS filas. Se usa su promedio y sale una sola fila.
    two_rows = (departures("2025-11-02 01:00-04:00") + departures("2025-11-02 01:00-05:00")) / 2
    assert _at(features, "2025-11-03 01:00-05:00", "dep_lag_1d") == two_rows
    days_before = [departures(f"2025-10-{day} 01:00-04:00") for day in (27, 28, 29, 30, 31)]
    days_before += [departures("2025-11-01 01:00-04:00"), two_rows]
    assert _at(features, "2025-11-03 01:00-05:00", "dep_media_7d") == pytest.approx(sum(days_before) / 7)

    # Día siguiente, a otra hora: el rezago de un día son las 24 horas de siempre.
    assert _at(features, "2025-11-03 05:00-05:00", "dep_lag_1d") == departures("2025-11-02 05:00-05:00")

    # Siete días después del cambio: la misma hora del reloj, 169 horas atrás.
    assert _at(features, "2025-11-04 09:00-05:00", "dep_lag_7d") == departures("2025-10-28 09:00-04:00")
    assert departures("2025-11-04 09:00-05:00") - departures("2025-10-28 09:00-04:00") == 7 * 24 + 1
    # Una semana después, el rezago de 7 días de la 01:00 también es el promedio de las dos filas.
    assert _at(features, "2025-11-09 01:00-05:00", "dep_lag_7d") == two_rows


def test_referencia_en_el_dia_de_23_horas_y_el_dia_siguiente():
    table = _grid("2026-02-27", "2026-03-17")
    features = lags_dias_previos(table)
    assert len(features) == len(table)
    assert (local_day(table["ts_hour"]) == pd.Timestamp("2026-03-08")).sum() == 23  # no existe la 02:00

    def departures(moment: str) -> int:
        return _at(table, moment, "departures")

    # El 8 de marzo a las 03:00 ya es horario de verano. "Un día antes" son las
    # 03:00 del 7 de marzo, que quedan 23 horas atrás (no 24).
    assert _at(features, "2026-03-08 03:00-04:00", "dep_lag_1d") == departures("2026-03-07 03:00-05:00")
    assert departures("2026-03-08 03:00-04:00") - departures("2026-03-07 03:00-05:00") == 23

    # Día siguiente, a las 02:00: esa hora no existió el día 8. El rezago de un
    # día queda vacío y el promedio usa los 6 días que sí la tienen.
    assert pd.isna(_at(features, "2026-03-09 02:00-04:00", "dep_lag_1d"))
    assert pd.isna(_at(features, "2026-03-09 02:00-04:00", "arr_lag_1d"))
    assert _at(features, "2026-03-09 02:00-04:00", "dep_lag_7d") == departures("2026-03-02 02:00-05:00")
    six_days = [departures(f"2026-03-0{day} 02:00-05:00") for day in (2, 3, 4, 5, 6, 7)]
    assert _at(features, "2026-03-09 02:00-04:00", "dep_media_7d") == pytest.approx(sum(six_days) / 6)

    # Día siguiente, a otra hora: el rezago de un día son las 24 horas de siempre.
    assert _at(features, "2026-03-09 05:00-04:00", "dep_lag_1d") == departures("2026-03-08 05:00-04:00")

    # Una semana después, a las 02:00: el rezago de 7 días cae en la hora que no existió.
    assert pd.isna(_at(features, "2026-03-15 02:00-04:00", "dep_lag_7d"))
    assert _at(features, "2026-03-15 02:00-04:00", "dep_lag_1d") == departures("2026-03-14 02:00-04:00")


def test_referencia_no_mezcla_estaciones_ni_depende_del_orden_de_las_filas():
    table = _grid("2025-10-24", "2025-11-05", stations=("A", "B"))
    features = lags_dias_previos(table)
    for station in ("A", "B"):
        day_before = _at(table, "2025-11-02 09:00-05:00", "departures", station=station)
        assert _at(features, "2025-11-03 09:00-05:00", "dep_lag_1d", station=station) == day_before
    assert _at(features, "2025-11-03 09:00-05:00", "dep_lag_1d", station="A") < 100_000
    assert _at(features, "2025-11-03 09:00-05:00", "dep_lag_1d", station="B") >= 100_000

    shuffled = table.sample(frac=1.0, random_state=0)
    from_shuffled = lags_dias_previos(shuffled).sort_values(KEY_COLUMNS).reset_index(drop=True)
    pd.testing.assert_frame_equal(from_shuffled, features.sort_values(KEY_COLUMNS).reset_index(drop=True))


def test_referencia_rechaza_fechas_sin_zona():
    table = _grid("2025-10-24", "2025-10-26")
    table["ts_hour"] = table["ts_hour"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="zona horaria"):
        lags_dias_previos(table)


# --------------------------------------------------------------------------- #
# d) Una fila por cada fila recibida, con llaves únicas
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", BUILDER_NAMES)
def test_constructor_devuelve_una_fila_por_cada_fila_recibida(name, sample):
    features = BUILDERS[name](sample.copy())
    check_builder_output(features, sample)  # lanza un error claro si la forma no es la acordada
    assert len(features) == len(sample)
    assert not features.duplicated(KEY_COLUMNS).any()
    assert set(KEY_COLUMNS) < set(features.columns)
    # En particular, el día de 25 horas conserva sus 25 filas por estación.
    long_day = features[local_day(features["ts_hour"]) == pd.Timestamp("2025-11-02")]
    assert (long_day.groupby("station_id").size() == 25).all()


def test_referencia_devuelve_solo_las_llaves_y_sus_seis_variables(sample):
    features = lags_dias_previos(sample.copy())
    assert list(features.columns) == KEY_COLUMNS + FEATURE_COLUMNS
    assert FEATURE_COLUMNS == ["dep_lag_1d", "dep_lag_7d", "arr_lag_1d", "arr_lag_7d", "dep_media_7d", "arr_media_7d"]
    assert (features[FEATURE_COLUMNS].dtypes == "float64").all()
    assert features.index.equals(sample.index)


# --------------------------------------------------------------------------- #
# e) Se rechaza un constructor que pierde, duplica o cambia filas
# --------------------------------------------------------------------------- #

def _join_without_grouping(station_hour: pd.DataFrame) -> pd.DataFrame:
    """ERROR típico: unir por día y hora local sin agrupar antes.

    La 01:00 repetida del día de 25 horas hace que la 01:00 del día siguiente
    encuentre dos filas de origen y salga duplicada.
    """
    local = station_hour["ts_hour"].dt.tz_localize(None)
    rows = station_hour[KEY_COLUMNS].assign(day=local.dt.normalize(), hour=local.dt.hour)
    source = rows.assign(day=rows["day"] + pd.Timedelta(days=1), dep_lag_1d=station_hour["departures"])
    source = source[["station_id", "day", "hour", "dep_lag_1d"]]
    return rows.merge(source, on=["station_id", "day", "hour"], how="left").drop(columns=["day", "hour"])


# Cada renglón: qué hace mal, constructor, texto que debe salir en el error.
BAD_SHAPES = [
    ("pierde filas (dropna)", lambda t: lags_dias_previos(t).dropna(), "filas recibidas no aparecen en la salida"),
    ("duplica filas (unión sin agrupar)", _join_without_grouping, "filas repiten una llave"),
    ("duplica todas las filas", lambda t: pd.concat([lags_dias_previos(t)] * 2), "filas repiten una llave"),
    ("sin columna llave", lambda t: lags_dias_previos(t).drop(columns="ts_hour"), "faltan las columnas llave"),
    ("cambia las llaves", lambda t: lags_dias_previos(t).assign(ts_hour=t["ts_hour"] + pd.Timedelta(hours=1)),
     "llaves de la salida no venían en la entrada"),
    ("quita la zona horaria", lambda t: lags_dias_previos(t).assign(ts_hour=t["ts_hour"].dt.tz_localize(None)),
     "zona horaria"),
    ("no trae variables", lambda t: t[KEY_COLUMNS], "ninguna variable"),
    ("no devuelve una tabla", lambda t: t["departures"], "debe devolver un DataFrame"),
]


@pytest.mark.parametrize("builder, expected_text", [pytest.param(*rest, id=label) for label, *rest in BAD_SHAPES])
def test_se_rechaza_un_constructor_con_la_forma_mal(builder, expected_text, sample):
    with pytest.raises(BuilderOutputError, match=expected_text):
        check_no_leakage(builder, sample, "2025-11-02")


def test_el_mensaje_de_filas_perdidas_dice_cuantas(sample):
    with pytest.raises(BuilderOutputError) as error:
        check_no_leakage(lambda t: lags_dias_previos(t).iloc[5:], sample, "2025-11-02")
    message = str(error.value)
    assert f"recibió {len(sample)} filas y devolvió {len(sample) - 5}" in message
    assert "5 filas recibidas no aparecen en la salida" in message


# --------------------------------------------------------------------------- #
# El registro
# --------------------------------------------------------------------------- #

def test_load_builders_encuentra_un_archivo_nuevo_de_src_features(tmp_path, monkeypatch):
    """Basta crear el archivo y usar @register: nadie tiene que anotarlo en otro lado."""
    import src.features as package

    (tmp_path / "variable_de_prueba.py").write_text(
        "from src.features.registry import register\n\n\n"
        "@register('variable_de_prueba')\n"
        "def variable_de_prueba(station_hour):\n"
        "    return station_hour\n",
        encoding="utf-8",
    )
    # Para la prueba, src/features abarca además una carpeta temporal con ese archivo.
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    monkeypatch.setattr(registry, "BUILDERS", {})  # registro vacío solo para esta prueba
    try:
        assert "variable_de_prueba" in registry.load_builders()
    finally:
        sys.modules.pop("src.features.variable_de_prueba", None)
        if hasattr(package, "variable_de_prueba"):
            delattr(package, "variable_de_prueba")
    assert "variable_de_prueba" not in BUILDERS  # el registro real no se tocó


@pytest.mark.parametrize("with_init", [True, False], ids=["con __init__.py", "sin __init__.py"])
def test_load_builders_encuentra_un_archivo_dentro_de_una_subcarpeta(with_init, tmp_path, monkeypatch):
    """Una subcarpeta como src/features/externas/ también se recorre, tenga o no __init__.py."""
    import src.features as package

    folder = tmp_path / "subcarpeta_de_prueba"
    folder.mkdir()
    if with_init:
        (folder / "__init__.py").write_text("", encoding="utf-8")
    (folder / "variable_en_subcarpeta.py").write_text(
        "from src.features.registry import register\n\n\n"
        "@register('variable_en_subcarpeta')\n"
        "def variable_en_subcarpeta(station_hour):\n"
        "    return station_hour\n",
        encoding="utf-8",
    )
    # Copia que deja Jupyter al editar un archivo: no es código del proyecto y no se importa.
    checkpoints = tmp_path / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "variable-checkpoint.py").write_text("raise RuntimeError('esto no se importa')\n", encoding="utf-8")

    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    monkeypatch.setattr(registry, "BUILDERS", {})  # registro vacío solo para esta prueba
    try:
        assert "variable_en_subcarpeta" in registry.load_builders()
    finally:
        for module in [name for name in sys.modules if name.startswith("src.features.subcarpeta_de_prueba")]:
            del sys.modules[module]
        if hasattr(package, "subcarpeta_de_prueba"):
            delattr(package, "subcarpeta_de_prueba")
    assert "variable_en_subcarpeta" not in BUILDERS  # el registro real no se tocó


def test_register_anota_la_funcion_y_la_devuelve_igual(monkeypatch):
    monkeypatch.setattr(registry, "BUILDERS", {})  # registro vacío solo para esta prueba

    def builder(station_hour):
        return station_hour

    assert registry.register("de_prueba")(builder) is builder
    assert registry.BUILDERS == {"de_prueba": builder}
    # Registrar otra vez la misma función (volver a correr una celda) no es un error.
    registry.register("de_prueba")(builder)
    assert registry.BUILDERS == {"de_prueba": builder}


def test_register_rechaza_dos_constructores_distintos_con_el_mismo_nombre(monkeypatch):
    monkeypatch.setattr(registry, "BUILDERS", {})

    def first(station_hour):
        return station_hour

    def second(station_hour):
        return station_hour

    registry.register("repetido")(first)
    with pytest.raises(ValueError, match="Ya hay un constructor registrado como 'repetido'"):
        registry.register("repetido")(second)
    assert registry.BUILDERS == {"repetido": first}
