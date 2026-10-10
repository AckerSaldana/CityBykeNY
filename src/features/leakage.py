"""Revisión de fuga de información en las variables (features).

Qué es la fuga
--------------
El sistema decide cuántas bicis mover ANTES de que empiece el periodo que
protege. En ese momento todavía no se sabe cuántos viajes habrá ese día. Si una
variable del modelo usa datos de ese mismo día o de días posteriores, aunque
sea de forma indirecta, el modelo "ve el futuro" al entrenar: sus métricas
salen muy buenas y en la operación real falla, porque ese dato aún no existe
cuando hay que decidir.

Regla del equipo (decisión 5 de CLAUDE.md): las variables de una fila cuyo día
local de Nueva York es D solo pueden depender de filas de station_hour de días
anteriores a D.

Cómo se revisa
--------------
No se lee el código del constructor. Se calculan las variables con los datos
originales y luego se hacen dos experimentos con un día de corte:

A) Cambiar valores. Se copia la tabla y se alteran mucho todos los conteos
   desde el día de corte en adelante (el día de corte incluido). Se calculan
   otra vez las variables con esa copia.
B) Quitar filas. Se copia la tabla y se le quita cerca de la mitad de las
   filas desde el día de corte en adelante (incluido), sin tocar ningún
   valor. Se calculan otra vez las variables con esa copia.

En los dos se comparan las variables de las filas del día de corte y de los
días anteriores (en B, las de las filas que quedaron). Esas filas solo pueden
mirar días previos al corte, y esos días no se tocaron. Una variable que
cumple la regla no tiene forma de enterarse del cambio y sale idéntica. Si
cambió, es que usó datos del mismo día o del futuro: tiene fuga.

Por qué hace falta B: una variable puede tener fuga sin leer ningún conteo.
"Cuántas horas con viajes lleva hoy la estación" o "cuántas horas faltan para
la siguiente fila" solo miran QUÉ FILAS EXISTEN. Si station_hour no trae fila
para las horas sin viajes (decisión abierta en CLAUDE.md), eso delata cuánta
demanda hubo ese día. El experimento A no lo ve, porque no cambia las filas.

Consecuencia de B: un rezago por posición, como `shift(24)`, sale marcado,
porque depende de cuántas filas hay y no de la fecha. Los rezagos se buscan
por fecha y hora local, como en src/features/lags.py.

Límites
-------
- Cada revisión prueba un solo día de corte. Por eso tests/test_no_leakage.py
  usa varios, incluidos los días de cambio de horario.
- Solo ve fugas que pasan por la tabla que se altera (hoy, station_hour), sea
  por sus conteos o por las filas que trae. No ve lo que un constructor lea
  de otro archivo ni lo que guarde de una corrida anterior.
- Vale para la tabla con la que se corre. Una variable que en esa tabla sale
  vacía (NaN) en todas las filas comparadas pasa sin que se haya probado nada;
  por ejemplo, un rezago más largo que el periodo de la muestra.
- El experimento B quita filas al azar con semilla fija. En una tabla con muy
  pocas filas desde el día de corte puede no quitar ninguna.
- Un constructor con azar sin semilla fija también sale marcado, porque no da
  lo mismo dos veces.

Cómo se extiende a fuentes externas (todavía NO está implementado)
-----------------------------------------------------------------
La idea es la misma: alterar todo lo que no se conocía al decidir y comprobar
que las variables no cambian. Ejemplo con el clima: para decidir el día D solo
existe el PRONÓSTICO emitido antes de D. La lluvia observada el día D se conoce
después, así que usarla es fuga. Cuando se agregue el clima, su tabla debe
guardar para qué día es cada dato y cuándo se emitió. La revisión alterará todo
dato emitido u observado desde el día de corte, y las variables de ese día y de
los anteriores no deben cambiar. Lo mismo vale para cualquier otra fuente
(afluencia del metro, capacidad de GBFS): cuenta la fecha en que el dato estuvo
disponible, no la fecha que describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import KEY_COLUMNS, FeatureBuilder
from src.paths import TZ

# Conteos de station_hour que se alteran: todos.
VALUE_COLUMNS = ["departures", "arrivals", "net_flow", "departures_member", "departures_casual"]

# Semilla fija: la alteración y las filas que se quitan son siempre las mismas.
SEED = 5


class BuilderOutputError(ValueError):
    """La salida de un constructor no tiene la forma acordada."""


def local_day(ts: pd.Series) -> pd.Series:
    """Día local de Nueva York de cada instante (medianoche, sin zona)."""
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"ts_hour debe traer zona horaria (tipo recibido: {ts.dtype}). Localízala a {TZ}.")
    return ts.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()


def _as_day(cutoff_date) -> pd.Timestamp:
    """Acepta "2025-11-02", un date o un Timestamp y devuelve ese día a medianoche."""
    day = pd.Timestamp(cutoff_date)
    if day.tz is not None:
        day = day.tz_convert(TZ).tz_localize(None)
    return day.normalize()


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """Llaves de una tabla en forma comparable: el instante se lleva a UTC."""
    instant = pd.DatetimeIndex(frame["ts_hour"]).tz_convert("UTC").as_unit("ns")
    return pd.MultiIndex.from_arrays([frame["station_id"].astype(str).to_numpy(), instant], names=KEY_COLUMNS)


def perturb_from_date(station_hour: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    """Copia de station_hour con todos los conteos alterados desde el día de corte (incluido).

    A cada conteo se le suma una cantidad grande, entre mil y cien mil, distinta
    en cada fila y en cada columna. Las llaves y las filas anteriores al corte
    quedan igual. El resultado es siempre el mismo (semilla fija).
    """
    columns = [column for column in VALUE_COLUMNS if column in station_hour.columns]
    if not columns:
        raise ValueError(f"station_hour no trae ninguna columna de conteos {VALUE_COLUMNS}: no hay nada que alterar.")
    altered = (local_day(station_hour["ts_hour"]) >= _as_day(cutoff_date)).to_numpy()
    rng = np.random.default_rng(SEED)
    out = station_hour.copy()
    for column in columns:
        shift = rng.integers(1_000, 100_000, size=len(out))
        out[column] = out[column] + np.where(altered, shift, 0)
    return out


def drop_rows_from_date(station_hour: pd.DataFrame, cutoff_date) -> pd.DataFrame:
    """Copia de station_hour sin cerca de la mitad de las filas desde el día de corte (incluido).

    Las filas anteriores al corte quedan todas y ningún valor cambia. Las filas
    que se quitan son siempre las mismas (semilla fija). El índice se renumera,
    como en una tabla recién leída.
    """
    from_cutoff = (local_day(station_hour["ts_hour"]) >= _as_day(cutoff_date)).to_numpy()
    keep = ~from_cutoff | (np.random.default_rng(SEED).random(len(station_hour)) < 0.5)
    return station_hour[keep].reset_index(drop=True)


def check_builder_output(features: pd.DataFrame, station_hour: pd.DataFrame) -> None:
    """Comprueba la forma de la salida de un constructor. Lanza BuilderOutputError si no cumple.

    Forma acordada: un DataFrame con las llaves station_id y ts_hour, una fila
    por cada fila recibida (ni más ni menos) y al menos una variable.
    """
    if not isinstance(features, pd.DataFrame):
        raise BuilderOutputError(f"El constructor debe devolver un DataFrame y devolvió {type(features).__name__}.")
    missing = [column for column in KEY_COLUMNS if column not in features.columns]
    if missing:
        raise BuilderOutputError(f"A la salida del constructor le faltan las columnas llave {missing}.")
    if not features.columns.is_unique:
        repeated = sorted(set(features.columns[features.columns.duplicated()]))
        raise BuilderOutputError(f"La salida del constructor repite nombres de columna: {repeated}.")
    if not isinstance(features["ts_hour"].dtype, pd.DatetimeTZDtype):
        raise BuilderOutputError(
            f"El constructor cambió ts_hour (tipo devuelto: {features['ts_hour'].dtype}). "
            "Debe devolverla igual que la recibió, con su zona horaria."
        )

    local_day(station_hour["ts_hour"])  # la tabla de entrada también debe traer zona horaria
    expected, got = _key_index(station_hour), _key_index(features)
    if expected.has_duplicates:
        raise ValueError("station_hour trae llaves (station_id, ts_hour) repetidas. Revisa la tabla de entrada.")

    problems: list[str] = []
    if len(features) != len(station_hour):
        problems.append(f"recibió {len(station_hour)} filas y devolvió {len(features)}")
    repeated_rows = int(got.duplicated().sum())
    if repeated_rows:
        problems.append(f"{repeated_rows} filas repiten una llave (suele pasar con un merge que multiplica filas)")
    lost = len(expected.difference(got))
    if lost:
        problems.append(f"{lost} filas recibidas no aparecen en la salida")
    unknown = len(got.difference(expected))
    if unknown:
        problems.append(f"{unknown} llaves de la salida no venían en la entrada")
    if len(features.columns) == len(KEY_COLUMNS):
        problems.append("no trae ninguna variable además de las llaves")
    if problems:
        raise BuilderOutputError(
            "La salida del constructor debe tener exactamente una fila por cada fila recibida, "
            "con las mismas llaves (station_id, ts_hour):\n- " + "\n- ".join(problems)
        )


def _features_in_input_order(builder: FeatureBuilder, station_hour: pd.DataFrame) -> pd.DataFrame:
    """Corre el constructor, revisa la forma y acomoda las variables en el orden de la entrada."""
    # Se le pasa una copia: así el constructor no puede ensuciar la tabla de quien llama.
    features = builder(station_hour.copy())
    check_builder_output(features, station_hour)
    by_key = features.drop(columns=KEY_COLUMNS).set_axis(_key_index(features), axis=0)
    return by_key.reindex(_key_index(station_hour))


def _differs(a: np.ndarray, b: np.ndarray) -> bool:
    """¿Cambió algún valor? Dos vacíos (NaN) en la misma fila cuentan como iguales."""
    empty_a, empty_b = pd.isna(a), pd.isna(b)
    if (empty_a != empty_b).any():
        return True
    return bool((a[~empty_a] != b[~empty_a]).any())


def check_no_leakage(builder: FeatureBuilder, station_hour: pd.DataFrame, cutoff_date) -> list[str]:
    """Devuelve las variables de `builder` que tienen fuga. Lista vacía = no tiene.

    Hace dos experimentos desde `cutoff_date` (día local de Nueva York,
    incluido): altera todos los conteos de station_hour y, aparte, le quita
    cerca de la mitad de las filas. En los dos comprueba que las variables de
    las filas de ese día y de los anteriores salgan idénticas. La explicación
    completa está al inicio de este archivo.

    Lanza BuilderOutputError si la salida del constructor no tiene la forma
    acordada, y ValueError si el día de corte queda fuera de los datos (la
    revisión pasaría sin probar nada).
    """
    cutoff = _as_day(cutoff_date)
    day = local_day(station_hour["ts_hour"])
    compared = (day <= cutoff).to_numpy()
    if not compared.any() or not (day >= cutoff).any():
        raise ValueError(
            f"El día de corte {cutoff:%Y-%m-%d} queda fuera de los datos "
            f"({day.min():%Y-%m-%d} a {day.max():%Y-%m-%d}). La revisión no probaría nada."
        )

    original = _features_in_input_order(builder, station_hour)
    # Experimento A: las mismas filas, con los conteos alterados desde el corte.
    altered = _features_in_input_order(builder, perturb_from_date(station_hour, cutoff))
    # Experimento B: los mismos conteos, con menos filas desde el corte. Solo se comparan las que quedaron.
    thinned = drop_rows_from_date(station_hour, cutoff)
    fewer = _features_in_input_order(builder, thinned).reindex(original.index)
    survived = compared & np.asarray(original.index.isin(_key_index(thinned)))

    for other in (altered, fewer):
        if set(original.columns) != set(other.columns):
            changed = sorted(set(original.columns) ^ set(other.columns))
            raise BuilderOutputError(
                f"El constructor devolvió columnas distintas al cambiar los datos: {changed}. "
                "Los nombres de las variables no pueden depender de los valores ni de las filas que haya."
            )

    leaking = []
    for column in original.columns:
        values = original[column].to_numpy(dtype=object)
        by_values = _differs(values[compared], altered[column].to_numpy(dtype=object)[compared])
        by_rows = _differs(values[survived], fewer[column].to_numpy(dtype=object)[survived])
        if by_values or by_rows:
            leaking.append(column)
    return leaking
