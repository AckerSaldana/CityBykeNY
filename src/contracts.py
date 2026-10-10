"""Contratos de datos: la única fuente de verdad de nombres y tipos.

Estos seis archivos conectan el trabajo de los dos integrantes. Quien los
produce debe escribirlos con `write_contract` (convierte a los tipos exactos y
valida antes de guardar) y quien los consume puede leerlos con `read_contract`.

Los tipos se definen como esquema de Arrow, que es lo que queda escrito en el
Parquet. Así la validación no depende de la versión de pandas.

Si un contrato cambia, se cambia AQUÍ y en CLAUDE.md, avisando al otro
integrante. Nunca en un solo lado.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.paths import TZ

# ts_hour, window_start y window_end: instante con zona America/New_York.
TS = pa.timestamp("us", tz=TZ)

MODELS = ("baseline", "lgbm")
SPLITS = ("val", "test")
POLICIES = ("none", "historical", "model")
RISK_LEVELS = ("alto", "medio", "bajo")
# Interpretación del equipo (pendiente de confirmar por logística): station_type
# usa las mismas categorías que la estrategia por tipo en config/operacion.yaml.
# OJO: no es el campo `station_type` de GBFS ("classic"/"lightweight").
STATION_TYPES = ("oficinas", "residencial", "recreativa", "mixta")

QUANTILE_COLUMNS = ("dep_q10", "dep_q50", "dep_q90", "arr_q10", "arr_q50", "arr_q90")


class ContractError(ValueError):
    """Un archivo o tabla no cumple su contrato."""


@dataclass(frozen=True)
class Contract:
    name: str                       # nombre corto, p. ej. "station_hour"
    schema: pa.Schema               # columnas y tipos exactos, en orden
    key: tuple[str, ...]            # columnas que identifican una fila (sin duplicados)
    nullable: frozenset = frozenset()          # columnas que admiten nulos
    allowed: dict = field(default_factory=dict)  # columna -> valores permitidos

    @property
    def filename(self) -> str:
        return f"{self.name}.parquet"

    @property
    def columns(self) -> list[str]:
        return list(self.schema.names)


STATIONS = Contract(
    name="stations",
    schema=pa.schema(
        [
            ("station_id", pa.string()),   # id de los viajes (= short_name en GBFS)
            ("name", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("capacity", pa.int64()),      # puede ser nulo
            ("station_type", pa.string()), # puede ser nulo
        ]
    ),
    key=("station_id",),
    nullable=frozenset({"capacity", "station_type"}),
    allowed={"station_type": STATION_TYPES},
)

STATION_HOUR = Contract(
    name="station_hour",
    schema=pa.schema(
        [
            ("station_id", pa.string()),
            ("ts_hour", TS),               # inicio de la hora, hora local de NY
            ("departures", pa.int64()),
            ("arrivals", pa.int64()),
            ("net_flow", pa.int64()),      # llegadas menos salidas
            ("departures_member", pa.int64()),
            ("departures_casual", pa.int64()),
        ]
    ),
    key=("station_id", "ts_hour"),
)

PREDICTIONS = Contract(
    name="predictions",
    schema=pa.schema(
        [
            ("station_id", pa.string()),
            ("ts_hour", TS),
            ("model", pa.string()),
            ("split", pa.string()),
            ("dep_q10", pa.float64()),
            ("dep_q50", pa.float64()),
            ("dep_q90", pa.float64()),
            ("arr_q10", pa.float64()),
            ("arr_q50", pa.float64()),
            ("arr_q90", pa.float64()),
        ]
    ),
    key=("station_id", "ts_hour", "model", "split"),
    allowed={"model": MODELS, "split": SPLITS},
)

RECOMMENDATIONS = Contract(
    name="recommendations",
    schema=pa.schema(
        [
            ("station_id", pa.string()),
            ("window_start", TS),
            ("window_end", TS),
            ("policy", pa.string()),
            ("start_inventory_target", pa.int64()),  # inventario con el que queda la estación tras el traslado
            ("bikes_to_move", pa.int64()),  # positivo es llevar, negativo es retirar
            ("risk_level", pa.string()),
            ("feasible", pa.bool_()),
        ]
    ),
    key=("station_id", "window_start", "policy"),
    allowed={"policy": POLICIES, "risk_level": RISK_LEVELS},
)

POLICY_CURVE = Contract(
    name="policy_curve",
    schema=pa.schema(
        [
            ("policy", pa.string()),
            ("budget", pa.int64()),
            ("risk_station_hours", pa.float64()),
            ("ci_low", pa.float64()),
            ("ci_high", pa.float64()),
        ]
    ),
    key=("policy", "budget"),
    allowed={"policy": POLICIES},
)

# Viajes entre pares de estaciones, sumados sobre todo el periodo. Es para la
# vista de ranking de estaciones y rutas. NO son rutas de rebalanceo (camiones).
ROUTES = Contract(
    name="routes",
    schema=pa.schema(
        [
            ("start_station_id", pa.string()),
            ("end_station_id", pa.string()),
            ("trips", pa.int64()),          # viajes de inicio a fin en el periodo
        ]
    ),
    key=("start_station_id", "end_station_id"),
)

CONTRACTS: dict[str, Contract] = {
    c.name: c for c in (STATIONS, STATION_HOUR, PREDICTIONS, RECOMMENDATIONS, POLICY_CURVE, ROUTES)
}


def get_contract(name: str) -> Contract:
    try:
        return CONTRACTS[name]
    except KeyError:
        raise KeyError(f"Contrato desconocido: {name!r}. Opciones: {sorted(CONTRACTS)}") from None


# --------------------------------------------------------------------------- #
# Validación
# --------------------------------------------------------------------------- #

def _count(mask) -> int:
    """Cuántas filas cumplen `mask` (los nulos no cuentan)."""
    total = pc.sum(mask).as_py()
    return int(total or 0)


def _check_values(t: pa.Table, name: str) -> list[str]:
    """Reglas de contenido de cada contrato (más allá de nombres y tipos)."""
    problems: list[str] = []

    def at_least_zero(*columns: str) -> None:
        for column in columns:
            n = _count(pc.less(t[column], 0))
            if n:
                problems.append(f"{column}: {n} valores negativos")

    def on_the_hour(*columns: str) -> None:
        for column in columns:
            off = pc.or_(
                pc.or_(pc.not_equal(pc.minute(t[column]), 0), pc.not_equal(pc.second(t[column]), 0)),
                pc.not_equal(pc.subsecond(t[column]), 0),
            )
            n = _count(off)
            if n:
                problems.append(f"{column}: {n} valores que no caen en hora exacta")

    if name == "stations":
        bad_lat = pc.or_(pc.less(t["lat"], -90), pc.greater(t["lat"], 90))
        bad_lon = pc.or_(pc.less(t["lon"], -180), pc.greater(t["lon"], 180))
        if _count(bad_lat) or _count(bad_lon):
            problems.append("lat/lon fuera de rango")
        at_least_zero("capacity")

    elif name == "station_hour":
        at_least_zero("departures", "arrivals", "departures_member", "departures_casual")
        on_the_hour("ts_hour")
        n = _count(pc.not_equal(t["net_flow"], pc.subtract(t["arrivals"], t["departures"])))
        if n:
            problems.append(f"net_flow: {n} filas donde no es llegadas menos salidas")
        by_type = pc.add(t["departures_member"], t["departures_casual"])
        n = _count(pc.greater(by_type, t["departures"]))
        if n:
            problems.append(f"departures: {n} filas donde miembros + casuales supera el total")

    elif name == "predictions":
        at_least_zero(*QUANTILE_COLUMNS)
        on_the_hour("ts_hour")
        for prefix in ("dep", "arr"):
            q10, q50, q90 = (t[f"{prefix}_q{q}"] for q in (10, 50, 90))
            n = _count(pc.or_(pc.greater(q10, q50), pc.greater(q50, q90)))
            if n:
                problems.append(f"{prefix}: {n} filas con cuantiles cruzados (debe ser q10 <= q50 <= q90)")

    elif name == "recommendations":
        at_least_zero("start_inventory_target")
        n = _count(pc.less_equal(t["window_end"], t["window_start"]))
        if n:
            problems.append(f"window_end: {n} filas donde no es posterior a window_start")
        moved_without_policy = pc.and_(pc.equal(t["policy"], "none"), pc.not_equal(t["bikes_to_move"], 0))
        n = _count(moved_without_policy)
        if n:
            problems.append(f"bikes_to_move: {n} filas con policy 'none' que mueven bicis")

    elif name == "policy_curve":
        at_least_zero("budget", "risk_station_hours", "ci_low")
        outside = pc.or_(
            pc.greater(t["ci_low"], t["risk_station_hours"]),
            pc.less(t["ci_high"], t["risk_station_hours"]),
        )
        n = _count(outside)
        if n:
            problems.append(f"ci_low/ci_high: {n} filas donde el intervalo no contiene risk_station_hours")

    elif name == "routes":
        n = _count(pc.less(t["trips"], 1))
        if n:
            problems.append(f"trips: {n} rutas con menos de 1 viaje")

    return problems


def _check_dense(t: pa.Table) -> list[str]:
    """station_hour es densa: una fila por estación y hora, con ceros, solo
    entre el primer y el último viaje observado de cada estación.

    Se cuenta en instantes, no en hora de reloj: el día de 25 horas del cambio
    de horario tiene 25 filas y el de 23 horas tiene 23.
    """
    problems: list[str] = []
    hours = t.group_by("station_id").aggregate([("ts_hour", "min"), ("ts_hour", "max"), ("ts_hour", "count")])
    span = pc.subtract(hours["ts_hour_max"], hours["ts_hour_min"]).cast(pa.int64())  # microsegundos
    expected = pc.add(pc.divide(span, 3_600_000_000), 1)
    n = _count(pc.not_equal(hours["ts_hour_count"], expected))
    if n:
        problems.append(
            f"{n} estaciones con horas faltantes: debe haber una fila por hora, con ceros, "
            "entre su primera y su última hora"
        )

    with_trips = t.filter(pc.greater(pc.add(t["departures"], t["arrivals"]), 0))
    trips = with_trips.group_by("station_id").aggregate([("ts_hour", "min"), ("ts_hour", "max")])
    trips = trips.select(["station_id", "ts_hour_min", "ts_hour_max"]).rename_columns(
        ["station_id", "first_trip", "last_trip"]
    )
    joined = hours.join(trips, keys="station_id", join_type="left outer")
    n = joined["first_trip"].null_count
    if n:
        problems.append(f"{n} estaciones sin ningún viaje")
    padded = pc.or_(
        pc.not_equal(joined["ts_hour_min"], joined["first_trip"]),
        pc.not_equal(joined["ts_hour_max"], joined["last_trip"]),
    )
    n = _count(padded)
    if n:
        problems.append(f"{n} estaciones con filas en cero antes de su primer viaje o después del último")
    return problems


def validate_table(table: pa.Table, contract: Contract | str) -> list[str]:
    """Devuelve la lista de problemas encontrados. Lista vacía = cumple el contrato."""
    c = get_contract(contract) if isinstance(contract, str) else contract
    problems: list[str] = []

    # 1) Nombres y orden de columnas.
    if list(table.schema.names) != c.columns:
        missing = [col for col in c.columns if col not in table.schema.names]
        extra = [col for col in table.schema.names if col not in c.columns]
        if missing:
            problems.append(f"faltan columnas: {missing}")
        if extra:
            problems.append(f"sobran columnas: {extra}")
        if not missing and not extra:
            problems.append(f"orden de columnas distinto: {table.schema.names} (esperado {c.columns})")
        return problems  # sin las columnas correctas no tiene sentido seguir

    # 2) Tipos exactos.
    for expected in c.schema:
        actual = table.schema.field(expected.name).type
        if actual != expected.type:
            problems.append(f"{expected.name}: tipo {actual} (esperado {expected.type})")
    if problems:
        return problems

    # Un archivo sin filas casi siempre es un paso anterior que falló en silencio.
    if table.num_rows == 0:
        return ["la tabla no tiene filas"]

    # 3) Nulos solo donde el contrato los permite.
    for column in c.columns:
        nulls = table[column].null_count
        if nulls and column not in c.nullable:
            problems.append(f"{column}: {nulls} nulos y el contrato no los permite")

    # 4) Los decimales no pueden ser NaN ni infinito.
    for f in c.schema:
        if pa.types.is_floating(f.type):
            n = _count(pc.invert(pc.is_finite(table[f.name])))
            if n:
                problems.append(f"{f.name}: {n} valores NaN o infinitos")

    # 5) Valores permitidos.
    for column, options in c.allowed.items():
        present = set(pc.unique(table[column]).to_pylist()) - {None}
        unexpected = sorted(present - set(options))
        if unexpected:
            problems.append(f"{column}: valores no permitidos {unexpected} (opciones {list(options)})")

    # 6) Sin filas duplicadas por llave.
    groups = table.group_by(list(c.key)).aggregate([]).num_rows
    if groups != table.num_rows:
        problems.append(f"{table.num_rows - groups} filas duplicadas por llave {list(c.key)}")

    # 7) Reglas de contenido.
    problems.extend(_check_values(table, c.name))

    # 8) Tabla densa. Solo tiene sentido revisarla si todo lo anterior está bien.
    if c.name == "station_hour" and not problems:
        problems.extend(_check_dense(table))
    return problems


def validate_file(path: str | Path, contract: Contract | str) -> list[str]:
    """Valida un Parquet en disco contra su contrato."""
    return validate_table(pq.read_table(path), contract)


# --------------------------------------------------------------------------- #
# Lectura y escritura
# --------------------------------------------------------------------------- #

def _same_kind(source: pa.DataType, target: pa.DataType) -> bool:
    """¿El tipo recibido es de la misma familia (texto, número, sí/no) que pide el contrato?"""
    if pa.types.is_dictionary(source):  # categorías de pandas
        source = source.value_type
    if pa.types.is_string(target):
        return pa.types.is_string(source) or pa.types.is_large_string(source) or pa.types.is_string_view(source)
    if pa.types.is_boolean(target):
        return pa.types.is_boolean(source)
    if pa.types.is_integer(target) or pa.types.is_floating(target):
        # Entero, decimal o el DECIMAL que devuelve sum() en DuckDB. Un decimal
        # con fracción hacia una columna entera lo rechaza la conversión de abajo.
        return pa.types.is_integer(source) or pa.types.is_floating(source) or pa.types.is_decimal(source)
    return True  # fechas: tienen su propia revisión


def to_contract_table(data: pd.DataFrame | pa.Table | pa.RecordBatchReader, contract: Contract | str) -> pa.Table:
    """Convierte a los tipos exactos del contrato. Falla con mensaje claro si no se puede.

    Recibe un DataFrame de pandas o una tabla de Arrow (también lo que devuelve
    `.arrow()` en DuckDB). Solo convierte dentro de la misma familia de tipo:
    no vuelve texto un número, ni número un texto, ni sí/no un número.
    """
    c = get_contract(contract) if isinstance(contract, str) else contract
    if isinstance(data, pa.RecordBatchReader):  # lo que devuelve .arrow() en DuckDB
        data = data.read_all()
    if not isinstance(data, (pd.DataFrame, pa.Table)):
        raise ContractError(
            f"[{c.name}] se esperaba un DataFrame de pandas o una tabla de Arrow y llegó {type(data).__name__}."
        )
    names = list(data.columns) if isinstance(data, pd.DataFrame) else list(data.schema.names)
    missing = [col for col in c.columns if col not in names]
    if missing:
        raise ContractError(f"[{c.name}] faltan columnas: {missing}")

    if isinstance(data, pd.DataFrame):
        table = pa.Table.from_pandas(data[c.columns], preserve_index=False)
    else:
        table = data.select(c.columns)

    # Una fecha sin zona se interpretaría como UTC al convertir y quedaría
    # desplazada 4 o 5 horas sin avisar. Mejor fallar aquí.
    for f in c.schema:
        if pa.types.is_timestamp(f.type):
            source = table.schema.field(f.name).type
            if not pa.types.is_timestamp(source) or source.tz is None:
                raise ContractError(
                    f"[{c.name}] {f.name} debe traer zona horaria (tipo recibido: {source}). "
                    f"Localízala a {TZ} antes de guardar."
                )

    # Tampoco se cambia de familia de tipo sin avisar. Un id leído como número
    # ya perdió los ceros finales ("5470.10" llega como 5470.1) y al volverlo
    # texto quedaría "5470.1", que es otra estación.
    for f in c.schema:
        column = table[f.name]
        if column.null_count == len(column):
            continue  # columna toda vacía: no hay ningún valor que se pueda dañar
        if not _same_kind(column.type, f.type):
            if pa.types.is_string(f.type):
                hint = (
                    'Un texto leído como número pierde los ceros finales (el id "5470.10" queda "5470.1"): '
                    "léelo como texto desde el origen."
                )
            else:
                hint = "Conviértela tú antes de guardar, revisando los valores."
            raise ContractError(
                f"[{c.name}] {f.name} llegó como {column.type} y el contrato pide {f.type}. "
                f"Esa conversión no se hace sola. {hint}"
            )
    try:
        return table.cast(c.schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise ContractError(f"[{c.name}] no se pudo convertir a los tipos del contrato: {error}") from error


def write_contract(data: pd.DataFrame | pa.Table, contract: Contract | str, path: str | Path) -> Path:
    """Convierte, valida y guarda. Si no cumple el contrato, no escribe nada."""
    c = get_contract(contract) if isinstance(contract, str) else contract
    table = to_contract_table(data, c)
    problems = validate_table(table, c)
    if problems:
        raise ContractError(f"[{c.name}] no cumple el contrato:\n- " + "\n- ".join(problems))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)  # reemplazo atómico: nunca queda un archivo a medias
    return path


def read_contract(contract: Contract | str, path: str | Path) -> pd.DataFrame:
    """Lee un Parquet de contrato como DataFrame.

    Las columnas enteras que admiten nulos (p. ej. capacity) se leen como
    `Int64` de pandas para que no se conviertan en decimales.
    """
    c = get_contract(contract) if isinstance(contract, str) else contract
    df = pq.read_table(path).to_pandas()
    for f in c.schema:
        if f.name in c.nullable and pa.types.is_integer(f.type) and f.name in df.columns:
            df[f.name] = df[f.name].astype("Int64")
    return df
