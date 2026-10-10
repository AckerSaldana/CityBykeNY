"""Métricas de pronóstico (decisión 6): MAE, RMSE, WAPE y cobertura del intervalo.

Sirve para cualquier modelo que escriba el contrato `predictions`: hoy la
línea base, después LightGBM.

Qué se mide, para salidas (dep) y para llegadas (arr) por separado:

- MAE y RMSE del pronóstico puntual, que es la columna q50.
- WAPE = suma de |real - q50| entre suma de real. Queda vacío (NaN) cuando
  la suma de real es 0. No se usa MAPE: explota con demanda cercana a cero.
- Cobertura = proporción de filas con q10 <= real <= q90. El intervalo va
  del 10 % al 90 %, así que la cobertura nominal es 80 %.
- Ancho medio del intervalo = promedio de q90 - q10.

Desgloses: general, por hora local del día, por franja del reporte, por tipo
de día (laborable, fin de semana, festivo) y por estación.

Las franjas del reporte son las mismas con las que la simulación reporta el
riesgo: las franjas de config/operacion.yaml, la franja de noche de
config/datos.yaml (`simulacion.franja_noche`) y "resto". Cada fila de estación
y hora cuenta en la franja donde empieza su hora.

La prueba final se usa una sola vez (decisión 4). Por eso `evaluate` se niega
a calificar el tramo "test", y cualquier fila que caiga en las fechas de la
prueba final, salvo que se le pida de forma explícita.

Al leer los números: los datos solo registran demanda atendida. Una hora en
cero puede ser una estación sin demanda, vacía, llena o cerrada.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.datos_config import Tramo, load_datos
from src.paths import DATOS_YAML, OPERACION_YAML, TZ
from src.sim.engine import NIGHT, OUTSIDE, Reglas, hour_labels
from src.sim.operacion import load_operacion

# Prefijo de las columnas en predictions -> columna con el valor real en station_hour.
TARGETS = {"dep": "departures", "arr": "arrivals"}
TARGET_LABELS = {"dep": "salidas", "arr": "llegadas"}
KEYS = ["station_id", "ts_hour"]

GROUP_BYS = ("general", "hora", "franja", "tipo_de_dia", "estacion")
DAY_TYPES = ("laborable", "fin de semana", "festivo")
ALL = "todas"              # valor de `group` en el desglose general
FINAL_SPLIT = "test"       # el tramo que se califica una sola vez
NOMINAL_COVERAGE = 0.80    # el intervalo va de q10 a q90

_HOUR_LABELS = np.array([f"{hour:02d}" for hour in range(24)])  # "00" a "23": ordena bien como texto

# Tabla larga de métricas: una fila por modelo, tramo, objetivo y grupo.
METRICS_SCHEMA = pa.schema(
    [
        ("model", pa.string()),
        ("split", pa.string()),
        ("target", pa.string()),       # "dep" o "arr"
        ("group_by", pa.string()),     # uno de GROUP_BYS
        ("group", pa.string()),        # "todas", hora "00" a "23", franja, tipo de día o station_id
        ("n", pa.int64()),             # filas de estación y hora calificadas
        ("mae", pa.float64()),
        ("rmse", pa.float64()),
        ("wape", pa.float64()),        # vacío si la suma de real es 0
        ("coverage", pa.float64()),
        ("mean_width", pa.float64()),
    ]
)
METRIC_COLUMNS = list(METRICS_SCHEMA.names)


class FinalTestError(RuntimeError):
    """Se intentó calificar la prueba final sin pedirlo de forma explícita."""


# --------------------------------------------------------------------------- #
# Hora local y tipo de día
# --------------------------------------------------------------------------- #

def local_time(ts_hour: pd.Series) -> pd.Series:
    """Fecha y hora del reloj de Nueva York de cada fila, ya sin zona.

    El día de 25 horas las dos 01:00 quedan con la misma hora de reloj.
    """
    if not isinstance(ts_hour.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"ts_hour debe traer zona horaria (tipo recibido: {ts_hour.dtype}). Localízala a {TZ}.")
    return ts_hour.dt.tz_convert(TZ).dt.tz_localize(None)


def federal_holidays(first_year: int, last_year: int) -> dict[date, str]:
    """Festivos federales de Estados Unidos, con su nombre (paquete `holidays`).

    Incluye el día en que se observa un festivo que cae en fin de semana.
    No incluye los festivos propios del estado de Nueva York.
    """
    return dict(holidays.US(years=range(first_year, last_year + 1)))


def day_type(ts_hour: pd.Series) -> pd.Series:
    """Tipo de día de cada fila: "laborable", "fin de semana" o "festivo".

    Se usa la fecha local de Nueva York. Sábado y domingo son siempre "fin de
    semana". "festivo" es un festivo federal que cae de lunes a viernes.
    """
    local = local_time(ts_hour)
    if local.empty:
        return pd.Series([], index=ts_hour.index, dtype="str")
    day = local.dt.normalize()
    festivos = pd.DatetimeIndex(list(federal_holidays(day.min().year, day.max().year)))
    weekend = (local.dt.dayofweek >= 5).to_numpy()
    kind = np.where(weekend, "fin de semana", np.where(day.isin(festivos).to_numpy(), "festivo", "laborable"))
    return pd.Series(kind, index=ts_hour.index)


# Nota que acompaña a la tabla por franja en los reportes de métricas.
FRANJA_NOTE = (
    "Las franjas son las del reporte de la simulación: las de `config/operacion.yaml`, la franja de noche de\n"
    "`config/datos.yaml` (`simulacion.franja_noche`) y \"resto\", que es lo que no cae en ninguna. Cada fila de\n"
    "estación y hora cuenta en la franja donde empieza su hora. Las filas de las franjas suman las del tramo."
)


def report_bands(operacion: str | Path = OPERACION_YAML, config: str | Path = DATOS_YAML) -> tuple[str, ...]:
    """Franja del reporte de cada hora de reloj (de la 0 a la 23), leída de los dos archivos de config.

    Las franjas de operación salen de `operacion` (config/operacion.yaml) y la
    franja de noche del bloque `simulacion` de `config` (config/datos.yaml).
    Lo que no cae en ninguna es "resto". Si `config` no trae el bloque
    `simulacion`, no hay franja de noche. Es la misma regla con la que la
    simulación reparte el riesgo (`src.sim.engine.hour_labels`).
    """
    franjas = Reglas.from_operacion(load_operacion(operacion)).franjas
    simulacion = load_datos(config).simulacion
    return hour_labels(franjas, None if simulacion is None else tuple(simulacion.franja_noche))


# --------------------------------------------------------------------------- #
# Cálculo
# --------------------------------------------------------------------------- #

def metrics_by_group(actual, q10, q50, q90, group=None) -> pd.DataFrame:
    """Métricas por grupo. Devuelve una fila por grupo (el grupo es el índice).

    Columnas: n, mae, rmse, wape, coverage y mean_width. Sin `group` se
    calcula una sola fila con todo.
    """
    actual, q10, q50, q90 = (np.asarray(values, dtype=float) for values in (actual, q10, q50, q90))
    error = actual - q50
    frame = pd.DataFrame(
        {
            "group": ALL if group is None else np.asarray(group),
            "n": 1,
            "actual": actual,
            "abs_error": np.abs(error),
            "sq_error": error**2,
            "covered": ((q10 <= actual) & (actual <= q90)).astype(float),
            "width": q90 - q10,
        }
    )
    sums = frame.groupby("group", sort=True).sum()
    n = sums["n"]
    return pd.DataFrame(
        {
            "n": n.astype("int64"),
            "mae": sums["abs_error"] / n,
            "rmse": np.sqrt(sums["sq_error"] / n),
            # Sin demanda real en el grupo no hay contra qué dividir: queda vacío.
            "wape": sums["abs_error"] / sums["actual"].where(sums["actual"] != 0),
            "coverage": sums["covered"] / n,
            "mean_width": sums["width"] / n,
        }
    )


def evaluate(
    predictions: pd.DataFrame,
    station_hour: pd.DataFrame,
    *,
    model: str,
    split: str,
    usar_prueba_final: bool = False,
    prueba_final: Tramo | None = None,
    exigir_todas: bool = True,
    franjas=None,
) -> pd.DataFrame:
    """Califica un modelo en un tramo. Devuelve la tabla larga (METRIC_COLUMNS).

    `predictions` sigue el contrato predictions y `station_hour` trae los
    valores reales. Se califican las filas de `predictions` con ese `model` y
    ese `split`; cada una debe tener su fila en `station_hour`.

    Uso normal, igual para la línea base y para LightGBM:

        val = datos.tramos["validacion"]
        reales = station_hour[val.contains(station_hour["ts_hour"])]
        evaluate(predictions, reales, model="lgbm", split="val")

    Argumentos opcionales:

    - `usar_prueba_final`: la prueba final solo se califica con `True`
      (decisión 4). Sin eso falla si `split` es "test" y también si alguna fila
      a calificar cae en las fechas de la prueba final o después, aunque su
      `split` diga otra cosa.
    - `prueba_final`: el tramo de la prueba final, para saber qué fechas
      proteger. Si no se pasa, se lee de config/datos.yaml.
    - `exigir_todas`: con `True` (lo normal) cada fila de `station_hour` debe
      tener su predicción. Así dos modelos se comparan sobre las mismas filas.
      Por eso a `station_hour` se le pasan solo las filas del tramo. Con
      `False` se califican solo las filas que el modelo sí predijo: esas
      métricas no se pueden comparar con las de otro modelo.
    - `franjas`: la franja del reporte de cada hora de reloj, 24 etiquetas de
      la hora 0 a la 23 (lo que devuelve `report_bands`). Si no se pasa, se
      lee de config/operacion.yaml y config/datos.yaml.

    Falla con ValueError si en las filas a calificar hay un valor vacío (un
    vacío contaría como error cero) o si q10 es mayor que q90.
    """
    if split == FINAL_SPLIT and not usar_prueba_final:
        raise FinalTestError(
            "La prueba final no se califica aquí. Decisión 4: las decisiones se toman con validación "
            "y la prueba final se usa una sola vez. Si de verdad es el momento de usarla, llama a "
            "evaluate(..., split='test', usar_prueba_final=True)."
        )
    rows = predictions[(predictions["model"] == model) & (predictions["split"] == split)]
    if rows.empty:
        raise ValueError(f"No hay predicciones con model={model!r} y split={split!r}.")
    if not usar_prueba_final:
        # El candado mira las fechas, no solo la etiqueta: una fila de la prueba final marcada "val" no pasa.
        final = load_datos().tramos["prueba"] if prueba_final is None else prueba_final
        inside = int((local_time(rows["ts_hour"]) >= pd.Timestamp(final.inicio)).sum())
        if inside:
            raise FinalTestError(
                f"{inside} predicciones con split={split!r} caen en la prueba final o después "
                f"(desde {final.inicio}). Decisión 4: no se califican sin usar_prueba_final=True."
            )

    actual = station_hour[KEYS + list(TARGETS.values())]
    scored = rows.merge(actual, on=KEYS, how="left", validate="one_to_one")
    missing = int(scored[TARGETS["dep"]].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} predicciones de model={model!r} y split={split!r} no tienen fila en station_hour. "
            "No se pueden calificar."
        )
    quantiles = [f"{prefix}_{q}" for prefix in TARGETS for q in ("q10", "q50", "q90")]
    empty = scored[list(TARGETS.values()) + quantiles].isna().sum()
    if empty.any():
        detail = ", ".join(f"{column}: {count}" for column, count in empty[empty > 0].items())
        raise ValueError(
            f"Hay valores vacíos en las filas que se califican ({detail}). "
            "No se califica: un vacío contaría como error cero."
        )
    for prefix in TARGETS:
        crossed = int((scored[f"{prefix}_q10"] > scored[f"{prefix}_q90"]).sum())
        if crossed:
            raise ValueError(
                f"{TARGET_LABELS[prefix]}: {crossed} filas con {prefix}_q10 mayor que {prefix}_q90. "
                "Ordena los cuantiles antes de calificar."
            )
    unscored = len(actual) - len(scored)
    if exigir_todas and unscored:
        raise ValueError(
            f"{unscored} filas de station_hour no tienen predicción de model={model!r} y split={split!r}. "
            "Pasa solo las filas del tramo. Si el modelo de verdad no predice todas, usa exigir_todas=False: "
            "esas métricas no se pueden comparar con las de otro modelo."
        )

    bands = np.array(report_bands() if franjas is None else list(franjas), dtype=object)
    if len(bands) != 24:
        raise ValueError(f"`franjas` debe traer 24 etiquetas, una por hora de reloj (trae {len(bands)}).")
    # Como en el reporte de la simulación: primero las franjas de operación, después la noche y el resto.
    band_order = [band for band in dict.fromkeys(bands.tolist()) if band not in (NIGHT, OUTSIDE)] + [NIGHT, OUTSIDE]
    hour = local_time(scored["ts_hour"]).dt.hour.to_numpy()
    groups = {
        "general": None,
        "hora": _HOUR_LABELS[hour],
        "franja": bands[hour],
        "tipo_de_dia": day_type(scored["ts_hour"]).to_numpy(),
        "estacion": scored["station_id"].to_numpy(),
    }
    parts = []
    for prefix, column in TARGETS.items():
        for group_by, group in groups.items():
            part = metrics_by_group(
                scored[column], scored[f"{prefix}_q10"], scored[f"{prefix}_q50"], scored[f"{prefix}_q90"], group
            )
            if group_by == "tipo_de_dia":  # en el orden de DAY_TYPES, no alfabético
                part = part.reindex([kind for kind in DAY_TYPES if kind in part.index])
            elif group_by == "franja":
                part = part.reindex([band for band in band_order if band in part.index])
            part = part.rename_axis("group").reset_index()
            part["group"] = part["group"].astype("str")
            part["model"], part["split"], part["target"], part["group_by"] = model, split, prefix, group_by
            parts.append(part[METRIC_COLUMNS])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Guardar y mostrar
# --------------------------------------------------------------------------- #

def write_metrics(metrics: pd.DataFrame, path: str | Path) -> Path:
    """Guarda la tabla larga de métricas en Parquet.

    Si el archivo ya existe, reemplaza las filas de cada modelo y tramo que
    vienen en `metrics` y conserva todas las demás: las de otros modelos y las
    del mismo modelo en otro tramo.
    """
    path = Path(path)
    metrics = metrics[METRIC_COLUMNS]
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        if list(existing.columns) != METRIC_COLUMNS:
            raise ValueError(
                f"{path} ya existe con otras columnas ({list(existing.columns)}). Bórralo y vuelve a correr."
            )
        replaced = set(zip(metrics["model"], metrics["split"]))
        others = existing[[key not in replaced for key in zip(existing["model"], existing["split"])]]
        if len(others):
            metrics = pd.concat([others, metrics], ignore_index=True)
    table = pa.Table.from_pandas(metrics, schema=METRICS_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)  # reemplazo atómico: nunca queda un archivo a medias
    return path


def markdown_table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> str:
    """Tabla en Markdown con las columnas alineadas (se lee bien también en la terminal).

    Las primeras `text_columns` columnas son texto y van a la izquierda; el
    resto son números y van a la derecha.
    """
    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]

    def render(cells: list[str]) -> str:
        padded = [
            cell.ljust(width) if i < text_columns else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = [("-" * width) if i < text_columns else ("-" * (width - 1) + ":") for i, width in enumerate(widths)]
    return "\n".join([render(header), "| " + " | ".join(rule) + " |", *(render(row) for row in rows)])


def _number(value: float) -> str:
    return "n/d" if pd.isna(value) else f"{value:.3f}"


def _percent(value: float) -> str:
    return "n/d" if pd.isna(value) else f"{100 * value:.1f} %"


def format_table(metrics: pd.DataFrame, group_by: str, label: str) -> str:
    """Tabla en Markdown de un desglose: primero las filas de salidas y después las de llegadas.

    `metrics` es la salida de `evaluate` para UN modelo y UN tramo. `label` es
    el título de la columna del grupo (por ejemplo "hora").
    """
    header = ["objetivo", label, "filas", "MAE", "RMSE", "WAPE", "cobertura", "ancho medio"]
    rows = [
        [
            TARGET_LABELS[row.target],
            str(row.group),
            f"{row.n:,}",
            _number(row.mae),
            _number(row.rmse),
            _percent(row.wape),
            _percent(row.coverage),
            _number(row.mean_width),
        ]
        for row in metrics[metrics["group_by"] == group_by].itertuples()
    ]
    return markdown_table(header, rows, text_columns=2)
