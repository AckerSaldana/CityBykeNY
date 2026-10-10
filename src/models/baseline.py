"""Línea base: promedio histórico por estación, día de la semana y hora (decisión 3).

Para cada tramo que se predice (validación y prueba final):

1. Historia = los tramos ANTERIORES completos. Para validación es solo
   entrenamiento. Para la prueba final es entrenamiento más validación.
   Nunca entra nada del tramo que se predice ni de después (decisión 5).
2. Con la historia se calcula, para salidas y para llegadas, la media y los
   cuantiles empíricos 10 % y 90 % de cada celda: estación, día de la semana
   y hora local de Nueva York.
3. Se predice cada fila de station_hour que cae en el tramo. La tabla es
   densa solo entre el primer y el último viaje de cada estación (decisión
   12), así que no todas las estaciones cubren todo el tramo.

Cuando una celda no tiene historia se usa un respaldo, en este orden:

- nivel 1: estación, día de la semana y hora;
- nivel 2: estación y hora, con toda la historia de la estación (una
  estación con menos de una semana de historia);
- nivel 3: día de la semana y hora, con las filas de todas las estaciones
  de la historia (una estación nueva). Es el promedio por estación del
  sistema en esa hora.

OJO con el nombre de las columnas: en la línea base q50 es la MEDIA de la
celda, no la mediana. Es lo que fija la decisión 3. q10 y q90 sí son los
cuantiles empíricos de la misma celda (el mismo nivel de respaldo que la
media). Con muchos ceros la media puede quedar fuera de [q10, q90]; entonces
q10 se baja a q50 o q90 se sube a q50, y se cuenta cuántas filas lo
necesitaron. Una diferencia de redondeo (CLIP_TOLERANCE) no se cuenta.

Cambios de horario: el día de la semana y la hora salen del reloj de Nueva
York. El día de 25 horas las dos 01:00 caen en la misma celda.

La prueba final NO se califica aquí (decisión 4). Sus predicciones se
escriben porque el dashboard y el simulador las necesitan, pero sus valores
reales no se leen: de las filas del tramo que se predice solo se usan la
estación y la hora. Las métricas son solo de validación.

Los datos solo registran demanda atendida (decisión 9).

Para saber después si las salidas siguen al día, junto a las métricas se
escribe reports/linea_base_entrada.json: las fechas del config y una huella
de la historia que se leyó (entrenamiento y validación). Una prueba la vuelve
a calcular y falla si station_hour o el config cambiaron después.

Uso:  python scripts/run_baseline.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import PREDICTIONS, SPLITS, read_contract, validate_file, write_contract
from src.data.datos_config import SPLIT_NAMES, TRAMOS, Datos, Tramo, load_datos
from src.models.metrics import (
    NOMINAL_COVERAGE,
    TARGET_LABELS,
    evaluate,
    federal_holidays,
    format_table,
    local_time,
    markdown_table,
    write_metrics,
)
from src.paths import DATA_PROCESSED, DATOS_YAML, ROOT

MODEL = "baseline"
Q_LOW, Q_HIGH = 0.10, 0.90
CLIP_TOLERANCE = 1e-9  # viajes; menos que esto es redondeo de la interpolación, no un recorte

# "val" -> "validacion", "test" -> "prueba": el tramo de config/datos.yaml de cada split.
TRAMO_OF_SPLIT = {split: name for name, split in SPLIT_NAMES.items()}
TRAMO_LABELS = {"entrenamiento": "entrenamiento", "validacion": "validación", "prueba": "prueba final"}

# Niveles de respaldo, del más específico al más general.
LEVELS = {
    1: "estación, día de la semana y hora",
    2: "estación y hora (toda la historia de la estación)",
    3: "día de la semana y hora con todas las estaciones de la historia",
}

_INPUT_COLUMNS = ["station_id", "ts_hour", "departures", "arrivals"]
# Estadísticas de una celda, en el orden en que se guardan.
_STATS = ["dep_q10", "dep_q50", "dep_q90", "arr_q10", "arr_q50", "arr_q90"]

REPORT_NAME = "metricas_validacion"
FINGERPRINT_NAME = "linea_base_entrada.json"   # huella de la entrada, junto a las métricas
CLEANING_REPORT = "limpieza.json"              # lo escribe build_tables junto a station_hour


class BaselineError(ValueError):
    """La línea base no se pudo calcular. El mensaje dice por qué."""


@dataclass(frozen=True)
class SplitSummary:
    """Qué se hizo al predecir un tramo. No contiene ningún valor real del tramo."""

    split: str
    tramo: Tramo
    span: str                    # el tramo predicho, como texto: nombre y fechas
    history_span: str            # los tramos usados como historia, como texto
    history_days: int            # días de calendario de la historia
    history_rows: int
    rows: int                    # filas predichas
    stations: int
    levels: dict[int, int]       # nivel de respaldo -> filas predichas
    clipped: dict[str, int]      # "dep_q10", "dep_q90", "arr_q10", "arr_q90" -> filas recortadas


# --------------------------------------------------------------------------- #
# Ajuste y predicción
# --------------------------------------------------------------------------- #

def _cells(station_hour: pd.DataFrame) -> pd.DataFrame:
    """Número de celda de cada fila en los tres niveles de respaldo.

    El día de la semana (0 = lunes) y la hora salen del reloj de Nueva York.
    """
    local = local_time(station_hour["ts_hour"])
    station = pd.factorize(station_hour["station_id"])[0]
    dow = local.dt.dayofweek.to_numpy()
    hour = local.dt.hour.to_numpy()
    return pd.DataFrame(
        {1: (station * 7 + dow) * 24 + hour, 2: station * 24 + hour, 3: dow * 24 + hour},
        index=station_hour.index,
    )


def _cell_stats(cell: np.ndarray, values: pd.DataFrame) -> pd.DataFrame:
    """Media y cuantiles 10 % y 90 % de salidas y llegadas por celda (el índice es la celda)."""
    grouped = values.groupby(cell)
    mean, low, high = grouped.mean(), grouped.quantile(Q_LOW), grouped.quantile(Q_HIGH)
    stats = pd.DataFrame(index=mean.index)
    for prefix in ("dep", "arr"):
        stats[f"{prefix}_q10"] = low[prefix]
        stats[f"{prefix}_q50"] = mean[prefix]  # la línea base pone la MEDIA en q50
        stats[f"{prefix}_q90"] = high[prefix]
    return stats[_STATS]


def _span(datos: Datos, names) -> str:
    """Nombre y fechas de uno o más tramos, como texto."""
    return " y ".join(
        f"{TRAMO_LABELS[name]} ({datos.tramos[name].inicio} a {datos.tramos[name].fin})" for name in names
    )


def predict_split(station_hour: pd.DataFrame, datos: Datos, split: str) -> tuple[pd.DataFrame, SplitSummary]:
    """Predicciones de la línea base para un tramo ("val" o "test").

    Devuelve la tabla con las columnas del contrato predictions (una fila por
    cada fila de `station_hour` dentro del tramo) y un resumen de lo hecho.
    En q50 va la MEDIA histórica de la celda, no la mediana.
    """
    if split not in TRAMO_OF_SPLIT:
        raise BaselineError(f"split desconocido: {split!r}. Opciones: {sorted(TRAMO_OF_SPLIT)}")
    missing = [column for column in _INPUT_COLUMNS if column not in station_hour.columns]
    if missing:
        raise BaselineError(f"A station_hour le faltan columnas: {missing}")
    name = TRAMO_OF_SPLIT[split]
    tramo = datos.tramos[name]
    previous = TRAMOS[: TRAMOS.index(name)]  # los tramos anteriores: la historia
    span, history_span = _span(datos, [name]), _span(datos, previous)

    cells = _cells(station_hour)  # revisa de paso que ts_hour traiga zona
    ts = station_hour["ts_hour"]
    in_history = np.zeros(len(station_hour), dtype=bool)
    for earlier in previous:
        in_history |= datos.tramos[earlier].contains(ts).to_numpy()
    in_target = tramo.contains(ts).to_numpy()
    hint = (
        "Revisa que station_hour se haya construido con este mismo config y que meses_crudos cubra ese tramo "
        "(python scripts/build_tables.py)."
    )
    if not in_history.any():
        raise BaselineError(f"No hay filas de station_hour en la historia: {history_span}. {hint}")
    if not in_target.any():
        raise BaselineError(f"No hay filas de station_hour en el tramo que se predice: {span}. {hint}")

    # Valores reales: SOLO de la historia. Del tramo que se predice no se lee ninguno.
    values = pd.DataFrame(
        {
            "dep": station_hour["departures"].to_numpy(dtype=float)[in_history],
            "arr": station_hour["arrivals"].to_numpy(dtype=float)[in_history],
        }
    )

    n = int(in_target.sum())
    picked = np.full((n, len(_STATS)), np.nan)
    level = np.zeros(n, dtype=np.int8)  # 0 = todavía sin valor
    for number in LEVELS:
        stats = _cell_stats(cells[number].to_numpy()[in_history], values)
        found = stats.reindex(cells[number].to_numpy()[in_target]).to_numpy()  # sin historia queda NaN
        use = (level == 0) & ~np.isnan(found[:, 0])
        picked[use] = found[use]
        level[use] = number
    if (level == 0).any():
        raise BaselineError(
            f"{int((level == 0).sum())} filas de {span} no tienen historia ni en el nivel 3: "
            "la historia no trae ese día de la semana y hora en ninguna estación. "
            "El tramo de entrenamiento debe cubrir al menos una semana completa."
        )

    out = station_hour.loc[in_target, ["station_id", "ts_hour"]].reset_index(drop=True)
    out["model"] = MODEL
    out["split"] = split
    clipped: dict[str, int] = {}
    for i, prefix in enumerate(("dep", "arr")):
        q10, q50, q90 = (np.maximum(picked[:, 3 * i + j], 0.0) for j in range(3))
        # La media puede quedar fuera de [q10, q90]. Se ordena moviendo el cuantil, no la media.
        # Solo cuenta como recorte un movimiento mayor que el redondeo de la interpolación.
        clipped[f"{prefix}_q10"] = int((q10 - q50 > CLIP_TOLERANCE).sum())
        clipped[f"{prefix}_q90"] = int((q50 - q90 > CLIP_TOLERANCE).sum())
        out[f"{prefix}_q10"] = np.minimum(q10, q50)
        out[f"{prefix}_q50"] = q50
        out[f"{prefix}_q90"] = np.maximum(q90, q50)
    out = out.sort_values(["station_id", "ts_hour"], kind="stable", ignore_index=True)

    summary = SplitSummary(
        split=split,
        tramo=tramo,
        span=span,
        history_span=history_span,
        history_days=sum((datos.tramos[earlier].fin - datos.tramos[earlier].inicio).days + 1 for earlier in previous),
        history_rows=int(in_history.sum()),
        rows=n,
        stations=int(out["station_id"].nunique()),
        levels={number: int((level == number).sum()) for number in LEVELS},
        clipped=clipped,
    )
    return out[PREDICTIONS.columns], summary


def predict_all(station_hour: pd.DataFrame, datos: Datos) -> tuple[pd.DataFrame, dict[str, SplitSummary]]:
    """Predicciones de la línea base para validación y prueba final, en una sola tabla."""
    tables, summaries = [], {}
    for split in SPLITS:
        table, summaries[split] = predict_split(station_hour, datos, split)
        tables.append(table)
    return pd.concat(tables, ignore_index=True), summaries


def save_predictions(baseline_rows: pd.DataFrame, path: str | Path) -> int:
    """Escribe predictions.parquet con el contrato. Devuelve cuántas filas de otros modelos conservó.

    Si el archivo ya existe, solo se reemplazan las filas de la línea base.
    """
    path = Path(path)
    kept = 0
    table = baseline_rows[PREDICTIONS.columns]
    if path.exists():
        existing = read_contract("predictions", path)
        if list(existing.columns) != PREDICTIONS.columns:
            raise BaselineError(
                f"{path} ya existe y no tiene las columnas del contrato predictions. "
                "No se toca: revísalo o bórralo y vuelve a correr."
            )
        others = existing[existing["model"] != MODEL]
        kept = len(others)
        if kept:
            table = pd.concat([others, table], ignore_index=True)
    write_contract(table, "predictions", path)
    return kept


# --------------------------------------------------------------------------- #
# ¿Las salidas siguen al día?
# --------------------------------------------------------------------------- #

def _dates(tramo: Tramo) -> dict[str, str]:
    return {"inicio": tramo.inicio.isoformat(), "fin": tramo.fin.isoformat()}


def input_fingerprint(station_hour: pd.DataFrame, datos: Datos) -> dict:
    """Huella de lo que leyó la línea base. Sirve para saber si sus salidas siguen al día.

    Guarda las fechas del periodo y de los tres tramos. De entrenamiento y de
    validación, que son la historia, guarda además cuántas filas hay, la suma
    de salidas y de llegadas y un número de control de estación, hora, salidas
    y llegadas (no depende del orden de las filas).

    De la prueba final solo van las fechas: ningún valor real (decisión 4).
    Basta pasar las filas anteriores a la prueba final; las demás no se leen.
    """
    out: dict = {"periodo": _dates(datos.periodo), "tramos": {name: _dates(datos.tramos[name]) for name in TRAMOS}}
    for name in TRAMOS[:-1]:  # la historia: todos los tramos menos la prueba final
        rows = station_hour.loc[datos.tramos[name].contains(station_hour["ts_hour"]), _INPUT_COLUMNS]
        out["tramos"][name].update(
            filas=len(rows),
            salidas=int(rows["departures"].sum()),
            llegadas=int(rows["arrivals"].sum()),
            control=int(pd.util.hash_pandas_object(rows, index=False).to_numpy().sum()),
        )
    return out


def _check_built_period(data_dir: Path, datos: Datos, config_path: str) -> None:
    """Falla si build_tables construyó las tablas de `data_dir` con otro periodo.

    build_tables deja el periodo que usó en reports/limpieza.json, junto a las
    tablas. Si ese archivo no está (por ejemplo, una carpeta de prueba), no hay
    con qué comparar y no se revisa.
    """
    cleaning = data_dir / "reports" / CLEANING_REPORT
    if not cleaning.exists():
        return
    try:
        built = json.loads(cleaning.read_text(encoding="utf-8"))["config"]["periodo"]
        built = {"inicio": str(built["inicio"]), "fin": str(built["fin"])}
    except (ValueError, KeyError, TypeError):
        raise BaselineError(
            f"No se pudo leer el periodo en {_short(cleaning)}. Vuelve a correr: python scripts/build_tables.py"
        ) from None
    mine = _dates(datos.periodo)
    if built != mine:
        raise BaselineError(
            f"{_short(data_dir / 'station_hour.parquet')} se construyó con el periodo {built['inicio']} a "
            f"{built['fin']} y {config_path} dice {mine['inicio']} a {mine['fin']}. "
            "Primero corre: python scripts/build_tables.py"
        )


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #

def _levels_table(summaries: dict[str, SplitSummary]) -> str:
    header = ["tramo", "filas predichas", "estaciones", "filas de historia", "nivel 1", "nivel 2", "nivel 3"]
    rows = [
        [s.split, f"{s.rows:,}", f"{s.stations:,}", f"{s.history_rows:,}", *(f"{s.levels[k]:,}" for k in LEVELS)]
        for s in summaries.values()
    ]
    return markdown_table(header, rows)


def _clipping_table(summaries: dict[str, SplitSummary]) -> str:
    header = ["tramo", "objetivo", "filas predichas", "q10 bajado a q50", "q90 subido a q50"]
    rows = []
    for s in summaries.values():
        for prefix, label in TARGET_LABELS.items():
            cells = [f"{s.clipped[f'{prefix}_{q}']:,} ({100 * s.clipped[f'{prefix}_{q}'] / s.rows:.2f} %)" for q in ("q10", "q90")]
            rows.append([s.split, label, f"{s.rows:,}", *cells])
    return markdown_table(header, rows, text_columns=2)


def _weekday_holidays(tramo: Tramo) -> str:
    """Festivos federales de lunes a viernes dentro del tramo, como texto."""
    found = [
        f"{day} ({name})"
        for day, name in sorted(federal_holidays(tramo.inicio.year, tramo.fin.year).items())
        if tramo.inicio <= day <= tramo.fin and day.weekday() < 5
    ]
    return ", ".join(found) if found else "ninguno"


def _theoretical_coverage(n: int) -> float:
    """Cobertura teórica, en %, del intervalo empírico de 10 % a 90 % de `n` valores ante un valor nuevo.

    Es la cuenta para una variable continua: nominal por (n - 1) / (n + 1).
    """
    return 100 * NOMINAL_COVERAGE * max(n - 1, 0) / (n + 1)


def build_report(summaries: dict[str, SplitSummary], metrics: pd.DataFrame) -> str:
    """Texto de metricas_validacion.md. No lleva fecha ni tiempos: dos corridas iguales dan el mismo archivo."""
    val, test = summaries["val"], summaries["test"]
    levels = "\n".join(f"- Nivel {number}: {text}." for number, text in LEVELS.items())
    few, many = val.history_days // 7, math.ceil(val.history_days / 7)
    weeks = f"{few} valores" if few == many else f"{few} o {many} valores"
    low, high = (f"{_theoretical_coverage(n):.0f}" for n in (few, many))
    theory = f"{low} %" if low == high else f"{low} a {high} %"
    return f"""# Línea base: métricas en validación

Este archivo lo escribe `python scripts/run_baseline.py`. No se edita a mano.

## Qué se calificó

- Modelo: línea base. Promedio histórico por estación, día de la semana y hora local de Nueva York (decisión 3).
- Historia usada para predecir validación: {val.history_span}, {val.history_rows:,} filas de estación y hora.
- Tramo calificado: {val.span}. {val.rows:,} filas de estación y hora, {val.stations:,} estaciones.
- Festivos de lunes a viernes en el tramo: {_weekday_holidays(val.tramo)}.
- La {test.span} NO se calificó.

Unidades: MAE, RMSE y ancho están en viajes por estación y hora. WAPE y cobertura van aquí en porcentaje
(en el Parquet van como proporción, de 0 a 1). La cobertura nominal del intervalo de q10 a q90 es {100 * NOMINAL_COVERAGE:.0f} %.

## General

{format_table(metrics, "general", "grupo")}

## Por tipo de día

{format_table(metrics, "tipo_de_dia", "tipo de día")}

"festivo" es un festivo federal que cae de lunes a viernes. Sábado y domingo son siempre "fin de semana".

## Por hora del día (hora local de Nueva York)

{format_table(metrics, "hora", "hora")}

El desglose por estación está en `{REPORT_NAME}.parquet`, en las filas con `group_by = "estacion"`.

## Respaldo cuando una celda no tiene historia

{levels}

{_levels_table(summaries)}

Las filas de nivel 3 no son un pronóstico de esa estación: llevan el promedio del sistema en ese día de la semana
y hora, puesto como marcador donde la estación no tiene historia.

## Recortes para ordenar los cuantiles

Filas donde la media quedó fuera de los cuantiles empíricos y hubo que mover q10 o q90 hasta la media.
Una diferencia que es solo redondeo del cálculo no cuenta como recorte.

{_clipping_table(summaries)}

Con 10 valores o menos en la celda, el cuantil 90 % nunca queda debajo de la media ni el 10 % arriba.
El recorte solo puede aparecer en celdas con 11 valores o más: por eso depende de cuántas semanas tiene la historia.

## Cómo leer esto

- Los datos solo registran demanda atendida: viajes que sí ocurrieron. No miden la demanda que no se pudo atender.
- Un cero puede ser una hora sin demanda, una estación vacía, llena o cerrada. El dato no los distingue.
- La tabla tiene filas solo entre el primer y el último viaje de cada estación. Las horas fuera de ese rango no se predicen ni se califican.
- En la línea base q50 es la media histórica de la celda, no la mediana. MAE, RMSE y WAPE se calculan contra esa media.
- La media no es el pronóstico que minimiza el error absoluto: ese es la mediana. Al comparar MAE o WAPE con el q50 de otro modelo que sí sea una mediana, parte de la diferencia viene de eso y no del modelo. Comparar también RMSE.
- q10 y q90 son los cuantiles empíricos 10 % y 90 % de la misma celda. Cobertura es la proporción de filas con q10 <= real <= q90.
- La historia de validación tiene {val.history_days} días. Una celda de nivel 1 tiene un valor por semana: {weeks}, o menos si la estación empezó después. q10 y q90 salen de esos pocos valores.
- Una cobertura general cercana a la nominal no quiere decir que el intervalo esté calibrado: cuenta como cubierta toda fila con real = q10 = 0. Con {weeks} por celda, el intervalo empírico de 10 % a 90 % cubre en teoría cerca de {theory} de los valores nuevos cuando la demanda no es cero (algo más con conteos pequeños). Ver la tabla por hora.
- Donde hubo recorte, q10 o q90 valen lo mismo que q50.
- WAPE = suma de |real - q50| entre suma de real. Aparece "n/d" cuando la suma de real es 0.
- WAPE puede pasar de 100 %: ocurre en horas con muy pocos viajes, donde la suma de errores supera a la suma de viajes. Quiere decir que predecir cero en todas las filas de ese grupo tendría menor MAE.
- La historia de validación es fija: no se actualiza con los días de validación ya transcurridos.
- La prueba final no se ha calificado. Sus predicciones están en predictions.parquet (split = "test"), pero no se compararon con los valores reales. Se usa una sola vez (decisión 4).
"""


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def _short(path: Path) -> str:
    """Ruta relativa a la raíz del repositorio, para mensajes más cortos."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_baseline.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python scripts/run_baseline.py [opciones]",
        description=(
            "Línea base: promedio histórico por estación, día de la semana y hora.\n"
            "Escribe predictions.parquet (validación y prueba final) y las métricas de VALIDACIÓN.\n"
            "La prueba final no se califica."
        ),
    )
    options = parser.add_argument_group("opciones")
    options.add_argument(
        "--datos", metavar="DIR", default=str(DATA_PROCESSED),
        help="carpeta con station_hour.parquet (por defecto data/processed)",
    )
    options.add_argument(
        "--salida", metavar="DIR",
        help=(
            "carpeta donde se escriben predictions.parquet y, en reports/, metricas_validacion.parquet y .md "
            "y linea_base_entrada.json (por defecto, la misma de --datos)"
        ),
    )
    options.add_argument(
        "--config", metavar="FILE", default=str(DATOS_YAML),
        help="archivo con el periodo y los tramos (por defecto config/datos.yaml)",
    )
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Devuelve 0 si todo salió bien y 1 si no se pudo correr."""
    args = build_parser().parse_args(argv)
    started = time.time()
    data_dir = Path(args.datos).expanduser()
    out_dir = Path(args.salida).expanduser() if args.salida else data_dir
    source = data_dir / "station_hour.parquet"
    predictions_path = out_dir / PREDICTIONS.filename
    metrics_path = out_dir / "reports" / f"{REPORT_NAME}.parquet"
    report_path = out_dir / "reports" / f"{REPORT_NAME}.md"
    fingerprint_path = out_dir / "reports" / FINGERPRINT_NAME

    if not source.exists():
        print(f"ERROR. No existe {_short(source)}. Primero corre: python scripts/build_tables.py")
        return 1
    try:
        datos = load_datos(args.config)
        _check_built_period(data_dir, datos, args.config)
        problems = validate_file(source, "station_hour")
        if problems:
            raise BaselineError(f"{_short(source)} no cumple el contrato station_hour:\n- " + "\n- ".join(problems))
        station_hour = read_contract("station_hour", source)
        predictions, summaries = predict_all(station_hour, datos)

        # Solo validación. A las métricas no llega ni una fila real de la prueba final.
        val = datos.tramos[TRAMO_OF_SPLIT["val"]]
        metrics = evaluate(
            predictions,
            station_hour[val.contains(station_hour["ts_hour"])],
            model=MODEL,
            split="val",
            prueba_final=datos.tramos[TRAMO_OF_SPLIT["test"]],
        )
        report = build_report(summaries, metrics)
        fingerprint = json.dumps(input_fingerprint(station_hour, datos), indent=2) + "\n"

        kept = save_predictions(predictions, predictions_path)
        write_metrics(metrics, metrics_path)
        report_path.write_text(report, encoding="utf-8")
        fingerprint_path.write_text(fingerprint, encoding="utf-8")
    except (ValueError, OSError) as error:  # config, contrato, línea base o archivo ilegible
        print(f"ERROR. {error}")
        return 1

    tramos = "; ".join(_span(datos, [name]) for name in TRAMOS)
    print("Línea base: promedio histórico por estación, día de la semana y hora. q50 es una media, no una mediana.")
    print(f"Entrada: {_short(source)}, {len(station_hour):,} filas, {station_hour['station_id'].nunique():,} estaciones.")
    print(f"Tramos: {tramos}.")
    outside = len(station_hour) - int(datos.periodo.contains(station_hour["ts_hour"]).sum())
    if outside:
        print(f"AVISO: {outside:,} filas de station_hour caen fuera del periodo y no se usaron.")

    print("\nFilas predichas y nivel de respaldo usado:")
    print("\n".join(f"  nivel {number}: {text}" for number, text in LEVELS.items()))
    print(_levels_table(summaries))
    print("\nFilas donde hubo que mover q10 o q90 hasta la media para ordenar los cuantiles:")
    print(_clipping_table(summaries))

    print("\nMétricas en VALIDACIÓN. La prueba final no se calificó (decisión 4).")
    print("Solo demanda atendida. MAE, RMSE y ancho en viajes por estación y hora.")
    for title, group_by, label in (
        ("General", "general", "grupo"),
        ("Por tipo de día", "tipo_de_dia", "tipo de día"),
        ("Por hora local", "hora", "hora"),
    ):
        print(f"\n{title}")
        print(format_table(metrics, group_by, label))

    print("\nEscrito:")
    counts = ", ".join(f"{split} {summaries[split].rows:,}" for split in summaries)
    print(f"  {_short(predictions_path)}: {len(predictions):,} filas de la línea base ({counts}); "
          f"se conservaron {kept:,} filas de otros modelos.")
    print(f"  {_short(metrics_path)}")
    print(f"  {_short(report_path)}")
    print(f"  {_short(fingerprint_path)}")
    print(f"Tiempo: {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
