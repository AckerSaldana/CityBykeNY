"""Modelo del proyecto (decisión 3): UN LightGBM global con cuantiles 10, 50 y 90.

Seis modelos (boosters): salidas y llegadas, cada una con q10, q50 y q90. Es
"global" porque un mismo modelo sirve a todas las estaciones; la estación
entra por sus variables, no hay un modelo por estación.

Es un pronóstico de un día para otro: las variables de una fila cuyo día local
es D usan solo filas de station_hour de días anteriores a D (decisión 5), más
el calendario de la propia fila y las variables fijas de la estación.

Variables (lista en FEATURES):

- src/features/historia.py, constructores registrados y revisados por
  tests/test_no_leakage.py: rezagos a la misma hora de 1, 2, 7 y 14 días;
  promedio y cuantiles de los 7 y 28 días previos; mismo día de la semana en
  las 4 semanas previas; totales de los días previos; calendario.
- src/features/estaticas.py: coordenadas y capacidad de stations.parquet y
  promedios de la estación calculados SOLO con el tramo de ajuste.

Tramos (decisión 4), con las fechas de config/datos.yaml:

- split "val": se ajusta con entrenamiento. El número de rondas de cada modelo
  se fija por parada temprana contra validación. Todo lo que se decidió
  (variables, hiperparámetros, rondas) se decidió con entrenamiento y
  validación; la prueba final no entra a esa corrida: se corta la tabla antes.
- split "test": se vuelve a ajustar con entrenamiento más validación, con los
  hiperparámetros y las rondas ya fijados. Las filas se escriben y NO se
  califican. De la prueba final nunca se lee un valor como objetivo (etiqueta)
  ni para una métrica ni para decidir algo. Sí entran como VARIABLES los
  conteos de días de la prueba ya transcurridos (el rezago de ayer para
  predecir hoy): es lo que permite la decisión 5 y lo que tendría el operador.
  Los conteos del último día de la prueba no se usan para nada.

Antes de escribir (CLAUDE.md): los cuantiles se recortan en 0 y se ordenan
q10 <= q50 <= q90 por fila. Las métricas se calculan sobre esos valores.

Aquí q50 es una MEDIANA condicional. En la línea base q50 es una media: al
comparar MAE o WAPE, parte de la diferencia viene de eso. Comparar también RMSE.

Los datos solo registran demanda atendida (decisión 9).

Uso:  python scripts/run_lgbm.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.contracts import PREDICTIONS, SPLITS, read_contract, validate_file, write_contract
from src.data.datos_config import TRAMOS, Datos, Tramo, load_datos
from src.features.estaticas import STATIC_COLUMNS, station_statics
from src.features.historia import (
    CALENDAR_COLUMNS,
    LAG_COLUMNS,
    TOTAL_COLUMNS,
    calendario,
    rezagos_misma_hora,
    totales_dias_previos,
)
from src.models import baseline
from src.models.baseline import TRAMO_OF_SPLIT, _check_built_period, _short, _span, input_fingerprint
from src.models.metrics import (
    FINAL_SPLIT,
    FRANJA_NOTE,
    KEYS,
    NOMINAL_COVERAGE,
    TARGET_LABELS,
    TARGETS,
    FinalTestError,
    _number,
    _percent,
    evaluate,
    format_table,
    local_time,
    markdown_table,
    report_bands,
    write_metrics,
)
from src.paths import DATA_PROCESSED, DATOS_YAML

MODEL = "lgbm"
BASELINE = baseline.MODEL
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
BOOSTER_NAMES = [f"{prefix}_{q}" for prefix in TARGETS for q in QUANTILES]  # el orden de las columnas del contrato

FEATURES = LAG_COLUMNS + TOTAL_COLUMNS + CALENDAR_COLUMNS + STATIC_COLUMNS

# Hiperparámetros. Se eligieron una vez con entrenamiento y validación (salidas, q50); ver metricas_lgbm.md.
SEED = 7
NUM_THREADS = 8
PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
MAX_ROUNDS = 1000       # tope de rondas por modelo
EARLY_STOPPING = 30     # rondas sin mejorar en validación antes de parar

REPORT_NAME = "metricas_lgbm"
METRICS_NAME = "metricas_validacion"          # el mismo Parquet de la línea base: una fila por modelo
FINGERPRINT_NAME = "lgbm_entrada.json"        # huella de la entrada y ajustes del modelo


class LgbmError(ValueError):
    """El modelo no se pudo ajustar o guardar. El mensaje dice por qué."""


@dataclass(frozen=True)
class SplitSummary:
    """Qué se hizo al predecir un tramo. No contiene ningún valor real de la prueba final."""

    split: str
    tramo: Tramo
    span: str
    history_span: str
    history_rows: int
    rows: int
    stations: int
    rows_without_history: int       # filas predichas sin ninguna fila de la estación a esa hora en los 28 días previos
    rounds: dict[str, int]          # "dep_q10" ... -> rondas del modelo
    negatives: dict[str, int]       # columna -> filas con valor negativo antes de recortar
    crossed: dict[str, int]         # "dep", "arr" -> filas con cuantiles cruzados después de recortar en 0


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #

def feature_matrix(station_hour: pd.DataFrame, history: np.ndarray, stations: pd.DataFrame | None) -> pd.DataFrame:
    """Matriz de variables (columnas FEATURES), una fila por fila de `station_hour`, en su mismo orden.

    `history` marca las filas del tramo de ajuste: solo con ellas se calculan
    los promedios fijos de cada estación. Los rezagos de cada fila miran días
    anteriores al suyo (lo prueba tests/test_no_leakage.py).
    """
    table = station_hour.reset_index(drop=True)
    parts = [
        rezagos_misma_hora(table)[LAG_COLUMNS],
        totales_dias_previos(table)[TOTAL_COLUMNS],
        calendario(table)[CALENDAR_COLUMNS],
        station_statics(table, table[np.asarray(history)], stations),
    ]
    return pd.concat(parts, axis=1)[FEATURES].astype(np.float32)


def _labels(station_hour: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Valores reales (objetivos) de las filas marcadas. Es el ÚNICO lugar donde el modelo lee un objetivo."""
    return {prefix: station_hour[column].to_numpy(dtype=float)[rows] for prefix, column in TARGETS.items()}


# --------------------------------------------------------------------------- #
# Ajuste, predicción y corrección de cuantiles
# --------------------------------------------------------------------------- #

def fit_boosters(
    x_fit: np.ndarray,
    y_fit: dict[str, np.ndarray],
    *,
    rounds: dict[str, int] | None = None,
    x_eval: np.ndarray | None = None,
    y_eval: dict[str, np.ndarray] | None = None,
    params: dict | None = None,
    num_threads: int = NUM_THREADS,
    max_rounds: int = MAX_ROUNDS,
    early_stopping: int = EARLY_STOPPING,
) -> tuple[dict[str, lgb.Booster], dict[str, int]]:
    """Ajusta los seis modelos. Devuelve los modelos y las rondas de cada uno.

    Con `rounds` se usan esas rondas tal cual (reajuste con rondas ya fijadas).
    Sin `rounds` hacen falta `x_eval` y `y_eval`: cada modelo para cuando deja
    de mejorar ahí y se queda con su mejor ronda.
    """
    if rounds is None and (x_eval is None or y_eval is None):
        raise LgbmError("Sin rondas fijadas hace falta un tramo de validación para la parada temprana.")
    settings = {**PARAMS, **(params or {}), "num_threads": num_threads}
    boosters: dict[str, lgb.Booster] = {}
    used: dict[str, int] = {}
    for prefix in TARGETS:
        # Un solo conjunto por objetivo: sus tres cuantiles comparten los mismos datos.
        train = lgb.Dataset(x_fit, y_fit[prefix], feature_name=FEATURES, params=settings, free_raw_data=False)
        valid = None if rounds is not None else lgb.Dataset(x_eval, y_eval[prefix], reference=train)
        for q, alpha in QUANTILES.items():
            name = f"{prefix}_{q}"
            if rounds is not None:
                booster = lgb.train({**settings, "alpha": alpha}, train, num_boost_round=int(rounds[name]))
                used[name] = int(rounds[name])
            else:
                booster = lgb.train(
                    {**settings, "alpha": alpha},
                    train,
                    num_boost_round=max_rounds,
                    valid_sets=[valid],
                    callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
                )
                used[name] = int(booster.best_iteration or booster.current_iteration())
            boosters[name] = booster
    return boosters, used


def predict_raw(boosters: dict[str, lgb.Booster], rounds: dict[str, int], x: np.ndarray) -> dict[str, np.ndarray]:
    """Salida de los seis modelos, sin corregir."""
    return {name: boosters[name].predict(x, num_iteration=rounds[name]) for name in BOOSTER_NAMES}


def postprocess(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    """Recorta en 0 y ordena q10 <= q50 <= q90 por fila.

    Devuelve los valores corregidos, cuántas filas eran negativas en cada
    columna y cuántas tenían cuantiles cruzados (ya recortadas) por objetivo.
    """
    fixed: dict[str, np.ndarray] = {}
    negatives = {name: int((raw[name] < 0).sum()) for name in BOOSTER_NAMES}
    crossed: dict[str, int] = {}
    for prefix in TARGETS:
        names = [f"{prefix}_{q}" for q in QUANTILES]
        values = np.column_stack([np.maximum(raw[name], 0.0) for name in names])
        ordered = np.sort(values, axis=1)
        crossed[prefix] = int((values != ordered).any(axis=1).sum())
        for i, name in enumerate(names):
            fixed[name] = ordered[:, i]
    return fixed, negatives, crossed


def predict_split(
    station_hour: pd.DataFrame,
    stations: pd.DataFrame | None,
    datos: Datos,
    split: str,
    *,
    rounds: dict[str, int] | None = None,
    params: dict | None = None,
    num_threads: int = NUM_THREADS,
) -> tuple[pd.DataFrame, SplitSummary, dict[str, lgb.Booster]]:
    """Predicciones de LightGBM para un tramo ("val" o "test"), con las columnas del contrato predictions.

    "val": ajusta con entrenamiento y fija las rondas con validación.
    "test": exige `rounds` (las que salieron de "val"), ajusta con
    entrenamiento más validación y no lee ningún objetivo de la prueba final.
    """
    if split not in TRAMO_OF_SPLIT:
        raise LgbmError(f"split desconocido: {split!r}. Opciones: {sorted(TRAMO_OF_SPLIT)}")
    if split == FINAL_SPLIT and rounds is None:
        raise LgbmError(
            "Para la prueba final las rondas deben venir ya fijadas con validación (decisión 4). "
            "Corre primero el split 'val' y pasa sus rondas."
        )
    needed = ["station_id", "ts_hour", *TARGETS.values()]
    missing = [column for column in needed if column not in station_hour.columns]
    if missing:
        raise LgbmError(f"A station_hour le faltan columnas: {missing}")
    name = TRAMO_OF_SPLIT[split]
    tramo = datos.tramos[name]
    previous = TRAMOS[: TRAMOS.index(name)]
    span, history_span = _span(datos, [name]), _span(datos, previous)

    # Nada posterior al tramo que se predice entra a esta corrida: al predecir validación no hay prueba final.
    table = station_hour.loc[station_hour["ts_hour"] < tramo.end, needed].reset_index(drop=True)
    ts = table["ts_hour"]
    in_history = np.zeros(len(table), dtype=bool)
    for earlier in previous:
        in_history |= datos.tramos[earlier].contains(ts).to_numpy()
    in_target = tramo.contains(ts).to_numpy()
    hint = "Revisa que station_hour se haya construido con este mismo config (python scripts/build_tables.py)."
    if not in_history.any():
        raise LgbmError(f"No hay filas de station_hour en la historia: {history_span}. {hint}")
    if not in_target.any():
        raise LgbmError(f"No hay filas de station_hour en el tramo que se predice: {span}. {hint}")

    x = feature_matrix(table, in_history, stations)
    x_target = x.to_numpy()[in_target]
    if split == FINAL_SPLIT:
        # Reajuste con rondas fijas. Los objetivos salen SOLO de la historia.
        fit_rows = in_history
        boosters, used = fit_boosters(
            x.to_numpy()[fit_rows], _labels(table, fit_rows), rounds=rounds, params=params, num_threads=num_threads
        )
    else:
        # Las rondas se fijan con validación: aquí sí se leen sus objetivos, y solo para eso.
        fit_rows = datos.tramos["entrenamiento"].contains(ts).to_numpy()
        boosters, used = fit_boosters(
            x.to_numpy()[fit_rows], _labels(table, fit_rows),
            x_eval=x_target, y_eval=_labels(table, in_target), params=params, num_threads=num_threads,
        )
    fixed, negatives, crossed = postprocess(predict_raw(boosters, used, x_target))

    out = table.loc[in_target, ["station_id", "ts_hour"]].reset_index(drop=True)
    out["model"] = MODEL
    out["split"] = split
    for column in BOOSTER_NAMES:
        out[column] = fixed[column]
    no_history = int((x.loc[in_target, "dias_con_historia_28d"].to_numpy() == 0).sum())
    out = out.sort_values(["station_id", "ts_hour"], kind="stable", ignore_index=True)
    summary = SplitSummary(
        split=split, tramo=tramo, span=span, history_span=history_span,
        history_rows=int(fit_rows.sum()), rows=int(in_target.sum()), stations=int(out["station_id"].nunique()),
        rows_without_history=no_history, rounds=used, negatives=negatives, crossed=crossed,
    )
    return out[PREDICTIONS.columns], summary, boosters


def predict_all(
    station_hour: pd.DataFrame,
    stations: pd.DataFrame | None,
    datos: Datos,
    *,
    params: dict | None = None,
    num_threads: int = NUM_THREADS,
) -> tuple[pd.DataFrame, dict[str, SplitSummary]]:
    """Predicciones para validación y prueba final. Las rondas de la prueba final son las fijadas en validación."""
    val, summary_val, _ = predict_split(station_hour, stations, datos, "val", params=params, num_threads=num_threads)
    test, summary_test, _ = predict_split(
        station_hour, stations, datos, "test", rounds=summary_val.rounds, params=params, num_threads=num_threads
    )
    assert list(SPLITS) == ["val", "test"]
    return pd.concat([val, test], ignore_index=True), {"val": summary_val, "test": summary_test}


def save_predictions(lgbm_rows: pd.DataFrame, path: str | Path) -> int:
    """Escribe predictions.parquet con el contrato. Devuelve cuántas filas de otros modelos conservó.

    Si el archivo ya existe, solo se reemplazan las filas de LightGBM: las de la línea base se quedan.
    """
    path = Path(path)
    kept = 0
    table = lgbm_rows[PREDICTIONS.columns]
    if path.exists():
        existing = read_contract("predictions", path)
        if list(existing.columns) != PREDICTIONS.columns:
            raise LgbmError(
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
# Pérdida de cuantil (pinball) y comparación con la línea base
# --------------------------------------------------------------------------- #

def pinball_loss(actual, predicted, alpha: float) -> float:
    """Pérdida de cuantil media: alpha * (real - pred) si el real queda arriba, (1 - alpha) * (pred - real) si abajo."""
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def quantile_scores(
    predictions: pd.DataFrame, station_hour: pd.DataFrame, *, model: str, split: str, prueba_final: Tramo
) -> pd.DataFrame:
    """Pérdida de cuantil de q10, q50 y q90 y proporción de reales fuera del intervalo, por objetivo.

    Como `evaluate`, se niega a calificar la prueba final: ni el split "test"
    ni filas que caigan en sus fechas.
    """
    rows = predictions[(predictions["model"] == model) & (predictions["split"] == split)]
    if split == FINAL_SPLIT or (local_time(rows["ts_hour"]) >= pd.Timestamp(prueba_final.inicio)).any():
        raise FinalTestError("La prueba final no se califica aquí (decisión 4).")
    scored = rows.merge(station_hour[KEYS + list(TARGETS.values())], on=KEYS, how="inner", validate="one_to_one")
    if len(scored) != len(rows) or len(scored) != len(station_hour):
        raise ValueError(f"Las filas de model={model!r} no son las mismas que las de station_hour en el tramo.")
    out = []
    for prefix, column in TARGETS.items():
        actual = scored[column].to_numpy(dtype=float)
        row = {"model": model, "target": prefix, "n": len(scored)}
        for q, alpha in QUANTILES.items():
            row[f"pinball_{q}"] = pinball_loss(actual, scored[f"{prefix}_{q}"], alpha)
        row["below_q10"] = float((actual < scored[f"{prefix}_q10"].to_numpy()).mean())
        row["above_q90"] = float((actual > scored[f"{prefix}_q90"].to_numpy()).mean())
        out.append(row)
    return pd.DataFrame(out)


_COMPARED = (("mae", "MAE", _number), ("rmse", "RMSE", _number), ("wape", "WAPE", _percent),
             ("coverage", "cobertura", _percent), ("mean_width", "ancho", _number))


def _paired(base: pd.DataFrame, model: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Las métricas de los dos modelos lado a lado, una fila por objetivo y grupo, en el orden de `model`."""
    keys = ["target", "group"]
    left = model[model["group_by"] == group_by]
    right = base[base["group_by"] == group_by]
    pair = left.merge(right, on=keys, how="left", suffixes=("_lgbm", "_base"), validate="one_to_one")
    if pair["n_base"].isna().any() or (pair["n_base"] != pair["n_lgbm"]).any():
        raise ValueError(f"Los dos modelos no se calificaron sobre las mismas filas en el desglose {group_by!r}.")
    return pair


def comparison_table(base: pd.DataFrame, model: pd.DataFrame, group_by: str, label: str) -> str:
    """Tabla en Markdown: línea base y LightGBM en las mismas filas, por objetivo y grupo."""
    header = ["objetivo", label, "filas"]
    for _, title, _ in _COMPARED:
        header += [f"{title} base", f"{title} lgbm"]
    rows = []
    for row in _paired(base, model, group_by).itertuples():
        cells = [TARGET_LABELS[row.target], str(row.group), f"{int(row.n_lgbm):,}"]
        for column, _, fmt in _COMPARED:
            cells += [fmt(getattr(row, f"{column}_base")), fmt(getattr(row, f"{column}_lgbm"))]
        rows.append(cells)
    return markdown_table(header, rows, text_columns=2)


def pinball_table(scores: pd.DataFrame) -> str:
    header = ["modelo", "objetivo", "filas", "pinball q10", "pinball q50", "pinball q90",
              "reales bajo q10", "reales sobre q90", "cobertura"]
    rows = [
        [
            row.model, TARGET_LABELS[row.target], f"{row.n:,}",
            f"{row.pinball_q10:.4f}", f"{row.pinball_q50:.4f}", f"{row.pinball_q90:.4f}",
            _percent(row.below_q10), _percent(row.above_q90), _percent(1 - row.below_q10 - row.above_q90),
        ]
        for row in scores.itertuples()
    ]
    return markdown_table(header, rows, text_columns=2)


def worse_lines(base: pd.DataFrame, model: pd.DataFrame, scores: pd.DataFrame) -> list[str]:
    """Frases que dicen dónde LightGBM sale peor que la línea base. Lista vacía si en ningún lado."""
    lines: list[str] = []
    names = {"general": "General", "tipo_de_dia": "Tipo de día", "hora": "Hora", "franja": "Franja"}
    for group_by, title in names.items():
        pair = _paired(base, model, group_by)
        for prefix, label in TARGET_LABELS.items():
            part = pair[pair["target"] == prefix]
            for column, metric in (("mae", "MAE"), ("rmse", "RMSE")):
                worse = part[part[f"{column}_lgbm"] > part[f"{column}_base"]]
                if len(worse):
                    detail = ", ".join(
                        f"{row.group} ({getattr(row, column + '_lgbm'):.3f} contra {getattr(row, column + '_base'):.3f})"
                        for row in worse.itertuples()
                    )
                    lines.append(f"{title}, {label}, {metric} mayor que la línea base en: {detail}.")
            far = part[(part["coverage_lgbm"] - NOMINAL_COVERAGE).abs() > (part["coverage_base"] - NOMINAL_COVERAGE).abs()]
            if len(far):
                detail = ", ".join(
                    f"{row.group} ({_percent(row.coverage_lgbm)} contra {_percent(row.coverage_base)})" for row in far.itertuples()
                )
                lines.append(
                    f"{title}, {label}, cobertura más lejos de {100 * NOMINAL_COVERAGE:.0f} % que la línea base en: {detail}."
                )
    pair = _paired(base, model, "estacion")
    for prefix, label in TARGET_LABELS.items():
        part = pair[pair["target"] == prefix]
        for column, metric in (("mae", "MAE"), ("rmse", "RMSE")):
            count = int((part[f"{column}_lgbm"] > part[f"{column}_base"]).sum())
            if count:
                lines.append(f"Por estación, {label}: {count:,} de {len(part):,} estaciones tienen {metric} mayor que con la línea base.")
    by_model = {(row.model, row.target): row for row in scores.itertuples()}
    for prefix, label in TARGET_LABELS.items():
        for q in QUANTILES:
            ours, theirs = getattr(by_model[(MODEL, prefix)], f"pinball_{q}"), getattr(by_model[(BASELINE, prefix)], f"pinball_{q}")
            if ours > theirs:
                lines.append(f"Pérdida de cuantil {q}, {label}: {ours:.4f} contra {theirs:.4f} de la línea base.")
    return lines


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #

def _rounds_table(summaries: dict[str, SplitSummary]) -> str:
    header = ["tramo", "filas de ajuste", "filas predichas", "estaciones", "sin historia de 28 días", *BOOSTER_NAMES]
    rows = [
        [s.split, f"{s.history_rows:,}", f"{s.rows:,}", f"{s.stations:,}", f"{s.rows_without_history:,}",
         *(f"{s.rounds[name]:,}" for name in BOOSTER_NAMES)]
        for s in summaries.values()
    ]
    return markdown_table(header, rows)


def _fixes_table(summaries: dict[str, SplitSummary]) -> str:
    header = ["tramo", "objetivo", "filas predichas", "q10 negativo", "q50 negativo", "q90 negativo", "cuantiles cruzados"]
    rows = []
    for s in summaries.values():
        for prefix, label in TARGET_LABELS.items():
            cells = [f"{s.negatives[f'{prefix}_{q}']:,}" for q in QUANTILES]
            rows.append([s.split, label, f"{s.rows:,}", *cells, f"{s.crossed[prefix]:,} ({100 * s.crossed[prefix] / s.rows:.2f} %)"])
    return markdown_table(header, rows, text_columns=2)


def build_report(
    summaries: dict[str, SplitSummary],
    metrics: pd.DataFrame,
    base_metrics: pd.DataFrame,
    scores: pd.DataFrame,
    params: dict,
) -> str:
    """Texto de metricas_lgbm.md. No lleva fecha ni tiempos: dos corridas iguales dan el mismo archivo."""
    val, test = summaries["val"], summaries["test"]
    worse = worse_lines(base_metrics, metrics, scores)
    worse_text = "\n".join(f"- {line}" for line in worse) if worse else (
        "- En ningún grupo de las tablas de arriba (general, tipo de día, hora, franja) ni en la pérdida de cuantil."
    )
    shown = ", ".join(f"{key} = {value}" for key, value in params.items() if key not in ("verbose", "objective"))
    return f"""# LightGBM con cuantiles: métricas en validación y comparación con la línea base

Este archivo lo escribe `python scripts/run_lgbm.py`. No se edita a mano.

## Qué se calificó

- Modelo: un LightGBM global con objetivo de cuantil. Seis modelos: salidas y llegadas, cada una con q10, q50 y q90 (decisión 3).
- Ajuste para validación: {val.history_span}, {val.history_rows:,} filas de estación y hora. No se tomó una muestra: entran todas.
- Tramo calificado: {val.span}. {val.rows:,} filas de estación y hora, {val.stations:,} estaciones.
- Las dos tablas comparan los dos modelos sobre las mismas {val.rows:,} filas. "base" es la línea base y "lgbm" es LightGBM.
- La {test.span} NO se calificó.

Unidades: MAE, RMSE, ancho y pérdida de cuantil están en viajes por estación y hora. WAPE y cobertura van en porcentaje.
La cobertura nominal del intervalo de q10 a q90 es {100 * NOMINAL_COVERAGE:.0f} %.

## General

{comparison_table(base_metrics, metrics, "general", "grupo")}

## Por tipo de día

{comparison_table(base_metrics, metrics, "tipo_de_dia", "tipo de día")}

"festivo" es un festivo federal que cae de lunes a viernes. Sábado y domingo son siempre "fin de semana".

## Por hora del día (hora local de Nueva York)

{comparison_table(base_metrics, metrics, "hora", "hora")}

## Por franja del día

{comparison_table(base_metrics, metrics, "franja", "franja")}

{FRANJA_NOTE}

El desglose por estación de los dos modelos está en `{METRICS_NAME}.parquet`, en las filas con `group_by = "estacion"`.

## Pérdida de cuantil (pinball) y lados del intervalo

{pinball_table(scores)}

La pérdida de cuantil de q50 es la mitad del MAE. "reales bajo q10" y "reales sobre q90" son la proporción de filas
con el valor real fuera del intervalo por cada lado; lo nominal es 10 % de cada lado.

## Dónde LightGBM sale peor que la línea base

{worse_text}

## Variables ({len(FEATURES)})

Rezagos y resúmenes a la misma hora local, de días anteriores al de la fila (`rezagos_misma_hora`):
{", ".join(LAG_COLUMNS)}.

Totales de días anteriores (`totales_dias_previos`): {", ".join(TOTAL_COLUMNS)}.

Calendario de la propia fila (`calendario`): {", ".join(CALENDAR_COLUMNS)}.

Fijas de la estación, con los promedios calculados solo con el tramo de ajuste: {", ".join(STATIC_COLUMNS)}.

No hay variables externas (clima u otras).

## Hiperparámetros y rondas

{shown}. Tope de {MAX_ROUNDS:,} rondas y parada temprana tras {EARLY_STOPPING} rondas sin mejorar en validación.

{_rounds_table(summaries)}

"sin historia de 28 días" son filas predichas de una estación sin ninguna fila a esa hora en los 28 días previos.
Las rondas de la prueba final son las de validación: no se volvieron a elegir.

Los hiperparámetros se eligieron una vez, con entrenamiento y validación, sobre salidas q50: se probaron 63, 255 y
1,023 hojas, tasa de aprendizaje 0.1 y 0.05, quitar los promedios fijos de la estación y quitar del ajuste los
primeros 14 o 28 días. Esas pruebas no leyeron la prueba final.

## Corrección antes de escribir

Los cuantiles se recortan en 0 y se ordenan q10 <= q50 <= q90 por fila. Filas que lo necesitaron:

{_fixes_table(summaries)}

## Cómo leer esto

- Los datos solo registran demanda atendida: viajes que sí ocurrieron. No miden la demanda que no se pudo atender.
- En LightGBM q50 es una mediana condicional. En la línea base q50 es la media de la celda, no la mediana. La mediana es el pronóstico que minimiza el error absoluto y la media el que minimiza el error cuadrático: parte de la diferencia en MAE y WAPE viene de eso y no del modelo. Comparar también RMSE.
- LightGBM pronostica de un día para otro: para el día D usa los conteos de los días anteriores a D, también los de validación ya transcurridos. La historia de la línea base es fija (solo entrenamiento). Parte de la diferencia viene de esa información más reciente.
- Las rondas de cada modelo se fijaron por parada temprana contra las mismas filas de validación que aquí se califican. Las métricas de validación de LightGBM son por eso algo favorables.
- Los promedios fijos de la estación incluyen, para cada fila de ajuste, su propio valor (uno entre decenas).
- Cobertura es la proporción de filas con q10 <= real <= q90. Una cobertura general cercana a la nominal no dice que el intervalo esté calibrado en cada hora: ver la tabla por hora y los dos lados del intervalo.
- WAPE = suma de |real - q50| entre suma de real. Pasa de 100 % en horas con muy pocos viajes.
- "festivo" es un solo día en validación: no generaliza.
- Un cero puede ser una hora sin demanda o una estación vacía, llena o cerrada. El dato no los distingue.
- La prueba final no se ha calificado. Sus predicciones están en predictions.parquet (model = "lgbm", split = "test"), hechas con un reajuste sobre {test.history_span} con las rondas ya fijadas. De la prueba final no se leyó ningún valor como objetivo, ni para una métrica, ni para decidir algo. Como variables sí entran los conteos de días de la prueba anteriores al día que se predice (pronóstico de un día para otro, decisión 5).
"""


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def model_fingerprint(station_hour: pd.DataFrame, datos: Datos, summaries: dict[str, SplitSummary], params: dict) -> dict:
    """Huella de la entrada (la misma de la línea base: fechas e historia) y ajustes con que se corrió el modelo."""
    return {
        "entrada": input_fingerprint(station_hour, datos),
        "modelo": {
            "variables": FEATURES,
            "parametros": {key: value for key, value in params.items()},
            "rondas": summaries["val"].rounds,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_lgbm.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python scripts/run_lgbm.py [opciones]",
        description=(
            "LightGBM global con cuantiles 10, 50 y 90 para salidas y llegadas.\n"
            "Escribe sus filas en predictions.parquet (validación y prueba final) y las métricas de VALIDACIÓN,\n"
            "comparadas con la línea base. La prueba final no se califica."
        ),
    )
    options = parser.add_argument_group("opciones")
    options.add_argument(
        "--datos", metavar="DIR", default=str(DATA_PROCESSED),
        help="carpeta con station_hour.parquet, stations.parquet y predictions.parquet con la línea base "
             "(por defecto data/processed)",
    )
    options.add_argument(
        "--salida", metavar="DIR",
        help="carpeta donde se escriben predictions.parquet y reports/ (por defecto, la misma de --datos)",
    )
    options.add_argument(
        "--config", metavar="FILE", default=str(DATOS_YAML),
        help="archivo con el periodo y los tramos (por defecto config/datos.yaml)",
    )
    options.add_argument(
        "--hilos", metavar="N", type=int, default=NUM_THREADS,
        help=f"hilos de LightGBM (por defecto {NUM_THREADS})",
    )
    options.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    return parser


def main(argv: list[str] | None = None, *, params: dict | None = None) -> int:
    """Punto de entrada. Devuelve 0 si todo salió bien y 1 si no se pudo correr.

    `params` cambia hiperparámetros; es para las pruebas, que usan tablas muy pequeñas.
    """
    args = build_parser().parse_args(argv)
    started = time.time()
    data_dir = Path(args.datos).expanduser()
    out_dir = Path(args.salida).expanduser() if args.salida else data_dir
    source = data_dir / "station_hour.parquet"
    stations_path = data_dir / "stations.parquet"
    predictions_path = out_dir / PREDICTIONS.filename
    metrics_path = out_dir / "reports" / f"{METRICS_NAME}.parquet"
    report_path = out_dir / "reports" / f"{REPORT_NAME}.md"
    fingerprint_path = out_dir / "reports" / FINGERPRINT_NAME
    settings = {**PARAMS, **(params or {}), "num_threads": args.hilos}

    for needed, command in ((source, "build_tables"), (stations_path, "build_tables"), (predictions_path, "run_baseline")):
        if not needed.exists():
            print(f"ERROR. No existe {_short(needed)}. Primero corre: python scripts/{command}.py")
            return 1
    try:
        datos = load_datos(args.config)
        _check_built_period(data_dir, datos, args.config)
        for path, contract in ((source, "station_hour"), (stations_path, "stations")):
            problems = validate_file(path, contract)
            if problems:
                raise LgbmError(f"{_short(path)} no cumple el contrato {contract}:\n- " + "\n- ".join(problems))
        station_hour = read_contract("station_hour", source)
        stations = read_contract("stations", stations_path)
        existing = read_contract("predictions", predictions_path)
        base_rows = existing[existing["model"] == BASELINE]
        if not ((base_rows["split"] == "val").any()):
            raise LgbmError(
                f"{_short(predictions_path)} no trae la línea base en validación: no hay con qué comparar. "
                "Primero corre: python scripts/run_baseline.py"
            )

        fit_started = time.time()
        predictions, summaries = predict_all(station_hour, stations, datos, params=params, num_threads=args.hilos)
        fit_seconds = time.time() - fit_started

        # Solo validación. A las métricas no llega ni una fila real de la prueba final.
        final = datos.tramos[TRAMO_OF_SPLIT["test"]]
        actual = station_hour[datos.tramos[TRAMO_OF_SPLIT["val"]].contains(station_hour["ts_hour"])]
        franjas = report_bands(config=args.config)
        metrics = evaluate(predictions, actual, model=MODEL, split="val", prueba_final=final, franjas=franjas)
        try:
            base_metrics = evaluate(
                base_rows, actual, model=BASELINE, split="val", prueba_final=final, franjas=franjas
            )
        except ValueError as error:
            raise LgbmError(
                f"La línea base de {_short(predictions_path)} no corresponde a este station_hour ({error}) "
                "Primero corre: python scripts/run_baseline.py"
            ) from None
        scores = pd.concat(
            [
                quantile_scores(base_rows, actual, model=BASELINE, split="val", prueba_final=final),
                quantile_scores(predictions, actual, model=MODEL, split="val", prueba_final=final),
            ],
            ignore_index=True,
        )
        report = build_report(summaries, metrics, base_metrics, scores, settings)
        fingerprint = json.dumps(model_fingerprint(station_hour, datos, summaries, settings), indent=2) + "\n"

        kept = save_predictions(predictions, predictions_path)
        write_metrics(metrics, metrics_path)
        report_path.write_text(report, encoding="utf-8")
        fingerprint_path.write_text(fingerprint, encoding="utf-8")
    except (ValueError, OSError) as error:
        print(f"ERROR. {error}")
        return 1

    val = summaries["val"]
    print("LightGBM global con cuantiles 10, 50 y 90, salidas y llegadas. q50 es una mediana; en la línea base es una media.")
    print(f"Entrada: {_short(source)}, {len(station_hour):,} filas, {station_hour['station_id'].nunique():,} estaciones.")
    print(f"Tramos: {'; '.join(_span(datos, [name]) for name in TRAMOS)}.")
    print(f"Variables: {len(FEATURES)}. Hilos: {args.hilos}. Semilla: {settings['seed']}.")
    print("\nAjuste, filas predichas y rondas de cada modelo (las de la prueba final son las fijadas en validación):")
    print(_rounds_table(summaries))
    print("\nFilas con cuantiles negativos o cruzados antes de corregir:")
    print(_fixes_table(summaries))

    print("\nMétricas en VALIDACIÓN, las dos sobre las mismas filas. La prueba final no se calificó (decisión 4).")
    print("Solo demanda atendida. MAE, RMSE y ancho en viajes por estación y hora.")
    print("\nGeneral, línea base (q50 es una media)")
    print(format_table(base_metrics, "general", "grupo"))
    print("\nGeneral, LightGBM (q50 es una mediana)")
    print(format_table(metrics, "general", "grupo"))
    print("\nPor tipo de día (base = línea base, lgbm = LightGBM)")
    print(comparison_table(base_metrics, metrics, "tipo_de_dia", "tipo de día"))
    print("\nPor hora local")
    print(comparison_table(base_metrics, metrics, "hora", "hora"))
    print("\nPor franja del día (las del reporte de la simulación)")
    print(comparison_table(base_metrics, metrics, "franja", "franja"))
    print("\nPérdida de cuantil (pinball) y lados del intervalo")
    print(pinball_table(scores))
    print("\nDónde LightGBM sale peor que la línea base:")
    worse = worse_lines(base_metrics, metrics, scores)
    print("\n".join(f"  - {line}" for line in worse) if worse else "  - En ningún grupo de estas tablas.")

    print("\nEscrito:")
    counts = ", ".join(f"{split} {summaries[split].rows:,}" for split in summaries)
    print(f"  {_short(predictions_path)}: {len(predictions):,} filas de LightGBM ({counts}); "
          f"se conservaron {kept:,} filas de otros modelos.")
    print(f"  {_short(metrics_path)} (filas de LightGBM en validación; las de la línea base se conservan)")
    print(f"  {_short(report_path)}")
    print(f"  {_short(fingerprint_path)}")
    print(f"Tiempo de variables, ajuste y predicción (validación y prueba final): {fit_seconds:.1f} s. "
          f"Total: {time.time() - started:.1f} s. Filas de ajuste en validación: {val.history_rows:,}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
