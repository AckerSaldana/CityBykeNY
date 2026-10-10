"""Piso de azar del WAPE y métricas de pronóstico por franja, en validación.

Responde: ¿cuánto WAPE se obtiene sin ningún modelo? Sirve de referencia para
leer el WAPE de la línea base y de LightGBM. **No entrena nada**: lee
`predictions.parquet` y `station_hour.parquet` tal como están.

Dos pronósticos ingenuos, los dos deterministas, calificados sobre las mismas
filas de validación que los modelos:

- **Media global del entrenamiento** ("media global"): para toda estación y
  toda hora se pronostica el mismo número, el promedio de salidas (o de
  llegadas) por estación y hora de TODAS las filas de `station_hour` del tramo
  de entrenamiento. No sabe de estaciones ni de horas.
- **Línea base revuelta entre estaciones** ("permutación"): dentro de cada
  hora del tramo (`ts_hour`), los pronósticos q50 de la línea base se reparten
  al azar entre las estaciones de esa hora (un sorteo con
  `simulacion.semilla` de config/datos.yaml; filas ordenadas por hora y
  estación antes de sortear, así el resultado se repite). Conserva el nivel de
  cada hora y pierde a qué estación le toca cada pronóstico.

El WAPE es el de `src/models/metrics.py`: suma de |real - pronóstico| entre
suma de real. Las franjas son las de config/operacion.yaml y la franja de
noche de config/datos.yaml, las mismas del reporte de la simulación de HOY
(`report_bands`).

La prueba final no se califica (decisión 4): solo se leen filas con
`split = "val"` del tramo de validación, y `evaluate` se niega a otra cosa.

Los datos solo registran demanda atendida.

Uso:  python scripts/run_piso_wape.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.contracts import TS
from src.data.datos_config import load_datos
from src.models.metrics import TARGET_LABELS, TARGETS, evaluate, markdown_table, report_bands
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML

REPORT_NAME = "piso_wape"
NAIVE_MEAN = "media global"
NAIVE_SHUFFLE = "permutación"
MODEL_LABELS = {"baseline": "línea base", "lgbm": "LightGBM", NAIVE_MEAN: NAIVE_MEAN, NAIVE_SHUFFLE: NAIVE_SHUFFLE}
GROUPS = {"general": "grupo", "franja": "franja", "tipo_de_dia": "tipo de día"}
QUANTILES = ("q10", "q50", "q90")


def _as_predictions(keys: pd.DataFrame, model: str, values: dict) -> pd.DataFrame:
    """Un pronóstico puntual con forma de `predictions`: q10 = q50 = q90 = el valor."""
    frame = keys[["station_id", "ts_hour"]].copy()
    frame["model"], frame["split"] = model, "val"
    for prefix in TARGETS:
        for q in QUANTILES:
            frame[f"{prefix}_{q}"] = values[prefix]
    return frame


def global_mean_forecast(keys: pd.DataFrame, training: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """El mismo número para todas las filas: la media por estación y hora del tramo de entrenamiento."""
    means = {prefix: float(training[column].mean()) for prefix, column in TARGETS.items()}
    return _as_predictions(keys, NAIVE_MEAN, means), means


def shuffled_forecast(baseline: pd.DataFrame, semilla: int) -> pd.DataFrame:
    """Los q50 de la línea base repartidos al azar entre las estaciones de cada hora.

    Salidas y llegadas se mueven juntas (una estación recibe el par completo de otra).
    """
    rows = baseline.sort_values(["ts_hour", "station_id"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(int(semilla))
    order = np.arange(len(rows))
    starts = np.flatnonzero(np.r_[True, rows["ts_hour"].to_numpy()[1:] != rows["ts_hour"].to_numpy()[:-1]])
    for low, high in zip(starts, np.r_[starts[1:], len(rows)]):
        order[low:high] = low + rng.permutation(high - low)
    values = {prefix: rows[f"{prefix}_q50"].to_numpy()[order] for prefix in TARGETS}
    return _as_predictions(rows, NAIVE_SHUFFLE, values)


def compute(predictions: pd.DataFrame, validation: pd.DataFrame, training: pd.DataFrame, semilla: int, franjas) -> dict:
    """Métricas de cada modelo y de los dos pronósticos ingenuos sobre las filas de validación.

    `predictions`: filas `split = "val"`. `validation` y `training`: filas de `station_hour` de cada tramo.
    Devuelve {"metricas": tabla larga, "medias": la media global de salidas y llegadas, "modelos": los que hay}.
    """
    models = [model for model in ("baseline", "lgbm") if (predictions["model"] == model).any()]
    if "baseline" not in models:
        raise ValueError('predictions no trae filas "baseline" de validación. Corre: python scripts/run_baseline.py')
    baseline = predictions[predictions["model"] == "baseline"]
    mean_rows, means = global_mean_forecast(baseline, training)
    naive = {NAIVE_MEAN: mean_rows, NAIVE_SHUFFLE: shuffled_forecast(baseline, semilla)}
    parts = [evaluate(predictions, validation, model=model, split="val", franjas=franjas) for model in models]
    parts += [evaluate(rows, validation, model=name, split="val", franjas=franjas) for name, rows in naive.items()]
    metrics = pd.concat(parts, ignore_index=True)
    return {"metricas": metrics[metrics["group_by"].isin(GROUPS)], "medias": means, "modelos": models}


def _pct(value: float) -> str:
    return "n/d" if pd.isna(value) else f"{100 * value:.1f} %"


def render(result: dict, datos, semilla: int, rows: dict) -> str:
    """El reporte en Markdown."""
    metrics, models = result["metricas"], result["modelos"]
    names = [*models, NAIVE_MEAN, NAIVE_SHUFFLE]
    train, val, final = (datos.tramos[name] for name in ("entrenamiento", "validacion", "prueba"))
    lines = [
        "# Piso de azar del WAPE y métricas de pronóstico por franja (validación)",
        "",
        "Este archivo lo escribe `python scripts/run_piso_wape.py`. No se edita a mano. **No entrena ningún "
        "modelo**: lee `predictions.parquet` y `station_hour.parquet` como están.",
        "",
        "## Qué se calificó",
        "",
        f"- Tramo: validación ({val.inicio} a {val.fin}), {rows['validacion']:,} filas de estación y hora. Todas las "
        "columnas comparan las mismas filas.",
        f"- La prueba final ({final.inicio} a {final.fin}) NO se calificó ni se leyó.",
        "- Línea base y LightGBM: las filas `split = \"val\"` de `predictions.parquet`, sin recalcular.",
        f"- **{NAIVE_MEAN}** (piso 1): el mismo número para toda estación y toda hora, la media por estación y hora "
        f"de las {rows['entrenamiento']:,} filas de `station_hour` del entrenamiento ({train.inicio} a {train.fin}): "
        f"{result['medias']['dep']:.4f} salidas y {result['medias']['arr']:.4f} llegadas.",
        f"- **{NAIVE_SHUFFLE}** (piso 2): dentro de cada hora del tramo, los q50 de la línea base repartidos al azar "
        f"entre las estaciones de esa hora (un solo sorteo, semilla {semilla}, `simulacion.semilla` de "
        "config/datos.yaml). Conserva el nivel de cada hora y pierde a qué estación le toca.",
        "- WAPE = suma de |real - pronóstico| entre suma de real. MAE en viajes por estación y hora.",
        "- Franjas: las de `config/operacion.yaml` de hoy y la franja de noche de `config/datos.yaml`; cada fila "
        "cuenta en la franja donde empieza su hora.",
        "- Los datos solo registran demanda atendida.",
        "",
    ]
    for group_by, label in GROUPS.items():
        title = {"general": "General", "franja": "Por franja del día", "tipo_de_dia": "Por tipo de día"}[group_by]
        part = metrics[metrics["group_by"] == group_by]
        table = []
        for prefix in TARGETS:
            target = part[part["target"] == prefix]
            for group in target["group"].drop_duplicates():
                cells = target[target["group"] == group].set_index("model")
                table.append(
                    [
                        TARGET_LABELS[prefix], str(group), f"{int(cells['n'].iloc[0]):,}",
                        *[_pct(cells.loc[name, "wape"]) for name in names],
                        *[f"{cells.loc[name, 'mae']:.3f}" for name in names],
                    ]
                )
        lines += [
            f"## {title}",
            "",
            markdown_table(
                ["objetivo", label, "filas", *[f"WAPE {MODEL_LABELS[name]}" for name in names],
                 *[f"MAE {MODEL_LABELS[name]}" for name in names]],
                table, text_columns=2,
            ),
            "",
        ]
    by_band = metrics[(metrics["group_by"] == "franja") & metrics["model"].isin(models)]
    lines += [
        "## Métricas completas por franja de cada modelo",
        "",
        markdown_table(
            ["modelo", "objetivo", "franja", "filas", "MAE", "RMSE", "WAPE", "cobertura", "ancho medio"],
            [
                [MODEL_LABELS[row.model], TARGET_LABELS[row.target], str(row.group), f"{row.n:,}", f"{row.mae:.3f}",
                 f"{row.rmse:.3f}", _pct(row.wape), _pct(row.coverage), f"{row.mean_width:.3f}"]
                for row in by_band.itertuples()
            ],
            text_columns=3,
        ),
        "",
        "La cobertura nominal del intervalo de q10 a q90 es 80 %. Los pronósticos ingenuos no tienen intervalo.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    data_dir = Path(argv[0]) if argv else DATA_PROCESSED
    datos = load_datos(DATOS_YAML)
    train, val = datos.tramos["entrenamiento"], datos.tramos["validacion"]
    semilla = int(datos.simulacion.semilla)

    def instant(value) -> pa.Scalar:
        return pa.scalar(value.to_pydatetime(), type=TS)

    def hours(tramo) -> pd.DataFrame:
        inside = (pc.field("ts_hour") >= instant(tramo.start)) & (pc.field("ts_hour") < instant(tramo.end))
        return pq.read_table(
            data_dir / "station_hour.parquet", columns=["station_id", "ts_hour", *TARGETS.values()], filters=inside
        ).to_pandas()

    predictions = pq.read_table(
        data_dir / "predictions.parquet",
        filters=(pc.field("split") == "val")
        & (pc.field("ts_hour") >= instant(val.start)) & (pc.field("ts_hour") < instant(val.end)),
    ).to_pandas()
    validation, training = hours(val), hours(train)
    franjas = report_bands(OPERACION_YAML, DATOS_YAML)
    result = compute(predictions, validation, training, semilla, franjas)
    reports = data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = {"validacion": int(len(validation)), "entrenamiento": int(len(training))}
    text = render(result, datos, semilla, rows)
    (reports / f"{REPORT_NAME}.md").write_text(text, encoding="utf-8")
    document = {
        "tramo": {"inicio": val.inicio.isoformat(), "fin": val.fin.isoformat()},
        "filas": rows,
        "semilla": semilla,
        "franja_de_cada_hora": list(franjas),
        "media_global_del_entrenamiento": result["medias"],
        "metricas": json.loads(result["metricas"].to_json(orient="records")),
    }
    (reports / f"{REPORT_NAME}.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(text)
    print(f"\nEscrito: {reports / (REPORT_NAME + '.md')} y {REPORT_NAME}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
