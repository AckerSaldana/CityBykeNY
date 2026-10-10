"""Prueba final (decisión 4): métricas de pronóstico del tramo de prueba y tabla validación contra prueba.

La prueba final se usa UNA sola vez. Este módulo hace dos cosas separadas:

1. Con `--usar-prueba-final`: califica las filas `split = "test"` de
   `predictions.parquet` (línea base y LightGBM, sin recalcular ni entrenar
   nada) con `src.models.metrics.evaluate(..., usar_prueba_final=True)`, y el
   piso "media global" igual que `src/models/piso_wape.py` (el mismo número
   para toda estación y hora: la media del tramo de entrenamiento). Lo guarda
   en `prueba_final/reports/metricas_prueba.json`. **Si ese archivo ya existe
   se niega a calificar otra vez.**
2. Siempre: arma la tabla de validación contra prueba con lo que ya está en
   disco, sin leer datos ni simular: `reports/simulacion.json` y
   `reports/piso_wape.json` (validación, congelados),
   `prueba_final/reports/simulacion.json` (la simulación del tramo de prueba,
   que se corre aparte con `scripts/run_simulation.py --tramo prueba
   --usar-prueba-final --banda <la de validación>`) y el archivo del paso 1.
   Escribe `reports/validacion_y_prueba.json` (lo lee el dashboard) y
   `reports/prueba_final.md`.

Los datos solo registran demanda atendida; los viajes bloqueados son simulados.

Uso:  python scripts/run_prueba_final.py [--usar-prueba-final]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.contracts import TS
from src.data.datos_config import load_datos
from src.models.metrics import TARGET_LABELS, TARGETS, evaluate, markdown_table, report_bands
from src.models.piso_wape import NAIVE_MEAN, QUANTILES
from src.paths import DATA_PROCESSED, DATOS_YAML, OPERACION_YAML
from src.sim.simulate import FINAL_FOLDER, resample_days

METRICS_FILE = "metricas_prueba.json"
TABLE_FILE = "validacion_y_prueba.json"
REPORT_FILE = "prueba_final.md"
GROUPS = {"general": "General", "franja": "Por franja del día", "tipo_de_dia": "Por tipo de día"}
MODEL_LABELS = {"baseline": "promedio histórico", "lgbm": "LightGBM", NAIVE_MEAN: NAIVE_MEAN}
FLAG = "--usar-prueba-final"


class PruebaFinalError(RuntimeError):
    """No se puede calificar o armar la tabla de la prueba final."""


# --------------------------------------------------------------------------- #
# 1. Métricas de pronóstico del tramo de prueba (una sola vez)
# --------------------------------------------------------------------------- #

def score_forecasts(data_dir: Path, datos) -> dict:
    """Califica línea base, LightGBM y la media global en el tramo de prueba. Lee datos; no entrena."""
    train, final = datos.tramos["entrenamiento"], datos.tramos["prueba"]

    def instant(value) -> pa.Scalar:
        return pa.scalar(value.to_pydatetime(), type=TS)

    def inside(tramo):
        return (pc.field("ts_hour") >= instant(tramo.start)) & (pc.field("ts_hour") < instant(tramo.end))

    def hours(tramo) -> pd.DataFrame:
        return pq.read_table(
            data_dir / "station_hour.parquet", columns=["station_id", "ts_hour", *TARGETS.values()],
            filters=inside(tramo),
        ).to_pandas()

    predictions = pq.read_table(
        data_dir / "predictions.parquet", filters=(pc.field("split") == "test") & inside(final)
    ).to_pandas()
    actual, training = hours(final), hours(train)
    franjas = report_bands(OPERACION_YAML, DATOS_YAML)
    models = [model for model in ("baseline", "lgbm") if (predictions["model"] == model).any()]
    if models != ["baseline", "lgbm"]:
        raise PruebaFinalError(f'predictions debe traer filas "baseline" y "lgbm" con split "test" (trae {models}).')
    # Piso "media global", igual que piso_wape: la media por estación y hora del tramo de entrenamiento.
    means = {prefix: float(training[column].mean()) for prefix, column in TARGETS.items()}
    naive = predictions.loc[predictions["model"] == "baseline", ["station_id", "ts_hour"]].copy()
    naive["model"], naive["split"] = NAIVE_MEAN, "test"
    for prefix in TARGETS:
        for q in QUANTILES:
            naive[f"{prefix}_{q}"] = means[prefix]
    parts = [
        evaluate(rows, actual, model=model, split="test", usar_prueba_final=True, franjas=franjas)
        for model, rows in [*((model, predictions) for model in models), (NAIVE_MEAN, naive)]
    ]
    metrics = pd.concat(parts, ignore_index=True)
    metrics = metrics[metrics["group_by"].isin(GROUPS)]
    return {
        "calificado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tramo": {"inicio": final.inicio.isoformat(), "fin": final.fin.isoformat()},
        "filas": {"prueba": int(len(actual)), "entrenamiento": int(len(training))},
        "franja_de_cada_hora": list(franjas),
        "media_global_del_entrenamiento": means,
        "metricas": json.loads(metrics.to_json(orient="records")),
    }


# --------------------------------------------------------------------------- #
# 2. Tabla validación contra prueba (solo lee reportes ya escritos)
# --------------------------------------------------------------------------- #

def _quantiles(samples: np.ndarray, mean: float, nivel: float) -> list[float]:
    tail = (1.0 - nivel) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return [float(min(low, mean)), float(max(high, mean))]


def mean_ci(values, draws: np.ndarray, nivel: float) -> tuple[float, list[float]]:
    """Media por día y su intervalo al remuestrear días completos (los mismos sorteos que la simulación)."""
    values = np.asarray(values, dtype="float64")
    mean = float(values.mean())
    return mean, _quantiles(values[draws].mean(axis=1), mean, nivel)


def ratio_ci(part, whole, draws: np.ndarray, nivel: float) -> tuple[float, list[float]]:
    """Cociente de medias (por ejemplo riesgo evitado entre riesgo sin intervención) y su intervalo.

    En cada remuestreo se sortean los mismos días para el numerador y el denominador.
    """
    part, whole = np.asarray(part, dtype="float64"), np.asarray(whole, dtype="float64")
    mean = float(part.mean() / whole.mean())
    return mean, _quantiles(part[draws].mean(axis=1) / whole[draws].mean(axis=1), mean, nivel)


def simulation_facts(report: dict, half_per_station: float) -> dict:
    """Las cifras de la simulación de un tramo, sacadas de su simulacion.json. No simula nada.

    `half_per_station`: el presupuesto por estación de la comparación "1,620 bicis al día" (en validación).
    Devuelve {clave: (valor, [bajo, alto] o None)} y, en "detalle", lo que no cabe en la tabla.
    """
    rows = {(row["politica"], row["presupuesto_por_estacion"]): row for row in report["resultados"]}
    reference = report["reglas"]["presupuesto_de_referencia"]
    budgets = sorted({per_station for _, per_station in rows})
    following = min(per_station for per_station in budgets if per_station > reference)
    days = len(report["tramo"]["dias_evaluados"])
    spec = report["intervalo"]
    draws = resample_days(days, int(spec["remuestreos"]), int(spec["semilla"]))
    nivel = float(spec["nivel"])

    def risk(policy: str, per_station: float) -> np.ndarray:
        return np.asarray(rows[(policy, per_station)]["riesgo_de_cada_dia"], dtype="float64")

    base = risk("none", 0.0)
    out, detail = {}, {
        "dias_evaluados": days,
        "primer_dia_evaluado": report["tramo"]["dias_evaluados"][0],
        "ultimo_dia_evaluado": report["tramo"]["dias_evaluados"][-1],
        "estaciones_simuladas": int(report["estaciones"]["simuladas"]),
        "banda": report["banda"]["elegida"],
        "banda_fijada_de_antemano": bool(report["banda"].get("fijada", False)),
        "riesgo_sin_intervencion_por_dia": float(base.mean()),
        "presupuesto_de_referencia_en_bicis": int(rows[("none", reference)]["presupuesto_del_dia_en_bicis"]),
        "presupuesto_siguiente_en_bicis": int(rows[("none", following)]["presupuesto_del_dia_en_bicis"]),
        "presupuesto_de_la_comparacion_1620_en_bicis": int(
            rows[("none", half_per_station)]["presupuesto_del_dia_en_bicis"]
        ),
        "nivel_del_intervalo": nivel,
    }
    for policy, name in (("historical", "historica"), ("model", "modelo")):
        at_reference = risk(policy, reference)
        out[f"pct_evitado_{name}"] = ratio_ci(base - at_reference, base, draws, nivel)
        out[f"ganancia_pct_al_duplicar_{name}"] = ratio_ci(at_reference - risk(policy, following), base, draws, nivel)
        out[f"bloqueados_{name}"] = mean_ci(rows[(policy, reference)]["viajes_bloqueados"]["cada_dia"], draws, nivel)
        detail[f"riesgo_por_dia_{name}"] = float(at_reference.mean())
    out["bloqueados_sin_intervencion"] = mean_ci(rows[("none", 0.0)]["viajes_bloqueados"]["cada_dia"], draws, nivel)
    for key, per_station in (("referencia", reference), ("1620", half_per_station)):
        difference = risk("model", per_station) - risk("historical", per_station)
        out[f"diferencia_modelo_menos_historica_{key}"] = mean_ci(difference, draws, nivel)
        detail[f"dias_modelo_con_menos_riesgo_{key}"] = int((difference < 0).sum())
    chance = report["piso_de_azar"]
    out["pct_evitado_azar"] = (chance["riesgo_simulado_evitado_pct"] / 100.0, None)
    detail["azar"] = {
        "corridas": chance["corridas"],
        "riesgo_por_dia": chance["riesgo_por_dia"],
        "viajes_bloqueados_por_dia": chance["viajes_bloqueados_por_dia"]["media"],
    }
    out["detalle"] = detail
    return out


def forecast_facts(metricas: list[dict], target: str = "dep") -> dict:
    """WAPE, MAE y cobertura generales de un objetivo, de la tabla larga de métricas (lista de registros)."""
    general = {
        row["model"]: row for row in metricas if row["group_by"] == "general" and row["target"] == target
    }
    out = {}
    for model, name in (("baseline", "promedio_historico"), ("lgbm", "modelo")):
        for metric, column in (("wape", "wape"), ("cobertura", "coverage"), ("mae", "mae")):
            out[f"{metric}_{name}"] = (float(general[model][column]), None)
    out["wape_media_global"] = (float(general[NAIVE_MEAN]["wape"]), None)
    return out


# clave -> (texto para la pantalla, unidad). El orden es el de la tabla.
CLAIMS = {
    "pct_evitado_historica": ("Riesgo simulado evitado con la regla histórica, presupuesto de referencia", "pct"),
    "pct_evitado_modelo": ("Riesgo simulado evitado con el modelo, presupuesto de referencia", "pct"),
    "diferencia_modelo_menos_historica_referencia": (
        "Horas de estación en riesgo por día, modelo menos regla histórica, presupuesto de referencia "
        "(negativo: el modelo tiene menos riesgo)", "horas"),
    "diferencia_modelo_menos_historica_1620": (
        "Horas de estación en riesgo por día, modelo menos regla histórica, con la mitad del presupuesto de "
        "referencia (1,620 bicis al día en validación; negativo: el modelo tiene menos riesgo)", "horas"),
    "ganancia_pct_al_duplicar_historica": (
        "Riesgo simulado evitado adicional (puntos porcentuales) al pasar del presupuesto de referencia al siguiente, "
        "regla histórica", "pct"),
    "ganancia_pct_al_duplicar_modelo": (
        "Riesgo simulado evitado adicional (puntos porcentuales) al pasar del presupuesto de referencia al siguiente, "
        "modelo", "pct"),
    "pct_evitado_azar": ("Riesgo simulado evitado moviendo las mismas bicis al azar", "pct"),
    "bloqueados_sin_intervencion": ("Viajes bloqueados por día (simulados), sin intervención", "viajes"),
    "bloqueados_historica": ("Viajes bloqueados por día (simulados), regla histórica", "viajes"),
    "bloqueados_modelo": ("Viajes bloqueados por día (simulados), modelo", "viajes"),
    "wape_promedio_historico": ("WAPE de salidas, promedio histórico", "pct"),
    "wape_modelo": ("WAPE de salidas, modelo (LightGBM)", "pct"),
    "wape_media_global": ("WAPE de salidas, media global (piso sin modelo)", "pct"),
    "cobertura_promedio_historico": ("Cobertura del intervalo de salidas (nominal 80 %), promedio histórico", "pct"),
    "cobertura_modelo": ("Cobertura del intervalo de salidas (nominal 80 %), modelo (LightGBM)", "pct"),
    "mae_promedio_historico": ("MAE de salidas (viajes por estación y hora), promedio histórico", "numero"),
    "mae_modelo": ("MAE de salidas (viajes por estación y hora), modelo (LightGBM)", "numero"),
}


def build_table(val_sim: dict, test_sim: dict, val_metrics: list[dict], test_metrics: list[dict]) -> dict:
    """El documento `validacion_y_prueba.json`: una fila por afirmación, validación junto a prueba."""
    half = next(
        row["presupuesto_por_estacion"] for row in val_sim["resultados"] if row["presupuesto_del_dia_en_bicis"] == 1620
    )
    sides = {}
    for name, sim, metrics in (("validacion", val_sim, val_metrics), ("prueba", test_sim, test_metrics)):
        sides[name] = {**simulation_facts(sim, half), **forecast_facts(metrics)}
    rows = []
    for key, (label, unit) in CLAIMS.items():
        (val, val_ci), (test, test_ci) = sides["validacion"][key], sides["prueba"][key]
        rows.append(
            {"clave": key, "afirmacion": label, "unidad": unit, "validacion": val, "prueba": test,
             "validacion_ic": val_ci, "prueba_ic": test_ci}
        )
    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "presupuesto_de_referencia": sides["validacion"]["detalle"]["presupuesto_de_referencia_en_bicis"],
        "filas": rows,
        # Lo que no cabe en las filas: días, estaciones, presupuestos en bicis de cada tramo y días con menos riesgo.
        "detalle": {name: side["detalle"] for name, side in sides.items()},
    }


# --------------------------------------------------------------------------- #
# Reporte en Markdown
# --------------------------------------------------------------------------- #

def _value(value, unit: str) -> str:
    if value is None:
        return "n/d"
    if unit == "pct":
        return f"{100 * value:.1f} %"
    if unit == "horas":
        return f"{value:+,.1f}"
    if unit == "viajes":
        return f"{value:,.1f}"
    return f"{value:.3f}"


def _cell(value, ci, unit: str) -> str:
    return _value(value, unit) + (f" ({_value(ci[0], unit)} a {_value(ci[1], unit)})" if ci else "")


def _forecast_tables(metricas: list[dict]) -> list[str]:
    frame = pd.DataFrame(metricas)
    names = ["baseline", "lgbm", NAIVE_MEAN]
    lines = []
    for group_by, title in GROUPS.items():
        part = frame[frame["group_by"] == group_by]
        table = []
        for prefix in TARGETS:
            target = part[part["target"] == prefix]
            for group in target["group"].drop_duplicates():
                cells = target[target["group"] == group].set_index("model")
                table.append(
                    [
                        TARGET_LABELS[prefix], str(group), f"{int(cells['n'].iloc[0]):,}",
                        *[_value(cells.loc[name, "wape"], "pct") for name in names],
                        *[f"{cells.loc[name, 'mae']:.3f}" for name in names],
                        *[f"{cells.loc[name, 'rmse']:.3f}" for name in names[:2]],
                        *[_value(cells.loc[name, "coverage"], "pct") for name in names[:2]],
                    ]
                )
        lines += [
            f"### {title}",
            "",
            markdown_table(
                ["objetivo", "grupo", "filas", *[f"WAPE {MODEL_LABELS[n]}" for n in names],
                 *[f"MAE {MODEL_LABELS[n]}" for n in names], *[f"RMSE {MODEL_LABELS[n]}" for n in names[:2]],
                 *[f"cobertura {MODEL_LABELS[n]}" for n in names[:2]]],
                table, text_columns=2,
            ),
            "",
        ]
    return lines


def render(table: dict, test_sim: dict, scored: dict) -> str:
    """El reporte `prueba_final.md`."""
    val, test = table["detalle"]["validacion"], table["detalle"]["prueba"]
    level = f"{test['nivel_del_intervalo'] * 100:g} %"
    curve = {}
    for row in test_sim["resultados"]:
        curve.setdefault(row["presupuesto_del_dia_en_bicis"], {})[row["politica"]] = row["riesgo_por_dia"]["media"]
    base = test["riesgo_sin_intervencion_por_dia"]

    def curve_cell(risk: float) -> str:
        return f"{risk:,.1f} ({(base - risk) / base * 100:.1f} % evitado)"

    lines = [
        "# Prueba final: validación junto a prueba",
        "",
        "Este archivo lo escribe `python scripts/run_prueba_final.py`. No se edita a mano.",
        "",
        f"**La prueba final se usó una sola vez** (decisión 4): pronóstico calificado el {scored['calificado_utc']} "
        f"(UTC) y simulación corrida el {test_sim['corrida']['generado_utc']} (UTC). Nada del modelo, de las reglas "
        "ni del simulador se cambió después de ver estos números. Las cifras de validación son las de la corrida "
        "congelada (`reports/simulacion.json` y `reports/piso_wape.json`); no se volvieron a correr.",
        "",
        "## La tabla",
        "",
        f"Entre paréntesis, el intervalo del {level} por remuestreo de días completos.",
        "",
        markdown_table(
            ["afirmación", f"validación ({val['dias_evaluados']} días)", f"prueba ({test['dias_evaluados']} días)"],
            [
                [row["afirmacion"], _cell(row["validacion"], row["validacion_ic"], row["unidad"]),
                 _cell(row["prueba"], row["prueba_ic"], row["unidad"])]
                for row in table["filas"]
            ],
        ),
        "",
        "Lo que no cabe en la tabla:",
        "",
        markdown_table(
            ["dato", "validación", "prueba"],
            [
                ["Días evaluados", f"{val['primer_dia_evaluado']} a {val['ultimo_dia_evaluado']}",
                 f"{test['primer_dia_evaluado']} a {test['ultimo_dia_evaluado']}"],
                ["Estaciones simuladas", f"{val['estaciones_simuladas']:,}", f"{test['estaciones_simuladas']:,}"],
                ["Banda de incertidumbre", f"{val['banda']:g} (elegida aquí)",
                 f"{test['banda']:g} (" + ("fijada de antemano" if test["banda_fijada_de_antemano"] else "ELEGIDA EN PRUEBA") + ")"],
                ["Horas de estación en riesgo por día, sin intervención",
                 f"{val['riesgo_sin_intervencion_por_dia']:,.1f}", f"{test['riesgo_sin_intervencion_por_dia']:,.1f}"],
                ["Horas en riesgo por día, regla histórica (referencia)", f"{val['riesgo_por_dia_historica']:,.1f}",
                 f"{test['riesgo_por_dia_historica']:,.1f}"],
                ["Horas en riesgo por día, modelo (referencia)", f"{val['riesgo_por_dia_modelo']:,.1f}",
                 f"{test['riesgo_por_dia_modelo']:,.1f}"],
                ["Presupuesto de referencia, bicis al día", f"{val['presupuesto_de_referencia_en_bicis']:,}",
                 f"{test['presupuesto_de_referencia_en_bicis']:,}"],
                ["Presupuesto siguiente, bicis al día", f"{val['presupuesto_siguiente_en_bicis']:,}",
                 f"{test['presupuesto_siguiente_en_bicis']:,}"],
                ["Presupuesto de la comparación \"1,620\", bicis al día",
                 f"{val['presupuesto_de_la_comparacion_1620_en_bicis']:,}",
                 f"{test['presupuesto_de_la_comparacion_1620_en_bicis']:,}"],
                ["Días en que el modelo tiene menos riesgo que la regla histórica (referencia)",
                 f"{val['dias_modelo_con_menos_riesgo_referencia']} de {val['dias_evaluados']}",
                 f"{test['dias_modelo_con_menos_riesgo_referencia']} de {test['dias_evaluados']}"],
                ["Días en que el modelo tiene menos riesgo (comparación \"1,620\")",
                 f"{val['dias_modelo_con_menos_riesgo_1620']} de {val['dias_evaluados']}",
                 f"{test['dias_modelo_con_menos_riesgo_1620']} de {test['dias_evaluados']}"],
                ["Viajes bloqueados por día (simulados), moviendo al azar",
                 f"{val['azar']['viajes_bloqueados_por_dia']:,.1f}", f"{test['azar']['viajes_bloqueados_por_dia']:,.1f}"],
                ["Horas en riesgo por día moviendo al azar (mínimo a máximo entre corridas)",
                 f"{val['azar']['riesgo_por_dia']['media']:,.1f} ({val['azar']['riesgo_por_dia']['minimo']:,.1f} a "
                 f"{val['azar']['riesgo_por_dia']['maximo']:,.1f})",
                 f"{test['azar']['riesgo_por_dia']['media']:,.1f} ({test['azar']['riesgo_por_dia']['minimo']:,.1f} a "
                 f"{test['azar']['riesgo_por_dia']['maximo']:,.1f})"],
            ],
        ),
        "",
        "## Cómo leerla",
        "",
        "- **Los días no son independientes**: el inventario pasa de un día al siguiente, hay patrón semanal y son "
        f"{val['dias_evaluados']} y {test['dias_evaluados']} días. Los intervalos son indicativos, no una garantía "
        "estadística, y pueden salir más angostos de lo debido.",
        "- Los datos solo registran demanda atendida. Los viajes bloqueados son un resultado de la simulación "
        "(posible demanda insatisfecha), no demanda medida. \"Riesgo simulado evitado\" es tiempo de estación bajo "
        "el umbral que la simulación deja de contar; no son viajes.",
        "- Los porcentajes de riesgo simulado evitado son contra la corrida sin intervención del mismo tramo. Su "
        "intervalo remuestrea a la vez el riesgo evitado y el riesgo sin intervención.",
        "- Las diferencias entre modelo y regla histórica son pareadas: mismos días, mismos viajes, mismo "
        "inventario inicial. Un intervalo que incluye el cero no permite decir que una política es mejor.",
        "- \"Al azar\" son 5 corridas que gastan el presupuesto de referencia en traslados al azar con los mismos "
        "límites; no tiene intervalo, solo el rango entre corridas.",
        "- WAPE, MAE y cobertura de la tabla son de salidas, sobre todas las filas de estación y hora del tramo. "
        "La media global es el mismo número para toda estación y hora (la media del entrenamiento, "
        f"{scored['media_global_del_entrenamiento']['dep']:.4f} salidas): es un piso, no tiene intervalo.",
        "- El inventario inicial es un supuesto (captura de GBFS de octubre de 2026) y el resultado depende de él "
        "en los dos tramos.",
        "",
        "## Qué cambió entre validación y prueba",
        "",
        f"- Días: validación evalúa {val['primer_dia_evaluado']} a {val['ultimo_dia_evaluado']} "
        f"({val['dias_evaluados']} días; el anterior es calentamiento) y la prueba {test['primer_dia_evaluado']} a "
        f"{test['ultimo_dia_evaluado']} ({test['dias_evaluados']} días; el {test_sim['tramo']['dias_de_calentamiento'][0]} "
        "es calentamiento).",
        "- Historia de los pronósticos: en validación, el entrenamiento; en la prueba, entrenamiento más "
        "validación. LightGBM se reajustó con esa historia y las rondas ya fijadas en validación. Sus rezagos en "
        "las filas de prueba usan los conteos observados de días de prueba anteriores (aprobado por el usuario).",
        "- Banda de incertidumbre: se eligió en validación (0) y en la prueba entró fijada con `--banda 0`. No se "
        "volvió a elegir y las otras candidatas no se simularon en la prueba.",
        f"- Estaciones simuladas: {val['estaciones_simuladas']:,} en validación y {test['estaciones_simuladas']:,} en "
        "la prueba (el conjunto son las estaciones con capacidad y con algún viaje en el tramo). El presupuesto "
        "es por estación, así que el total diario en bicis cambia con ese número (tabla de arriba).",
        "- Lo que NO cambió: las reglas de `config/operacion.yaml`, el bloque `simulacion` de `config/datos.yaml`, "
        "el archivo congelado de GBFS (`config/gbfs_congelado.csv`: misma capacidad y misma ocupación inicial "
        "supuesta), el código del motor y de las políticas, y las predicciones de `predictions.parquet`.",
        "- La validación sirvió para decidir (banda, rondas de LightGBM, reglas); la prueba no decidió nada.",
        "",
        "## Curva de la prueba por presupuesto",
        "",
        f"Horas de estación en riesgo por día (sin intervención: {base:,.1f}). El detalle completo está en "
        f"`{FINAL_FOLDER}/reports/simulacion.md`.",
        "",
        markdown_table(
            ["bicis al día", "regla histórica", "modelo (LightGBM)"],
            [
                [f"{budget:,}", curve_cell(risks["historical"]), curve_cell(risks["model"])]
                for budget, risks in sorted(curve.items()) if budget
            ],
        ),
        "",
        "## Métricas de pronóstico en la prueba",
        "",
        f"Tramo {scored['tramo']['inicio']} a {scored['tramo']['fin']}, {scored['filas']['prueba']:,} filas de estación "
        "y hora; las tres columnas califican las mismas filas. Franjas de `config/operacion.yaml` de hoy. La media "
        "global no tiene intervalo. Las de validación, con las mismas franjas, están en `reports/piso_wape.md`.",
        "",
        *_forecast_tables(scored["metricas"]),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    use_final = FLAG in argv
    rest = [arg for arg in argv if arg != FLAG]
    data_dir = Path(rest[0]) if rest else DATA_PROCESSED
    reports, final_reports = data_dir / "reports", data_dir / FINAL_FOLDER / "reports"
    scored_path = final_reports / METRICS_FILE
    try:
        if use_final:
            if scored_path.exists():
                raise PruebaFinalError(
                    f"{scored_path} ya existe: la prueba final ya se calificó y se usa una sola vez (decisión 4). "
                    f"Para rehacer solo la tabla, corre sin {FLAG}."
                )
            scored = score_forecasts(data_dir, load_datos(DATOS_YAML))
            final_reports.mkdir(parents=True, exist_ok=True)
            scored_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        needed = {
            "val_sim": reports / "simulacion.json", "val_piso": reports / "piso_wape.json",
            "test_sim": final_reports / "simulacion.json", "scored": scored_path,
        }
        missing = [str(path) for path in needed.values() if not path.exists()]
        if missing:
            raise PruebaFinalError("Faltan: " + ", ".join(missing) + ". Ver el inicio de src/models/prueba_final.py.")
        docs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in needed.items()}
        table = build_table(docs["val_sim"], docs["test_sim"], docs["val_piso"]["metricas"], docs["scored"]["metricas"])
        text = render(table, docs["test_sim"], docs["scored"])
    except PruebaFinalError as error:
        print(f"ERROR. {error}")
        return 1
    (reports / TABLE_FILE).write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (reports / REPORT_FILE).write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nEscrito: {reports / TABLE_FILE} y {reports / REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
