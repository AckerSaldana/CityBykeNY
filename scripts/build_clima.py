"""Clima observado (Open-Meteo) contra la demanda del sistema: horas con lluvia y horas secas.

    python scripts/build_clima.py

Escribe data/processed/app/clima_demanda.parquet (la comparación por hora del
día y tipo de día) y data/processed/reports/clima.json (las cifras de
docs/clima.md). La descarga queda en data/raw/clima (no va en Git).
Es clima OBSERVADO: describe, no alimenta al modelo. No vuelve a correr ni a
calificar la prueba final; solo lee sus pronósticos ya guardados.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clima as cl  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((cl.ROOT / "config" / "datos.yaml").read_text())
    start, end = str(cfg["periodo"]["inicio"]), str(cfg["periodo"]["fin"])
    t_start, t_end = str(cfg["tramos"]["prueba"]["inicio"]), str(cfg["tramos"]["prueba"]["fin"])

    path = cl.download_weather(start, end)
    weather, meta = cl.load_weather(path)
    print(f"Clima: {len(weather):,} horas, {int(weather.precipitation.isna().sum())} sin precipitación; {meta}")

    hours = cl.join_hours(cl.system_demand(start, end), weather)
    cells = cl.compare_rain_dry(hours)
    total = cl.weighted_difference(cells)
    by_type = cl.summary_by_day_type(cells)
    temp = cl.temperature_correlation(hours)
    print("Lluvia contra seco:", total)
    print("Por tipo de día:", by_type)
    print("Temperatura:", temp)

    days = cl.daily_test_errors(weather, t_start, t_end)
    print(days.round(3).to_string(index=False))

    out = cells.copy()
    out["umbral_lluvia_mm"] = cl.RAIN_MM
    out["periodo_inicio"], out["periodo_fin"] = start, end
    (cl.PROCESSED / "app").mkdir(parents=True, exist_ok=True)
    out.to_parquet(cl.PROCESSED / "app" / "clima_demanda.parquet", index=False)

    days_json = days.assign(fecha=days["fecha"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    report = {
        "fuente": {"endpoint": cl.ENDPOINT, "latitud_pedida": cl.LAT, "longitud_pedida": cl.LON,
                   "variables": cl.VARIABLES, "zona_horaria": cl.TIMEZONE, "respuesta": meta,
                   "tipo": "clima observado (reanálisis), no pronosticado"},
        "periodo": {"inicio": start, "fin": end},
        "umbral_lluvia_mm": cl.RAIN_MM, "horas_minimas_por_lado": cl.MIN_HOURS,
        "horas": {"total": int(len(hours)), "con_lluvia": int(hours.rain.sum()),
                  "precipitacion_total_mm": float(hours.precipitation.sum())},
        "lluvia_contra_seco": total, "por_tipo_de_dia": by_type, "temperatura": temp,
        "prueba": {"inicio": t_start, "fin": t_end, "dias": days_json},
    }
    (cl.PROCESSED / "reports").mkdir(parents=True, exist_ok=True)
    (cl.PROCESSED / "reports" / "clima.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Listo.")


if __name__ == "__main__":
    main()
