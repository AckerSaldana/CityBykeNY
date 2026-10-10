"""Empleos y trabajadores residentes a 300 m de cada estación, y su relación con el patrón de uso.

    python scripts/build_station_context.py          # LODES y comparación
    python scripts/build_station_context.py --osm    # además, una consulta a OpenStreetMap

Escribe data/processed/station_context.parquet y la tabla del dashboard
data/processed/app/contexto_estaciones.parquet. Las descargas quedan en
data/raw/lodes (no van en Git). Los números para docs/contexto_estaciones.md
se imprimen y se guardan en data/processed/reports/contexto_estaciones.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import station_context as sc  # noqa: E402

PROCESSED = sc.ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--osm", action="store_true", help="consulta parques, atracciones y universidades")
    args = parser.parse_args()
    t0 = time.time()
    resumen = {}

    usados = sc.download_lodes("ny")
    for clase, info in usados.items():
        print(f"{clase}: {info['archivo']} ({info['bytes'] / 1e6:.1f} MB), año {info['anio']}")
    anio = usados["wac"]["anio"]
    if usados["rac"]["anio"] != anio:
        raise SystemExit("WAC y RAC no son del mismo año.")
    resumen["archivos"] = {c: {"archivo": i["archivo"], "anio": i["anio"], "bytes": i["bytes"]}
                           for c, i in usados.items()}

    stations = pd.read_parquet(PROCESSED / "stations.parquet")
    stations["station_id"] = stations["station_id"].astype(str)
    con_coord = stations[stations.lat.notna() & stations.lon.notna()]
    bbox = sc.station_bbox(con_coord)
    blocks = sc.load_blocks(usados["wac"]["ruta"], usados["rac"]["ruta"], usados["xwalk"]["ruta"], bbox)
    print(f"Manzanas en la caja: {len(blocks):,}; empleos {blocks.empleos.sum():,}; "
          f"residentes {blocks.residentes.sum():,}")

    ctx = sc.sum_within(con_coord, blocks)
    ctx = stations[["station_id"]].merge(ctx, on="station_id", how="left")
    ctx[["station_id", "empleos_300m", "residentes_300m", "proporcion_empleos"]].to_parquet(
        PROCESSED / "station_context.parquet", index=False)
    resumen["estaciones"] = {"total": len(ctx), "con_manzanas": int(ctx.empleos_300m.notna().sum()),
                             "con_proporcion": int(ctx.proporcion_empleos.notna().sum()),
                             "manzanas_en_caja": len(blocks)}
    print(resumen["estaciones"])

    uso = sc.usage_pattern()
    df = (stations[["station_id", "name"]].merge(ctx, on="station_id", how="left")
          .merge(uso[["station_id", "am_indice", "pm_indice", "prop_casual", "prop_fin_de_semana",
                      "station_type", "poca_actividad"]], on="station_id", how="left"))
    resumen["tipos"] = df.station_type.value_counts(dropna=False).to_dict()
    resumen["tipos"] = {str(k): int(v) for k, v in resumen["tipos"].items()}

    cor = {}
    for nombre, sub in {"todas": df, "sin_poca_actividad": df[df.poca_actividad == False]}.items():  # noqa: E712
        p, n = sc.pearson(sub.proporcion_empleos, sub.am_indice)
        s, _ = sc.spearman(sub.proporcion_empleos, sub.am_indice)
        cor[nombre] = {"pearson": round(p, 4), "spearman": round(s, 4), "n": n}
    resumen["proporcion_vs_am_indice"] = cor
    print("proporcion_empleos vs am_indice:", cor)

    med = sc.medians_by_type(df)
    resumen["medianas_por_tipo"] = json.loads(med.to_json(orient="records"))
    print(med.to_string(index=False))

    if args.osm:
        try:
            t1 = time.time()
            puntos = sc.osm_points(bbox)
            osm = {"segundos": round(time.time() - t1, 1), "servidor": puntos.attrs.get("servidor"), "puntos": puntos.clase.value_counts().to_dict(),
                   "spearman": {}}
            # df conserva el orden y el índice de stations (uniones por la izquierda con ids únicos)
            df["lugares_300m"] = pd.Series(sc.count_within(con_coord, puntos), index=con_coord.index)
            for clase in ["parque", "atraccion", "universidad"]:
                df[f"{clase}_300m"] = pd.Series(
                    sc.count_within(con_coord, puntos[puntos.clase == clase]), index=con_coord.index)
            activas = df[df.poca_actividad == False]  # noqa: E712
            for col in ["lugares_300m", "parque_300m", "atraccion_300m", "universidad_300m"]:
                for y in ["prop_casual", "prop_fin_de_semana"]:
                    s, n = sc.spearman(activas[col], activas[y])
                    osm["spearman"][f"{col} vs {y}"] = {"spearman": round(s, 4), "n": n}
            osm["estaciones_con_alguno"] = int((activas.lugares_300m > 0).sum())
            resumen["osm"] = osm
            print("OSM:", json.dumps(osm, ensure_ascii=False, indent=1))
        except Exception as e:  # la consulta es opcional: si falla, se dice y se sigue
            resumen["osm"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print("OSM no se pudo consultar:", resumen["osm"]["error"])

    app = df[["name", "station_type", "proporcion_empleos", "am_indice", "empleos_300m", "residentes_300m"]].copy()
    app = app.rename(columns={"name": "estacion", "station_type": "tipo", "empleos_300m": "empleos",
                              "residentes_300m": "residentes"})
    app["anio_datos"] = anio
    app = app[app.tipo.notna()]
    (PROCESSED / "app").mkdir(exist_ok=True)
    app.to_parquet(PROCESSED / "app" / "contexto_estaciones.parquet", index=False)

    resumen["segundos"] = round(time.time() - t0, 1)
    (PROCESSED / "reports").mkdir(exist_ok=True)
    (PROCESSED / "reports" / "contexto_estaciones.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo en {resumen['segundos']} s")


if __name__ == "__main__":
    main()
