"""Recalibra con el tramo de entrenamiento lo que docs/reglas_operacion.md marca [Falta validar].

Se corre desde la raíz del repositorio:

    python notebooks/b/calibrar_reglas.py

Lee (no escribe nada):
- config/operacion.yaml: franjas, umbral de riesgo, ocupación de respaldo y estaciones con valet.
- notebooks/b/tipo_estacion.sql: tipo de cada estación. Las fechas del tramo salen de ahí.
- data/processed/station_hour.parquet y data/processed/stations.parquet (capacidad).

Imprime:
1. Cuántas estaciones quedan de cada tipo y los cortes de percentiles.
2. Por franja y tipo, qué parte de las estación-día no tiene ningún inventario inicial que
   aguante la franja sin riesgo.
3. En esas estación-día, las horas en riesgo según la ocupación de respaldo (acotada entre
   bicis_minimas y capacidad - lugares_libres_minimos), la actual y la mejor.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[2]
SQL = RAIZ / "notebooks" / "b" / "tipo_estacion.sql"
YAML = RAIZ / "config" / "operacion.yaml"
STATION_HOUR = RAIZ / "data" / "processed" / "station_hour.parquet"
STATIONS = RAIZ / "data" / "processed" / "stations.parquet"
TIPOS = ("oficinas", "residencial", "recreativa", "mixta")
REJILLA = np.round(np.arange(0.05, 0.951, 0.05), 2)


def _minutos(hora: str) -> int:
    h, m = hora.split(":")
    return int(h) * 60 + int(m)


def tipos_de_estacion(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str, str]:
    """Corre la consulta de tipos. Devuelve la tabla y las fechas del tramo."""
    texto = SQL.read_text(encoding="utf-8")
    t0, t1 = re.findall(r"TIMESTAMPTZ '([0-9-]+ [0-9:]+)' AS t[01]", texto)
    sentencias = [s for s in texto.split(";") if s.strip()]
    for s in sentencias[:-1]:
        con.execute(s)
    return con.execute(sentencias[-1]).df(), t0, t1


def main() -> None:
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    bmin = cfg["umbral_riesgo"]["bicis_minimas"]
    dmin = cfg["umbral_riesgo"]["lugares_libres_minimos"]
    estrategia = cfg["estrategia_por_tipo"]
    clave = next(k for k in estrategia["oficinas"] if k.startswith("ocupacion_"))
    valet = set(cfg.get("excepciones_valet", {}).get("estaciones", {}))

    con = duckdb.connect()
    con.execute("SET TimeZone = 'America/New_York'")
    tipos, t0, t1 = tipos_de_estacion(con)
    print(f"Tramo: {t0} a {t1} (excluido). Umbral: menos de {bmin} bicis o menos de {dmin} lugares libres.")
    print("\n1. Tipos de estación")
    print(tipos["station_type"].value_counts().to_string())
    print(f"cortes: ocasionales {tipos['corte_casual'].iloc[0]}, fin de semana {tipos['corte_fds'].iloc[0]}")

    capacidad = pd.read_parquet(STATIONS, columns=["station_id", "capacity"]).dropna()
    capacidad = capacidad[capacidad["capacity"] > 0]
    neto = con.execute(f"""
        SELECT station_id, CAST(ts_hour AS DATE) AS fecha, hour(ts_hour) AS h,
               sum(arrivals - departures) AS neto
        FROM read_parquet('{STATION_HOUR.as_posix()}')
        WHERE ts_hour >= TIMESTAMPTZ '{t0}' AND ts_hour < TIMESTAMPTZ '{t1}'
        GROUP BY 1, 2, 3
    """).df()
    neto = (neto.merge(tipos[["station_id", "station_type"]], on="station_id")
                .merge(capacidad, on="station_id"))
    neto = neto[~neto["station_id"].isin(valet)]  # las de valet no cuentan en sus horas: se dejan fuera
    print(f"estaciones con tipo y capacidad (sin las de valet): {neto['station_id'].nunique()}")

    print("\n2 y 3. Estación-día sin solución y horas en riesgo según el respaldo")
    for franja in cfg["franjas"]:
        a, b = _minutos(franja["inicio"]) // 60, -(-_minutos(franja["fin"]) // 60)
        w = neto[(neto["h"] >= a) & (neto["h"] < b)]
        tabla = w.pivot_table(index=["station_id", "fecha", "station_type", "capacity"], columns="h",
                              values="neto", aggfunc="sum").reindex(columns=range(a, b)).fillna(0)
        flujos = tabla.to_numpy()
        cap = tabla.index.get_level_values("capacity").to_numpy().astype(float)
        tipo = tabla.index.get_level_values("station_type").to_numpy()
        acumulado = np.cumsum(flujos, axis=1)
        sin_solucion = (bmin - np.minimum(0, acumulado.min(axis=1))) > (cap - dmin - np.maximum(0, acumulado.max(axis=1)))
        print(f"\n  franja {franja['nombre']} ({franja['inicio']} a {franja['fin']}): "
              f"{100 * sin_solucion.mean():.1f}% de las estación-día sin solución")
        for t in TIPOS:
            m = sin_solucion & (tipo == t)
            if m.sum() < 20:
                print(f"    {t:12s} {100 * sin_solucion[tipo == t].mean():5.1f}% sin solución; muy pocas para calibrar")
                continue

            def horas(f: float) -> float:
                inv = np.clip(np.round(f * cap[m]), bmin, cap[m] - dmin)
                total = np.zeros(m.sum())
                for k in range(b - a):
                    total += (inv < bmin) | (cap[m] - inv < dmin)
                    inv = np.clip(inv + flujos[m, k], 0, cap[m])
                return float(total.mean())

            actual = estrategia[t][clave][franja["nombre"]]
            res = {f: horas(f) for f in REJILLA}
            mejor = min(res, key=res.get)
            print(f"    {t:12s} {100 * sin_solucion[tipo == t].mean():5.1f}% sin solución (n={m.sum()}) | "
                  f"actual {actual:.2f}: {horas(actual):.2f} h | mejor {mejor:.2f}: {res[mejor]:.2f} h | "
                  f"a la mitad: {res[0.5]:.2f} h")


if __name__ == "__main__":
    main()
