"""Tablas pequeñas para el dashboard.

El dashboard nunca lee las tablas grandes (station_hour tiene 5 millones de
filas y predictions 3 millones). Este módulo las resume una sola vez y deja el
resultado en `data/processed/app/`. `app/datos.py` solo lee esa carpeta.

    python scripts/build_app_tables.py                      # datos reales
    python scripts/build_app_tables.py --datos data/sample --salida /tmp/app

Qué lee de la carpeta de datos: los seis contratos y, si existen,
`reports/simulacion.json`, `reports/piso_wape.json` (el error de pronosticar
con una sola media), `station_context.parquet` (empleos y residentes cerca de
cada estación) y `reports/validacion_y_prueba.json` (validación y prueba final
lado a lado). Todo lo que falte se escribe vacío o como `null` (el dashboard
muestra "Sin dato" o no dibuja ese bloque); nada de esto hace fallar la
construcción.

Decisión 4: lo observado solo se cruza con el pronóstico en el tramo de
validación (`split = "val"`). Las filas de la prueba final nunca se leen para
comparar ni para calcular métricas. Las tablas descriptivas de demanda
(estaciones, mapa, rutas) usan todo el periodo, sin pronóstico.

Definiciones que fija este módulo (y que el dashboard solo muestra):

- Tipo de día: sábado y domingo son "fin_de_semana"; lo demás, "laborable"
  (un festivo entre semana cuenta como laborable).
- Usuarios: las salidas vienen separadas en miembros y ocasionales. Las
  llegadas y el flujo neto NO traen esa separación en los datos: en el mapa,
  el color (flujo neto) es siempre el de todos los usuarios.
- Bloques de hora para el error de pronóstico: madrugada 00-05, mañana 06-11,
  tarde 12-17, noche 18-23.
- Volumen de la estación: tercios (bajo, medio, alto) según las salidas
  anteriores al tramo de validación.
- Presupuesto de referencia: el del reporte de simulación. Sin reporte, el
  presupuesto de la curva más cercano a las bicis recibidas por día en
  `recommendations` (por contrato esa tabla sale del presupuesto de referencia).
- Presupuesto a partir del cual casi no ayuda: el primer presupuesto de la
  curva desde el cual el siguiente escalón evita, por cada bici extra, menos
  del 10 % de lo que evitaba cada bici en el primer escalón
  (`UMBRAL_GANANCIA`).
- Estación de ejemplo (la que abre "Pronóstico por estación"): entre las
  estaciones de oficinas con demanda alta, la de más salidas que tenga algún día
  laborable en que la regla histórica movió bicis en todas las franjas y todas
  tuvieron solución. De esos días se abre el de error de pronóstico más típico:
  el más cercano a la mediana del error diario de esa estación en sus días
  laborables simulados (error diario: suma de |observado - pronóstico| entre la
  suma de lo observado, salidas, modelo). Si empatan, el que tiene más franjas
  con menos riesgo que sin intervenir y luego el de más bicis movidas (ver
  `_estacion_de_ejemplo`). Si ninguna cumple, la estación con más salidas.
- Rondas: la franja cuyo momento de intervención es el más temprano del día es
  la "ronda nocturna"; las demás, la "ronda de mediodía". La hora sale del
  reporte de simulación.
- Mitad del presupuesto: el presupuesto de la curva más cercano a la mitad del
  de referencia, si queda entre 40 % y 60 % de él.
- Rutas más usadas: las `RUTAS_EN_PANTALLA` rutas con más viajes entre
  estaciones distintas; su distancia es en línea recta entre las dos estaciones.
- Inventario antes del traslado: el inventario justo después del traslado
  menos las bicis movidas (positivo es llevar).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / "data" / "processed"
SALIDA = DATOS / "app"

ZONA = "America/New_York"
UMBRAL_GANANCIA = 0.10  # fracción de la ganancia por bici del primer escalón
TOP = 100  # filas que se guardan en los rankings (la página muestra 15)
RUTAS_EN_PANTALLA = 15  # rutas entre estaciones distintas que resume `rutas_mas_usadas`
BLOQUES = [("madrugada", 0, 5), ("manana", 6, 11), ("tarde", 12, 17), ("noche", 18, 23)]
POLITICAS = ["none", "historical", "model"]

# Archivos que se escriben. La app los busca con estos mismos nombres.
TABLAS = [
    "estaciones", "estacion_hora", "demanda_hora", "top_estaciones", "top_rutas",
    "pronostico", "recomendaciones", "curva", "metricas", "riesgo_horas",
    "riesgo_niveles",
]
INDICADORES = "indicadores.json"


def _f(x):
    """Número de Python o None (para el JSON)."""
    if x is None or pd.isna(x):
        return None
    return float(x)


def _escribir(df: pd.DataFrame, ruta: Path, grupo: int | None = None) -> None:
    tabla = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    tabla = tabla.replace_schema_metadata(None)
    pq.write_table(tabla, ruta, compression="zstd", row_group_size=grupo)


def _leer_json(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta, en kilómetros (fórmula del haversine)."""
    lat1, lon1, lat2, lon2 = (np.radians(pd.to_numeric(x, errors="coerce").astype(float))
                              for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def _hora(texto) -> int | None:
    try:
        return int(str(texto).split(":")[0])
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Demanda (descriptivo, todo el periodo)
# --------------------------------------------------------------------------- #

def _demanda(con, datos: Path, salida: Path) -> dict:
    sh = f"read_parquet('{datos / 'station_hour.parquet'}')"
    st = f"read_parquet('{datos / 'stations.parquet'}')"
    con.execute(f"""
        CREATE TEMP TABLE sh AS
        SELECT station_id, CAST(ts_hour AS TIMESTAMP) AS ts, departures, arrivals,
               net_flow, departures_member, departures_casual
        FROM {sh}""")
    con.execute("""
        CREATE TEMP TABLE shd AS
        SELECT *, CAST(ts AS DATE) AS fecha, hour(ts) AS hora,
               CASE WHEN isodow(ts) >= 6 THEN 'fin_de_semana' ELSE 'laborable' END AS tipo_dia
        FROM sh""")

    estaciones = con.execute(f"""
        WITH t AS (
            SELECT station_id, sum(departures) AS salidas, sum(arrivals) AS llegadas,
                   count(DISTINCT fecha) AS dias
            FROM shd GROUP BY 1)
        SELECT s.station_id AS estacion_id, s.name AS nombre, s.lat, s.lon,
               CAST(s.capacity AS DOUBLE) AS capacidad, s.station_type AS tipo,
               CAST(coalesce(t.salidas, 0) AS BIGINT) AS salidas,
               CAST(coalesce(t.llegadas, 0) AS BIGINT) AS llegadas,
               CAST(coalesce(t.dias, 0) AS BIGINT) AS dias_con_datos,
               round(coalesce(t.salidas / nullif(t.dias, 0), 0), 2) AS salidas_dia,
               round(coalesce(t.llegadas / nullif(t.dias, 0), 0), 2) AS llegadas_dia
        FROM {st} s LEFT JOIN t USING (station_id)
        ORDER BY 1""").df()
    estaciones["flujo_neto_dia"] = (estaciones["llegadas_dia"] - estaciones["salidas_dia"]).round(2)

    estacion_hora = con.execute("""
        SELECT station_id AS estacion_id, hora, tipo_dia,
               CAST(round(avg(departures), 3) AS FLOAT) AS salidas,
               CAST(round(avg(departures_member), 3) AS FLOAT) AS salidas_miembros,
               CAST(round(avg(departures_casual), 3) AS FLOAT) AS salidas_ocasionales,
               CAST(round(avg(arrivals), 3) AS FLOAT) AS llegadas,
               CAST(round(avg(net_flow), 3) AS FLOAT) AS flujo_neto
        FROM shd GROUP BY 1, 2, 3 ORDER BY 3, 2, 1""").df()
    _escribir(estacion_hora, salida / "estacion_hora.parquet")

    # Sistema completo: viajes por hora, promedio entre los días de cada tipo.
    demanda_hora = con.execute("""
        WITH d AS (
            SELECT fecha, tipo_dia, hora, sum(departures) AS s, sum(departures_member) AS m,
                   sum(departures_casual) AS c, sum(arrivals) AS a
            FROM shd GROUP BY 1, 2, 3)
        SELECT tipo_dia, hora, round(avg(s), 1) AS salidas, round(avg(m), 1) AS salidas_miembros,
               round(avg(c), 1) AS salidas_ocasionales, round(avg(a), 1) AS llegadas,
               count(*) AS dias
        FROM d GROUP BY 1, 2 ORDER BY 1, 2""").df()
    _escribir(demanda_hora, salida / "demanda_hora.parquet")

    top = estaciones.sort_values(["salidas", "estacion_id"], ascending=[False, True]).head(TOP).copy()
    top.insert(0, "puesto", range(1, len(top) + 1))
    _escribir(top, salida / "top_estaciones.parquet")

    rutas = con.execute(f"""
        SELECT r.start_station_id AS origen_id, a.name AS origen,
               r.end_station_id AS destino_id, b.name AS destino, r.trips AS viajes,
               a.lat AS origen_lat, a.lon AS origen_lon, b.lat AS destino_lat, b.lon AS destino_lon
        FROM read_parquet('{datos / 'routes.parquet'}') r
        LEFT JOIN {st} a ON a.station_id = r.start_station_id
        LEFT JOIN {st} b ON b.station_id = r.end_station_id
        ORDER BY r.trips DESC, 1, 3 LIMIT {TOP}""").df()
    rutas.insert(0, "puesto", range(1, len(rutas) + 1))
    rutas["origen"] = rutas["origen"].fillna(rutas["origen_id"])
    rutas["destino"] = rutas["destino"].fillna(rutas["destino_id"])
    rutas["misma_estacion"] = rutas["origen_id"] == rutas["destino_id"]
    rutas["ruta"] = rutas["origen"] + " → " + rutas["destino"]
    _escribir(rutas, salida / "top_rutas.parquet")
    # Qué tan largas son las rutas más usadas entre estaciones distintas.
    distintas = rutas[~rutas["misma_estacion"]].dropna(
        subset=["origen_lat", "origen_lon", "destino_lat", "destino_lon"]).head(RUTAS_EN_PANTALLA)
    km = _km(distintas["origen_lat"], distintas["origen_lon"], distintas["destino_lat"], distintas["destino_lon"])
    rutas_mas_usadas = {"rutas": int(len(distintas)),
                        "distancia_mediana_km": _f(km.median()) if len(distintas) else None,
                        "distancia_maxima_km": _f(km.max()) if len(distintas) else None}

    pares, viajes_rutas = con.execute(
        f"SELECT count(*), sum(trips) FROM read_parquet('{datos / 'routes.parquet'}')").fetchone()
    ini, fin, dias = con.execute("SELECT min(fecha), max(fecha), count(DISTINCT fecha) FROM shd").fetchone()
    return {
        "estaciones_df": estaciones,
        "resumen": {
            "viajes_analizados": int(estaciones["salidas"].sum()),
            "llegadas": int(estaciones["llegadas"].sum()),
            "estaciones": int(len(estaciones)),
            "estaciones_con_capacidad": int(estaciones["capacidad"].notna().sum()),
            "estaciones_con_tipo": int(estaciones["tipo"].notna().sum()),
            "pares_de_estaciones": int(pares or 0),
            "viajes_en_rutas": int(viajes_rutas or 0),
            "periodo": {"inicio": str(ini), "fin": str(fin), "dias": int(dias)},
            "rutas_mas_usadas": rutas_mas_usadas,
        },
    }


# --------------------------------------------------------------------------- #
# Pronóstico contra observado (solo validación)
# --------------------------------------------------------------------------- #

def _pronostico(con, datos: Path, salida: Path, estaciones: pd.DataFrame) -> dict:
    pr = f"read_parquet('{datos / 'predictions.parquet'}')"
    # Solo split = 'val'. La prueba final no entra (decisión 4).
    cols = ", ".join(
        f"CAST(max({c}_{q}) FILTER (model = '{m}') AS FLOAT) AS {pref}_{nom}_{q}"
        for m, pref in (("baseline", "base"), ("lgbm", "modelo"))
        for c, nom in (("dep", "sal"), ("arr", "lle"))
        for q in ("q10", "q50", "q90"))
    con.execute(f"""
        CREATE TEMP TABLE pv AS
        WITH p AS (
            SELECT station_id, CAST(ts_hour AS TIMESTAMP) AS ts, {cols}
            FROM {pr} WHERE split = 'val' GROUP BY 1, 2)
        SELECT p.station_id AS estacion_id, p.ts, CAST(p.ts AS DATE) AS fecha,
               CAST(hour(p.ts) AS TINYINT) AS hora,
               CAST(o.departures AS INTEGER) AS sal_obs, CAST(o.arrivals AS INTEGER) AS lle_obs,
               p.* EXCLUDE (station_id, ts)
        FROM p JOIN sh o ON o.station_id = p.station_id AND o.ts = p.ts""")
    tabla = con.execute("SELECT * FROM pv ORDER BY estacion_id, ts").df()
    _escribir(tabla, salida / "pronostico.parquet", grupo=8192)

    if tabla.empty:
        val = {"inicio": None, "fin": None, "dias": 0}
        corte = None
    else:
        corte = tabla["fecha"].min()
        val = {"inicio": str(pd.Timestamp(corte).date()), "fin": str(pd.Timestamp(tabla["fecha"].max()).date()),
               "dias": int(tabla["fecha"].nunique())}

    # Tercios de volumen con las salidas ANTERIORES a validación.
    con.execute(f"""
        CREATE TEMP TABLE vol AS
        WITH e AS (SELECT DISTINCT estacion_id FROM pv),
        h AS (SELECT station_id, sum(departures) AS s FROM sh
              WHERE ts < {"TIMESTAMP '" + str(pd.Timestamp(corte)) + "'" if corte is not None else "ts"}
              GROUP BY 1)
        SELECT e.estacion_id, coalesce(h.s, 0) AS previas,
               ntile(3) OVER (ORDER BY coalesce(h.s, 0), e.estacion_id) AS tercio
        FROM e LEFT JOIN h ON h.station_id = e.estacion_id""")
    vol = con.execute("SELECT * FROM vol ORDER BY 1").df()
    nombres = {1: "bajo", 2: "medio", 3: "alto"}
    estaciones = estaciones.merge(vol.rename(columns={"previas": "salidas_previas"}), on="estacion_id", how="left")
    estaciones["volumen"] = estaciones["tercio"].map(nombres)
    estaciones["tiene_pronostico"] = estaciones["tercio"].notna()
    # Sin viajes antes de validación: su pronóstico histórico es un marcador
    # (promedio del sistema), no un pronóstico de esa estación.
    estaciones["sin_historia"] = estaciones["tiene_pronostico"] & (estaciones["salidas_previas"].fillna(0) == 0)
    estaciones = estaciones.drop(columns=["tercio", "salidas_previas"])

    bloque = "CASE " + " ".join(f"WHEN hora BETWEEN {a} AND {b} THEN '{n}'" for n, a, b in BLOQUES) + " END"
    partes = []
    for modelo, pref in (("baseline", "base"), ("lgbm", "modelo")):
        for objetivo, nom in (("salidas", "sal"), ("llegadas", "lle")):
            partes.append(f"""
                SELECT '{modelo}' AS modelo, '{objetivo}' AS objetivo, estacion_id, hora,
                       {nom}_obs AS y, {pref}_{nom}_q10 AS q10, {pref}_{nom}_q50 AS q50, {pref}_{nom}_q90 AS q90
                FROM pv WHERE {pref}_{nom}_q50 IS NOT NULL""")
    largo = " UNION ALL ".join(partes)
    medidas = """count(*) AS n, avg(abs(y - q50)) AS mae, sqrt(avg((y - q50) * (y - q50))) AS rmse,
                 sum(abs(y - q50)) / nullif(sum(y), 0) AS wape,
                 avg(CASE WHEN y >= q10 AND y <= q90 THEN 1.0 ELSE 0.0 END) AS cobertura,
                 avg(q90 - q10) AS ancho"""
    metricas = con.execute(f"""
        WITH l AS ({largo}), lv AS (SELECT l.*, v.tercio, {bloque} AS bloque FROM l JOIN vol v USING (estacion_id))
        SELECT 'general' AS corte, 'todas' AS grupo, 0 AS orden, modelo, objetivo, {medidas} FROM lv GROUP BY ALL
        UNION ALL
        SELECT 'bloque', bloque, min(hora), modelo, objetivo, {medidas} FROM lv GROUP BY bloque, modelo, objetivo
        UNION ALL
        SELECT 'hora', lpad(CAST(hora AS VARCHAR), 2, '0'), hora, modelo, objetivo, {medidas} FROM lv GROUP BY hora, modelo, objetivo
        UNION ALL
        SELECT 'volumen', CASE tercio WHEN 1 THEN 'bajo' WHEN 2 THEN 'medio' ELSE 'alto' END, tercio, modelo, objetivo, {medidas}
        FROM lv GROUP BY tercio, modelo, objetivo
        ORDER BY 1, 3, 4, 5""").df()
    _escribir(metricas, salida / "metricas.parquet")

    general = {}
    for fila in metricas[metricas["corte"] == "general"].itertuples():
        general.setdefault(fila.modelo, {})[fila.objetivo] = {
            "n": int(fila.n), "mae": _f(fila.mae), "rmse": _f(fila.rmse), "wape": _f(fila.wape),
            "cobertura": _f(fila.cobertura), "ancho": _f(fila.ancho)}
    # Días con pronóstico de cada estación y su error diario (salidas, modelo): sirve para abrir
    # "Pronóstico por estación" en un día típico.
    dias = (tabla.assign(_e=(tabla["sal_obs"] - tabla["modelo_sal_q50"]).abs().where(tabla["modelo_sal_q50"].notna()),
                         _o=tabla["sal_obs"].where(tabla["modelo_sal_q50"].notna()))
            .groupby(["estacion_id", "fecha"], as_index=False)[["_e", "_o"]].sum(min_count=1))
    dias["error"] = (dias["_e"] / dias["_o"].where(dias["_o"] > 0)).astype(float)
    dias = dias[["estacion_id", "fecha", "error"]]
    return {"estaciones_df": estaciones, "validacion": val, "pronostico": general, "dias_df": dias}


# --------------------------------------------------------------------------- #
# Simulación: recomendaciones, curva e indicadores
# --------------------------------------------------------------------------- #

def _recomendaciones(con, datos: Path, salida: Path, reporte: dict, estaciones: pd.DataFrame):
    ruta = datos / "recommendations.parquet"
    franjas = {}
    for f in (reporte.get("reglas") or {}).get("franjas") or []:
        if _hora(f.get("inicio")) is not None:
            franjas[_hora(f["inicio"])] = f.get("nombre")
    rec = con.execute(f"""
        SELECT station_id AS estacion_id, CAST(CAST(window_start AS TIMESTAMP) AS DATE) AS fecha,
               CAST(hour(CAST(window_start AS TIMESTAMP)) AS INTEGER) AS hora_inicio,
               CAST(hour(CAST(window_end AS TIMESTAMP)) AS INTEGER) AS hora_fin,
               policy AS politica, CAST(bikes_to_move AS INTEGER) AS bicis,
               risk_level AS nivel_riesgo, feasible AS factible,
               CAST(start_inventory_target AS INTEGER) AS inventario
        FROM read_parquet('{ruta}') ORDER BY 1, 2, 3, 5""").df() if ruta.exists() else pd.DataFrame(
        columns=["estacion_id", "fecha", "hora_inicio", "hora_fin", "politica", "bicis",
                 "nivel_riesgo", "factible", "inventario"])
    # Bicis en la estación justo antes del traslado (el contrato trae las de después).
    rec["inventario_antes"] = (rec["inventario"] - rec["bicis"]).astype("int32")
    rec.insert(2, "franja", [franjas.get(h) or ("manana" if h < 12 else "tarde") for h in rec["hora_inicio"]])
    _escribir(rec, salida / "recomendaciones.parquet", grupo=8192)

    simuladas = set(rec["estacion_id"].unique())
    estaciones["simulada"] = estaciones["estacion_id"].isin(simuladas)

    # Reparto de niveles de riesgo por franja y por tipo de estación.
    r = rec.merge(estaciones[["estacion_id", "tipo"]], on="estacion_id", how="left")
    r["tipo"] = r["tipo"].fillna("sin_dato")
    partes = []
    for corte, col in (("franja", "franja"), ("tipo_estacion", "tipo")):
        g = (r.groupby([col, "politica", "nivel_riesgo"]).size().unstack("nivel_riesgo", fill_value=0)
             .reindex(columns=["alto", "medio", "bajo"], fill_value=0).reset_index()
             .rename(columns={col: "grupo"}))
        g.insert(0, "corte", corte)
        partes.append(g)
    niveles = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    if niveles.empty:
        niveles = pd.DataFrame(columns=["corte", "grupo", "politica", "alto", "medio", "bajo"])
    niveles["total"] = niveles[["alto", "medio", "bajo"]].sum(axis=1)
    niveles["pct_alto"] = (niveles["alto"] / niveles["total"].where(niveles["total"] > 0)).astype(float)
    niveles = niveles.sort_values(["corte", "grupo", "politica"])
    _escribir(niveles, salida / "riesgo_niveles.parquet")
    return rec, estaciones


def _curva(datos: Path, salida: Path) -> pd.DataFrame:
    ruta = datos / "policy_curve.parquet"
    cols = ["politica", "presupuesto", "riesgo", "ic_bajo", "ic_alto"]
    if ruta.exists():
        curva = pq.read_table(ruta).to_pandas()
        curva.columns = cols
    else:
        curva = pd.DataFrame(columns=cols)
    base = curva[curva["politica"] == "none"].set_index("presupuesto")["riesgo"]
    curva["riesgo_sin_intervencion"] = curva["presupuesto"].map(base).astype(float)
    curva["evitado"] = curva["riesgo_sin_intervencion"] - curva["riesgo"]
    curva["pct_evitado"] = curva["evitado"] / curva["riesgo_sin_intervencion"].where(curva["riesgo_sin_intervencion"] > 0)
    orden = {p: i for i, p in enumerate(POLITICAS)}
    curva = curva.sort_values("presupuesto", kind="stable")
    curva = curva.iloc[curva["politica"].map(orden).fillna(9).argsort(kind="stable")]
    _escribir(curva, salida / "curva.parquet")
    return curva


def _franjas(reglas: dict) -> list[dict]:
    """Las franjas del reporte con su ronda: la intervención más temprana del día es la
    "nocturna"; las demás, la de "mediodia". Sin momento de intervención, sin ronda."""
    franjas = [{"nombre": f.get("nombre"), "inicio": f.get("inicio"), "fin": f.get("fin"),
                "momento_intervencion": f.get("momento_intervencion"),
                "hora_intervencion": _hora(f.get("momento_intervencion")), "ronda": None}
               for f in reglas.get("franjas") or []]
    horas = [f["hora_intervencion"] for f in franjas if f["hora_intervencion"] is not None]
    for f in franjas:
        if f["hora_intervencion"] is not None:
            f["ronda"] = "nocturna" if f["hora_intervencion"] == min(horas) else "mediodia"
    return franjas


def _siguiente_presupuesto(curva: pd.DataFrame, ref, presupuestos: list) -> dict:
    """El presupuesto que sigue al de referencia en la curva y cuánto riesgo evitado agrega
    (fracción del riesgo sin intervención), por política."""
    mayores = [b for b in presupuestos if ref is not None and b > ref]
    dato = {"presupuesto": None, "factor": None, "puntos": {}}
    if not mayores or not ref:
        return dato
    dato["presupuesto"], dato["factor"] = int(mayores[0]), mayores[0] / ref
    for politica in ("historical", "model"):
        c = curva[curva["politica"] == politica].set_index("presupuesto")["pct_evitado"]
        if ref in c.index and mayores[0] in c.index and not pd.isna(c[mayores[0]] - c[ref]):
            dato["puntos"][politica] = float(c[mayores[0]] - c[ref])
    return dato


def _evita(curva: pd.DataFrame, presupuesto) -> tuple[str | None, float | None]:
    """El modelo contra la regla histórica con un presupuesto de la curva: "mas", "menos" o
    "igual" (las bandas se traslapan), y la diferencia de riesgo (modelo menos regla)."""
    c = curva[curva["presupuesto"] == presupuesto].set_index("politica")
    if presupuesto is None or not {"historical", "model"} <= set(c.index):
        return None, None
    h, m = c.loc["historical"], c.loc["model"]
    if any(pd.isna(x) for x in (h["riesgo"], m["riesgo"], h["ic_bajo"], h["ic_alto"], m["ic_bajo"], m["ic_alto"])):
        return None, None
    evita = "mas" if m["ic_alto"] < h["ic_bajo"] else "menos" if m["ic_bajo"] > h["ic_alto"] else "igual"
    return evita, float(m["riesgo"] - h["riesgo"])


def _mitad(curva: pd.DataFrame, ref) -> dict:
    """La comparación con la mitad del presupuesto de referencia (ver el encabezado)."""
    dato = {"presupuesto": None, "modelo_evita": None, "diferencia_de_riesgo": None}
    if not ref:
        return dato
    cerca = [b for b in sorted(curva["presupuesto"].unique().tolist()) if 0.4 * ref <= b <= 0.6 * ref]
    if not cerca:
        return dato
    b = min(cerca, key=lambda x: (abs(x - ref / 2), x))
    evita, diferencia = _evita(curva, b)
    return {"presupuesto": int(b), "modelo_evita": evita, "diferencia_de_riesgo": diferencia}


def _umbral(curva: pd.DataFrame) -> dict:
    """Presupuesto desde el cual cada bici extra evita muy poco (ver el encabezado)."""
    por_politica = {}
    for politica in ("historical", "model"):
        c = curva[curva["politica"] == politica].sort_values("presupuesto")
        b, r = c["presupuesto"].tolist(), c["riesgo"].tolist()
        pasos = [{"desde": int(b[i]), "hasta": int(b[i + 1]),
                  "horas_evitadas_por_bici": (r[i] - r[i + 1]) / (b[i + 1] - b[i])}
                 for i in range(len(b) - 1) if b[i + 1] > b[i]]
        dato = {"presupuesto": None, "ganancia_inicial": None, "ganancia_despues": None, "pasos": pasos}
        if pasos and pasos[0]["horas_evitadas_por_bici"] > 0:
            inicial = pasos[0]["horas_evitadas_por_bici"]
            dato["ganancia_inicial"] = inicial
            for paso in pasos[1:]:
                if paso["horas_evitadas_por_bici"] < UMBRAL_GANANCIA * inicial:
                    dato["presupuesto"] = paso["desde"]
                    dato["ganancia_despues"] = paso["horas_evitadas_por_bici"]
                    break
        por_politica[politica] = dato
    return {"umbral_relativo": UMBRAL_GANANCIA, "politicas": por_politica}


def _limitaciones(reporte: dict) -> list[str]:
    """Primera oración de cada limitación del reporte, sin las que nombran tablas o
    columnas ni las de más de 140 letras (no caben en un renglón)."""
    lineas = []
    for texto in reporte.get("limitaciones") or []:
        texto = str(texto).replace("**", "")
        primera = re.split(r"(?<=[a-záéíóúñ\)])\.\s", texto, maxsplit=1)[0].strip()
        if "`" in primera or '"' in primera or not primera or len(primera) > 140:
            continue
        lineas.append(primera if primera.endswith(".") else primera + ".")
    return lineas


def _simulacion(reporte: dict, curva: pd.DataFrame, rec: pd.DataFrame, salida: Path) -> dict:
    reglas = reporte.get("reglas") or {}
    presupuestos = sorted(curva["presupuesto"].unique().tolist())
    ref = reglas.get("presupuesto_de_referencia_en_bicis")
    if ref is None and presupuestos and not rec.empty:
        # Sin reporte: el presupuesto más cercano a las bicis recibidas por día.
        mov = rec[(rec["politica"] != "none") & (rec["bicis"] > 0)]
        if not mov.empty:
            por_dia = mov.groupby(["politica", "fecha"])["bicis"].sum().groupby("politica").mean().max()
            ref = min(presupuestos, key=lambda b: (abs(b - por_dia), b))
    if ref is not None and int(ref) not in presupuestos:
        ref = None
    ref = None if ref is None else int(ref)

    resultados = {(r.get("politica"), r.get("presupuesto_del_dia_en_bicis")): r
                  for r in reporte.get("resultados") or []}
    referencia = {"presupuesto": ref, "bicis_por_estacion": _f(reglas.get("presupuesto_de_referencia")),
                  "riesgo_sin_intervencion": None, "politicas": {}}
    referencia["siguiente"] = _siguiente_presupuesto(curva, ref, presupuestos)
    horas = []
    if ref is not None:
        en_ref = curva[curva["presupuesto"] == ref].set_index("politica")
        sin = _f(en_ref["riesgo"].get("none"))
        referencia["riesgo_sin_intervencion"] = sin
        for politica in ("historical", "model"):
            if politica not in en_ref.index:
                continue
            fila = en_ref.loc[politica]
            dato = {"riesgo": _f(fila["riesgo"]), "ic_bajo": _f(fila["ic_bajo"]), "ic_alto": _f(fila["ic_alto"]),
                    "evitado": _f(fila["evitado"]), "evitado_ic_bajo": None, "evitado_ic_alto": None,
                    "pct_evitado": _f(fila["pct_evitado"]), "pct_ic_bajo": None, "pct_ic_alto": None,
                    "viajes_bloqueados_por_dia": None}
            res = resultados.get((politica, ref)) or {}
            ev = res.get("riesgo_simulado_evitado_por_dia") or {}
            if sin and ev.get("ic_bajo") is not None and ev.get("ic_alto") is not None:
                dato["evitado_ic_bajo"], dato["evitado_ic_alto"] = _f(ev["ic_bajo"]), _f(ev["ic_alto"])
                dato["pct_ic_bajo"], dato["pct_ic_alto"] = ev["ic_bajo"] / sin, ev["ic_alto"] / sin
            dato["viajes_bloqueados_por_dia"] = _f((res.get("viajes_bloqueados") or {}).get("por_dia"))
            referencia["politicas"][politica] = dato
        # Horas de riesgo por franja, por causa y (si el reporte lo trae) por tipo de estación.
        claves = (("franja", "riesgo_por_franja"), ("causa", "riesgo_por_tipo"),
                  ("tipo_estacion", "riesgo_por_tipo_de_estacion"), ("tipo_estacion", "riesgo_por_tipo_estacion"))
        for politica in POLITICAS:
            res = resultados.get((politica, ref)) or {}
            for corte, clave in claves:
                for grupo, v in (res.get(clave) or {}).items():
                    v = v if isinstance(v, dict) else {"media": v}
                    v = v.get("riesgo_por_dia", v)  # por tipo de estación el reporte anida la estadística
                    if corte == "tipo_estacion" and str(grupo) == "sin tipo":
                        grupo = "sin_dato"
                    horas.append({"corte": corte, "grupo": str(grupo), "politica": politica,
                                  "riesgo": _f(v.get("media")), "ic_bajo": _f(v.get("ic_bajo")),
                                  "ic_alto": _f(v.get("ic_alto"))})
    horas_df = pd.DataFrame(horas, columns=["corte", "grupo", "politica", "riesgo", "ic_bajo", "ic_alto"])
    for col in ("riesgo", "ic_bajo", "ic_alto"):
        horas_df[col] = horas_df[col].astype(float)
    # "Resto del día" solo existe si las franjas no cubren las 24 horas: sin riesgo, no se guarda.
    resto = (horas_df["corte"] == "franja") & (horas_df["grupo"] == "resto")
    if resto.any() and not (horas_df.loc[resto, "riesgo"].fillna(0) > 0).any():
        horas_df = horas_df[~resto]
    _escribir(horas_df, salida / "riesgo_horas.parquet")

    # Sensibilidad al inventario inicial.
    filas = []
    for f in (reporte.get("sensibilidad_al_inventario_inicial") or {}).get("filas") or []:
        sin = _f(f.get("riesgo_sin_intervencion_por_dia"))
        fila = {"inventario_inicial": f.get("inventario_inicial"), "riesgo_sin_intervencion": sin}
        for politica, v in (f.get("por_politica") or {}).items():
            ev = _f(v.get("riesgo_simulado_evitado_por_dia"))
            fila[politica] = {"riesgo": _f(v.get("riesgo_por_dia")), "evitado": ev,
                              "pct_evitado": ev / sin if sin and ev is not None else None}
        filas.append(fila)
    rango = {}
    for politica in ("historical", "model"):
        pcts = [f[politica]["pct_evitado"] for f in filas if f.get(politica) and f[politica]["pct_evitado"] is not None]
        evs = [f[politica]["evitado"] for f in filas if f.get(politica) and f[politica]["evitado"] is not None]
        if pcts:
            rango[politica] = {"pct_min": min(pcts), "pct_max": max(pcts),
                               "evitado_min": min(evs), "evitado_max": max(evs), "escenarios": len(pcts)}

    # Piso de azar: mover las mismas bicis sin información.
    azar = reporte.get("piso_de_azar") or {}
    ev_azar = _f((azar.get("riesgo_simulado_evitado_por_dia") or {}).get("media"))
    r_azar = _f((azar.get("riesgo_por_dia") or {}).get("media"))
    sin_azar = r_azar + ev_azar if r_azar is not None and ev_azar is not None else None
    piso = {"presupuesto": azar.get("presupuesto_del_dia_en_bicis"), "corridas": azar.get("corridas"),
            "riesgo": r_azar, "evitado": ev_azar,
            "pct_evitado": ev_azar / sin_azar if sin_azar else None}

    tramo = reporte.get("tramo") or {}
    dias = sorted(rec["fecha"].astype(str).unique().tolist()) if not rec.empty else []
    return {
        "simulacion": {
            "con_reporte": bool(reporte),
            "dias_evaluados": len(tramo.get("dias_evaluados") or dias),
            "inicio": (tramo.get("dias_evaluados") or dias or [None])[0],
            "fin": (tramo.get("dias_evaluados") or dias or [None])[-1],
            "estaciones_simuladas": (reporte.get("estaciones") or {}).get("simuladas")
            or (int(rec["estacion_id"].nunique()) if not rec.empty else None),
            "reglas_provisionales": (reglas.get("estado") == "provisional") if reglas.get("estado") else None,
            "bicis_minimas": reglas.get("bicis_minimas"),
            "lugares_libres_minimos": reglas.get("lugares_libres_minimos"),
            "nivel_del_intervalo": (reporte.get("intervalo") or {}).get("nivel"),
            "presupuestos": [int(b) for b in presupuestos],
            "politicas": [p for p in POLITICAS if p in set(curva["politica"])],
            "franjas": _franjas(reglas),
        },
        "referencia": referencia,
        "sensibilidad": {"filas": filas, "rango": rango},
        "piso_de_azar": piso,
        "presupuesto_umbral": _umbral(curva),
        "limitaciones": _limitaciones(reporte),
    }


def _comparacion(pronostico: dict, referencia: dict, curva: pd.DataFrame) -> dict:
    """El modelo contra la regla histórica: error de pronóstico y riesgo, lado a lado."""
    base = (pronostico.get("baseline") or {}).get("salidas") or {}
    lgbm = (pronostico.get("lgbm") or {}).get("salidas") or {}
    hist = referencia["politicas"].get("historical") or {}
    modelo = referencia["politicas"].get("model") or {}
    mejor = None
    if base.get("mae") is not None and lgbm.get("mae") is not None:
        mejor = lgbm["mae"] < base["mae"]
    evita = None  # "mas", "menos" o "igual" (los intervalos se traslapan)
    if hist.get("riesgo") is not None and modelo.get("riesgo") is not None:
        if modelo["ic_alto"] < hist["ic_bajo"]:
            evita = "mas"
        elif modelo["ic_bajo"] > hist["ic_alto"]:
            evita = "menos"
        else:
            evita = "igual"
    return {"modelo_pronostica_mejor": mejor, "modelo_evita": evita,
            "diferencia_de_riesgo": (modelo["riesgo"] - hist["riesgo"]) if evita else None,
            "mitad": _mitad(curva, referencia.get("presupuesto"))}


def _piso_del_pronostico(datos: Path) -> dict:
    """Error de pronosticar sin información (reports/piso_wape.json), en validación:
    {"media_global": {"salidas": {"wape", "mae"}, "llegadas": {...}}, "permutacion": {...}}."""
    claves = {"media global": "media_global", "permutación": "permutacion"}
    objetivos = {"dep": "salidas", "arr": "llegadas"}
    piso = {}
    for m in _leer_json(datos / "reports" / "piso_wape.json").get("metricas") or []:
        if (m.get("group_by") == "general" and m.get("split") == "val"
                and m.get("model") in claves and m.get("target") in objetivos):
            piso.setdefault(claves[m["model"]], {})[objetivos[m["target"]]] = {
                "wape": _f(m.get("wape")), "mae": _f(m.get("mae"))}
    return piso


def _contexto(datos: Path, estaciones: pd.DataFrame) -> dict:
    """Empleos alrededor de la estación típica de cada tipo (mediana), con
    station_context.parquet: {"por_tipo": {tipo: {"estaciones", "proporcion_empleos"}}}."""
    ruta = datos / "station_context.parquet"
    if not ruta.exists():
        return {"por_tipo": {}}
    ctx = pq.read_table(ruta, columns=["station_id", "proporcion_empleos"]).to_pandas()
    ctx["station_id"] = ctx["station_id"].astype(str)
    t = estaciones[["estacion_id", "tipo"]].merge(ctx, left_on="estacion_id", right_on="station_id")
    t = t[t["tipo"].notna() & t["proporcion_empleos"].notna()]
    por_tipo = {str(tipo): {"estaciones": int(len(g)), "proporcion_empleos": _f(g["proporcion_empleos"].median())}
                for tipo, g in t.groupby("tipo")}
    return {"por_tipo": por_tipo}


def _estacion_de_ejemplo(rec: pd.DataFrame, estaciones: pd.DataFrame, dias_con_pronostico: pd.DataFrame,
                         politica: str = "historical") -> dict:
    """La estación y el día con que abre "Pronóstico por estación" (regla en el encabezado)."""
    con_pronostico = estaciones[estaciones["tiene_pronostico"]].sort_values(
        ["salidas", "estacion_id"], ascending=[False, True])
    respaldo = {"estacion_id": con_pronostico["estacion_id"].iloc[0] if len(con_pronostico) else None,
                "fecha": None, "cumple_la_regla": False}
    candidatas = con_pronostico[(con_pronostico["tipo"] == "oficinas") & (con_pronostico["volumen"] == "alto")
                                & con_pronostico["simulada"]]
    if rec.empty or candidatas.empty:
        return respaldo
    r = rec[rec["estacion_id"].isin(set(candidatas["estacion_id"]))]
    # Solo días en que la estación también tiene pronóstico: la página dibuja las dos cosas.
    con = dias_con_pronostico.assign(fecha=pd.to_datetime(dias_con_pronostico["fecha"]))
    if "error" not in con.columns:
        con["error"] = float("nan")
    r = r.assign(fecha=pd.to_datetime(r["fecha"])).merge(con, on=["estacion_id", "fecha"])
    if r.empty:
        return respaldo
    laborable = pd.to_datetime(r["fecha"]).dt.weekday < 5
    franjas = rec["franja"].nunique()
    con_solucion = r[laborable].groupby(["estacion_id", "fecha"])["factible"].all()
    orden = {"bajo": 0, "medio": 1, "alto": 2}
    sin_mover = r[r["politica"] == "none"].set_index(["estacion_id", "fecha", "franja"])["nivel_riesgo"].map(orden)
    p = r[laborable & (r["politica"] == politica)].assign(mueve=lambda d: d["bicis"] != 0, movidas=lambda d: d["bicis"].abs())
    antes = sin_mover.reindex(pd.MultiIndex.from_frame(p[["estacion_id", "fecha", "franja"]])).to_numpy()
    p["mejora"] = p["nivel_riesgo"].map(orden).to_numpy() < antes
    dias = p.groupby(["estacion_id", "fecha"]).agg(franjas=("franja", "nunique"), mueve=("mueve", "all"),
                                                    movidas=("movidas", "sum"), mejoras=("mejora", "sum"))
    dias = dias[(dias["franjas"] == franjas) & dias["mueve"] & con_solucion.reindex(dias.index).fillna(False)]
    if dias.empty:
        return respaldo
    buenas = set(dias.index.get_level_values("estacion_id"))
    estacion_id = next(e for e in candidatas["estacion_id"] if e in buenas)  # la de más salidas
    # De los días que cumplen, el de error de pronóstico más cercano a la mediana de la estación
    # en sus días laborables simulados: un día típico, no uno raro.
    errores = (r[laborable & (r["estacion_id"] == estacion_id)].drop_duplicates("fecha")
               .set_index("fecha")["error"].astype(float))
    mediana = errores.median()
    del_dia = dias.loc[estacion_id].reset_index()
    del_dia["error"] = del_dia["fecha"].map(errores)
    del_dia["lejos"] = (del_dia["error"] - mediana).abs().fillna(float("inf")) if pd.notna(mediana) else 0.0
    del_dia = del_dia.sort_values(["lejos", "mejoras", "movidas", "fecha"], ascending=[True, False, False, True])
    elegido = del_dia.iloc[0]
    return {"estacion_id": estacion_id, "fecha": str(pd.Timestamp(elegido["fecha"]).date()),
            "cumple_la_regla": True, "error_del_dia": _f(elegido["error"]), "error_mediano": _f(mediana),
            "dias_laborables_simulados": int(errores.notna().sum())}


UNIDADES = ("pct", "horas", "viajes", "numero")


def _par(x) -> list | None:
    """Un intervalo como [bajo, alto], venga como lista o como diccionario."""
    if isinstance(x, dict):
        x = [x.get("ic_bajo", x.get("bajo")), x.get("ic_alto", x.get("alto"))]
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return None
    try:
        par = [_f(x[0]), _f(x[1])]
    except (TypeError, ValueError):
        return None
    return None if None in par else par


def _validacion_y_prueba(datos: Path) -> dict | None:
    """Copia reports/validacion_y_prueba.json (validación y prueba final, lado a
    lado). Sin archivo, o si no trae filas que se puedan leer, es `None`."""
    crudo = _leer_json(datos / "reports" / "validacion_y_prueba.json")
    filas = []
    for f in crudo.get("filas") or []:
        if not isinstance(f, dict) or not str(f.get("afirmacion") or "").strip():
            continue
        try:
            validacion, prueba = _f(f.get("validacion")), _f(f.get("prueba"))
        except (TypeError, ValueError):
            continue
        filas.append({"clave": str(f.get("clave") or ""), "afirmacion": str(f["afirmacion"]).strip(),
                      "unidad": f.get("unidad") if f.get("unidad") in UNIDADES else "numero",
                      "validacion": validacion, "prueba": prueba,
                      "validacion_ic": _par(f.get("validacion_ic")), "prueba_ic": _par(f.get("prueba_ic"))})
    if not filas:
        return None
    presupuesto = crudo.get("presupuesto_de_referencia")
    entero = lambda x: int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None  # noqa: E731
    detalle = crudo.get("detalle") if isinstance(crudo.get("detalle"), dict) else {}
    por_periodo = {periodo: entero((detalle.get(periodo) or {}).get("presupuesto_de_referencia_en_bicis"))
                   for periodo in ("validacion", "prueba")}
    return {"presupuesto_de_referencia": entero(presupuesto), "presupuestos": por_periodo, "filas": filas}


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #

def build(datos: Path | str = DATOS, salida: Path | str = SALIDA) -> dict:
    """Construye todas las tablas de la app. Devuelve el diccionario de indicadores."""
    datos, salida = Path(datos), Path(salida)
    for nombre in ("stations", "station_hour", "predictions", "routes"):
        if not (datos / f"{nombre}.parquet").exists():
            raise FileNotFoundError(f"Falta {datos / (nombre + '.parquet')}")
    salida.mkdir(parents=True, exist_ok=True)
    reporte = _leer_json(datos / "reports" / "simulacion.json")

    con = duckdb.connect()
    try:
        con.execute(f"SET TimeZone = '{ZONA}'")
        demanda = _demanda(con, datos, salida)
        pron = _pronostico(con, datos, salida, demanda["estaciones_df"])
        rec, estaciones = _recomendaciones(con, datos, salida, reporte, pron["estaciones_df"])
    finally:
        con.close()
    _escribir(estaciones, salida / "estaciones.parquet")
    curva = _curva(datos, salida)
    sim = _simulacion(reporte, curva, rec, salida)

    indicadores = {
        # data/sample es sintética: la app lo dice mientras la use.
        "sinteticos": datos.resolve().name == "sample",
        **demanda["resumen"],
        "validacion": pron["validacion"],
        "pronostico": pron["pronostico"],
        **sim,
        "comparacion": _comparacion(pron["pronostico"], sim["referencia"], curva),
        "pronostico_piso": _piso_del_pronostico(datos),
        "contexto": _contexto(datos, estaciones),
        "estacion_de_ejemplo": _estacion_de_ejemplo(rec, estaciones, pron["dias_df"]),
        "validacion_y_prueba": _validacion_y_prueba(datos),
        "bloques_de_hora": [{"nombre": n, "desde": a, "hasta": b} for n, a, b in BLOQUES],
    }
    (salida / INDICADORES).write_text(
        json.dumps(indicadores, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return indicadores
