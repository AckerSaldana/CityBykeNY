"""El constructor de las tablas pequeñas del dashboard (src/app_tables.py).

Se corre sobre data/sample hacia una carpeta temporal. Comprueba que las tablas
existen, que son pequeñas, que traen las columnas que usa app/datos.py y que
los indicadores se calculan a partir de las tablas (no están escritos a mano).

    pytest tests/test_app_tables.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"

COLUMNAS = {
    "estaciones": ["estacion_id", "nombre", "lat", "lon", "capacidad", "tipo", "salidas", "llegadas",
                   "dias_con_datos", "salidas_dia", "llegadas_dia", "flujo_neto_dia", "volumen",
                   "tiene_pronostico", "sin_historia", "simulada"],
    "estacion_hora": ["estacion_id", "hora", "tipo_dia", "salidas", "salidas_miembros",
                      "salidas_ocasionales", "llegadas", "flujo_neto"],
    "demanda_hora": ["tipo_dia", "hora", "salidas", "salidas_miembros", "salidas_ocasionales", "llegadas", "dias"],
    "top_rutas": ["puesto", "origen_id", "origen", "destino_id", "destino", "viajes", "origen_lat",
                  "origen_lon", "destino_lat", "destino_lon", "misma_estacion", "ruta"],
    "recomendaciones": ["estacion_id", "fecha", "franja", "hora_inicio", "hora_fin", "politica", "bicis",
                        "nivel_riesgo", "factible", "inventario", "inventario_antes"],
    "curva": ["politica", "presupuesto", "riesgo", "ic_bajo", "ic_alto", "riesgo_sin_intervencion",
              "evitado", "pct_evitado"],
    "metricas": ["corte", "grupo", "orden", "modelo", "objetivo", "n", "mae", "rmse", "wape", "cobertura", "ancho"],
    "riesgo_horas": ["corte", "grupo", "politica", "riesgo", "ic_bajo", "ic_alto"],
    "riesgo_niveles": ["corte", "grupo", "politica", "alto", "medio", "bajo", "total", "pct_alto"],
}


@pytest.fixture(scope="module")
def construido(tmp_path_factory):
    salida = tmp_path_factory.mktemp("app")
    return salida, app_tables.build(SAMPLE, salida)


def leer(carpeta: Path, nombre: str) -> pd.DataFrame:
    return pd.read_parquet(carpeta / f"{nombre}.parquet")


def test_escribe_todas_las_tablas_y_son_pequenas(construido):
    salida, _ = construido
    for nombre in app_tables.TABLAS:
        ruta = salida / f"{nombre}.parquet"
        assert ruta.exists(), nombre
        assert ruta.stat().st_size < 1_000_000, nombre
    assert (salida / app_tables.INDICADORES).exists()


@pytest.mark.parametrize("nombre", sorted(COLUMNAS))
def test_columnas_esperadas(construido, nombre):
    assert list(leer(construido[0], nombre).columns) == COLUMNAS[nombre]


def test_los_indicadores_salen_de_las_tablas(construido):
    salida, ind = construido
    sh = pd.read_parquet(SAMPLE / "station_hour.parquet")
    stations = pd.read_parquet(SAMPLE / "stations.parquet")
    assert ind == json.loads((salida / app_tables.INDICADORES).read_text(encoding="utf-8"))
    assert ind["sinteticos"] is True
    assert ind["viajes_analizados"] == int(sh["departures"].sum())
    assert ind["estaciones"] == len(stations)
    assert ind["estaciones_con_capacidad"] == int(stations["capacity"].notna().sum())

    # El % de riesgo evitado sale de la curva, con el presupuesto de referencia.
    curva = pd.read_parquet(SAMPLE / "policy_curve.parquet")
    ref = ind["referencia"]["presupuesto"]
    assert ref in set(curva["budget"])
    en_ref = curva[curva["budget"] == ref].set_index("policy")["risk_station_hours"]
    esperado = (en_ref["none"] - en_ref["historical"]) / en_ref["none"]
    assert ind["referencia"]["politicas"]["historical"]["pct_evitado"] == pytest.approx(esperado)


def test_las_metricas_coinciden_con_un_calculo_aparte(construido):
    salida, ind = construido
    pred = pd.read_parquet(SAMPLE / "predictions.parquet")
    sh = pd.read_parquet(SAMPLE / "station_hour.parquet")
    val = pred[(pred["split"] == "val") & (pred["model"] == "baseline")].merge(sh, on=["station_id", "ts_hour"])
    error = (val["departures"] - val["dep_q50"]).abs()
    dato = ind["pronostico"]["baseline"]["salidas"]
    assert dato["n"] == len(val)
    assert dato["mae"] == pytest.approx(error.mean(), rel=1e-5)
    assert dato["wape"] == pytest.approx(error.sum() / val["departures"].sum(), rel=1e-5)
    dentro = (val["departures"] >= val["dep_q10"]) & (val["departures"] <= val["dep_q90"])
    assert dato["cobertura"] == pytest.approx(dentro.mean(), abs=1e-4)
    m = leer(salida, "metricas")
    assert set(m["corte"]) == {"general", "bloque", "hora", "volumen"}
    assert set(m["modelo"]) <= {"baseline", "lgbm"}


def test_la_prueba_final_no_entra_al_pronostico(construido):
    """Decisión 4: lo observado solo se cruza con el pronóstico en validación."""
    salida, ind = construido
    pred = pd.read_parquet(SAMPLE / "predictions.parquet")
    local = pred["ts_hour"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    de_val = set(local[pred["split"] == "val"])
    pron = leer(salida, "pronostico")
    assert set(pron["ts"]) == de_val
    solo_prueba = set(local[pred["split"] == "test"]) - de_val
    assert solo_prueba and not (set(pron["ts"]) & solo_prueba)
    assert ind["validacion"]["dias"] == pron["fecha"].nunique()
    # Ordenada por estación: así el dashboard lee una sola sin cargar todo.
    assert pron["estacion_id"].is_monotonic_increasing


def test_estacion_por_hora_separa_usuarios_solo_en_salidas(construido):
    eh = leer(construido[0], "estacion_hora")
    assert set(eh["tipo_dia"]) == {"laborable", "fin_de_semana"}
    assert not eh.duplicated(["estacion_id", "hora", "tipo_dia"]).any()
    suma = eh["salidas_miembros"] + eh["salidas_ocasionales"]
    assert (suma <= eh["salidas"] + 0.01).all()


def test_el_umbral_de_presupuesto_se_calcula_con_la_curva():
    curva = pd.DataFrame({
        "politica": ["historical"] * 4,
        "presupuesto": [0, 10, 20, 40],
        "riesgo": [100.0, 60.0, 40.0, 39.0],  # 4.0, 2.0 y 0.05 horas por bici
    })
    dato = app_tables._umbral(curva)["politicas"]["historical"]
    assert dato["presupuesto"] == 20
    assert dato["ganancia_inicial"] == pytest.approx(4.0)
    assert dato["ganancia_despues"] == pytest.approx(0.05)
    # Si la ganancia nunca cae bajo el umbral, no hay presupuesto que reportar.
    curva["riesgo"] = [100.0, 60.0, 30.0, 0.0]
    assert app_tables._umbral(curva)["politicas"]["historical"]["presupuesto"] is None


def test_sin_reporte_ni_tablas_simuladas_no_falla(tmp_path):
    """Lo que falta queda vacío o nulo ("Sin dato" en la app), no tumba la construcción."""
    datos = tmp_path / "datos"
    datos.mkdir()
    for nombre in ("stations", "station_hour", "predictions", "routes"):
        shutil.copy(SAMPLE / f"{nombre}.parquet", datos / f"{nombre}.parquet")
    ind = app_tables.build(datos, tmp_path / "app")
    assert ind["referencia"]["presupuesto"] is None
    assert ind["referencia"]["politicas"] == {}
    assert ind["limitaciones"] == []
    assert ind["sensibilidad"]["filas"] == []
    assert ind["comparacion"]["modelo_evita"] is None
    assert leer(tmp_path / "app", "curva").empty
    assert leer(tmp_path / "app", "recomendaciones").empty


def test_con_reporte_toma_referencia_intervalo_sensibilidad_y_limitaciones(tmp_path):
    datos = tmp_path / "datos"
    shutil.copytree(SAMPLE, datos)
    curva = pd.read_parquet(SAMPLE / "policy_curve.parquet")
    ref = int(sorted(curva["budget"].unique())[2])
    sin = float(curva[(curva["policy"] == "none") & (curva["budget"] == ref)]["risk_station_hours"].iloc[0])
    reporte = {
        "reglas": {"estado": "provisional", "presupuesto_de_referencia_en_bicis": ref,
                   "franjas": [{"nombre": "manana", "inicio": "06:00", "fin": "12:00"}]},
        "resultados": [{"politica": "historical", "presupuesto_del_dia_en_bicis": ref,
                        "riesgo_simulado_evitado_por_dia": {"media": 1.0, "ic_bajo": sin * 0.1, "ic_alto": sin * 0.3},
                        "riesgo_por_franja": {"manana": {"media": 5.0, "ic_bajo": 4.0, "ic_alto": 6.0},
                                              "resto": {"media": 0.0, "ic_bajo": 0.0, "ic_alto": 0.0}}}],
        "piso_de_azar": {"corridas": 5, "presupuesto_del_dia_en_bicis": ref,
                         "riesgo_por_dia": {"media": 92.0}, "riesgo_simulado_evitado_por_dia": {"media": 8.0}},
        "sensibilidad_al_inventario_inicial": {"filas": [
            {"inventario_inicial": "a", "riesgo_sin_intervencion_por_dia": 100.0,
             "por_politica": {"historical": {"riesgo_por_dia": 80.0, "riesgo_simulado_evitado_por_dia": 20.0}}},
            {"inventario_inicial": "b", "riesgo_sin_intervencion_por_dia": 200.0,
             "por_politica": {"historical": {"riesgo_por_dia": 100.0, "riesgo_simulado_evitado_por_dia": 100.0}}}]},
        "limitaciones": ["La distancia es en línea recta. Ignora ríos.", "La tabla `predictions` no cuenta."],
    }
    (datos / "reports").mkdir()
    (datos / "reports" / "simulacion.json").write_text(json.dumps(reporte), encoding="utf-8")
    piso = {"metricas": [{"model": "media global", "split": "val", "target": "dep", "group_by": "general",
                          "group": "todas", "wape": 1.16, "mae": 3.5},
                         {"model": "media global", "split": "val", "target": "dep", "group_by": "franja",
                          "group": "noche", "wape": 2.8, "mae": 2.7}]}
    (datos / "reports" / "piso_wape.json").write_text(json.dumps(piso), encoding="utf-8")
    ind = app_tables.build(datos, tmp_path / "app")

    assert ind["referencia"]["presupuesto"] == ref
    hist = ind["referencia"]["politicas"]["historical"]
    assert hist["pct_ic_bajo"] == pytest.approx(0.1) and hist["pct_ic_alto"] == pytest.approx(0.3)
    assert ind["sensibilidad"]["rango"]["historical"]["pct_min"] == pytest.approx(0.2)
    assert ind["sensibilidad"]["rango"]["historical"]["pct_max"] == pytest.approx(0.5)
    assert ind["limitaciones"] == ["La distancia es en línea recta."]
    assert ind["simulacion"]["reglas_provisionales"] is True
    horas = leer(tmp_path / "app", "riesgo_horas")
    assert horas.to_dict("records") == [{"corte": "franja", "grupo": "manana", "politica": "historical",
                                         "riesgo": 5.0, "ic_bajo": 4.0, "ic_alto": 6.0}]
    # El piso de azar y el error de una sola media salen de los reportes; "resto" sin riesgo no se guarda.
    assert ind["piso_de_azar"]["pct_evitado"] == pytest.approx(0.08)
    assert ind["pronostico_piso"]["media_global"]["salidas"]["wape"] == pytest.approx(1.16)
    assert ind["contexto"]["por_tipo"] == {}


def test_inventario_antes_y_estacion_de_ejemplo(construido):
    salida, ind = construido
    rec = leer(salida, "recomendaciones")
    assert (rec["inventario_antes"] + rec["bicis"] == rec["inventario"]).all()
    sin_mover = rec[rec["politica"] == "none"]
    assert (sin_mover["inventario_antes"] == sin_mover["inventario"]).all()

    # La estación de ejemplo sale de las tablas; si ninguna cumple la regla, es la de más salidas con pronóstico.
    est = leer(salida, "estaciones")
    ejemplo = ind["estacion_de_ejemplo"]
    con_pronostico = est[est["tiene_pronostico"]].sort_values(["salidas", "estacion_id"], ascending=[False, True])
    if not ejemplo["cumple_la_regla"]:
        assert ejemplo["estacion_id"] == con_pronostico["estacion_id"].iloc[0] and ejemplo["fecha"] is None


def test_la_estacion_de_ejemplo_sigue_la_regla():
    """Oficinas, demanda alta, traslado en todas las franjas, con solución y en día laborable."""
    est = pd.DataFrame({"estacion_id": ["A", "B", "C", "D"], "salidas": [900, 800, 700, 600],
                        "tipo": ["residencial", "oficinas", "oficinas", "oficinas"],
                        "volumen": ["alto", "alto", "alto", "medio"], "tiene_pronostico": True, "simulada": True})

    def filas(estacion, fecha, bicis, factible=(True, True), riesgo=("alto", "alto")):
        return [{"estacion_id": estacion, "fecha": pd.Timestamp(fecha), "franja": f, "politica": p,
                 "bicis": b if p == "historical" else 0, "factible": ok,
                 "nivel_riesgo": r if p == "historical" else "alto"}
                for f, b, ok, r in zip(("manana", "tarde"), bicis, factible, riesgo) for p in ("none", "historical")]

    rec = pd.DataFrame(
        filas("A", "2026-09-08", (5, -5))                                   # no es de oficinas
        + filas("B", "2026-09-08", (5, 0))                                  # una franja sin traslado
        + filas("B", "2026-09-09", (5, -5), factible=(True, False))         # una franja sin solución
        + filas("B", "2026-09-12", (9, -9))                                 # sábado
        + filas("C", "2026-09-08", (9, -9))                                 # cumple, sin bajar el riesgo
        + filas("C", "2026-09-10", (2, -2), riesgo=("medio", "alto"))       # cumple y baja el riesgo en una franja
        + filas("D", "2026-09-08", (5, -5)))                                # demanda media
    dias = rec[["estacion_id", "fecha"]].drop_duplicates()

    def elegido(dias):
        e = app_tables._estacion_de_ejemplo(rec, est, dias)
        return {k: e[k] for k in ("estacion_id", "fecha", "cumple_la_regla")}

    # Sin error de pronóstico por día: el día con más franjas que bajan el riesgo.
    assert elegido(dias) == {"estacion_id": "C", "fecha": "2026-09-10", "cumple_la_regla": True}
    # Con error: de los días que cumplen, el más cercano a la mediana de la estación (un día típico).
    mas = pd.DataFrame(filas("C", "2026-09-09", (3, 0)) + filas("C", "2026-09-11", (4, -4)))  # el 9 no cumple
    rec = pd.concat([rec, mas], ignore_index=True)
    dias = rec[["estacion_id", "fecha"]].drop_duplicates()
    errores = {"2026-09-08": 0.31, "2026-09-09": 0.30, "2026-09-10": 0.90, "2026-09-11": 0.10}  # mediana: 0.305
    con_error = dias.assign(error=dias["fecha"].dt.strftime("%Y-%m-%d").map(errores))
    tipico = app_tables._estacion_de_ejemplo(rec, est, con_error)
    assert (tipico["estacion_id"], tipico["fecha"]) == ("C", "2026-09-08")
    assert tipico["error_del_dia"] == pytest.approx(0.31) and tipico["error_mediano"] == pytest.approx(0.305)
    assert tipico["dias_laborables_simulados"] == 4
    # Si ninguna cumple: la de más salidas.
    solo_a = app_tables._estacion_de_ejemplo(rec[rec["estacion_id"] == "A"], est, dias)
    assert solo_a == {"estacion_id": "A", "fecha": None, "cumple_la_regla": False}


def test_rondas_siguiente_presupuesto_mitad_y_distancia_de_rutas(construido):
    _, ind = construido
    # Rondas: la intervención más temprana es la nocturna; las demás, de mediodía.
    franjas = app_tables._franjas({"franjas": [
        {"nombre": "tarde", "momento_intervencion": "14:00", "inicio": "14:00", "fin": "22:00"},
        {"nombre": "manana", "momento_intervencion": "05:00", "inicio": "06:00", "fin": "14:00"},
        {"nombre": "rara", "inicio": "22:00", "fin": "23:00"}]})
    assert [(f["nombre"], f["ronda"], f["hora_intervencion"]) for f in franjas] == [
        ("tarde", "mediodia", 14), ("manana", "nocturna", 5), ("rara", None, None)]

    curva = pd.DataFrame({
        "politica": ["historical"] * 4 + ["model"] * 4, "presupuesto": [0, 50, 100, 190] * 2,
        "riesgo": [100, 70, 50, 46, 100, 55, 49, 47], "ic_bajo": [95, 66, 45, 42, 95, 50, 44, 43],
        "ic_alto": [105, 74, 55, 50, 105, 60, 54, 51], "pct_evitado": [0, .30, .50, .54, 0, .45, .51, .53]})
    siguiente = app_tables._siguiente_presupuesto(curva, 100, [0, 50, 100, 190])
    assert siguiente["presupuesto"] == 190 and siguiente["factor"] == pytest.approx(1.9)
    assert siguiente["puntos"] == pytest.approx({"historical": 0.04, "model": 0.02})
    assert app_tables._siguiente_presupuesto(curva, 190, [0, 50, 100, 190])["presupuesto"] is None
    assert app_tables._mitad(curva, 100) == {"presupuesto": 50, "modelo_evita": "mas", "diferencia_de_riesgo": -15.0}
    assert app_tables._mitad(curva, 190)["modelo_evita"] == "igual"  # 100 queda entre 40 % y 60 % de 190
    assert app_tables._mitad(curva, 50)["presupuesto"] is None

    # Distancia de las rutas más usadas entre estaciones distintas, en línea recta.
    rutas = pd.read_parquet(construido[0] / "top_rutas.parquet")
    r = rutas[~rutas["misma_estacion"]].head(app_tables.RUTAS_EN_PANTALLA)
    km = app_tables._km(r["origen_lat"], r["origen_lon"], r["destino_lat"], r["destino_lon"])
    assert ind["rutas_mas_usadas"] == {"rutas": len(r), "distancia_mediana_km": pytest.approx(km.median()),
                                       "distancia_maxima_km": pytest.approx(km.max())}
    un_grado = app_tables._km(pd.Series([0.0]), pd.Series([0.0]), pd.Series([1.0]), pd.Series([0.0]))
    assert un_grado[0] == pytest.approx(111.2, abs=0.1)


def test_copia_la_validacion_y_la_prueba_final_solo_si_existe(tmp_path):
    datos = tmp_path / "datos"
    shutil.copytree(SAMPLE, datos)
    assert app_tables.build(datos, tmp_path / "app")["validacion_y_prueba"] is None

    (datos / "reports").mkdir()
    falso = {"presupuesto_de_referencia": 172, "filas": [
        {"clave": "evitado", "afirmacion": "Riesgo simulado evitado", "unidad": "pct", "validacion": 0.5,
         "prueba": 0.4, "validacion_ic": [0.4, 0.6], "prueba_ic": None},
        {"clave": "raro", "afirmacion": "Unidad desconocida", "unidad": "litros", "validacion": 3, "prueba": None},
        {"clave": "sin_texto", "afirmacion": "", "unidad": "pct", "validacion": 1, "prueba": 1}]}
    (datos / "reports" / "validacion_y_prueba.json").write_text(json.dumps(falso), encoding="utf-8")
    copia = app_tables.build(datos, tmp_path / "app")["validacion_y_prueba"]
    assert copia["presupuesto_de_referencia"] == 172
    assert copia["presupuestos"] == {"validacion": None, "prueba": None}
    assert [f["clave"] for f in copia["filas"]] == ["evitado", "raro"]
    assert copia["filas"][0]["validacion_ic"] == [0.4, 0.6] and copia["filas"][0]["prueba_ic"] is None
    assert copia["filas"][1]["unidad"] == "numero" and copia["filas"][1]["prueba"] is None

    # Un archivo que no se puede leer no rompe la construcción.
    (datos / "reports" / "validacion_y_prueba.json").write_text("{esto no es json", encoding="utf-8")
    assert app_tables.build(datos, tmp_path / "app")["validacion_y_prueba"] is None


def test_es_determinista(construido, tmp_path):
    salida, _ = construido
    app_tables.build(SAMPLE, tmp_path)
    for ruta in sorted(salida.iterdir()):
        assert ruta.read_bytes() == (tmp_path / ruta.name).read_bytes(), ruta.name


def test_el_script_acepta_datos_y_salida(tmp_path):
    corrida = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_app_tables.py"), "--datos", str(SAMPLE), "--salida", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert corrida.returncode == 0, corrida.stderr
    assert (tmp_path / "indicadores.json").exists()
