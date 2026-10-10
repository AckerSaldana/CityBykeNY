"""Acceso a los datos del tablero. Es el ÚNICO archivo que sabe dónde están.

Las páginas no leen archivos: llaman a las funciones de aquí. Todas leen
tablas pequeñas, ya resumidas, de `data/processed/app/`. Esas tablas se
construyen (y se reconstruyen después de cada simulación) con:

    python scripts/build_app_tables.py

Reglas para las páginas:

- Los números se escriben con `num()` y `pct()`; lo que falta, con `texto()`
  (muestra "Sin dato").
- Las claves internas (políticas, franjas, tipos...) se traducen con los
  diccionarios de ETIQUETAS de abajo. En pantalla no van nombres de columnas
  ni de archivos.
- Ninguna cifra se escribe a mano: todo sale de `indicadores()` y de las
  tablas. Cuando cambie la simulación solo se vuelve a correr el constructor.
- Cada página empieza con `datos.encabezado("Título")`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

# Carpeta con las tablas pequeñas. Las pruebas la cambian por una temporal.
CARPETA = Path(__file__).resolve().parents[1] / "data" / "processed" / "app"

LINEA = "Resultados de simulación con datos de validación"
SIN_DATO = "Sin dato"
CONSTRUIR = "python scripts/build_app_tables.py"

# --------------------------------------------------------------------------- #
# Etiquetas (clave interna -> texto en pantalla)
# --------------------------------------------------------------------------- #

POLITICAS = {"none": "Sin intervención", "historical": "Regla histórica", "model": "Modelo"}
MODELOS = {"baseline": "Promedio histórico", "lgbm": "Modelo"}
FRANJAS = {"manana": "Mañana", "tarde": "Tarde", "noche": "Noche", "resto": "Resto del día"}
RIESGOS = {"alto": "Alto", "medio": "Medio", "bajo": "Bajo"}
TIPOS_DIA = {"laborable": "Laborable", "fin_de_semana": "Fin de semana"}
USUARIOS = {"todos": "Todos", "miembros": "Miembros", "ocasionales": "Ocasionales"}
TIPOS_ESTACION = {"oficinas": "Oficinas", "residencial": "Residencial", "recreativa": "Recreativa",
                  "mixta": "Mixta", "sin_dato": SIN_DATO}
BLOQUES = {"madrugada": "Madrugada (00 a 05)", "manana": "Mañana (06 a 11)",
           "tarde": "Tarde (12 a 17)", "noche": "Noche (18 a 23)"}
VOLUMENES = {"bajo": "Demanda baja", "medio": "Demanda media", "alto": "Demanda alta"}
CAUSAS = {"pocas_bicis": "Casi vacía", "pocos_lugares": "Casi llena"}
OBJETIVOS = {"salidas": "Salidas", "llegadas": "Llegadas"}

# Colores fijos para que las cinco páginas se vean iguales.
COLORES_POLITICA = {"none": "#8A8F98", "historical": "#E69F00", "model": "#0072B2"}
COLORES_RIESGO = {"alto": "#D55E00", "medio": "#E69F00", "bajo": "#009E73"}
COLORES_MODELO = {"baseline": "#E69F00", "lgbm": "#0072B2"}
COLOR_OBSERVADO = "#222222"
COLOR_SE_VACIA = "#D55E00"  # flujo neto negativo: salen más bicis de las que llegan
COLOR_SE_LLENA = "#0072B2"  # flujo neto positivo: llegan más de las que salen

_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]
_DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


# --------------------------------------------------------------------------- #
# Formato
# --------------------------------------------------------------------------- #

def _falta(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def num(x, decimales: int = 0) -> str:
    """Número con separador de miles y, como mucho, un decimal: 15,017,726 o 4.8."""
    if _falta(x):
        return SIN_DATO
    decimales = max(0, min(int(decimales), 1))
    return f"{float(x):,.{decimales}f}"


def pct(x, decimales: int = 1) -> str:
    """Recibe una fracción (0.539) y devuelve "53.9 %"."""
    if _falta(x):
        return SIN_DATO
    decimales = max(0, min(int(decimales), 1))
    return f"{float(x) * 100:,.{decimales}f} %"


def rango_pct(a, b, decimales: int = 0) -> str:
    """ "31 % a 54 %", o "Sin dato" si falta alguno."""
    if _falta(a) or _falta(b):
        return SIN_DATO
    return f"{pct(a, decimales)} a {pct(b, decimales)}"


def texto(x) -> str:
    """El valor como texto, o "Sin dato" si viene vacío."""
    if _falta(x) or str(x).strip() == "":
        return SIN_DATO
    return str(x)


def etiqueta(mapa: dict, clave) -> str:
    """Traduce una clave interna con uno de los diccionarios; lo desconocido es "Sin dato"."""
    if _falta(clave):
        return SIN_DATO
    return mapa.get(clave, SIN_DATO)


def hora(h) -> str:
    """6 -> "06:00"."""
    return SIN_DATO if _falta(h) else f"{int(h):02d}:00"


def fecha(d, con_dia: bool = True) -> str:
    """Fecha en español: "miércoles 2 de septiembre de 2026"."""
    if _falta(d):
        return SIN_DATO
    d = pd.Timestamp(d)
    base = f"{d.day} de {_MESES[d.month - 1]} de {d.year}"
    return f"{_DIAS[d.weekday()]} {base}" if con_dia else base


def accion(bicis, hora_inicio=None) -> str:
    """Texto de la tarjeta de acción: "Llevar 5 bicis antes de las 06:00"."""
    if _falta(bicis):
        return SIN_DATO
    bicis = int(bicis)
    if bicis == 0:
        return "Sin traslado"
    verbo = "Llevar" if bicis > 0 else "Retirar"
    unidad = "bici" if abs(bicis) == 1 else "bicis"
    cuando = "" if _falta(hora_inicio) else f" antes de las {hora(hora_inicio)}"
    return f"{verbo} {num(abs(bicis))} {unidad}{cuando}"


# --------------------------------------------------------------------------- #
# Lectura con caché
# --------------------------------------------------------------------------- #

def _faltan_tablas():
    """Mensaje amable (no un error de Python) cuando falta construir las tablas."""
    mensaje = ("Todavía no están listas las tablas del tablero. Desde la carpeta del proyecto corre "
               f"`{CONSTRUIR}` y vuelve a cargar esta página.")
    st.info(mensaje)
    st.stop()
    raise FileNotFoundError(mensaje)  # solo se llega aquí fuera de Streamlit


def _ruta(nombre: str) -> tuple[str, float]:
    """Ruta del archivo y su fecha de cambio. La fecha entra a la llave de la
    caché: al reconstruir las tablas, la app toma las nuevas sin reiniciar."""
    ruta = CARPETA / nombre
    if not ruta.exists():
        _faltan_tablas()
    return str(ruta), ruta.stat().st_mtime


@st.cache_data(show_spinner=False)
def _tabla(ruta: str, cambio: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


@st.cache_data(show_spinner=False, max_entries=200)
def _tabla_de_estacion(ruta: str, cambio: float, estacion_id: str) -> pd.DataFrame:
    # El archivo está ordenado por estación: el filtro lee solo su pedazo.
    return pq.read_table(ruta, filters=[("estacion_id", "=", estacion_id)]).to_pandas()


@st.cache_data(show_spinner=False)
def _json(ruta: str, cambio: float) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _leer(nombre: str) -> pd.DataFrame:
    return _tabla(*_ruta(f"{nombre}.parquet")).copy()


def indicadores() -> dict:
    """Todas las cifras sueltas del tablero (ver app/LEEME.md para las llaves)."""
    return _json(*_ruta("indicadores.json"))


def estaciones() -> pd.DataFrame:
    """Una fila por estación: estacion_id, nombre, lat, lon, capacidad, tipo,
    salidas, llegadas, dias_con_datos, salidas_dia, llegadas_dia, flujo_neto_dia,
    volumen, tiene_pronostico, sin_historia, simulada."""
    return _leer("estaciones")


def estacion(estacion_id: str) -> dict:
    """Los datos de una estación como diccionario ({} si no existe)."""
    e = _tabla(*_ruta("estaciones.parquet"))
    fila = e[e["estacion_id"] == estacion_id]
    return {} if fila.empty else fila.iloc[0].to_dict()


def opciones_de_estacion(solo_con_pronostico: bool = True) -> dict:
    """{estacion_id: nombre} para un selector, de mayor a menor número de salidas.
    Dos estaciones con el mismo nombre se distinguen con su clave."""
    e = estaciones()
    if solo_con_pronostico:
        e = e[e["tiene_pronostico"]]
    e = e.sort_values(["salidas", "estacion_id"], ascending=[False, True])
    repetido = e["nombre"].duplicated(keep=False)
    nombres = e["nombre"].where(~repetido, e["nombre"] + " (" + e["estacion_id"] + ")")
    return dict(zip(e["estacion_id"], nombres))


def demanda_mapa(hora_del_dia: int | None = None, tipo_dia: str = "laborable",
                 usuario: str = "todos") -> pd.DataFrame:
    """Una fila por estación para el mapa: estacion_id, nombre, lat, lon,
    capacidad, tipo, salidas, llegadas, flujo_neto.

    Con una hora, son promedios de esa hora; con `None`, del día completo.
    `usuario` ("todos", "miembros", "ocasionales") solo cambia las salidas: los
    datos no separan las llegadas por tipo de usuario, así que llegadas y flujo
    neto son siempre de todos los usuarios.
    """
    return _demanda_mapa(*_ruta("estacion_hora.parquet"), *_ruta("estaciones.parquet"),
                         hora_del_dia, tipo_dia, usuario).copy()


@st.cache_data(show_spinner=False, max_entries=200)
def _demanda_mapa(ruta, cambio, ruta_est, cambio_est, hora_del_dia, tipo_dia, usuario) -> pd.DataFrame:
    d = _tabla(ruta, cambio)
    d = d[d["tipo_dia"] == tipo_dia]
    columna = {"miembros": "salidas_miembros", "ocasionales": "salidas_ocasionales"}.get(usuario, "salidas")
    if hora_del_dia is not None:
        d = d[d["hora"] == int(hora_del_dia)]
    d = (d.assign(salidas=d[columna]).groupby("estacion_id", as_index=False)[["salidas", "llegadas", "flujo_neto"]]
         .sum().round(2))
    e = _tabla(ruta_est, cambio_est)[["estacion_id", "nombre", "lat", "lon", "capacidad", "tipo"]]
    return e.merge(d, on="estacion_id", how="inner")


def demanda_sistema() -> pd.DataFrame:
    """Viajes de todo el sistema por hora: tipo_dia, hora, salidas,
    salidas_miembros, salidas_ocasionales, llegadas, dias (promedio por día)."""
    return _leer("demanda_hora")


def top_estaciones(n: int = 15) -> pd.DataFrame:
    """Las estaciones con más salidas: puesto y las columnas de `estaciones()`."""
    return _leer("top_estaciones").head(n)


def top_rutas(n: int = 15, con_ida_y_vuelta: bool = True) -> pd.DataFrame:
    """Las rutas con más viajes: puesto, origen_id, origen, destino_id, destino,
    viajes, origen_lat, origen_lon, destino_lat, destino_lon, misma_estacion,
    ruta ("A → B"). Con `con_ida_y_vuelta=False` se quitan los viajes que salen
    y vuelven a la misma estación."""
    r = _leer("top_rutas")
    if not con_ida_y_vuelta:
        r = r[~r["misma_estacion"]]
    return r.head(n)


def pronostico_estacion(estacion_id: str, objetivo: str = "salidas") -> pd.DataFrame:
    """Pronóstico y observado de una estación, por hora, SOLO en validación.

    Columnas: ts (hora local de Nueva York), fecha, hora, modelo ("baseline" o
    "lgbm"), q10, q50, q90, observado. `objetivo` es "salidas" o "llegadas".
    Vacío si la estación no tiene pronóstico.
    """
    t = _tabla_de_estacion(*_ruta("pronostico.parquet"), estacion_id)
    nom = "lle" if objetivo == "llegadas" else "sal"
    partes = []
    for modelo, pref in (("baseline", "base"), ("lgbm", "modelo")):
        p = t[["ts", "fecha", "hora"]].copy()
        p["modelo"] = modelo
        for q in ("q10", "q50", "q90"):
            p[q] = t[f"{pref}_{nom}_{q}"].astype(float)
        p["observado"] = t[f"{nom}_obs"]
        partes.append(p[p["q50"].notna()])
    return pd.concat(partes, ignore_index=True)


def recomendaciones_estacion(estacion_id: str) -> pd.DataFrame:
    """Lo que hizo cada política en una estación, con el presupuesto de referencia.

    Columnas: estacion_id, fecha, franja, hora_inicio, hora_fin, politica, bicis
    (positivo es llevar, negativo es retirar), nivel_riesgo, factible, inventario
    (bicis en la estación justo después del traslado). Vacío si no se simuló.
    """
    return _tabla_de_estacion(*_ruta("recomendaciones.parquet"), estacion_id).copy()


def curva() -> pd.DataFrame:
    """Riesgo contra presupuesto: politica, presupuesto (bicis al día), riesgo
    (horas de estación en riesgo por día), ic_bajo, ic_alto,
    riesgo_sin_intervencion, evitado, pct_evitado (fracción)."""
    return _leer("curva")


def _por_corte(nombre: str, corte: str | None) -> pd.DataFrame:
    t = _leer(nombre)
    return t if corte is None else t[t["corte"] == corte].reset_index(drop=True)


def metricas(corte: str | None = None) -> pd.DataFrame:
    """Error del pronóstico en validación: corte, grupo, orden, modelo, objetivo,
    n, mae, rmse, wape (fracción), cobertura (fracción), ancho.
    Cortes: "general", "bloque" (grupo = llave de BLOQUES), "hora" ("00".."23")
    y "volumen" (llave de VOLUMENES)."""
    return _por_corte("metricas", corte)


def riesgo_horas(corte: str | None = None) -> pd.DataFrame:
    """Horas de estación en riesgo por día, con el presupuesto de referencia:
    corte, grupo, politica, riesgo, ic_bajo, ic_alto. Cortes: "franja" (FRANJAS),
    "causa" (CAUSAS) y "tipo_estacion" (TIPOS_ESTACION). Puede venir vacía."""
    return _por_corte("riesgo_horas", corte)


def riesgo_niveles(corte: str | None = None) -> pd.DataFrame:
    """Cuántas franjas de estación quedaron en cada nivel de riesgo: corte,
    grupo, politica, alto, medio, bajo, total, pct_alto (fracción).
    Cortes: "franja" (FRANJAS) y "tipo_estacion" (TIPOS_ESTACION)."""
    return _por_corte("riesgo_niveles", corte)


def sensibilidad() -> pd.DataFrame:
    """Cómo cambia el resultado con el inventario inicial supuesto:
    inventario_inicial, politica, riesgo_sin_intervencion, riesgo, evitado,
    pct_evitado. Vacía si la simulación no lo reportó."""
    filas = []
    for f in indicadores()["sensibilidad"]["filas"]:
        for politica in ("historical", "model"):
            if f.get(politica):
                filas.append({"inventario_inicial": f["inventario_inicial"], "politica": politica,
                              "riesgo_sin_intervencion": f["riesgo_sin_intervencion"], **f[politica]})
    return pd.DataFrame(filas, columns=["inventario_inicial", "politica", "riesgo_sin_intervencion",
                                        "riesgo", "evitado", "pct_evitado"])


def limitaciones(n: int = 5) -> list[str]:
    """Las primeras `n` limitaciones que escribió la simulación (puede venir vacía)."""
    return list(indicadores().get("limitaciones") or [])[:n]


# --------------------------------------------------------------------------- #
# Textos que se arman con las cifras
# --------------------------------------------------------------------------- #

def linea_discreta() -> str:
    """La única nota de contexto que lleva cada página."""
    try:
        sinteticos = indicadores().get("sinteticos")
    except FileNotFoundError:
        sinteticos = False
    return LINEA + (" · datos sintéticos de ejemplo" if sinteticos else "")


def encabezado(titulo: str) -> None:
    """Título de la página y, debajo, la línea discreta. Toda página empieza así."""
    st.title(titulo)
    st.caption(linea_discreta())


def hallazgos() -> list[str]:
    """Los tres hallazgos de la página de inicio, armados con las cifras."""
    ind = indicadores()
    ref = ind["referencia"]
    hist = ref["politicas"].get("historical") or {}
    modelo = ref["politicas"].get("model") or {}
    presupuesto = ref.get("presupuesto")

    # 1. Cuánto riesgo evita planear con presupuesto.
    rangos = [r for r in ind["sensibilidad"]["rango"].values()]
    if presupuesto is None or not (hist or modelo):
        uno = f"Riesgo simulado evitado al planear con presupuesto: {SIN_DATO}."
    elif rangos:
        bajo, alto = min(r["pct_min"] for r in rangos), max(r["pct_max"] for r in rangos)
        uno = (f"Planear los traslados con {num(presupuesto)} bicis al día evita entre "
               f"{rango_pct(bajo, alto).replace(' a ', ' y ')} del riesgo simulado, según el inventario "
               "inicial que se suponga.")
    else:
        mejor = max((p for p in (hist, modelo) if p), key=lambda p: p["pct_evitado"] or 0)
        uno = (f"Planear los traslados con {num(presupuesto)} bicis al día evita "
               f"{pct(mejor['pct_evitado'], 0)} del riesgo simulado.")

    # 2. El modelo contra la regla histórica.
    comp = ind["comparacion"]
    base = (ind["pronostico"].get("baseline") or {}).get("salidas") or {}
    lgbm = (ind["pronostico"].get("lgbm") or {}).get("salidas") or {}
    if comp["modelo_pronostica_mejor"] is None or comp["modelo_evita"] is None:
        dos = f"Comparación entre el modelo y la regla histórica: {SIN_DATO}."
    else:
        error = (f"El modelo pronostica {'mejor' if comp['modelo_pronostica_mejor'] else 'peor'} que el promedio "
                 f"histórico (error de {pct(lgbm['wape'])} contra {pct(base['wape'])})")
        riesgo = {"igual": "no evita más riesgo", "mas": "evita más riesgo", "menos": "evita menos riesgo"}[comp["modelo_evita"]]
        nexo = ("pero" if comp["modelo_pronostica_mejor"] != (comp["modelo_evita"] == "mas") else "y")
        dos = (f"{error}, {nexo} {riesgo}: {pct(modelo['pct_evitado'])} contra "
               f"{pct(hist['pct_evitado'])} de la regla histórica.")

    # 3. Desde qué presupuesto deja de ayudar.
    umbral = ind["presupuesto_umbral"]["politicas"]
    u = umbral.get("historical") or umbral.get("model") or {}
    if u.get("presupuesto") is not None:
        tres = (f"Más allá de {num(u['presupuesto'])} bicis al día, mover más casi no ayuda: cada bici extra "
                f"evita {num(u['ganancia_despues'], 1)} horas de riesgo, contra {num(u['ganancia_inicial'], 1)} "
                "al principio.")
    elif u.get("ganancia_inicial") is not None:
        tres = "En los presupuestos simulados, cada bici extra todavía reduce el riesgo."
    else:
        tres = f"Presupuesto a partir del cual mover más deja de ayudar: {SIN_DATO}."
    return [uno, dos, tres]
