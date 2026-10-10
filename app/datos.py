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
- Cada página empieza con `datos.encabezado("Título")`. Las páginas que solo
  describen viajes (mapa, estaciones y rutas, contexto) usan
  `datos.encabezado("Título", "viajes")`.
  Inicio usa `datos.encabezado("Título", "inicio")`.
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
LINEA_INICIO = "Resultados de simulación: validación y prueba final"  # Inicio, cuando ya hay prueba final
LINEA_VIAJES = "Viajes reales"
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
RONDAS = {"nocturna": "ronda nocturna", "mediodia": "ronda de mediodía"}
AZAR = "Mover al azar"
PERIODOS = {"validacion": "validación", "prueba": "la prueba final"}

# Colores fijos para que las cinco páginas se vean iguales.
COLORES_POLITICA = {"none": "#8A8F98", "historical": "#E69F00", "model": "#0072B2"}
COLORES_RIESGO = {"alto": "#D55E00", "medio": "#E69F00", "bajo": "#009E73"}
COLORES_MODELO = {"baseline": "#E69F00", "lgbm": "#0072B2"}
COLOR_AZAR = "#CC79A7"
COLOR_OBSERVADO = "#222222"
# Mapa de demanda: par propio, distinto de los colores de las políticas.
COLOR_SE_VACIA = "#7B3294"  # morado. Flujo neto negativo: salen más bicis de las que llegan
COLOR_SE_LLENA = "#008837"  # verde. Flujo neto positivo: llegan más de las que salen
# Plan de la ronda: otro par propio.
COLOR_LLEVAR = "#44AA99"    # verde azulado: bicis que se llevan a la estación
COLOR_RETIRAR = "#882255"   # vino: bicis que se retiran

# --- Estilo y textos comunes a todas las páginas -----------------------------
# Paleta fija: Sin intervención gris, Regla histórica naranja, Modelo azul
# (COLORES_POLITICA) y Mover al azar rosa (COLOR_AZAR).
FUENTE_EJE = 15       # tamaño de las etiquetas de los ejes y la leyenda (se verá en video)
FUENTE_TITULO = 17    # tamaño de los títulos de ejes y leyenda
QUE_ES_RIESGO = "una estación que pasa una hora casi vacía o casi llena"
TITULO_DESPLEGABLE = "Cómo se calculó"


def estilo_grafica(grafica):
    """Letras grandes en ejes y leyenda de una gráfica de Altair. Úsala en todas."""
    return (
        grafica.configure_axis(labelFontSize=FUENTE_EJE, titleFontSize=FUENTE_TITULO)
        .configure_legend(labelFontSize=FUENTE_EJE, titleFontSize=FUENTE_TITULO)
        .configure_text(fontSize=FUENTE_EJE)
    )


def como_se_calculo(lineas) -> None:
    """Desplegable al final de una sección con las aclaraciones técnicas y los límites."""
    lineas = [linea for linea in lineas if linea]
    if lineas:
        with st.expander(TITULO_DESPLEGABLE):
            st.markdown("\n".join(f"- {linea}" for linea in lineas))

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


def accion(bicis, hora_inicio=None, ronda: dict | None = None) -> str:
    """Texto de la tarjeta de acción: "Llevar 5 bicis en la ronda nocturna (lista a las 05:00)".

    `ronda` es una entrada de `rondas()`: {"nombre", "hora"}. Sin ella (la simulación no
    reportó el momento de intervención) queda "Llevar 5 bicis antes de las 06:00".
    """
    if _falta(bicis):
        return SIN_DATO
    bicis = int(bicis)
    if bicis == 0:
        return "Sin traslado"
    verbo = "Llevar" if bicis > 0 else "Retirar"
    unidad = "bici" if abs(bicis) == 1 else "bicis"
    if ronda and ronda.get("nombre") and not _falta(ronda.get("hora")):
        cuando = f" en la {ronda['nombre']} (lista a las {hora(ronda['hora'])})"
    else:
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


def tabla_pequena(ruta) -> pd.DataFrame:
    """Lee una tabla agregada pequeña por su ruta. La usan secciones aparte, como la de contexto."""
    ruta = Path(ruta)
    return _tabla(str(ruta), ruta.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=200)
def _tabla_de_estacion(ruta: str, cambio: float, estacion_id: str) -> pd.DataFrame:
    # El archivo está ordenado por estación: el filtro lee solo su pedazo.
    return pq.read_table(ruta, filters=[("estacion_id", "=", estacion_id)]).to_pandas()


@st.cache_data(show_spinner=False)
def _json(ruta: str, cambio: float) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def ruta_de_contexto() -> Path:
    """Dónde está la tabla de la página de contexto (puede no existir todavía)."""
    return CARPETA / "contexto_estaciones.parquet"


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
    (bicis en la estación justo después del traslado) e inventario_antes (justo
    antes). Vacío si no se simuló.
    """
    return _tabla_de_estacion(*_ruta("recomendaciones.parquet"), estacion_id).copy()


def estacion_de_ejemplo() -> dict:
    """La estación y el día con que abre "Pronóstico por estación":
    {"estacion_id", "fecha"}. La elige el constructor de las tablas con una regla
    fija (oficinas, demanda alta, traslado en todas las franjas y con solución,
    en día laborable; de esos días, el de error de pronóstico más típico de la
    estación); si ninguna cumple, es la estación con más salidas."""
    return indicadores().get("estacion_de_ejemplo") or {}


def rondas() -> dict:
    """En qué ronda se hace el traslado de cada franja: {franja: {"nombre", "hora"}}.
    La primera intervención del día es la ronda nocturna y la otra, la de mediodía;
    la hora es el momento de intervención que reportó la simulación. Vacío si no lo reportó."""
    franjas = (indicadores().get("simulacion") or {}).get("franjas") or []
    return {f.get("nombre"): {"nombre": RONDAS[f["ronda"]], "hora": f.get("hora_intervencion")}
            for f in franjas if f.get("ronda") in RONDAS and not _falta(f.get("hora_intervencion"))}


def inventario(antes, despues) -> str:
    """ "13 bicis antes, 2 después": las bicis en la estación antes y después del traslado."""
    if _falta(antes) or _falta(despues):
        return SIN_DATO
    unidad = "bici" if int(antes) == 1 else "bicis"
    return f"{num(antes)} {unidad} antes, {num(despues)} después"


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


DEMANDA_ATENDIDA = "Los datos solo registran demanda atendida."


def limitaciones(n: int = 6) -> list[str]:
    """Hasta `n` limitaciones de la simulación (vacía si no hubo simulación).

    Las dos primeras van siempre: el resultado depende del inventario inicial
    supuesto (con el rango que reportó la simulación, si lo hay) y los datos solo
    registran demanda atendida. Después, las que escribió la simulación.
    """
    ind = indicadores()
    reporte = [str(x) for x in ind.get("limitaciones") or []]
    if not reporte and (ind.get("referencia") or {}).get("presupuesto") is None:
        return []
    rango = ((ind.get("sensibilidad") or {}).get("rango") or {}).get("historical") or {}
    inicial = "El resultado depende del inventario inicial supuesto"
    if rango.get("pct_min") is not None and rango.get("pct_max") is not None:
        inicial += (f": con la {POLITICAS['historical'].lower()}, el riesgo simulado evitado va de "
                    f"{rango_pct(rango['pct_min'], rango['pct_max'])} según el supuesto")
    atendida = next((x for x in reporte if "demanda atendida" in x.lower()), DEMANDA_ATENDIDA)
    resto = [x for x in reporte if x != atendida and "inventario inicial" not in x.lower()]
    return ([inicial + ".", atendida] + resto)[:n]


# Las filas de la comparación de Inicio, en orden: clave del archivo, texto en pantalla y unidad
# (None deja la del archivo). Las demás filas del archivo no se muestran.
FILAS_DE_COMPARACION = [
    ("pct_evitado_historica", "Riesgo simulado evitado con la regla histórica", None),
    ("pct_evitado_modelo", "Riesgo simulado evitado con el modelo", None),
    ("pct_evitado_azar", "Riesgo simulado evitado moviendo las mismas bicis al azar", None),
    ("ganancia_pct_al_duplicar_historica", "Puntos extra de riesgo simulado evitado {subir}, regla histórica", "puntos"),
    ("bloqueados_sin_intervencion", "Viajes bloqueados por día en la simulación, sin intervención", None),
    ("bloqueados_historica", "Viajes bloqueados por día en la simulación, con la regla histórica", None),
    ("wape_promedio_historico", "Error del pronóstico de salidas, promedio histórico", None),
    ("wape_modelo", "Error del pronóstico de salidas, modelo (LightGBM)", None),
]


def _casi_el_doble(factor) -> bool:
    return not _falta(factor) and 1.7 <= float(factor) < 2


def _subir_el_presupuesto(ind: dict) -> str:
    """ "al casi duplicar el presupuesto", si el que sigue al de referencia es casi el doble."""
    factor = ((ind.get("referencia") or {}).get("siguiente") or {}).get("factor")
    return "al casi duplicar el presupuesto" if _casi_el_doble(factor) else "al pasar al siguiente presupuesto"


def _filas_de_prueba(ind: dict) -> dict:
    """Las filas de validación y prueba final, crudas, por su clave ({} si no hay archivo)."""
    return {f.get("clave"): f for f in (ind.get("validacion_y_prueba") or {}).get("filas") or [] if f.get("clave")}


def validacion_y_prueba() -> dict:
    """Validación y prueba final lado a lado: {"presupuesto": bicis o None,
    "presupuestos": {"validacion", "prueba"}, "filas": [{"afirmacion", "validacion",
    "prueba"}]} con los valores ya escritos según su unidad. Solo las filas de
    FILAS_DE_COMPARACION, en ese orden. Sin el archivo de la prueba final, "filas" va vacía."""
    ind = indicadores()
    crudo = ind.get("validacion_y_prueba") or {}

    def escribir(valor, intervalo, unidad) -> str:
        def uno(x, decimales_pct=1):
            if unidad == "pct":
                return pct(x, decimales_pct)
            if unidad == "puntos":  # diferencia entre dos porcentajes
                return SIN_DATO if _falta(x) else num(float(x) * 100, 1)
            if unidad in ("horas", "viajes"):
                return num(x)
            return num(x, 0 if not _falta(x) and float(x) == int(float(x)) else 1)
        texto_valor = uno(valor)
        if texto_valor == SIN_DATO:
            return SIN_DATO
        if unidad in ("horas", "viajes", "puntos"):
            texto_valor += f" {unidad}"
        if intervalo and not any(_falta(x) for x in intervalo):
            texto_valor += f" ({uno(intervalo[0])} a {uno(intervalo[1])})"
        return texto_valor

    por_clave = _filas_de_prueba(ind)
    filas = []
    for clave, afirmacion, unidad in FILAS_DE_COMPARACION:
        f = por_clave.get(clave)
        if f is None:
            continue
        unidad = unidad or f.get("unidad")
        filas.append({"afirmacion": afirmacion.format(subir=_subir_el_presupuesto(ind)),
                      "validacion": escribir(f.get("validacion"), f.get("validacion_ic"), unidad),
                      "prueba": escribir(f.get("prueba"), f.get("prueba_ic"), unidad)})
    return {"presupuesto": crudo.get("presupuesto_de_referencia"),
            "presupuestos": crudo.get("presupuestos") or {}, "filas": filas}


def hay_prueba_final() -> bool:
    """Si ya se construyó la comparación de validación contra prueba final."""
    try:
        return bool(validacion_y_prueba()["filas"])
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# Textos que se arman con las cifras
# --------------------------------------------------------------------------- #

def periodo_en_meses(inicio, fin) -> str:
    """ "julio a septiembre de 2026" (o "julio de 2026", o con los dos años si cambian)."""
    if _falta(inicio) or _falta(fin):
        return SIN_DATO
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)
    if a.year != b.year:
        return f"{_MESES[a.month - 1]} de {a.year} a {_MESES[b.month - 1]} de {b.year}"
    if a.month != b.month:
        return f"{_MESES[a.month - 1]} a {_MESES[b.month - 1]} de {b.year}"
    return f"{_MESES[b.month - 1]} de {b.year}"


def meses_de_historia() -> str:
    """Los meses anteriores a la validación ("julio a agosto de 2026"), o "Sin dato".
    No detiene la página si faltan las tablas: la usa la sección de contexto."""
    ruta = CARPETA / "indicadores.json"
    if not ruta.exists():
        return SIN_DATO
    ind = _json(str(ruta), ruta.stat().st_mtime)
    inicio, corte = (ind.get("periodo") or {}).get("inicio"), (ind.get("validacion") or {}).get("inicio")
    if _falta(inicio) or _falta(corte) or pd.Timestamp(corte) <= pd.Timestamp(inicio):
        return SIN_DATO
    return periodo_en_meses(inicio, pd.Timestamp(corte) - pd.Timedelta(days=1))


def _filas_validas(ind: dict) -> bool:
    claves = _filas_de_prueba(ind)
    return any(clave in claves for clave, _, _ in FILAS_DE_COMPARACION)


def linea_discreta(clase: str = "simulacion") -> str:
    """La única nota de contexto que lleva cada página.

    "simulacion": páginas con resultados de la simulación o del pronóstico.
    "inicio": la página de inicio; cuando ya hay prueba final, lo dice.
    "viajes": páginas que solo describen los viajes del periodo analizado.
    """
    try:
        ind = indicadores()
    except FileNotFoundError:
        ind = {}
    sinteticos = ind.get("sinteticos")
    linea = LINEA_INICIO if clase == "inicio" and _filas_validas(ind) else LINEA
    if clase == "viajes":
        periodo = ind.get("periodo") or {}
        meses = periodo_en_meses(periodo.get("inicio"), periodo.get("fin"))
        # Los datos de muestra son inventados: no se presentan como viajes reales.
        linea = "Viajes de ejemplo" if sinteticos else LINEA_VIAJES
        linea += " del periodo analizado" if meses == SIN_DATO else f" de {meses}"
    return linea + (" · datos sintéticos de ejemplo" if sinteticos else "")


def encabezado(titulo: str, clase: str = "simulacion") -> None:
    """Título de la página y, debajo, la línea discreta. Toda página empieza así."""
    st.title(titulo)
    st.caption(linea_discreta(clase))


def _hallazgo_de_contexto(ind: dict) -> str:
    """Empleos alrededor de las estaciones de oficinas contra las residenciales."""
    por_tipo = (ind.get("contexto") or {}).get("por_tipo") or {}
    oficinas = (por_tipo.get("oficinas") or {}).get("proporcion_empleos")
    residencial = (por_tipo.get("residencial") or {}).get("proporcion_empleos")
    if oficinas is None or residencial is None:
        return f"Empleos alrededor de las estaciones de oficinas y de las residenciales: {SIN_DATO}."
    return (f"A 300 metros de la estación típica de oficinas, los empleos son {pct(oficinas, 0)} de la suma de "
            f"empleos y trabajadores residentes; en la residencial, {pct(residencial, 0)}. El tipo de estación "
            "sale del patrón de uso y los datos del Censo lo confirman por separado.")


def _veredicto(valor, intervalo) -> str | None:
    """De la diferencia de riesgo "modelo menos regla histórica" y su intervalo: "mas" (el
    modelo evita más: todo el intervalo es negativo), "menos" o "igual" (el intervalo incluye
    el cero). Sin intervalo no se puede decir: None."""
    if _falta(valor) or not intervalo or any(_falta(x) for x in intervalo):
        return None
    bajo, alto = sorted(float(x) for x in intervalo)
    return "mas" if alto < 0 else "menos" if bajo > 0 else "igual"


def modelo_contra_regla(ind: dict | None = None) -> dict:
    """¿El modelo evita más riesgo simulado que la regla histórica?

    {"referencia": {periodo: veredicto}, "mitad": {periodo: veredicto}} con los periodos
    ("validacion", "prueba") de los que hay dato y veredicto "mas", "menos" o "igual".
    "mitad" es con la mitad del presupuesto de referencia. Sale de la diferencia entre las dos
    políticas y su intervalo (archivo de validación y prueba final); en validación, si no hay
    archivo, de si se traslapan las bandas de la curva.
    """
    ind = indicadores() if ind is None else ind
    filas = _filas_de_prueba(ind)
    comp = ind.get("comparacion") or {}
    presupuesto = (ind.get("referencia") or {}).get("presupuesto")
    prefijo = "diferencia_modelo_menos_historica_"
    en_ref = filas.get(prefijo + "referencia") or {}
    en_mitad = {}
    for clave, f in filas.items():  # la fila de la mitad lleva el presupuesto en la clave
        resto = clave[len(prefijo):] if clave.startswith(prefijo) else ""
        if resto.isdigit() and presupuesto and 0.4 <= int(resto) / presupuesto <= 0.6:
            en_mitad = f
    de_la_curva = {"referencia": comp.get("modelo_evita"), "mitad": (comp.get("mitad") or {}).get("modelo_evita")}
    resultado = {}
    for nombre, fila in (("referencia", en_ref), ("mitad", en_mitad)):
        por_periodo = {}
        validacion = _veredicto(fila.get("validacion"), fila.get("validacion_ic")) or de_la_curva[nombre]
        if validacion:
            por_periodo["validacion"] = validacion
        prueba = _veredicto(fila.get("prueba"), fila.get("prueba_ic"))
        if prueba and validacion:
            por_periodo["prueba"] = prueba
        resultado[nombre] = por_periodo
    return resultado


def _donde(periodos: list[str], negado: bool = False) -> str:
    """ "en validación", "en los dos periodos" o "en ninguno de los dos periodos"."""
    if len(periodos) == 1:
        return f"en {PERIODOS[periodos[0]]}"
    return "en ninguno de los dos periodos" if negado else "en los dos periodos"


def frase_de_la_curva() -> str:
    """La frase bajo la curva de Inicio (validación). Vacía si no hay nada que decir."""
    ind = indicadores()
    historica, modelo = POLITICAS["historical"], POLITICAS["model"]
    politicas = (ind.get("referencia") or {}).get("politicas") or {}
    if not (politicas.get("historical") and politicas.get("model")):
        return ""
    veredictos = modelo_contra_regla(ind)
    en_ref, en_mitad = veredictos["referencia"].get("validacion"), veredictos["mitad"].get("validacion")
    if en_ref == "igual" and en_mitad == "mas":
        return (f"Con el presupuesto de referencia, {historica} y {modelo} dan casi lo mismo; "
                f"con la mitad, el {modelo.lower()} evita más riesgo simulado.")
    if en_ref == "igual":
        return f"Las líneas de {historica} y {modelo} se enciman: su resultado es casi el mismo."
    a, b = politicas["historical"].get("pct_evitado"), politicas["model"].get("pct_evitado")
    if _falta(a) or _falta(b):
        return ""
    return (f"Con el presupuesto de referencia, la {historica.lower()} evita {pct(a)} del riesgo simulado "
            f"y el {modelo.lower()}, {pct(b)}.")


def _hallazgo_del_riesgo(ind: dict) -> str:
    """1. Cuánto riesgo simulado evita planear con presupuesto, y contra mover al azar."""
    ref = ind.get("referencia") or {}
    politicas = ref.get("politicas") or {}
    clave = next((p for p in ("historical", "model") if politicas.get(p)), None)
    presupuesto = ref.get("presupuesto")
    validacion = (politicas.get(clave) or {}).get("pct_evitado")
    if presupuesto is None or _falta(validacion):
        return f"Riesgo simulado evitado al planear con presupuesto: {SIN_DATO}."
    filas = _filas_de_prueba(ind)
    prueba = (filas.get({"historical": "pct_evitado_historica", "model": "pct_evitado_modelo"}[clave]) or {}).get("prueba")
    evitado = {"validacion": validacion, **({} if _falta(prueba) else {"prueba": prueba})}
    al_azar = {"validacion": (ind.get("piso_de_azar") or {}).get("pct_evitado"),
               "prueba": (filas.get("pct_evitado_azar") or {}).get("prueba")}
    cifras = " y ".join(f"{pct(v)} en {PERIODOS[p]}" for p, v in evitado.items())
    if len(evitado) == 2:
        frase = f"Planear los traslados con el presupuesto de referencia ({num(presupuesto)} bicis al día en validación) evita "
        frase += (f"más de la mitad del riesgo simulado en los dos periodos: {cifras}"
                  if min(evitado.values()) > 0.5 else f"{cifras.replace(' en ', ' del riesgo simulado en ', 1)}")
    else:
        frase = f"Planear los traslados con {num(presupuesto)} bicis al día evita "
        frase += (f"más de la mitad del riesgo simulado en validación ({pct(validacion)})"
                  if validacion > 0.5 else f"{pct(validacion)} del riesgo simulado en validación")
    veces = [evitado[p] / al_azar[p] for p in evitado if not _falta(al_azar[p]) and al_azar[p] > 0]
    if veces and min(veces) >= 1.5:
        bajo, alto = round(min(veces)), round(max(veces))
        cuantas = f"unas {num(bajo)} veces" if bajo == alto else f"entre {num(bajo)} y {num(alto)} veces"
        solo = "" if len(veces) == len(evitado) else " (en validación)"
        frase += f". Es {cuantas} lo que evita mover las mismas bicis al azar{solo}"
    elif veces:
        frase += f". Mover las mismas bicis al azar evita {pct(al_azar['validacion'])} en validación"
    return frase + "."


def _hallazgo_del_presupuesto(ind: dict) -> str:
    """2. Cuánto agrega pasar del presupuesto de referencia al siguiente."""
    ref = ind.get("referencia") or {}
    siguiente = ref.get("siguiente") or {}
    filas = _filas_de_prueba(ind)
    en_validacion = [v for v in (siguiente.get("puntos") or {}).values() if not _falta(v)]
    en_prueba = [f["prueba"] for c in ("ganancia_pct_al_duplicar_historica", "ganancia_pct_al_duplicar_modelo")
                 if not _falta((f := filas.get(c) or {}).get("prueba"))]
    if ref.get("presupuesto") is None or siguiente.get("presupuesto") is None or not en_validacion:
        return f"Lo que agrega subir el presupuesto más allá del de referencia: {SIN_DATO}."
    tope = math.floor(max(max(en_validacion + en_prueba), 0) * 100) + 1  # "menos de" el entero que sigue
    pasar = f"de {num(ref['presupuesto'])} a {num(siguiente['presupuesto'])} bicis al día en validación"
    sujeto = (f"Casi duplicar el presupuesto ({pasar})" if _casi_el_doble(siguiente.get("factor"))
              else f"Subir el presupuesto {pasar}")
    donde = _donde(["validacion", "prueba"] if en_prueba else ["validacion"])
    return (f"{sujeto} agrega menos de {num(tope)} {'punto' if tope == 1 else 'puntos'} de riesgo simulado "
            f"evitado {donde}.")


def _hallazgo_del_modelo(ind: dict) -> str:
    """3. Con qué presupuesto basta la regla histórica y con cuál gana el modelo."""
    veredictos = modelo_contra_regla(ind)
    historica, modelo = POLITICAS["historical"].lower(), POLITICAS["model"].lower()

    def clausula(por_periodo: dict, inicio: str, basta: str, gana: str, mixto: str) -> tuple[str, bool]:
        ganan = [p for p, v in por_periodo.items() if v == "mas"]
        pierden = [p for p in por_periodo if p not in ganan]
        if not ganan:
            return f"{inicio}{basta} {_donde(pierden, negado=True)}", False
        if not pierden:
            return f"{inicio}{gana} {_donde(ganan)}", True
        return f"{inicio}{mixto} en {PERIODOS[ganan[0]]}, pero no en {PERIODOS[pierden[0]]}", True

    if not veredictos["referencia"]:
        return f"Comparación entre el {modelo} y la {historica}: {SIN_DATO}."
    frase, gana_en_referencia = clausula(
        veredictos["referencia"], "Con el presupuesto de referencia",
        f" basta la {historica}: el {modelo} no evita más riesgo simulado",
        f", el {modelo} evita más riesgo simulado que la {historica}",
        f", el {modelo} evita más riesgo simulado que la {historica}")
    if veredictos["mitad"]:
        si = "" if gana_en_referencia else "sí "
        tambien = "también " if gana_en_referencia else ""
        segunda, _ = clausula(
            veredictos["mitad"], "con la mitad del presupuesto",
            f", el {modelo} tampoco evita más" if not gana_en_referencia else f", el {modelo} no evita más",
            f", el {modelo} {si}{tambien}evita más",
            f", el {modelo} evita más")
        frase += f"; {segunda}"
    return frase + "."


def hallazgos() -> list[str]:
    """Los cuatro hallazgos de la página de inicio, armados con las cifras. Con la prueba
    final hablan de los dos periodos; sin ella, solo de validación; sin cifra, "Sin dato"."""
    ind = indicadores()
    return [_hallazgo_del_riesgo(ind), _hallazgo_del_presupuesto(ind), _hallazgo_del_modelo(ind),
            _hallazgo_de_contexto(ind)]
