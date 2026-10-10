"""Página "Plan de la ronda".

Para un día simulado, una ronda (nocturna o de mediodía) y una política: la
lista de estaciones a visitar, con cuántas bicis llevar o retirar en cada una,
y el mapa de esas estaciones. Todo sale de `datos.recomendaciones()`,
`datos.estaciones()` y `datos.indicadores()`; aquí solo se filtra, se da
formato y se dibuja.
"""

import pandas as pd
import pydeck as pdk
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

POLITICAS_CON_TRASLADO = ("historical", "model")  # la primera es la que se muestra al abrir
ORDEN_RIESGO = {"alto": 0, "medio": 1, "bajo": 2}
FILAS_VISIBLES = 200  # la tabla muestra las primeras; debajo se dice cuántas más hay
ZOOM = 10.8  # el mismo acercamiento del mapa de demanda
ACCIONES = {"llevar": ("Llevar", datos.COLOR_LLEVAR, "se llevan bicis a la estación"),
            "retirar": ("Retirar", datos.COLOR_RETIRAR, "se retiran bicis de la estación")}
# Mapas base de Carto: no necesitan llave de Mapbox.
MAPAS_BASE = {"light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
              "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"}


def _tema() -> str:
    """ "light" o "dark", según el tema con que se está viendo la app."""
    try:
        tema = st.context.theme.type
    except Exception:  # versiones de Streamlit sin esa información
        tema = None
    return tema if tema in MAPAS_BASE else "light"


def _rgb(color: str, alfa: int = 210) -> list[int]:
    color = color.lstrip("#")
    return [int(color[i:i + 2], 16) for i in (0, 2, 4)] + [alfa]


def _bicis(n) -> str:
    """ "1 bici" o "1,250 bicis"."""
    return f"{datos.num(n)} {'bici' if int(n) == 1 else 'bicis'}"


def _accion(bicis) -> str:
    """ "Llevar 5" o "Retirar 3"."""
    return f"{ACCIONES['llevar' if bicis > 0 else 'retirar'][0]} {datos.num(abs(int(bicis)))}"


def _nombre_de_ronda(franja: str, filas: pd.DataFrame, rondas: dict) -> str:
    """ "Ronda nocturna (lista a las 05:00)"; si la simulación no reportó el momento de
    intervención, la franja que prepara: "Antes de la franja de la mañana (06:00 a 14:00)"."""
    ronda = rondas.get(franja)
    if ronda:
        nombre = ronda["nombre"]
        return f"{nombre[0].upper()}{nombre[1:]} (lista a las {datos.hora(ronda['hora'])})"
    primera = filas[filas["franja"] == franja].iloc[0]
    return (f"Antes de la franja de la {datos.etiqueta(datos.FRANJAS, franja).lower()} "
            f"({datos.hora(primera['hora_inicio'])} a {datos.hora(primera['hora_fin'])})")


def _vista() -> pdk.ViewState:
    """Vista fija sobre la ciudad: el centro de todas las estaciones, no solo las de la ronda,
    así el mapa no se mueve al cambiar de día, de ronda o de política."""
    todas = datos.estaciones()
    todas = todas[todas["lat"].notna() & todas["lon"].notna()]
    return pdk.ViewState(latitude=float(todas["lat"].median()), longitude=float(todas["lon"].median()),
                         zoom=ZOOM, min_zoom=ZOOM - 2, max_zoom=ZOOM + 5, pitch=0, bearing=0)


def _mapa(plan: pd.DataFrame) -> pdk.Deck:
    con_lugar = plan[plan["lat"].notna() & plan["lon"].notna()]
    maximo = float(con_lugar["bicis"].abs().max() or 0)
    escala = (con_lugar["bicis"].abs() / maximo) ** 0.5 if maximo > 0 else 0.0
    puntos = pd.DataFrame({
        "lon": con_lugar["lon"].astype(float),
        "lat": con_lugar["lat"].astype(float),
        "radio": (60 + 240 * escala).astype(float).round(1),
        "color": con_lugar["bicis"].map(
            lambda b: _rgb(ACCIONES["llevar" if b > 0 else "retirar"][1])),
        "nombre": con_lugar["nombre"].map(datos.texto),
        "accion": con_lugar["bicis"].map(lambda b: f"{ACCIONES['llevar' if b > 0 else 'retirar'][0]} {_bicis(abs(b))}"),
        "inventario": [datos.inventario(a, d) for a, d in zip(con_lugar["inventario_antes"], con_lugar["inventario"])],
        "riesgo": con_lugar["nivel_riesgo"].map(lambda r: datos.etiqueta(datos.RIESGOS, r)),
        "tipo": con_lugar["tipo"].map(lambda t: datos.etiqueta(datos.TIPOS_ESTACION, t)),
    })
    # Las grandes se dibujan primero para que no tapen a las chicas.
    puntos = puntos.sort_values("radio", ascending=False).reset_index(drop=True)
    capa = pdk.Layer(
        "ScatterplotLayer",
        data=puntos,
        get_position="[lon, lat]",
        get_radius="radio",
        get_fill_color="color",
        radius_min_pixels=3,
        radius_max_pixels=26,
        stroked=True,
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=0.5,
        pickable=True,
        auto_highlight=True,
    )
    # La rueda del ratón no acerca el mapa (así la página se puede recorrer); se acerca con
    # doble clic y se arrastra para moverlo.
    control = pdk.View(type="MapView", controller={"scrollZoom": False, "dragRotate": False, "touchRotate": False})
    globo = {
        "html": ("<b>{nombre}</b><br/>"
                 "{accion}<br/>"
                 "En la estación: {inventario}<br/>"
                 "Riesgo simulado: {riesgo}<br/>"
                 "Tipo: {tipo}"),
        "style": {"backgroundColor": "#FFFFFF", "color": "#222222", "fontSize": "15px",
                  "border": "1px solid #D0D4DA", "borderRadius": "6px"},
    }
    return pdk.Deck(layers=[capa], views=[control], initial_view_state=_vista(), map_style=MAPAS_BASE[_tema()],
                    tooltip=globo)


datos.encabezado("Plan de la ronda")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene
rec = datos.recomendaciones()
rec = rec[rec["politica"].isin(POLITICAS_CON_TRASLADO) & rec["fecha"].notna()]
if rec.empty:
    st.info("La simulación todavía no tiene traslados que mostrar.")
    st.stop()
rec = rec.assign(dia=pd.to_datetime(rec["fecha"]).dt.normalize())

# --------------------------------------------------------------------------- #
# Selectores
# --------------------------------------------------------------------------- #

dias = sorted(rec["dia"].unique())
rondas = datos.rondas()  # en qué ronda (y a qué hora queda lista) se hace el traslado de cada franja
franjas = list(rec.sort_values("hora_inicio")["franja"].drop_duplicates())
# Las opciones son los textos en pantalla; aquí se regresan a su clave.
ronda_por_texto = {_nombre_de_ronda(f, rec, rondas): f for f in franjas}
politica_por_texto = {datos.POLITICAS[p]: p for p in POLITICAS_CON_TRASLADO if (rec["politica"] == p).any()}

col_dia, col_ronda, col_politica = st.columns([2, 3, 2])
dia = col_dia.selectbox("Día", dias, format_func=datos.fecha, key="plan_dia")
texto_ronda = col_ronda.radio("Ronda", list(ronda_por_texto), horizontal=True, key="plan_ronda")
texto_politica = col_politica.radio("Política", list(politica_por_texto), horizontal=True, key="plan_politica")
franja, politica = ronda_por_texto[texto_ronda], politica_por_texto[texto_politica]

plan = rec[(rec["dia"] == dia) & (rec["franja"] == franja) & (rec["politica"] == politica)
           & rec["bicis"].notna() & (rec["bicis"] != 0)]
plan = plan.merge(datos.estaciones()[["estacion_id", "nombre", "tipo", "lat", "lon"]], on="estacion_id", how="left")
plan = (plan.assign(_riesgo=plan["nivel_riesgo"].map(ORDEN_RIESGO).fillna(len(ORDEN_RIESGO)),
                    _movidas=plan["bicis"].abs())
        .sort_values(["_riesgo", "_movidas", "nombre"], ascending=[True, False, True]).reset_index(drop=True))

# --------------------------------------------------------------------------- #
# Titular e indicadores
# --------------------------------------------------------------------------- #

ronda = rondas.get(franja)
esta_ronda = f"la {ronda['nombre']}" if ronda else "esta ronda"
n = len(plan)
if n == 0:
    st.subheader(f"No hay estaciones a visitar en {esta_ronda} del {datos.fecha(dia)}")
    st.info("Con esta selección, la simulación no movió bicis.")
    st.stop()

sustantivo = "Esta es la estación" if n == 1 else f"Estas son las {datos.num(n)} estaciones"
st.subheader(f"{sustantivo} a visitar en {esta_ronda} del {datos.fecha(dia)}")

llevar = int(plan.loc[plan["bicis"] > 0, "bicis"].sum())
retirar = int(-plan.loc[plan["bicis"] < 0, "bicis"].sum())
m1, m2, m3 = st.columns(3)
m1.metric("Estaciones a visitar", datos.num(n))
m2.metric("Bicis a llevar", datos.num(llevar))
m3.metric("Bicis a retirar", datos.num(retirar))

presupuesto = (ind.get("referencia") or {}).get("presupuesto")
st.markdown(f"Resultado de la simulación con el presupuesto de referencia ({datos.num(presupuesto)} bicis al día "
            f"en todo el sistema), con la política **{texto_politica}**, en días de validación.")

# --------------------------------------------------------------------------- #
# Tabla
# --------------------------------------------------------------------------- #

st.subheader("Estaciones a visitar")
visibles = plan.head(FILAS_VISIBLES)
tabla = pd.DataFrame({
    "Estación": visibles["nombre"].map(datos.texto),
    "Tipo": visibles["tipo"].map(lambda t: datos.etiqueta(datos.TIPOS_ESTACION, t)),
    "Acción": visibles["bicis"].map(_accion),
    "Bicis antes": visibles["inventario_antes"].map(datos.num),
    "Bicis después": visibles["inventario"].map(datos.num),
    "Riesgo simulado": visibles["nivel_riesgo"].map(lambda r: datos.etiqueta(datos.RIESGOS, r)),
})
st.dataframe(tabla, hide_index=True, width="stretch", height=460)
orden = "Ordenadas por riesgo simulado (alto primero) y luego por bicis movidas."
if n > FILAS_VISIBLES:
    resto = n - FILAS_VISIBLES
    orden += (f" Se muestran las primeras {datos.num(FILAS_VISIBLES)}; hay {datos.num(resto)} "
              f"{'estación' if resto == 1 else 'estaciones'} más.")
st.caption(orden)

# --------------------------------------------------------------------------- #
# Mapa
# --------------------------------------------------------------------------- #

st.subheader("Dónde están")
cuantas = {"llevar": int((plan["bicis"] > 0).sum()), "retirar": int((plan["bicis"] < 0).sum())}
leyenda = " &nbsp;·&nbsp; ".join(
    f"<span style='white-space:nowrap'><span style='color:{color}'>●</span> <b>{nombre}</b>: {explicacion} "
    f"({datos.num(cuantas[clave])})</span>"
    for clave, (nombre, color, explicacion) in ACCIONES.items())
st.markdown(f"<span style='font-size:1.1rem'>{leyenda}. El tamaño del punto son las bicis movidas.</span>",
            unsafe_allow_html=True)
sin_lugar = int((plan["lat"].isna() | plan["lon"].isna()).sum())
if sin_lugar == n:
    st.info("Estas estaciones no tienen ubicación conocida: no se pueden dibujar.")
else:
    st.pydeck_chart(_mapa(plan), height=560)

# --------------------------------------------------------------------------- #
# Notas técnicas
# --------------------------------------------------------------------------- #

datos.como_se_calculo([
    "Solo aparecen las estaciones donde la política elegida movió bicis; las demás no se visitan.",
    "Bicis antes y bicis después son las que hay en la estación, en la simulación, justo antes y justo después "
    "del traslado.",
    "El nivel de riesgo simulado es el de la franja que prepara la ronda, con las bicis que había antes del "
    f"traslado: dice qué tan urgente es la visita. Riesgo: {datos.QUE_ES_RIESGO}.",
    (f"Cada ronda queda lista a la hora que reportó la simulación: "
     + "; ".join(f"{r['nombre']}, {datos.hora(r['hora'])}" for r in rondas.values()) + ".") if rondas else "",
    f"El presupuesto de referencia ({datos.num(presupuesto)} bicis al día) se reparte entre las rondas del día.",
    "" if sin_lugar in (0, n) else f"{datos.num(sin_lugar)} de estas estaciones no tienen ubicación y no salen en el mapa.",
    *datos.limitaciones(2),
])
