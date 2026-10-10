"""Página "Mapa de demanda": un punto por estación y la demanda por hora del día.

El tamaño del punto son las salidas promedio y el color, el flujo neto (si la
estación se vacía o se llena). Todo sale de `datos.demanda_mapa()` y
`datos.demanda_sistema()`; aquí solo se da formato y se dibuja.
"""

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

TODO_EL_DIA = "Todo el día"
HORA_INICIAL = 8  # la página abre en la hora pico de la mañana de un día laborable
COLOR_EQUILIBRIO = "#8A8F98"
# Con menos de esto el flujo neto se escribe 0.0: la estación está en equilibrio.
UMBRAL_EQUILIBRIO = 0.05
ESTADOS = {
    "se_vacia": ("Se vacía", datos.COLOR_SE_VACIA, "salen más bicis de las que llegan"),
    "se_llena": ("Se llena", datos.COLOR_SE_LLENA, "llegan más bicis de las que salen"),
    "equilibrio": ("En equilibrio", COLOR_EQUILIBRIO, "salen y llegan casi las mismas"),
}
# Colores que se leen igual en tema claro y en tema oscuro.
COLORES_DIA = {"laborable": "#CC79A7", "fin_de_semana": "#009E73"}
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


def _rgb(color: str, alfa: int = 190) -> list[int]:
    color = color.lstrip("#")
    return [int(color[i:i + 2], 16) for i in (0, 2, 4)] + [alfa]


def _con_signo(x) -> str:
    """Flujo neto con signo: "+1.2", "-0.4" o "0.0"."""
    escrito = datos.num(x, 1)
    return f"+{escrito}" if not datos._falta(x) and x >= UMBRAL_EQUILIBRIO else escrito


def _estado(flujo) -> str:
    if datos._falta(flujo) or abs(flujo) < UMBRAL_EQUILIBRIO:
        return "equilibrio"
    return "se_vacia" if flujo < 0 else "se_llena"


def _puntos(mapa: pd.DataFrame) -> pd.DataFrame:
    """Lo que se manda al mapa: posición, tamaño, color y los textos ya escritos."""
    estado = mapa["flujo_neto"].map(_estado)
    maximo = float(mapa["salidas"].max() or 0)
    escala = (mapa["salidas"].clip(lower=0) / maximo) ** 0.5 if maximo > 0 else 0.0
    puntos = pd.DataFrame({
        "lon": mapa["lon"].astype(float),
        "lat": mapa["lat"].astype(float),
        "radio": (40 + 260 * escala).astype(float).round(1),
        "color": estado.map(lambda e: _rgb(ESTADOS[e][1])),
        "nombre": mapa["nombre"].map(datos.texto),
        "salidas": mapa["salidas"].map(lambda x: datos.num(x, 1)),
        "llegadas": mapa["llegadas"].map(lambda x: datos.num(x, 1)),
        "flujo": mapa["flujo_neto"].map(_con_signo),
        "estado": estado.map(lambda e: ESTADOS[e][0]),
        "capacidad": mapa["capacidad"].map(datos.num),
        "tipo": mapa["tipo"].map(lambda t: datos.etiqueta(datos.TIPOS_ESTACION, t)),
    })
    # Las grandes se dibujan primero para que no tapen a las chicas.
    return puntos.sort_values("radio", ascending=False).reset_index(drop=True)


def _mapa(puntos: pd.DataFrame, unidad: str) -> pdk.Deck:
    capa = pdk.Layer(
        "ScatterplotLayer",
        data=puntos,
        get_position="[lon, lat]",
        get_radius="radio",
        get_fill_color="color",
        radius_min_pixels=2,
        radius_max_pixels=28,
        stroked=True,
        get_line_color=[255, 255, 255, 160],
        line_width_min_pixels=0.5,
        pickable=True,
        auto_highlight=True,
    )
    vista = pdk.ViewState(latitude=float(puntos["lat"].median()), longitude=float(puntos["lon"].median()),
                          zoom=10.8, pitch=0)
    globo = {
        "html": ("<b>{nombre}</b><br/>"
                 f"Salidas {unidad}: " "{salidas}<br/>"
                 f"Llegadas {unidad}: " "{llegadas}<br/>"
                 "Flujo neto: {flujo} ({estado})<br/>"
                 "Capacidad: {capacidad}<br/>"
                 "Tipo: {tipo}"),
        "style": {"backgroundColor": "#FFFFFF", "color": "#222222", "fontSize": "13px",
                  "border": "1px solid #D0D4DA", "borderRadius": "6px"},
    }
    return pdk.Deck(layers=[capa], initial_view_state=vista, map_style=MAPAS_BASE[_tema()], tooltip=globo)


def _demanda_por_hora(usuario: str, hora_elegida: int | None) -> alt.Chart:
    columna = {"miembros": "salidas_miembros", "ocasionales": "salidas_ocasionales"}.get(usuario, "salidas")
    sistema = datos.demanda_sistema()
    serie = pd.DataFrame({
        "hora": sistema["hora"].astype(int),
        "viajes": sistema[columna].astype(float),
        "dia": sistema["tipo_dia"].map(lambda t: datos.etiqueta(datos.TIPOS_DIA, t)),
        "hora_texto": sistema["hora"].map(datos.hora),
        "viajes_texto": sistema[columna].map(datos.num),
    })
    dias = [datos.TIPOS_DIA[c] for c in COLORES_DIA]
    base = alt.Chart(serie).encode(
        x=alt.X("hora:Q", title="Hora del día", scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(values=list(range(0, 24, 3)), labelExpr="format(datum.value, '02') + ':00'",
                              grid=False)),
        y=alt.Y("viajes:Q", title="Viajes iniciados por hora (promedio)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("dia:N", title=None, scale=alt.Scale(domain=dias, range=list(COLORES_DIA.values())),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("dia:N", title="Tipo de día"), alt.Tooltip("hora_texto:N", title="Hora"),
                 alt.Tooltip("viajes_texto:N", title="Viajes promedio")],
    )
    capas = [base.mark_line(strokeWidth=2.5), base.mark_circle(size=45, opacity=1)]
    if hora_elegida is not None:
        marca = pd.DataFrame({"hora": [int(hora_elegida)]})
        capas.insert(0, alt.Chart(marca).mark_rule(color=COLOR_EQUILIBRIO, strokeDash=[4, 4]).encode(x="hora:Q"))
    return alt.layer(*capas).properties(height=330)


datos.encabezado("Mapa de demanda", "viajes")

# --- Selectores ------------------------------------------------------------- #
horas = [None, *range(24)]
c_hora, c_dia, c_usuario = st.columns([2, 2, 3])
hora_elegida = c_hora.selectbox("Hora del día", horas, index=horas.index(HORA_INICIAL), key="mapa_hora",
                                format_func=lambda h: TODO_EL_DIA if h is None else datos.hora(h))
tipo_dia = c_dia.radio("Tipo de día", list(datos.TIPOS_DIA), index=list(datos.TIPOS_DIA).index("laborable"),
                       key="mapa_tipo_dia", horizontal=True,
                       format_func=datos.TIPOS_DIA.get)
usuario = c_usuario.radio("Tipo de usuario", list(datos.USUARIOS), key="mapa_usuario", horizontal=True,
                          format_func=datos.USUARIOS.get)
st.caption("El tipo de usuario solo cambia las salidas: los datos no separan las llegadas por tipo de usuario. "
           "Es demanda atendida (viajes que sí se hicieron), no toda la demanda.")

mapa = datos.demanda_mapa(hora_elegida, tipo_dia, usuario)
mapa = mapa[mapa["lat"].notna() & mapa["lon"].notna()]
if mapa.empty:
    st.info("No hay viajes registrados para esta selección.")
    st.stop()

unidad = "al día" if hora_elegida is None else "en esa hora"
momento = "todo el día" if hora_elegida is None else f"a las {datos.hora(hora_elegida)}"
puntos = _puntos(mapa)
estados = mapa["flujo_neto"].map(_estado)

# --- Mapa y demanda por hora ------------------------------------------------ #
c_mapa, c_lado = st.columns([3, 2], gap="large")

with c_mapa:
    st.subheader(f"Estaciones, {datos.TIPOS_DIA[tipo_dia].lower()}, {momento}")
    st.pydeck_chart(_mapa(puntos, unidad), height=540)
    leyenda = " &nbsp;·&nbsp; ".join(
        f"<span style='white-space:nowrap'><span style='color:{color}'>●</span> <b>{nombre}</b>: {explicacion}</span>"
        for nombre, color, explicacion in ESTADOS.values())
    st.markdown(f"<small>{leyenda}.<br/>El tamaño del punto son las salidas promedio {unidad}; "
                "el color usa a todos los usuarios.</small>", unsafe_allow_html=True)

with c_lado:
    m1, m2, m3 = st.columns([3, 2, 2])
    m1.metric(f"Salidas {unidad}", datos.num(mapa["salidas"].sum()))
    m2.metric("Se vacían", datos.num((estados == "se_vacia").sum()))
    m3.metric("Se llenan", datos.num((estados == "se_llena").sum()))
    st.caption(f"Promedios de {datos.num(len(mapa))} estaciones; las dos últimas cifras cuentan estaciones.")

    st.subheader("Demanda por hora del día")
    st.altair_chart(_demanda_por_hora(usuario, hora_elegida), width="stretch")
    st.caption(f"Viajes iniciados en todo el sistema ({datos.USUARIOS[usuario].lower()}), promedio por día.")
