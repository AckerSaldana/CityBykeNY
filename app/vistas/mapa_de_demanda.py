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
ZOOM = 10.8  # acercamiento con que abre el mapa
COLOR_EQUILIBRIO = "#8A8F98"
# Con menos de esto el flujo neto se escribe 0.0: la estación está en equilibrio.
UMBRAL_EQUILIBRIO = 0.05
ESTADOS = {
    "se_vacia": ("Se vacía", datos.COLOR_SE_VACIA, "salen más bicis de las que llegan"),
    "se_llena": ("Se llena", datos.COLOR_SE_LLENA, "llegan más bicis de las que salen"),
    "equilibrio": ("En equilibrio", COLOR_EQUILIBRIO, "salen y llegan casi las mismas"),
}
# Par propio de esta gráfica: ni los colores de las políticas ni los del mapa.
COLORES_DIA = {"laborable": "#1B7F8C", "fin_de_semana": "#A0522D"}
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


def _vista() -> pdk.ViewState:
    """Vista fija sobre la ciudad: el centro de todas las estaciones, no solo las de la selección,
    así el mapa no se mueve al cambiar de hora o de tipo de día."""
    todas = datos.estaciones()
    todas = todas[todas["lat"].notna() & todas["lon"].notna()]
    return pdk.ViewState(latitude=float(todas["lat"].median()), longitude=float(todas["lon"].median()),
                         zoom=ZOOM, min_zoom=ZOOM - 2, max_zoom=ZOOM + 5, pitch=0, bearing=0)


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
    # La rueda del ratón no acerca el mapa (así la página se puede recorrer); se acerca con los
    # botones o con doble clic, y se arrastra para moverlo.
    control = pdk.View(type="MapView", controller={"scrollZoom": False, "dragRotate": False, "touchRotate": False})
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
    return pdk.Deck(layers=[capa], views=[control], initial_view_state=_vista(), map_style=MAPAS_BASE[_tema()],
                    tooltip=globo)


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
    return datos.estilo_grafica(alt.layer(*capas).properties(height=330))


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

se_vacian, se_llenan = int((estados == "se_vacia").sum()), int((estados == "se_llena").sum())
dia_texto = datos.TIPOS_DIA[tipo_dia].lower()


def _titular_del_pico() -> str:
    """A qué hora hay más viajes en el tipo de día elegido; sin datos, un título neutro."""
    columna = {"miembros": "salidas_miembros", "ocasionales": "salidas_ocasionales"}.get(usuario, "salidas")
    sistema = datos.demanda_sistema()
    del_dia = sistema[(sistema["tipo_dia"] == tipo_dia) & sistema[columna].notna()]
    if del_dia.empty or float(del_dia[columna].max()) <= 0:
        return "Demanda por hora del día"
    pico = del_dia.loc[del_dia[columna].idxmax(), "hora"]
    return f"En día {dia_texto}, la hora con más viajes es la de las {datos.hora(pico)}"


with c_mapa:
    cuando = "En todo el día" if hora_elegida is None else f"A las {datos.hora(hora_elegida)}"
    st.subheader(f"{cuando} en día {dia_texto}, {datos.num(se_vacian)} "
                 f"{'estación se vacía' if se_vacian == 1 else 'estaciones se vacían'} y "
                 f"{datos.num(se_llenan)} {'se llena' if se_llenan == 1 else 'se llenan'}")
    # La leyenda va ARRIBA del mapa: primero cómo se lee, después el mapa.
    leyenda = " &nbsp;·&nbsp; ".join(
        f"<span style='white-space:nowrap'><span style='color:{color};font-size:1.3em'>●</span> "
        f"<b>{nombre}</b>: {explicacion}</span>"
        for nombre, color, explicacion in ESTADOS.values())
    st.markdown(f"{leyenda}. El tamaño del punto son las salidas promedio {unidad}.", unsafe_allow_html=True)
    st.pydeck_chart(_mapa(puntos, unidad), height=540)

with c_lado:
    st.metric(f"Salidas {unidad}", datos.num(mapa["salidas"].sum()))

    st.subheader(_titular_del_pico())
    st.altair_chart(_demanda_por_hora(usuario, hora_elegida), width="stretch")

datos.como_se_calculo([
    f"El mapa muestra promedios de {datos.num(len(mapa))} estaciones, {dia_texto}, {momento}.",
    "El color del punto es el flujo neto (llegadas menos salidas) de todos los usuarios.",
    "El tipo de usuario solo cambia las salidas: los datos no separan las llegadas por tipo de usuario.",
    f"La gráfica son los viajes iniciados en todo el sistema ({datos.USUARIOS[usuario].lower()}), promedio por día.",
    "Es demanda atendida (viajes que sí se hicieron), no toda la demanda.",
])
