"""Página "Estaciones y rutas": las estaciones con más salidas y las rutas con más viajes.

Todo es descriptivo y de todo el periodo analizado. Son viajes de usuarios
entre estaciones (demanda atendida), no recorridos de camiones.
"""

import math

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

CUANTAS = 15
COLOR_BARRA = datos.COLOR_SE_LLENA
COLOR_ORIGEN = [230, 159, 0]    # naranja: donde empieza el viaje
COLOR_DESTINO = [0, 114, 178]   # azul: donde termina
ALTO_FILA = 28
ALTO_MAPA = 520
ANCHO_MAPA = 900  # ancho aproximado del mapa en pantalla, en pixeles, para calcular el acercamiento


def _nombres_unicos(nombres: pd.Series, claves: pd.Series) -> pd.Series:
    """Nombres para el eje: si dos se repiten, se distinguen con su clave."""
    nombres = nombres.map(datos.texto)
    repetido = nombres.duplicated(keep=False)
    return nombres.where(~repetido, nombres + " (" + claves.astype(str) + ")")


def _barras(tabla: pd.DataFrame, valor: str, titulo_valor: str, titulo_eje: str,
            detalles: list[tuple[str, str]]) -> alt.LayerChart:
    """Barras horizontales, de mayor a menor, con la cifra al final de cada barra.

    `tabla` trae "nombre", la columna `valor` y "cifra" (el valor ya escrito).
    `detalles` son pares (columna, título) que se agregan al recuadro de ayuda.
    """
    orden = tabla["nombre"].tolist()
    tope = float(tabla[valor].max()) * 1.02 or 1.0
    base = alt.Chart(tabla).encode(
        y=alt.Y("nombre:N", sort=orden, title=titulo_eje,
                # El título va arriba y horizontal: en vertical se encima con los nombres largos.
                axis=alt.Axis(labelLimit=380, labelPadding=6, ticks=False, titleAngle=0,
                              titleAnchor="start", titleAlign="right", titleX=-6, titleY=-10)),
        x=alt.X(f"{valor}:Q", title=titulo_valor, scale=alt.Scale(domain=[0, tope], nice=False),
                axis=alt.Axis(format=",.0f", tickCount=5)),
        tooltip=[alt.Tooltip("nombre:N", title=titulo_eje), alt.Tooltip("cifra:N", title=titulo_valor)]
                + [alt.Tooltip(f"{columna}:N", title=titulo) for columna, titulo in detalles],
    )
    barras = base.mark_bar(color=COLOR_BARRA, cornerRadiusEnd=3)
    # La cifra va dentro de la barra, en blanco: se lee igual con tema claro u oscuro.
    cifras = base.mark_text(align="right", baseline="middle", dx=-6, fontSize=12, fontWeight="bold",
                            color="white").encode(text="cifra:N")
    return (barras + cifras).properties(height=ALTO_FILA * len(tabla) + 50, padding={"top": 22})


def _vista(extremos: pd.DataFrame) -> pdk.ViewState:
    """Centro y acercamiento para que quepan justo las rutas que se muestran (su caja, con margen)."""
    oeste, este = float(extremos["lon"].min()), float(extremos["lon"].max())
    sur, norte = float(extremos["lat"].min()), float(extremos["lat"].max())

    def mercator(lat: float) -> float:
        return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))

    margen = 0.75  # las rutas ocupan tres cuartos del mapa
    # A nivel 0 el mundo (360 grados) mide 512 pixeles; cada nivel lo duplica.
    por_ancho = math.log2(ANCHO_MAPA * margen * 360 / (512 * max(este - oeste, 1e-4)))
    por_alto = math.log2(ALTO_MAPA * margen * 360 / (512 * max(mercator(norte) - mercator(sur), 1e-4)))
    return pdk.ViewState(latitude=(sur + norte) / 2, longitude=(oeste + este) / 2,
                         zoom=min(por_ancho, por_alto, 15), pitch=30)


def _mapa(rutas: pd.DataFrame) -> pdk.Deck | None:
    """Arcos de las rutas (del origen, naranja, al destino, azul). Los viajes que
    vuelven a la misma estación se dibujan como un punto. Sin coordenadas, no hay mapa."""
    columnas = ["origen_lat", "origen_lon", "destino_lat", "destino_lon"]
    r = rutas.dropna(subset=columnas)
    if r.empty:
        return None
    r = r.assign(ancho=6 + 10 * r["viajes"] / r["viajes"].max(),
                 radio=60 + 140 * r["viajes"] / r["viajes"].max())
    campos = ["nombre", "cifra", "ancho", "radio", "misma_estacion", *columnas]
    arcos = r.loc[~r["misma_estacion"], campos]
    puntos = r.loc[r["misma_estacion"], campos]
    capas = []
    if not arcos.empty:
        capas.append(pdk.Layer(
            "ArcLayer", data=arcos, get_source_position=["origen_lon", "origen_lat"],
            get_target_position=["destino_lon", "destino_lat"], get_source_color=COLOR_ORIGEN,
            get_target_color=COLOR_DESTINO, get_width="ancho", get_height=0.6, pickable=True,
            auto_highlight=True))
        # Un punto en cada extremo: el arco de una ruta corta se ve aunque el mapa abarque toda la ciudad.
        for lado, color in (("origen", COLOR_ORIGEN), ("destino", COLOR_DESTINO)):
            capas.append(pdk.Layer(
                "ScatterplotLayer", data=arcos, get_position=[f"{lado}_lon", f"{lado}_lat"], get_radius=40,
                radius_min_pixels=5, radius_max_pixels=9, get_fill_color=color + [230], stroked=True,
                get_line_color=[255, 255, 255, 200], line_width_min_pixels=1, pickable=True))
    if not puntos.empty:
        capas.append(pdk.Layer(
            "ScatterplotLayer", data=puntos, get_position=["origen_lon", "origen_lat"],
            get_radius="radio", radius_min_pixels=4, get_fill_color=COLOR_ORIGEN + [200],
            pickable=True, auto_highlight=True))
    extremos = pd.DataFrame({"lon": pd.concat([r["origen_lon"], r["destino_lon"]]),
                             "lat": pd.concat([r["origen_lat"], r["destino_lat"]])})
    vista = _vista(extremos)
    # Sin estilo fijo, el mapa base (CARTO, no pide llave) sigue el tema claro u oscuro de la app.
    return pdk.Deck(layers=capas, initial_view_state=vista, map_provider="carto", map_style=None,
                    tooltip={"html": "<b>{nombre}</b><br/>{cifra} viajes"})


datos.encabezado("Estaciones y rutas", "viajes")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene
periodo = ind.get("periodo") or {}

st.markdown(
    "Viajes hechos por usuarios entre estaciones durante todo el periodo analizado "
    f"({datos.fecha(periodo.get('inicio'), con_dia=False)} a {datos.fecha(periodo.get('fin'), con_dia=False)}). "
    "Solo se cuenta la demanda atendida; no son recorridos de camiones."
)

a, b, c = st.columns(3)
a.metric("Viajes analizados", datos.num(ind.get("viajes_analizados")))
b.metric("Estaciones", datos.num(ind.get("estaciones")))
c.metric("Pares de estaciones con viajes", datos.num(ind.get("pares_de_estaciones")))

# --------------------------------------------------------------------------- #
# Estaciones con más salidas
# --------------------------------------------------------------------------- #

st.subheader(f"Las {CUANTAS} estaciones con más salidas")

est = datos.top_estaciones(CUANTAS)
if est.empty:
    st.info(f"Estaciones con más salidas: {datos.SIN_DATO}.")
else:
    est = pd.DataFrame({
        "nombre": _nombres_unicos(est["nombre"], est["estacion_id"]),
        "salidas": est["salidas"].astype(float),
        "cifra": est["salidas"].map(datos.num),
        "por_dia": est["salidas_dia"].map(lambda x: datos.num(x, 1)),
        "llegadas_txt": est["llegadas"].map(datos.num),
        "lugares": est["capacidad"].map(datos.num),
        "tipo_txt": est["tipo"].map(lambda t: datos.etiqueta(datos.TIPOS_ESTACION, t)),
    })
    st.altair_chart(
        _barras(est, "salidas", "Salidas en el periodo", "Estación",
                [("por_dia", "Salidas por día"), ("llegadas_txt", "Llegadas en el periodo"),
                 ("lugares", "Lugares de anclaje"), ("tipo_txt", "Tipo de estación")]),
        width="stretch")

# --------------------------------------------------------------------------- #
# Rutas con más viajes
# --------------------------------------------------------------------------- #

st.subheader(f"Las {CUANTAS} rutas con más viajes")

con_vueltas = st.toggle(
    "Incluir viajes que salen y vuelven a la misma estación", value=False,
    help="Son paseos que terminan donde empezaron. Si se incluyen, ocupan los primeros lugares.")

rutas = datos.top_rutas(CUANTAS, con_ida_y_vuelta=con_vueltas)
if rutas.empty:
    st.info(f"Rutas con más viajes: {datos.SIN_DATO}.")
else:
    origen, destino = rutas["origen"].map(datos.texto), rutas["destino"].map(datos.texto)
    nombre = (origen + " → " + destino).where(~rutas["misma_estacion"], origen + " (ida y vuelta)")
    rutas = rutas.assign(
        nombre=_nombres_unicos(nombre, rutas["origen_id"].astype(str) + " a " + rutas["destino_id"].astype(str)),
        viajes=rutas["viajes"].astype(float),
        cifra=rutas["viajes"].map(datos.num),
        origen_txt=origen, destino_txt=destino,
    )
    st.altair_chart(
        _barras(rutas, "viajes", "Viajes en el periodo", "Ruta (origen → destino)",
                [("origen_txt", "Origen"), ("destino_txt", "Destino")]),
        width="stretch")
    deck = _mapa(rutas)
    if deck is not None:
        st.pydeck_chart(deck, height=ALTO_MAPA)
        st.caption("Dónde están estas rutas. Cada arco va del origen (naranja) al destino (azul); es más "
                   "grueso cuanto más viajes tiene la ruta. Son rutas cortas: acerca el mapa para verlas una por una."
                   + (" Los puntos naranjas sin arco son viajes que vuelven a la misma estación."
                      if con_vueltas else ""))

    total = ind.get("viajes_en_rutas")
    if total:
        suma = rutas["viajes"].sum()
        st.caption(f"Estas {datos.num(len(rutas))} rutas suman {datos.num(suma)} viajes, "
                   f"{datos.pct(suma / total)} de todos los viajes del periodo.")
