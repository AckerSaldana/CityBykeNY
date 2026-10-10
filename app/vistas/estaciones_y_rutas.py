"""Página "Estaciones y rutas": las estaciones con más salidas y las rutas con más viajes.

Todo es descriptivo y de todo el periodo analizado. Son viajes de usuarios
entre estaciones (demanda atendida), no recorridos de camiones.
"""

import altair as alt
import pandas as pd
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

CUANTAS = 15
COLOR_BARRA = datos.COLOR_SE_LLENA
ALTO_FILA = 28
CORTA_KM = 1.0  # con una mediana menor, las rutas más usadas se describen como tramos cortos


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


def _distancia(resumen: dict) -> str:
    """Qué tan largas son las rutas más usadas entre estaciones distintas (lo calcula el constructor)."""
    mediana, cuantas = resumen.get("distancia_mediana_km"), resumen.get("rutas")
    if datos._falta(mediana) or not cuantas:
        return f"Distancia de las rutas más usadas: {datos.SIN_DATO}."
    escrita = f"{datos.num(mediana * 1000)} metros" if mediana < 1 else f"{datos.num(mediana, 1)} km"
    cuales = f"Las {datos.num(cuantas)} rutas más usadas entre estaciones distintas"
    if mediana < CORTA_KM:
        return (f"{cuales} son tramos cortos entre estaciones vecinas: su distancia mediana es de {escrita} "
                "en línea recta.")
    return f"{cuales} tienen una distancia mediana de {escrita} en línea recta."


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
    st.markdown(_distancia(ind.get("rutas_mas_usadas") or {}))

    total = ind.get("viajes_en_rutas")
    if total:
        suma = rutas["viajes"].sum()
        st.caption(f"Estas {datos.num(len(rutas))} rutas suman {datos.num(suma)} viajes, "
                   f"{datos.pct(suma / total)} de todos los viajes del periodo.")
