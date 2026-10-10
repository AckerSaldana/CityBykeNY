"""Página "Inicio": el problema, cuatro indicadores, la curva de riesgo contra
presupuesto y tres hallazgos. Ninguna cifra está escrita a mano: todas salen de
app/datos.py, así que cambian solas al reconstruir las tablas."""

import altair as alt
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

EJE_X = "Bicis movidas por día"
EJE_Y = "Horas de estación en riesgo por día"
COLOR_REFERENCIA = "#8A8F98"
ALTO = 340  # alto del área de la gráfica, en pixeles

datos.encabezado("Rebalanceo de Citi Bike en Nueva York")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene
referencia = ind.get("referencia") or {}
presupuesto = referencia.get("presupuesto")

st.markdown(
    "A lo largo del día hay estaciones que se quedan vacías y otras que se llenan; la decisión es "
    "**cuántas bicis mover a cada estación y en qué franja horaria**."
)


# --------------------------------------------------------------------------- #
# Cuatro indicadores
# --------------------------------------------------------------------------- #

def indicador(columna, titulo: str, valor: str, nota: str) -> None:
    with columna:
        st.metric(titulo, valor, border=True)
        st.caption(nota)


periodo = ind.get("periodo") or {}
simulacion = ind.get("simulacion") or {}
politicas = referencia.get("politicas") or {}
# El indicador usa el modelo; si no se simuló, la regla histórica.
clave = next((p for p in ("model", "historical") if politicas.get(p)), None)
evitado = politicas.get(clave) or {}
error = ((ind.get("pronostico") or {}).get("lgbm") or {}).get("salidas") or {}

if periodo.get("inicio") and periodo.get("fin"):
    nota_viajes = f"Del {datos.fecha(periodo['inicio'], False)} al {datos.fecha(periodo['fin'], False)}"
else:
    nota_viajes = f"Periodo: {datos.SIN_DATO}"

nota_evitado = datos.SIN_DATO
if clave is not None:
    nota_evitado = f"{datos.POLITICAS[clave]}, con {datos.num(presupuesto)} bicis movidas al día"
    intervalo = datos.rango_pct(evitado.get("pct_ic_bajo"), evitado.get("pct_ic_alto"))
    if intervalo != datos.SIN_DATO:
        nota_evitado += f" · intervalo de {intervalo}"

c1, c2, c3, c4 = st.columns(4)
indicador(c1, "Viajes analizados", datos.num(ind.get("viajes_analizados")), nota_viajes)
indicador(c2, "Estaciones", datos.num(ind.get("estaciones")),
          f"{datos.num(simulacion.get('estaciones_simuladas'))} en la simulación")
indicador(c3, "Riesgo simulado evitado", datos.pct(evitado.get("pct_evitado")), nota_evitado)
indicador(c4, "Error del pronóstico (WAPE)", datos.pct(error.get("wape")),
          f"Salidas por hora, con el modelo · cobertura del intervalo: {datos.pct(error.get('cobertura'))}")


# --------------------------------------------------------------------------- #
# Curva de riesgo contra presupuesto
# --------------------------------------------------------------------------- #

st.subheader("Riesgo simulado contra presupuesto")

curva = datos.curva().dropna(subset=["presupuesto", "riesgo"])
curva = curva[curva["politica"].isin(datos.POLITICAS)].sort_values(["politica", "presupuesto"])

if curva.empty:
    st.info(f"Curva de riesgo contra presupuesto: {datos.SIN_DATO}.")
else:
    st.markdown(
        "Cada línea es una forma de decidir los traslados. Mientras más baja, menos horas pasan las "
        "estaciones casi vacías o casi llenas. La banda es el intervalo de la simulación."
    )
    claves = [p for p in datos.POLITICAS if p in set(curva["politica"])]
    nombres = [datos.POLITICAS[p] for p in claves]
    colores = [datos.COLORES_POLITICA[p] for p in claves]

    # Textos ya formateados para la etiqueta que aparece al pasar el cursor.
    curva["Política"] = curva["politica"].map(datos.POLITICAS)
    curva["bicis_txt"] = curva["presupuesto"].map(datos.num)
    curva["riesgo_txt"] = curva["riesgo"].map(datos.num)
    curva["intervalo_txt"] = [
        datos.SIN_DATO if datos.SIN_DATO in (a, b) else f"{a} a {b}"
        for a, b in zip(curva["ic_bajo"].map(datos.num), curva["ic_alto"].map(datos.num))
    ]
    curva["evitado_txt"] = curva["pct_evitado"].map(datos.pct)

    escala = alt.Scale(domain=nombres, range=colores)
    color = alt.Color("Política:N", scale=escala,
                      legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=3,
                                        symbolOpacity=1, labelFontSize=13))
    x = alt.X("presupuesto:Q", title=EJE_X, axis=alt.Axis(format=",.0f", grid=False, tickCount=6))
    y = alt.Y("riesgo:Q", title=EJE_Y, axis=alt.Axis(format=",.0f", tickCount=6), scale=alt.Scale(zero=True))
    base = alt.Chart(curva)

    banda = base.mark_area(opacity=0.15).encode(
        x=x, y=alt.Y("ic_bajo:Q", title=EJE_Y), y2="ic_alto:Q", color=color)
    sin_intervencion = datos.POLITICAS["none"]
    lineas = base.mark_line(strokeWidth=2).encode(
        x=x, y=y, color=color,
        strokeDash=alt.condition(alt.datum["Política"] == sin_intervencion, alt.value([6, 4]), alt.value([1, 0])))
    puntos = base.mark_point(size=70, filled=True, opacity=1).encode(
        x=x, y=y, color=color,
        tooltip=[alt.Tooltip("Política:N", title="Política"),
                 alt.Tooltip("bicis_txt:N", title=EJE_X),
                 alt.Tooltip("riesgo_txt:N", title=EJE_Y),
                 alt.Tooltip("intervalo_txt:N", title="Intervalo"),
                 alt.Tooltip("evitado_txt:N", title="Riesgo simulado evitado")])
    capas = [banda, lineas, puntos]

    if presupuesto is not None:
        # La etiqueta va hacia el lado donde hay espacio.
        a_la_derecha = presupuesto <= curva["presupuesto"].max() / 2
        # A media altura entre la línea sin intervención y las otras, para no tapar ninguna.
        en_referencia = curva[(curva["presupuesto"] == presupuesto) & (curva["politica"] != "none")]["riesgo"]
        altura = (curva["riesgo"].max() + en_referencia.max()) / 2 if len(en_referencia) else curva["riesgo"].max() / 2
        marca = alt.Chart(alt.Data(values=[{
            "presupuesto": presupuesto,
            "altura": float(altura),
            "texto": f"Presupuesto de referencia: {datos.num(presupuesto)} bicis",
        }]))
        capas.append(marca.mark_rule(color=COLOR_REFERENCIA, strokeDash=[4, 4], strokeWidth=1.5)
                     .encode(x="presupuesto:Q"))
        capas.append(marca.mark_text(color=COLOR_REFERENCIA, fontSize=12, baseline="middle",
                                     align="left" if a_la_derecha else "right",
                                     dx=6 if a_la_derecha else -6)
                     .encode(x="presupuesto:Q", y="altura:Q", text="texto:N"))

    st.altair_chart(alt.layer(*capas).properties(height=ALTO), width="stretch", height=ALTO + 110)


# --------------------------------------------------------------------------- #
# Tres hallazgos
# --------------------------------------------------------------------------- #

st.subheader("Hallazgos")
for numero, frase in enumerate(datos.hallazgos(), start=1):
    st.markdown(f"**{numero}.** {frase}")
