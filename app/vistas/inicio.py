"""Página "Inicio": el problema, cuatro indicadores, cuatro hallazgos, la curva de riesgo contra
presupuesto (con el punto de mover al azar) y, cuando ya se corrió la prueba final, la
comparación de validación contra prueba final. Cada sección abre con una frase que dice el hallazgo. Ninguna cifra está escrita a
mano: todas salen de app/datos.py, así que cambian solas al reconstruir las tablas."""

import altair as alt
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

EJE_X = "Bicis movidas por día"
EJE_Y = "Horas de estación en riesgo por día"
COLOR_REFERENCIA = "#8A8F98"
ALTO = 360  # alto del área de la gráfica, en pixeles

datos.encabezado("Rebalanceo de Citi Bike en Nueva York", "inicio")

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
# El indicador usa la regla histórica; si no se simuló, el modelo.
clave = next((p for p in ("historical", "model") if politicas.get(p)), None)
evitado = politicas.get(clave) or {}
pronostico = ind.get("pronostico") or {}
error = (pronostico.get("lgbm") or {}).get("salidas") or {}
error_promedio = (pronostico.get("baseline") or {}).get("salidas") or {}
error_media_global = (((ind.get("pronostico_piso") or {}).get("media_global") or {}).get("salidas") or {})
# Con casi el mismo resultado (intervalos que se traslapan), las dos líneas de la curva se enciman.
mismo_resultado = (ind.get("comparacion") or {}).get("modelo_evita") == "igual"

if periodo.get("inicio") and periodo.get("fin"):
    nota_viajes = f"Del {datos.fecha(periodo['inicio'], False)} al {datos.fecha(periodo['fin'], False)}"
else:
    nota_viajes = f"Periodo: {datos.SIN_DATO}"

# Riesgo simulado evitado: el rango entre los inventarios iniciales supuestos, no solo el valor más alto.
# Bajo cada indicador va una sola línea; el detalle baja al desplegable del final de la página.
valor_evitado, nota_evitado = datos.pct(evitado.get("pct_evitado"), 0), datos.SIN_DATO
detalle_evitado = []
if clave is not None:
    rango = ((ind.get("sensibilidad") or {}).get("rango") or {}).get(clave) or {}
    nota_evitado = f"{datos.POLITICAS[clave]}, con {datos.num(presupuesto)} bicis movidas al día"
    detalle_evitado.append(f"Riesgo simulado evitado con la {datos.POLITICAS[clave].lower()} y el presupuesto de "
                           f"referencia: {datos.pct(evitado.get('pct_evitado'))}.")
    if rango.get("pct_min") is not None and rango.get("pct_max") is not None:
        valor_evitado = datos.rango_pct(rango["pct_min"], rango["pct_max"])
        detalle_evitado.append("El rango del indicador depende del inventario inicial que se suponga.")
    if clave == "historical" and politicas.get("model"):
        detalle_evitado.append(f"El {datos.POLITICAS['model'].lower()} da prácticamente el mismo resultado."
                               if mismo_resultado else
                               f"Con el {datos.POLITICAS['model'].lower()}: {datos.pct(politicas['model'].get('pct_evitado'))}.")
    al_azar = (ind.get("piso_de_azar") or {}).get("pct_evitado")
    if al_azar is not None:
        detalle_evitado.append(f"Mover las mismas bicis al azar evita {datos.pct(al_azar)}.")

nota_error = f"Salidas, con el {datos.MODELOS['lgbm'].lower()}: {datos.pct(error.get('wape'))}"
detalle_error = []
if error_promedio.get("wape") is not None:
    detalle_error.append(f"Error del pronóstico de salidas con el {datos.MODELOS['baseline'].lower()}: "
                         f"{datos.pct(error_promedio['wape'])}.")
if error_media_global.get("wape") is not None:
    detalle_error.append("Error del pronóstico de salidas con una sola media para todo el sistema: "
                         f"{datos.pct(error_media_global['wape'])}.")

c1, c2, c3, c4 = st.columns(4)
indicador(c1, "Viajes analizados", datos.redondo(ind.get("viajes_analizados")),
          f"{datos.num(ind.get('viajes_analizados'))} viajes")
indicador(c2, "Estaciones", datos.redondo(ind.get("estaciones")),
          f"{datos.num(ind.get('estaciones'))} en total; {datos.num(simulacion.get('estaciones_simuladas'))} en la simulación")
indicador(c3, "Riesgo simulado evitado", valor_evitado, nota_evitado)
indicador(c4, "Error del pronóstico", datos.pct(error.get("wape"), 0), nota_error)


# --------------------------------------------------------------------------- #
# Cuatro hallazgos (a la vista sin bajar)
# --------------------------------------------------------------------------- #

def titular_de_hallazgos() -> str:
    """El hallazgo principal, con la cifra de los datos; sin cifra, un título neutro."""
    fraccion = evitado.get("pct_evitado")
    if clave is None or datos._falta(fraccion) or fraccion <= 0:
        return "Hallazgos"
    if fraccion > 0.5:
        return "Planear los traslados evita más de la mitad del riesgo simulado"
    return f"Planear los traslados evita {datos.pct(fraccion, 0)} del riesgo simulado"


st.subheader(titular_de_hallazgos())
frases = datos.hallazgos()
izquierda, derecha = st.columns(2, gap="large")
for numero, frase in enumerate(frases, start=1):
    with (izquierda if numero % 2 else derecha):
        st.markdown(f"**{numero}.** {frase}")


# --------------------------------------------------------------------------- #
# Curva de riesgo contra presupuesto
# --------------------------------------------------------------------------- #

def titular_de_la_curva() -> str:
    """Qué dice la curva sobre el modelo y la regla histórica en validación; sin veredicto, un título neutro."""
    if not (politicas.get("historical") and politicas.get("model")):
        return "Riesgo simulado contra presupuesto"
    veredicto = datos.modelo_contra_regla(ind)["referencia"].get("validacion")
    historica, modelo = datos.POLITICAS["historical"], datos.POLITICAS["model"]
    if veredicto == "igual":
        return f"{historica} y {modelo} evitan casi el mismo riesgo simulado"
    if veredicto == "mas":
        return f"El {modelo.lower()} evita más riesgo simulado que la {historica.lower()}"
    if veredicto == "menos":
        return f"La {historica.lower()} evita más riesgo simulado que el {modelo.lower()}"
    return "Riesgo simulado contra presupuesto"


st.subheader(titular_de_la_curva())

curva = datos.curva().dropna(subset=["presupuesto", "riesgo"])
curva = curva[curva["politica"].isin(datos.POLITICAS)].sort_values(["politica", "presupuesto"])

if curva.empty:
    st.info(f"Curva de riesgo contra presupuesto: {datos.SIN_DATO}.")
else:
    st.caption(f"{datos.explica_riesgo()} Mientras más baja la línea, mejor.")
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
                                        symbolOpacity=1))
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

    # Mover las mismas bicis al azar: un punto con su nombre, en el presupuesto con que se simuló.
    azar = ind.get("piso_de_azar") or {}
    if azar.get("riesgo") is not None and azar.get("presupuesto") is not None:
        a_la_izquierda = azar["presupuesto"] > curva["presupuesto"].max() / 2
        punto_azar = alt.Chart(alt.Data(values=[{
            "presupuesto": azar["presupuesto"], "riesgo": float(azar["riesgo"]), "nombre": datos.AZAR,
            "bicis_txt": datos.num(azar["presupuesto"]), "riesgo_txt": datos.num(azar["riesgo"]),
            "evitado_txt": datos.pct(azar.get("pct_evitado")),
        }]))
        capas.append(punto_azar.mark_point(size=110, filled=True, opacity=1, shape="diamond", color=datos.COLOR_AZAR)
                     .encode(x="presupuesto:Q", y="riesgo:Q",
                             tooltip=[alt.Tooltip("nombre:N", title="Política"),
                                      alt.Tooltip("bicis_txt:N", title=EJE_X),
                                      alt.Tooltip("riesgo_txt:N", title=EJE_Y),
                                      alt.Tooltip("evitado_txt:N", title="Riesgo simulado evitado")]))
        capas.append(punto_azar.mark_text(color=datos.COLOR_AZAR, fontSize=datos.FUENTE_EJE, fontWeight="bold", baseline="top",
                                          align="right" if a_la_izquierda else "left",
                                          dx=-9 if a_la_izquierda else 9, dy=5)
                     .encode(x="presupuesto:Q", y="riesgo:Q", text="nombre:N"))

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
        capas.append(marca.mark_text(color=COLOR_REFERENCIA, fontSize=datos.FUENTE_EJE, baseline="middle",
                                     align="left" if a_la_derecha else "right",
                                     dx=6 if a_la_derecha else -6)
                     .encode(x="presupuesto:Q", y="altura:Q", text="texto:N"))

    st.altair_chart(datos.estilo_grafica(alt.layer(*capas).properties(height=ALTO)), width="stretch",
                    height=ALTO + 130)
    frase = datos.frase_de_la_curva() if {"historical", "model"} <= set(claves) else ""
    datos.como_se_calculo([
        "Cada línea es una forma de decidir los traslados; la banda es el intervalo de la simulación.",
        frase,
        nota_viajes + "." if not nota_viajes.endswith(".") else nota_viajes,
        *detalle_evitado,
        "El error del pronóstico es el error relativo por hora y estación, en validación.",
        *detalle_error,
    ])


# --------------------------------------------------------------------------- #
# Validación y prueba final, lado a lado (solo cuando ya existe la prueba final)
# --------------------------------------------------------------------------- #

comparacion = datos.validacion_y_prueba()
if comparacion["filas"]:
    st.subheader("Validación y prueba final")
    lineas = ["| | Validación | Prueba final |", "|:--|--:|--:|"]
    junto = lambda valor: valor.replace(" ", "&nbsp;")  # la cifra y su intervalo, en un solo renglón
    lineas += [f"| {f['afirmacion']} | {junto(f['validacion'])} | {junto(f['prueba'])} |"
               for f in comparacion["filas"]]
    st.markdown("\n".join(lineas))
    notas = []
    por_periodo = comparacion["presupuestos"]
    if por_periodo.get("validacion") is not None and por_periodo.get("prueba") is not None:
        notas.append(f"Presupuesto de referencia: {datos.num(por_periodo['validacion'])} bicis movidas al día en "
                     f"validación y {datos.num(por_periodo['prueba'])} en la prueba final.")
    elif comparacion["presupuesto"] is not None:
        notas.append(f"Con {datos.num(comparacion['presupuesto'])} bicis movidas al día.")
    if any("(" in f["validacion"] or "(" in f["prueba"] for f in comparacion["filas"]):
        notas.append("Entre paréntesis, el intervalo.")
    datos.como_se_calculo(notas)
