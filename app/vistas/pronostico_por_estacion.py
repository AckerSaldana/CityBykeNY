"""Página "Pronóstico por estación".

Para una estación y un día de validación: el pronóstico del modelo por hora
(línea central y rango 10 a 90), lo observado y el promedio histórico; debajo,
una tarjeta por franja con lo que hizo UNA política en la simulación (se elige
con un selector; abre con la regla histórica).

La página abre con la estación y el día de ejemplo que elige el constructor de
las tablas (`datos.estacion_de_ejemplo()`), no con un id escrito aquí.
"""

import altair as alt
import pandas as pd
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

SERIE_MODELO = "Modelo"
SERIE_RANGO = "Rango esperado"
SERIE_PROMEDIO = datos.MODELOS["baseline"]
SERIE_OBSERVADO = "Observado"
COLOR_TEXTO_RIESGO = {"alto": "#FFFFFF", "medio": "#1A1A1A", "bajo": "#FFFFFF"}
POLITICAS_CON_TRASLADO = ("historical", "model")  # la primera es la que se muestra al abrir
COMO_LEER = "La franja es el rango esperado; los puntos, lo que pasó."

datos.encabezado("Pronóstico por estación")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene


def por_hora(tabla: pd.DataFrame) -> pd.DataFrame:
    """Una fila por hora del día (el día del cambio de horario repite una hora: se promedia)."""
    return (tabla.groupby("hora", as_index=False)[["q10", "q50", "q90", "observado"]].mean()
            .astype({"hora": int}))


def color_observado() -> str:
    """El color de lo observado; con el tema oscuro, uno claro para que se vea."""
    try:
        oscuro = st.context.theme.type == "dark"
    except Exception:
        oscuro = False
    return "#F2F2F2" if oscuro else datos.COLOR_OBSERVADO


def grafica(modelo: pd.DataFrame, promedio: pd.DataFrame, titulo_y: str) -> alt.Chart:
    """Pronóstico del modelo con su rango, promedio histórico y observado, por hora del día."""
    series = [SERIE_MODELO, SERIE_RANGO, SERIE_PROMEDIO, SERIE_OBSERVADO]
    colores = [datos.COLORES_MODELO["lgbm"], "#9CC7E4", datos.COLORES_MODELO["baseline"], color_observado()]
    color = alt.Color("Serie:N", scale=alt.Scale(domain=series, range=colores),
                      legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=4,
                                        symbolSize=300, symbolOpacity=1))
    eje_x = alt.X("Hora:Q", title="Hora del día", scale=alt.Scale(domain=[0, 23], nice=False),
                  axis=alt.Axis(values=list(range(24)), grid=False,
                                labelExpr="datum.value < 10 ? '0' + datum.value : '' + datum.value"))
    eje_y = alt.Y("Valor:Q", title=titulo_y, scale=alt.Scale(zero=True))

    def serie(tabla, columna, nombre):
        return pd.DataFrame({"Hora": tabla["hora"], "Valor": tabla[columna], "Serie": nombre})

    rango = pd.DataFrame({"Hora": modelo["hora"], "Bajo": modelo["q10"], "Alto": modelo["q90"],
                          "Serie": SERIE_RANGO})
    capas = [
        alt.Chart(rango).mark_area(opacity=0.35).encode(
            x=eje_x, y=alt.Y("Bajo:Q", title=titulo_y), y2="Alto:Q", color=color),
    ]
    if not promedio.empty:
        capas.append(alt.Chart(serie(promedio, "q50", SERIE_PROMEDIO)).mark_line(
            strokeWidth=1.5, strokeDash=[5, 4]).encode(x=eje_x, y=eje_y, color=color))
    capas.append(alt.Chart(serie(modelo, "q50", SERIE_MODELO)).mark_line(strokeWidth=3).encode(
        x=eje_x, y=eje_y, color=color))
    capas.append(alt.Chart(serie(modelo, "observado", SERIE_OBSERVADO)).mark_point(
        filled=True, size=110, opacity=1, stroke="#808080", strokeWidth=1).encode(x=eje_x, y=eje_y, color=color))

    # Una franja invisible por hora que muestra todos los valores al pasar el cursor.
    detalle = pd.DataFrame({
        "Hora": modelo["hora"], "Desde": modelo["hora"] - 0.5, "Hasta": modelo["hora"] + 0.5,
        "Hora del día": modelo["hora"].map(datos.hora),
        SERIE_MODELO: modelo["q50"].map(lambda v: datos.num(v, 1)),
        SERIE_RANGO: [f"{datos.num(a, 1)} a {datos.num(b, 1)}" for a, b in zip(modelo["q10"], modelo["q90"])],
        SERIE_OBSERVADO: modelo["observado"].map(lambda v: datos.num(v, 1)),
    })
    campos = ["Hora del día", SERIE_MODELO, SERIE_RANGO, SERIE_OBSERVADO]
    if not promedio.empty:
        referencia = promedio.set_index("hora")["q50"]
        detalle[SERIE_PROMEDIO] = [datos.num(referencia.get(h), 1) for h in modelo["hora"]]
        campos.insert(3, SERIE_PROMEDIO)
    capas.append(alt.Chart(detalle).mark_rect(opacity=0).encode(
        x=alt.X("Desde:Q", title="Hora del día"), x2="Hasta:Q",
        tooltip=[alt.Tooltip(f"{c}:N") for c in campos]))
    return datos.estilo_grafica(alt.layer(*capas).properties(height=380))


def tarjeta(fila: pd.Series, riesgo_sin_mover, ronda: dict | None = None) -> None:
    """Lo que hizo una política en una franja: la acción en grande y, en un renglón, los niveles de riesgo.

    El nivel de la tabla es el previsto con el inventario ANTES del traslado de esa política (no el que
    queda después): por eso se escribe "Riesgo previsto" y no "Riesgo tras el traslado".
    """
    nivel = fila["nivel_riesgo"]
    with st.container(border=True):
        st.caption(f"{datos.etiqueta(datos.FRANJAS, fila['franja'])} · "
                   f"{datos.hora(fila['hora_inicio'])} a {datos.hora(fila['hora_fin'])}")
        st.markdown(f"### {datos.accion(fila['bicis'], fila['hora_inicio'], ronda)}")
        fondo = datos.COLORES_RIESGO.get(nivel, "#8A8F98")
        letra = COLOR_TEXTO_RIESGO.get(nivel, "#FFFFFF")
        st.markdown(
            f'<span style="background:{fondo};color:{letra};padding:3px 12px;border-radius:12px;'
            f'font-weight:600">Riesgo previsto: {datos.etiqueta(datos.RIESGOS, nivel).lower()}.</span> '
            f"Sin intervenir: {datos.etiqueta(datos.RIESGOS, riesgo_sin_mover).lower()}",
            unsafe_allow_html=True)
        st.markdown(f"En la estación: **{datos.inventario(fila['inventario_antes'], fila['inventario'])}**")
        if not datos._falta(fila["factible"]) and not bool(fila["factible"]):
            st.markdown("Ningún inventario inicial evita los dos riesgos (quedar casi vacía y quedar casi "
                        "llena) en esta franja: conviene una segunda visita durante la franja.")
        elif nivel == "alto" and not datos._falta(fila["bicis"]) and int(fila["bicis"]) == 0:
            st.markdown("El presupuesto no alcanzó para esta estación.")


# --------------------------------------------------------------------------- #
# Selectores
# --------------------------------------------------------------------------- #

opciones = datos.opciones_de_estacion()
if not opciones:
    st.info("No hay estaciones con pronóstico.")
    st.stop()

# Al abrir: la estación y el día de ejemplo (una de oficinas con mucha demanda, traslado en todas
# las franjas y con solución, en día laborable; de esos días, el de error de pronóstico más típico
# de la estación). Si no existe, la primera de la lista (la de más salidas).
ejemplo = datos.estacion_de_ejemplo()
if "pron_estacion" not in st.session_state and ejemplo.get("estacion_id") in opciones:
    st.session_state["pron_estacion"] = ejemplo["estacion_id"]
    if ejemplo.get("fecha") and "pron_dia" not in st.session_state:
        st.session_state["pron_dia"] = pd.Timestamp(ejemplo["fecha"])

col_estacion, col_dia, col_objetivo = st.columns([3, 2, 2])
estacion_id = col_estacion.selectbox("Estación", list(opciones), format_func=opciones.get, key="pron_estacion")
objetivo = col_objetivo.radio("Viajes", list(datos.OBJETIVOS), format_func=datos.OBJETIVOS.get,
                              horizontal=True, key="pron_objetivo")

ficha = datos.estacion(estacion_id)
pron = datos.pronostico_estacion(estacion_id, objetivo)
rec = datos.recomendaciones_estacion(estacion_id)

# Días de validación de la estación: los que tienen pronóstico o simulación.
dias_pron = set(pd.to_datetime(pron["fecha"]).dt.normalize())
dias_rec = set(pd.to_datetime(rec["fecha"]).dt.normalize()) if not rec.empty else set()
dias = sorted(dias_pron | dias_rec)
if not dias:
    st.info("Esta estación no tiene pronóstico en los días de validación.")
    st.stop()
if st.session_state.get("pron_dia") not in dias:
    # Al cambiar de estación se conserva el día si existe; si no, el primero con pronóstico y simulación.
    completos = sorted(dias_pron & dias_rec) or sorted(dias_pron) or dias
    st.session_state["pron_dia"] = completos[0]
dia = col_dia.selectbox("Día", dias, format_func=datos.fecha, key="pron_dia")

del_dia = pron[pd.to_datetime(pron["fecha"]).dt.normalize() == dia]
modelo_dia = del_dia[del_dia["modelo"] == "lgbm"]
promedio_dia = del_dia[del_dia["modelo"] == "baseline"]

# --------------------------------------------------------------------------- #
# Ficha de la estación y gráfica
# --------------------------------------------------------------------------- #

capacidad = ficha.get("capacidad")
nombre_objetivo = datos.OBJETIVOS[objetivo]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Capacidad", datos.SIN_DATO if datos._falta(capacidad) else f"{datos.num(capacidad)} lugares")
m2.metric("Tipo de estación", datos.etiqueta(datos.TIPOS_ESTACION, ficha.get("tipo")))
m3.metric(f"{nombre_objetivo} pronosticadas en el día",
          datos.num(modelo_dia["q50"].sum()) if not modelo_dia.empty else datos.SIN_DATO)
m4.metric(f"{nombre_objetivo} observadas en el día",
          datos.num(modelo_dia["observado"].sum()) if not modelo_dia.empty else datos.SIN_DATO)

if ficha.get("sin_historia"):
    st.caption("Esta estación no tiene historia propia: su pronóstico es el promedio del sistema, no el suyo.")

if modelo_dia.empty:
    st.subheader(f"{nombre_objetivo} por hora")
    st.info("No hay pronóstico del modelo para esta estación en este día.")
else:
    horas_del_dia = por_hora(modelo_dia)
    con_dato = horas_del_dia.dropna(subset=["q10", "q90", "observado"])
    dentro = int(((con_dato["observado"] >= con_dato["q10"]) & (con_dato["observado"] <= con_dato["q90"])).sum())
    if len(con_dato):
        st.subheader(f"Lo que pasó cayó dentro del rango esperado en {datos.num(dentro)} de "
                     f"{datos.num(len(con_dato))} horas")
    else:
        st.subheader(f"{nombre_objetivo} por hora")
    st.caption(COMO_LEER)
    st.altair_chart(grafica(horas_del_dia, por_hora(promedio_dia) if not promedio_dia.empty else promedio_dia,
                            f"{nombre_objetivo} por hora"), width="stretch")
    datos.como_se_calculo([
        "El rango esperado va del percentil 10 al 90 del pronóstico del modelo.",
        "Los datos solo registran la demanda atendida: un viaje que no se pudo hacer porque la "
        "estación estaba vacía o llena no aparece en lo observado.",
    ])

# --------------------------------------------------------------------------- #
# Tarjetas de acción
# --------------------------------------------------------------------------- #

rec_dia = rec[pd.to_datetime(rec["fecha"]).dt.normalize() == dia] if not rec.empty else rec
titular_traslados = st.empty()  # se escribe abajo, cuando ya se eligió la política
titular_traslados.subheader("Traslados simulados por franja")
if rec.empty or not ficha.get("simulada", True):
    st.info("Esta estación no se simuló: no tiene capacidad conocida o no tuvo viajes en esos días.")
elif rec_dia.empty:
    st.info("Este día no entró en la simulación para esta estación.")
else:
    presupuesto = (ind.get("referencia") or {}).get("presupuesto")
    # Las opciones son los textos en pantalla; aquí se regresan a su clave.
    por_texto = {datos.POLITICAS[p]: p for p in POLITICAS_CON_TRASLADO}
    politica = por_texto[st.radio("Política", list(por_texto), horizontal=True, key="pron_politica")]
    sin_mover = rec_dia[rec_dia["politica"] == "none"].set_index("franja")["nivel_riesgo"]
    filas = rec_dia[rec_dia["politica"] == politica].sort_values("hora_inicio")
    rondas = datos.rondas()  # en qué ronda (y a qué hora queda lista) se hace el traslado de cada franja
    if filas.empty:
        st.caption(datos.SIN_DATO)
    else:
        movidas = int(filas["bicis"].dropna().abs().sum())
        nombre_politica = datos.POLITICAS[politica]
        titular_traslados.subheader(
            f"{nombre_politica}: sin traslados en esta estación ese día" if movidas == 0 else
            f"{nombre_politica}: {datos.num(movidas)} {'bici movida' if movidas == 1 else 'bicis movidas'} "
            "en esta estación ese día")
        st.caption(f"Riesgo: {datos.QUE_ES_RIESGO}.")
        for columna, (_, fila) in zip(st.columns(len(filas)), filas.iterrows()):
            with columna:
                tarjeta(fila, sin_mover.get(fila["franja"]), rondas.get(fila["franja"]))
    datos.como_se_calculo([
        f"Presupuesto de referencia: {datos.num(presupuesto)} bicis al día en todo el sistema.",
        "El riesgo previsto es el nivel de riesgo simulado con las bicis que hay en la estación antes del "
        "traslado de esa franja, siguiendo la política elegida; no es el nivel que queda después de moverlas.",
        f"«{datos.POLITICAS['none']}» es el mismo nivel en la simulación en la que no se mueve ninguna bici.",
    ])
