"""Página "Confiabilidad": cuánto confiar en el pronóstico y en la simulación.

Cuatro bloques, de arriba hacia abajo:

1. Promedio histórico contra modelo (error y cobertura, salidas y llegadas).
2. Error del pronóstico por bloque de horas y por volumen de demanda.
3. Riesgo simulado por franja y por tipo de estación, con las tres políticas.
4. Las limitaciones que escribió la simulación.

Todas las cifras salen de app/datos.py. Lo que falta se escribe "Sin dato".
"""

import altair as alt
import pandas as pd
import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

# El intervalo del pronóstico va del percentil 10 al 90.
P_BAJO, P_ALTO = 10, 90
COBERTURA_ESPERADA = (P_ALTO - P_BAJO) / 100

# Medidas del pronóstico: texto en pantalla, si es porcentaje y título del eje.
MEDIDAS = {
    "wape": ("Error relativo", True, "Error relativo"),
    "mae": ("Error medio", False, "Error medio (viajes por estación y hora)"),
    "cobertura": (f"Cobertura del intervalo {P_BAJO} a {P_ALTO}", True, "Horas dentro del intervalo"),
}
ALTURA = 300
GRIS = "#8A8F98"  # líneas de referencia: se ve en tema claro y en oscuro


def color_de_texto() -> str:
    """Color de las cifras sobre las barras, según el tema claro u oscuro."""
    try:
        oscuro = st.context.theme.type == "dark"
    except Exception:
        oscuro = False
    return "#FAFAFA" if oscuro else "#31333F"


def formato(valor, es_pct: bool, decimales: int = 1) -> str:
    return datos.pct(valor, decimales) if es_pct else datos.num(valor, decimales)


def sin_dato() -> None:
    st.caption(datos.SIN_DATO)


def barras(tabla: pd.DataFrame, *, grupos: dict, series: dict, colores: dict, columna_serie: str,
           columna_valor: str, titulo_eje: str, es_pct: bool, titulo_grupo: str, titulo_serie: str,
           decimales: int = 1, con_intervalo: bool = False, linea=None):
    """Barras agrupadas: un grupo por valor de `grupo` y una barra por serie.

    `grupos` y `series` son los diccionarios de etiquetas; su orden es el de la
    gráfica. Devuelve None si no hay nada que dibujar.
    """
    t = tabla[tabla["grupo"].isin(list(grupos)) & tabla[columna_serie].isin(list(series))]
    t = t[t[columna_valor].notna()]
    if t.empty:
        return None
    d = pd.DataFrame({
        "grupo": t["grupo"].map(grupos),
        "serie": t[columna_serie].map(series),
        "valor": t[columna_valor].astype(float),
        "texto": [formato(v, es_pct, decimales) for v in t[columna_valor]],
    })
    if con_intervalo:
        d["bajo"] = t["ic_bajo"].astype(float).to_numpy()
        d["alto"] = t["ic_alto"].astype(float).to_numpy()
        d["intervalo"] = [datos.SIN_DATO if pd.isna(a) or pd.isna(b) else f"{datos.num(a)} a {datos.num(b)}"
                          for a, b in zip(d["bajo"], d["alto"])]
    orden_grupos = [v for v in dict.fromkeys(grupos.values()) if v in set(d["grupo"])]
    orden_series = [series[k] for k in series if series[k] in set(d["serie"])]
    gama = [colores[k] for k in series if series[k] in set(d["serie"])]

    eje_y = alt.Axis(title=titulo_eje, labelExpr="format(datum.value * 100, ',.0f') + ' %'") if es_pct \
        else alt.Axis(title=titulo_eje, format=",.0f" if d["valor"].max() >= 10 else ",.1f")
    x = alt.X("grupo:N", sort=orden_grupos, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160))
    desplazamiento = alt.XOffset("serie:N", sort=orden_series)
    ayuda = [alt.Tooltip("grupo:N", title=titulo_grupo), alt.Tooltip("serie:N", title=titulo_serie),
             alt.Tooltip("texto:N", title=titulo_eje)]
    if con_intervalo:
        ayuda.append(alt.Tooltip("intervalo:N", title="Intervalo de confianza"))

    base = alt.Chart(d)
    capas = [base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=x, xOffset=desplazamiento, y=alt.Y("valor:Q", axis=eje_y, scale=alt.Scale(zero=True)),
        color=alt.Color("serie:N", sort=orden_series, scale=alt.Scale(domain=orden_series, range=gama),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=ayuda)]
    if con_intervalo and d[["bajo", "alto"]].notna().all(axis=None):
        capas.append(base.mark_rule(color=GRIS, strokeWidth=1.5).encode(
            x=x, xOffset=desplazamiento, y="bajo:Q", y2="alto:Q", tooltip=ayuda))
        capas.append(base.mark_text(dy=-6, fontSize=11, baseline="bottom", color=color_de_texto()).encode(
            x=x, xOffset=desplazamiento, y="alto:Q", text="texto:N"))
    else:
        capas.append(base.mark_text(dy=-4, fontSize=11, baseline="bottom", color=color_de_texto()).encode(
            x=x, xOffset=desplazamiento, y="valor:Q", text="texto:N"))
    if linea is not None:
        capas.append(alt.Chart(pd.DataFrame({"valor": [linea]})).mark_rule(
            strokeDash=[4, 4], color=GRIS).encode(y="valor:Q"))
    return alt.layer(*capas).properties(height=ALTURA)


def mostrar(grafica) -> None:
    if grafica is None:
        sin_dato()
    else:
        st.altair_chart(grafica, width="stretch")


datos.encabezado("Confiabilidad")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene
metricas = datos.metricas()

# --------------------------------------------------------------------------- #
# 1. Promedio histórico contra modelo
# --------------------------------------------------------------------------- #

st.subheader("El pronóstico: promedio histórico contra modelo")

general = metricas[metricas["corte"] == "general"]
validacion = ind.get("validacion") or {}
horas_evaluadas = general["n"].max() if not general.empty else None
st.markdown(
    f"Los dos pronósticos se comparan con lo observado en {datos.num(validacion.get('dias'))} días de "
    f"validación (del {datos.fecha(validacion.get('inicio'), con_dia=False)} al "
    f"{datos.fecha(validacion.get('fin'), con_dia=False)}), sobre {datos.num(horas_evaluadas)} horas de "
    "estación. Lo observado es demanda atendida: un viaje que no se pudo hacer no aparece en los datos."
)


def cifra(modelo: str, objetivo: str, medida: str) -> str:
    fila = general[(general["modelo"] == modelo) & (general["objetivo"] == objetivo)]
    valor = fila[medida].iloc[0] if not fila.empty else None
    return formato(valor, MEDIDAS[medida][1])


for columna, objetivo in zip(st.columns(2, gap="large"), datos.OBJETIVOS):
    with columna:
        lineas = [f"| {datos.OBJETIVOS[objetivo]} | " + " | ".join(datos.MODELOS.values()) + " |",
                  "|:--|" + "--:|" * len(datos.MODELOS)]
        for medida, (nombre, _, _) in MEDIDAS.items():
            sigla = {"wape": " (WAPE)", "mae": " (MAE)"}.get(medida, "")
            unidad = ", en viajes" if medida == "mae" else ""
            lineas.append(f"| {nombre}{sigla}{unidad} | "
                          + " | ".join(cifra(m, objetivo, medida) for m in datos.MODELOS) + " |")
        st.markdown("\n".join(lineas))

st.markdown(
    "- **Error relativo (WAPE):** la suma de los errores dividida entre los viajes observados. Menor es mejor.\n"
    "- **Error medio (MAE):** cuántos viajes se aleja el pronóstico de lo observado, en promedio, por "
    "estación y hora. Menor es mejor.\n"
    f"- **Cobertura del intervalo {P_BAJO} a {P_ALTO}:** porcentaje de horas en que lo observado cayó dentro "
    f"del rango pronosticado. Lo esperado es {datos.pct(COBERTURA_ESPERADA, 0)}; por debajo, el rango es "
    "demasiado estrecho."
)

# --------------------------------------------------------------------------- #
# 2. Error por bloque de horas y por volumen de demanda
# --------------------------------------------------------------------------- #

st.subheader("Dónde se equivoca más el pronóstico")

c_objetivo, c_medida = st.columns(2)
# Las opciones son los textos en pantalla; aquí se regresan a su clave.
por_texto = {v: k for k, v in datos.OBJETIVOS.items()}
por_nombre = {v[0]: k for k, v in MEDIDAS.items()}
objetivo = por_texto[c_objetivo.radio("Viajes", list(por_texto), horizontal=True)]
medida = por_nombre[c_medida.radio("Medida", list(por_nombre), horizontal=True)]
_, es_pct, titulo_eje = MEDIDAS[medida]
del_objetivo = metricas[metricas["objetivo"] == objetivo]
linea = COBERTURA_ESPERADA if medida == "cobertura" else None

c_bloque, c_volumen = st.columns(2, gap="large")
for columna, corte, grupos, titulo, titulo_grupo in (
        (c_bloque, "bloque", datos.BLOQUES, "Por bloque de horas", "Bloque de horas"),
        (c_volumen, "volumen", datos.VOLUMENES, "Por volumen de demanda de la estación", "Volumen de demanda")):
    with columna:
        st.markdown(f"**{titulo}**")
        mostrar(barras(del_objetivo[del_objetivo["corte"] == corte], grupos=grupos, series=datos.MODELOS,
                       colores=datos.COLORES_MODELO, columna_serie="modelo", columna_valor=medida,
                       titulo_eje=titulo_eje, es_pct=es_pct, titulo_grupo=titulo_grupo,
                       titulo_serie="Pronóstico", linea=linea))

if medida == "cobertura":
    st.caption(f"La línea punteada marca la cobertura esperada, {datos.pct(COBERTURA_ESPERADA, 0)}.")
elif medida == "wape":
    st.caption("El error relativo se divide entre los viajes observados: donde hay pocos viajes, un error de "
               "un solo viaje pesa mucho.")
else:
    st.caption("El error medio se mide en viajes por estación y hora: no descuenta que unas estaciones "
               "tienen más viajes que otras.")

# --------------------------------------------------------------------------- #
# 3. Riesgo simulado por franja y por tipo de estación
# --------------------------------------------------------------------------- #

st.subheader("El riesgo simulado, por franja y por tipo de estación")

referencia = ind.get("referencia") or {}
politicas = referencia.get("politicas") or {}
partes = [f"{datos.POLITICAS['none'].lower()}, {datos.num(referencia.get('riesgo_sin_intervencion'))}"]
partes += [f"{datos.POLITICAS[p].lower()}, {datos.num((politicas.get(p) or {}).get('riesgo'))}"
           for p in ("historical", "model")]
st.markdown(
    "Una hora de estación está en riesgo cuando la estación queda casi vacía o casi llena. Riesgo simulado "
    f"con un presupuesto de {datos.num(referencia.get('presupuesto'))} bicis al día, en horas de estación en "
    f"riesgo por día: {'; '.join(partes)}."
)


def riesgo_por(corte: str, grupos: dict, titulo_grupo: str):
    """La gráfica de riesgo de un corte y la frase que dice qué muestra.

    Si la simulación reportó horas de riesgo para ese corte, se dibujan las
    horas. Si no, la proporción de franjas de estación en riesgo alto.
    """
    horas = datos.riesgo_horas(corte)
    # Un grupo sin riesgo en ninguna política (como "Resto del día" cuando las franjas cubren las 24 horas) no se dibuja.
    con_riesgo = horas.groupby("grupo")["riesgo"].transform(lambda r: (r.fillna(0) > 0).any()) if not horas.empty else []
    horas = horas[con_riesgo] if not horas.empty else horas
    if not horas.empty:
        grafica = barras(horas, grupos=grupos, series=datos.POLITICAS, colores=datos.COLORES_POLITICA,
                         columna_serie="politica", columna_valor="riesgo",
                         titulo_eje="Horas de estación en riesgo por día", es_pct=False, decimales=0,
                         titulo_grupo=titulo_grupo, titulo_serie="Política", con_intervalo=True)
        if grafica is not None:
            return grafica, ("Horas de estación en riesgo por día. La línea vertical es el intervalo de "
                             "confianza.")
    niveles = datos.riesgo_niveles(corte)
    grafica = barras(niveles, grupos=grupos, series=datos.POLITICAS, colores=datos.COLORES_POLITICA,
                     columna_serie="politica", columna_valor="pct_alto",
                     titulo_eje="Franjas de estación en riesgo alto", es_pct=True, decimales=0,
                     titulo_grupo=titulo_grupo, titulo_serie="Política")
    return grafica, ("La simulación no reporta horas de riesgo con este desglose. Se muestra el porcentaje "
                     "de franjas de estación que quedaron en riesgo alto.")


c_franja, c_tipo = st.columns(2, gap="large")
for columna, corte, grupos, titulo, titulo_grupo in (
        (c_franja, "franja", datos.FRANJAS, "Por franja del día", "Franja"),
        (c_tipo, "tipo_estacion", datos.TIPOS_ESTACION, "Por tipo de estación", "Tipo de estación")):
    with columna:
        st.markdown(f"**{titulo}**")
        grafica, nota = riesgo_por(corte, grupos, titulo_grupo)
        mostrar(grafica)
        if grafica is not None:
            st.caption(nota)

# --------------------------------------------------------------------------- #
# 4. Limitaciones
# --------------------------------------------------------------------------- #

st.subheader("Limitaciones de la simulación")

limitaciones = datos.limitaciones()
if limitaciones:
    st.markdown("\n".join(f"- {datos.texto(linea)}" for linea in limitaciones))
else:
    sin_dato()
