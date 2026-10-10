"""Vista 3: Pronóstico por estación, con acción sugerida.

Esqueleto casi vacío a propósito. Lee los datos, deja elegir una estación para
comprobar que las tablas se pueden filtrar y deja escrito lo que falta por
construir. La gráfica del pronóstico todavía no existe.
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

st.set_page_config(page_title="Pronóstico por estación", layout="wide")

st.title("Pronóstico por estación")
# OJO: el pronóstico que hay en los datos es de periodos de evaluación que ya
# pasaron (validación o prueba), no de horas futuras. Por eso se habla en pasado.
st.write(
    "Pregunta que responde: para una estación y un día del periodo evaluado "
    "(validación o prueba), ¿cuántas salidas y llegadas se esperaban hora por "
    "hora, cuántas se observaron y qué convenía hacer? La acción sugerida dice "
    "si había que llevar o retirar bicis, cuántas y antes de qué franja horaria."
)

# Aviso de datos sintéticos. Va en todas las páginas; no lo quites.
datos.show_synthetic_warning()

# --- Datos de esta vista ---
# stations: una fila por estación (capacity y station_type pueden venir vacíos).
# station_hour: lo observado por estación y hora (solo demanda atendida).
# predictions: lo esperado por estación y hora, en días que ya pasaron.
#   - ts_hour: inicio de la hora, en hora local de Nueva York, con zona horaria.
#     Para el eje de una gráfica de st.line_chart, st.bar_chart, st.area_chart
#     o altair usa datos.hora_para_grafica(predictions["ts_hour"]). Con zona,
#     el navegador la convierte a la hora de la computadora.
#   - model: "baseline" (promedio histórico) o "lgbm" (modelo).
#   - split: "val" (validación) o "test" (prueba final). No hay horas futuras.
#   - dep_q10, dep_q50, dep_q90: salidas esperadas; rango bajo, centro y rango alto.
#   - arr_q10, arr_q50, arr_q90: lo mismo para las llegadas.
# recommendations: la acción sugerida por estación, franja horaria y política.
#   - bikes_to_move: positivo = llevar bicis; negativo = retirar bicis.
#   - feasible: si es False, ningún inventario inicial evita a la vez que la
#     estación se vacíe y que se llene.
stations = datos.load_stations()
station_hour = datos.load_station_hour()
predictions = datos.load_predictions()
recommendations = datos.load_recommendations()

# --- Selector de estación ---
# En la lista se ve "nombre (identificador)". La variable station_id guarda
# solo el identificador, que es texto (por ejemplo "6289.06").
names = dict(zip(stations["station_id"], stations["name"]))
station_id = st.selectbox(
    "Estación",
    options=list(names),
    format_func=lambda option: f"{names[option]} ({option})",
)

# Filas de la estación elegida en cada tabla.
history = station_hour[station_hour["station_id"] == station_id]
forecast = predictions[predictions["station_id"] == station_id]
actions = recommendations[recommendations["station_id"] == station_id]

# --- Prueba de lectura ---
st.subheader("Datos leídos")
datos.show_data_check(
    {
        "stations": stations,
        "station_hour": station_hour,
        "predictions": predictions,
        "recommendations": recommendations,
    }
)
st.write(
    f"Filas de la estación elegida ({station_id}): "
    f"{len(history)} de historia, {len(forecast)} de pronóstico y "
    f"{len(actions)} de recomendaciones."
)
with st.expander("Ver las primeras filas del pronóstico de la estación elegida"):
    st.dataframe(forecast.head(5), hide_index=True)
with st.expander("Ver las primeras filas de las recomendaciones de la estación elegida"):
    st.dataframe(actions.head(5), hide_index=True)

# --- Pendiente ---
# Lista en lenguaje normal de lo que hay que construir en esta vista.
st.subheader("Pendiente")
st.markdown(
    """
- Gráfica por hora de la estación elegida: salidas y llegadas esperadas (centro
  del pronóstico) con una banda entre el rango bajo y el rango alto.
- Poner en la misma gráfica lo observado en esas horas, para comparar.
- Agregar selectores de modelo (línea base o modelo) y de fechas.
- Acción sugerida en una frase por franja horaria, por ejemplo "llevar N bicis
  antes de la franja de la mañana" o "retirar N bicis", con su nivel de riesgo.
  El número sale de `bikes_to_move`: positivo es llevar, negativo es retirar.
- Si `feasible` es falso, decir con claridad que ningún inventario inicial evita
  a la vez que la estación se vacíe y que se llene.
- Mostrar la capacidad y el tipo de la estación. Pueden faltar: mostrar
  "Sin dato".
- Recordar junto a la gráfica que lo observado es solo demanda atendida: si la
  estación estuvo vacía pudo haber posible demanda insatisfecha.
"""
)
