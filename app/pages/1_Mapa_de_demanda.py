"""Vista 1: Mapa de demanda.

Esqueleto casi vacío a propósito. Lee los datos, comprueba que llegaron y deja
escrito lo que falta por construir. El mapa todavía no existe.
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

st.set_page_config(page_title="Mapa de demanda", layout="wide")

st.title("Mapa de demanda")
st.write(
    "Pregunta que responde: ¿en qué zonas de la ciudad y a qué horas salen y "
    "llegan más bicis? Sirve para ver de un vistazo dónde se concentra la "
    "demanda y qué estaciones tienden a vaciarse o a llenarse."
)

# Aviso de datos sintéticos. Va en todas las páginas; no lo quites.
# La lista dice qué tablas lee esta vista. Si son todas reales, en lugar del
# aviso sale una nota corta que lo dice.
# OJO: si la vista empieza a leer otra tabla, agrégala a esta lista.
datos.show_synthetic_warning(["stations", "station_hour"])

# --- Datos de esta vista ---
# stations: una fila por estación, con su ubicación (lat, lon).
#   OJO: capacity y station_type pueden venir vacíos. Hay que decidir cómo se
#   muestran esas estaciones (por ejemplo "Sin dato"), no esconderlas.
# station_hour: salidas y llegadas observadas por estación y hora.
#   OJO: ts_hour es hora local de Nueva York y trae zona horaria. La hora del
#   día se saca con station_hour["ts_hour"].dt.hour.
#   En tablas (st.dataframe) y en plotly, ts_hour se usa tal cual. En el eje de
#   una gráfica de st.line_chart, st.bar_chart, st.area_chart o altair usa
#   datos.hora_para_grafica(station_hour["ts_hour"]): con zona, el navegador la
#   convierte a la hora de la computadora y la gráfica sale corrida.
stations = datos.load_stations()
station_hour = datos.load_station_hour()

# --- Prueba de lectura ---
st.subheader("Datos leídos")
datos.show_data_check({"stations": stations, "station_hour": station_hour})

# --- Pendiente ---
# Lista en lenguaje normal de lo que hay que construir en esta vista.
st.subheader("Pendiente")
st.markdown(
    """
- Dibujar un mapa con un punto por estación (columnas `lat` y `lon`).
- Dar tamaño al punto según las salidas promedio de la hora elegida.
- Dar color al punto según el flujo neto: estaciones que tienden a vaciarse
  (salen más bicis de las que llegan) y estaciones que tienden a llenarse.
- Agregar un selector de hora del día y otro de tipo de día (entre semana o
  fin de semana).
- Al pasar el cursor por un punto, mostrar nombre, capacidad y tipo de
  estación. La capacidad y el tipo pueden faltar: mostrar "Sin dato".
- Escribir junto al mapa que solo se ve demanda atendida. Una estación vacía no
  registra salidas, así que ahí puede haber posible demanda insatisfecha que el
  mapa no muestra.
"""
)
