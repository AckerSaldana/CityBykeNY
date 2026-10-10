"""Página de inicio del dashboard de rebalanceo de Citi Bike.

Cómo abrir la app, desde la carpeta raíz del proyecto:

    streamlit run app/Inicio.py

Streamlit arma solo el menú de la izquierda: esta página aparece como "Inicio"
y cada archivo de la carpeta pages/ es una vista. El número con el que empieza
el nombre de cada archivo da el orden en el menú.

Este archivo es un esqueleto: está casi vacío a propósito. Lo que falta por
construir está en la lista "Pendiente", al final.
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

# Título de la pestaña del navegador y ancho completo de la pantalla.
# Debe ser la primera instrucción de Streamlit de la página.
st.set_page_config(page_title="Rebalanceo Citi Bike", layout="wide")

st.title("Rebalanceo de Citi Bike en Nueva York")
st.write(
    "Este tablero ayuda a decidir cuántas bicis mover a cada estación y en qué "
    "franja horaria, para prevenir estaciones vacías o llenas."
)

# Aviso de datos sintéticos. Va en todas las páginas y no se quita a mano:
# desaparece solo cuando los archivos reales estén en data/processed.
# Inicio lee las seis tablas. Por eso va sin lista: muestra el aviso completo
# mientras quede alguna tabla sintética.
datos.show_synthetic_warning()

# Nota de simulación. También va en todas las páginas y tampoco se quita a
# mano. Sale cuando recommendations o policy_curve ya son reales: dice que son
# resultados de una simulación, no mediciones. Sin lista, igual que el aviso.
datos.show_simulation_note()

# --- Vistas ---
# Enlaces a las tres vistas. Son las mismas que salen en el menú de la izquierda.
# OJO: si cambias el nombre de un archivo de pages/, cambia también su enlace
# aquí. Si un enlace apunta a un archivo que ya no existe, en su lugar sale un
# mensaje en rojo que dice cuál vista no se encontró.
st.subheader("Vistas")
datos.page_link("pages/1_Mapa_de_demanda.py", label="Mapa de demanda")
st.caption("¿En qué zonas y a qué horas salen y llegan más bicis?")
datos.page_link("pages/2_Ranking_de_estaciones_y_rutas.py", label="Ranking de estaciones y rutas")
st.caption(
    "¿Qué estaciones se atienden primero, cuántas bicis se llevan o se retiran "
    "y entre qué estaciones viajan más los usuarios?"
)
datos.page_link("pages/3_Pronostico_por_estacion.py", label="Pronóstico por estación")
# El pronóstico cubre periodos de evaluación que ya pasaron, no horas futuras.
# Las recomendaciones salen de la simulación: solo existen en los días simulados.
st.caption(
    "Para una estación y un día del periodo evaluado: ¿qué se esperaba hora por "
    "hora? En los días simulados: ¿cuántas bicis llevó o retiró cada política "
    "en la simulación?"
)

# Lenguaje honesto (decisión 9 del equipo). Los datos solo registran viajes que
# sí ocurrieron. Si una estación estaba vacía, no queda registro de quien no
# encontró bici. Por eso nunca se afirma cuántos viajes más habría.
st.info(
    "Los datos solo registran **demanda atendida**: viajes que sí ocurrieron. "
    "Por eso este tablero habla de *posible demanda insatisfecha* y de "
    "*riesgo simulado evitado*."
)

# --- Prueba de lectura ---
# Se leen las seis tablas y se muestra cuántas filas trae cada una. Si falta
# un archivo, aquí aparece un mensaje que dice cómo generarlo.
# OJO: routes son viajes de usuarios entre estaciones, no rutas de camiones.
st.subheader("Datos leídos")
tables = {
    "stations": datos.load_stations(),
    "station_hour": datos.load_station_hour(),
    "predictions": datos.load_predictions(),
    "recommendations": datos.load_recommendations(),
    "policy_curve": datos.load_policy_curve(),
    "routes": datos.load_routes(),
}
datos.show_data_check(tables)

# --- Pendiente ---
st.subheader("Pendiente")
st.markdown(
    """
- Explicar en dos o tres frases cómo se lee el tablero y en qué orden conviene
  ver las vistas.
- Poner aquí los indicadores principales. Los que salgan de `recommendations` o
  de `policy_curve` son resultados de una simulación: se presentan como riesgo
  simulado y riesgo simulado evitado, con su intervalo, nunca como mediciones.
  Con datos sintéticos no se presenta ninguna cifra como resultado.
- Decidir dónde va la curva de riesgo contra presupuesto (tabla `policy_curve`):
  aquí o en la vista de ranking. Su valor son horas de estación en riesgo
  simulado por día (promedio de los días evaluados).
- Quitar la sección "Datos leídos" cuando ya no haga falta.
"""
)
