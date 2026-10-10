"""Entrada del tablero de rebalanceo de Citi Bike.

    streamlit run app/Inicio.py

Este archivo solo arma el menú de la izquierda. El contenido de cada página
está en la carpeta vistas/, un archivo por página, en el orden de la lista.
Para cambiar un título del menú, cámbialo aquí.
"""

import streamlit as st

# Debe ser la primera instrucción de Streamlit. Vale para todas las páginas.
st.set_page_config(page_title="Rebalanceo Citi Bike", layout="wide")

PAGINAS = [
    st.Page("vistas/inicio.py", title="Inicio", url_path="inicio", default=True),
    st.Page("vistas/mapa_de_demanda.py", title="Mapa de demanda", url_path="mapa"),
    st.Page("vistas/estaciones_y_rutas.py", title="Estaciones y rutas", url_path="estaciones"),
    st.Page("vistas/pronostico_por_estacion.py", title="Pronóstico por estación", url_path="pronostico"),
    st.Page("vistas/confiabilidad.py", title="Confiabilidad", url_path="confiabilidad"),
    st.Page("vistas/contexto.py", title="Contexto de las estaciones", url_path="contexto"),
]

st.navigation(PAGINAS).run()
