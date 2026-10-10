"""Vista 2: Ranking de estaciones y rutas.

Esqueleto casi vacío a propósito. Lee los datos, comprueba que llegaron y deja
escrito lo que falta por construir. El ranking todavía no existe.

IMPORTANTE: hoy no hay datos de rutas en ningún contrato de datos. La tabla de
recomendaciones dice cuántas bicis recibe o entrega cada estación, pero no de
cuál estación salen ni a cuál van. Es una decisión pendiente del equipo.
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

st.set_page_config(page_title="Ranking de estaciones y rutas", layout="wide")

st.title("Ranking de estaciones y rutas")
st.write(
    "Pregunta que responde: ¿qué estaciones hay que atender primero y cuántas "
    "bicis hay que llevar o retirar en cada franja horaria? Sirve para repartir "
    "un presupuesto limitado de bicis trasladadas por día."
)

# Aviso de datos sintéticos. Va en todas las páginas; no lo quites.
datos.show_synthetic_warning()

# --- Datos de esta vista ---
# stations: nombre y ubicación de cada estación (capacity y station_type
#   pueden venir vacíos).
# recommendations: una fila por estación, franja horaria y política.
#   - window_start y window_end: inicio y fin de la franja, en hora local de
#     Nueva York, con zona horaria.
#   - policy: "none" (no intervenir), "historical" (regla histórica) o
#     "model" (modelo predictivo).
#   - bikes_to_move: positivo = llevar bicis a la estación; negativo = retirar
#     bicis de la estación; cero = no mover nada.
#   - risk_level: "alto", "medio" o "bajo".
#   - feasible: si es False, ningún inventario inicial evita a la vez que la
#     estación se vacíe y que se llene. Hay que avisarlo, no esconderlo.
# policy_curve: estaciones-hora en riesgo para cada política y presupuesto
#   diario, con su intervalo (ci_low, ci_high).
stations = datos.load_stations()
recommendations = datos.load_recommendations()
policy_curve = datos.load_policy_curve()

# --- Prueba de lectura ---
st.subheader("Datos leídos")
datos.show_data_check(
    {"stations": stations, "recommendations": recommendations, "policy_curve": policy_curve}
)

# --- Pendiente ---
# Lista en lenguaje normal de lo que hay que construir en esta vista.
st.subheader("Pendiente")
st.markdown(
    """
- Tabla de estaciones ordenada por urgencia: primero las de riesgo alto y, dentro
  de cada nivel, las que más bicis necesitan mover.
- Agregar selectores de día, franja horaria y política.
- Separar con claridad "llevar bicis" (número positivo) de "retirar bicis"
  (número negativo).
- Marcar las estaciones donde ningún inventario inicial evita a la vez que se
  vacíen y que se llenen (`feasible` en falso).
- Unir con la tabla de estaciones para mostrar el nombre y no solo el
  identificador.
- **Rutas: todavía no existen datos de rutas.** Ningún contrato de datos dice de
  qué estación salen las bicis ni a cuál van; `recommendations` solo dice cuántas
  recibe o entrega cada estación. El equipo debe decidir si se agrega un archivo
  de rutas y con qué columnas. Hasta entonces esta vista no puede mostrar rutas.
- Decidir si la curva de riesgo contra presupuesto (tabla `policy_curve`) va en
  esta vista o en Inicio. Al compararla con la política de no intervenir, hablar
  de riesgo simulado evitado.
"""
)
