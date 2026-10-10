"""Vista 2: Ranking de estaciones y rutas.

Esqueleto casi vacío a propósito. Lee los datos, comprueba que llegaron y deja
escrito lo que falta por construir. El ranking todavía no existe.

Qué quiere decir "rutas" en esta vista:

- SÍ hay datos de viajes entre pares de estaciones (tabla routes): cuántos
  viajes de usuarios hubo de una estación a otra en todo el periodo.
- NO hay rutas de rebalanceo (camiones). La tabla de recomendaciones dice
  cuántas bicis recibe o entrega cada estación, pero no de cuál estación salen
  ni a cuál van. Eso queda fuera por ahora (decisión 11 del equipo).
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

st.set_page_config(page_title="Ranking de estaciones y rutas", layout="wide")

st.title("Ranking de estaciones y rutas")
st.write(
    "Pregunta que responde: ¿qué estaciones hay que atender primero y cuántas "
    "bicis hay que llevar o retirar en cada franja horaria? Sirve para repartir "
    "un presupuesto limitado de bicis trasladadas por día. Segunda pregunta: "
    "¿entre qué pares de estaciones viajan más los usuarios?"
)

# Aviso de datos sintéticos. Va en todas las páginas; no lo quites.
# La lista dice qué tablas lee esta vista. Si son todas reales, en lugar del
# aviso sale una nota corta que lo dice.
# OJO: si la vista empieza a leer otra tabla, agrégala a esta lista.
datos.show_synthetic_warning(["stations", "recommendations", "policy_curve", "routes"])

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
# routes: viajes de usuarios entre pares de estaciones. NO son rutas de camiones.
#   - start_station_id: estación donde empezó el viaje.
#   - end_station_id: estación donde terminó. Puede ser la misma del inicio.
#   - trips: cuántos viajes hubo de la primera a la segunda en todo el periodo.
#     Es un total: no trae día ni hora, así que no se puede filtrar por fecha.
#   De A a B y de B a A son dos filas distintas. Los pares sin viajes no
#   tienen fila.
stations = datos.load_stations()
recommendations = datos.load_recommendations()
policy_curve = datos.load_policy_curve()
routes = datos.load_routes()

# --- Prueba de lectura ---
st.subheader("Datos leídos")
datos.show_data_check(
    {
        "stations": stations,
        "recommendations": recommendations,
        "policy_curve": policy_curve,
        "routes": routes,
    }
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
- **Rutas con más viajes:** tabla de los pares de estaciones con más viajes
  (tabla `routes`), con el nombre de la estación de inicio y el de la estación
  de fin. Hay que unir dos veces con la tabla de estaciones: una por
  `start_station_id` y otra por `end_station_id`.
- Mapa con una línea por cada uno de los pares con más viajes.
- Decidir qué se hace con los viajes que empiezan y terminan en la misma
  estación: caben en la tabla, pero no se pueden dibujar como línea en el mapa.
- Escribir junto a las rutas que son viajes de usuarios que sí ocurrieron (solo
  demanda atendida) y que el total es de todo el periodo: no se puede filtrar
  por día ni por hora.
- **Las rutas de esta vista no son rutas de camiones.** Las rutas de rebalanceo
  quedan fuera por ahora: `recommendations` dice cuántas bicis recibe o entrega
  cada estación, no de cuál estación salen ni a cuál van.
- Decidir si la curva de riesgo contra presupuesto (tabla `policy_curve`) va en
  esta vista o en Inicio. Al compararla con la política de no intervenir, hablar
  de riesgo simulado evitado.
"""
)
