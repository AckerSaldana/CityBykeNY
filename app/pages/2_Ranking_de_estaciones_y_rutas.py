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
tables_used = ["stations", "recommendations", "policy_curve", "routes"]
datos.show_synthetic_warning(tables_used)
# Nota de simulación. Va en todas las páginas, con la misma lista; no la
# quites. Sale cuando recommendations o policy_curve ya son reales: dice que
# son resultados de una simulación en días de validación, no mediciones.
datos.show_simulation_note(tables_used)

# --- Datos de esta vista ---
# stations: nombre y ubicación de cada estación (capacity y station_type
#   pueden venir vacíos).
# recommendations: lo que hizo cada política en la SIMULACIÓN de inventario.
#   Una fila por estación, día, franja horaria y política. No son mediciones.
#   OJO: antes de mostrar o sumar hay que elegir un día y una política. Si no,
#   la misma estación sale repetida y las sumas mezclan días y políticas.
#   - window_start y window_end: inicio y fin de la franja, en hora local de
#     Nueva York, con zona horaria. El día de la fila sale de window_start:
#     recommendations["window_start"].dt.date
#   - policy: "none" (no intervenir), "historical" (regla histórica) o
#     "model" (decide con el pronóstico). OJO: mientras no exista LightGBM,
#     ese pronóstico es la línea base, no un modelo entrenado. La nota de
#     simulación dice cuál se usó.
#   - bikes_to_move: bicis que esa política movió en la simulación, con el
#     presupuesto de referencia. Positivo = llevar bicis a la estación;
#     negativo = retirar bicis de la estación; cero = no movió nada.
#     OJO: cero con riesgo alto casi siempre quiere decir que el presupuesto
#     no alcanzó, no que la estación esté bien.
#   - start_inventory_target: bicis con las que la estación queda en la
#     simulación justo después de ese traslado (las que tenía más
#     bikes_to_move). No es el inventario ideal.
#   - risk_level: "alto", "medio" o "bajo". Se calcula con el pronóstico y con
#     el inventario de ANTES del traslado.
#   - feasible: si es False, según el pronóstico ningún inventario inicial
#     evita a la vez que la estación se vacíe y que se llene. Hay que
#     avisarlo, no esconderlo. Es igual en las tres políticas.
#   OJO: hay estaciones sin recomendaciones. No se simularon: no tienen
#   capacidad registrada o no tuvieron viajes en los días simulados.
# policy_curve: curva de riesgo contra presupuesto. Sale de la misma
#   simulación. Una fila por política y presupuesto diario.
#   - budget: total de bicis que se pueden trasladar en un día en toda la red.
#   - risk_station_hours: horas de estación en riesgo simulado POR DÍA
#     (promedio de los días evaluados, contando las 24 horas).
#   - ci_low, ci_high: intervalo que sale de sortear esos días. Los días no
#     son independientes: es indicativo, no una garantía.
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
- Agregar selectores de día, franja horaria y política. Hacen falta antes que
  cualquier tabla: `recommendations` trae una fila por estación, día, franja
  horaria y política. Sin elegir día y política, la misma estación sale
  repetida y las sumas mezclan días y políticas. El día sale de
  `window_start`. La sección "Datos leídos" dice cuántos días y políticas hay.
- Tabla de estaciones ordenada por urgencia, para el día, la franja y la
  política elegidos: primero las de riesgo alto y, dentro de cada nivel, las
  que más bicis movió esa política (`bikes_to_move`).
- `bikes_to_move` es lo que la política movió en la simulación con el
  presupuesto de referencia. Una estación con riesgo alto y cero bicis movidas
  casi siempre quiere decir que el presupuesto no alcanzó, no que la estación
  esté bien: mostrarla, no esconderla.
- `start_inventory_target` son las bicis con las que la estación queda justo
  después del traslado. No presentarlo como inventario ideal.
- Las estaciones sin recomendaciones no se simularon (sin capacidad registrada
  o sin viajes en los días simulados). Si aparecen en la tabla, decir "no se
  simuló", no "sin riesgo".
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
  de riesgo simulado evitado. El eje vertical son horas de estación en riesgo
  simulado por día (promedio de los días evaluados). La banda entre `ci_low` y
  `ci_high` es indicativa: los días no son independientes.
"""
)
