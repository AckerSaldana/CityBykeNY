"""Vista 3: Pronóstico por estación, con acción sugerida.

Esqueleto casi vacío a propósito. Lee los datos, deja elegir una estación para
comprobar que las tablas se pueden filtrar y deja escrito lo que falta por
construir. La gráfica del pronóstico todavía no existe.

La acción sugerida sale de la tabla de recomendaciones, que es el resultado de
una SIMULACIÓN: dice cuántas bicis movió cada política, no una medición.
"""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

st.set_page_config(page_title="Pronóstico por estación", layout="wide")

st.title("Pronóstico por estación")
# OJO: el pronóstico que hay en los datos es de periodos de evaluación que ya
# pasaron (validación o prueba final), no de horas futuras. Por eso se habla
# en pasado.
# OJO: la acción sugerida sale de la simulación. Solo existe en los días
# simulados, que son de validación: la prueba final no se simula.
st.write(
    "Pregunta que responde: para una estación y un día del periodo evaluado "
    "(validación o prueba final), ¿cuántas salidas y llegadas se esperaban hora "
    "por hora? En los días de validación se muestra también cuántas se "
    "observaron. La acción sugerida sale de la simulación y solo existe en los "
    "días simulados: dice si cada política llevó o retiró bicis, cuántas y antes "
    "de qué franja horaria."
)

# Aviso de datos sintéticos. Va en todas las páginas; no lo quites.
# La lista dice qué tablas lee esta vista. Si son todas reales, en lugar del
# aviso sale una nota corta que lo dice.
# OJO: si la vista empieza a leer otra tabla, agrégala a esta lista.
tables_used = ["stations", "station_hour", "predictions", "recommendations"]
datos.show_synthetic_warning(tables_used)
# Nota de simulación. Va en todas las páginas, con la misma lista; no la
# quites. Sale cuando recommendations ya es real: dice que son resultados de
# una simulación en días de validación, no mediciones.
datos.show_simulation_note(tables_used)

# Decisión 4 del equipo: la prueba final se usa una sola vez. La regla queda
# escrita en pantalla para que nadie compare antes de tiempo. No la quites.
st.info(
    "**Regla del equipo (decisión 4): lo observado se compara con el "
    "pronóstico solo en los días de validación** (`split` = `val`). En los "
    "días de prueba final (`split` = `test`) se muestra el pronóstico solo: "
    "sin lo observado al lado y sin calcular su error. La prueba final se usa "
    "una sola vez, en la evaluación final del proyecto. La simulación de "
    "inventario tampoco se corre sobre la prueba final."
)

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
#     OJO: lo observado se compara solo con "val" (ver forecast_val, abajo).
#   - dep_q10, dep_q50, dep_q90: salidas esperadas; rango bajo, centro y rango alto.
#     OJO: con model "baseline" el centro (q50) es el promedio histórico, no
#     la mediana.
#   - arr_q10, arr_q50, arr_q90: lo mismo para las llegadas.
# recommendations: lo que hizo cada política en la SIMULACIÓN de inventario.
#   Una fila por estación, día, franja horaria y política. No son mediciones.
#   OJO: antes de mostrar una acción hay que elegir un día y una política.
#   - window_start: inicio de la franja. El día de la fila sale de aquí:
#     recommendations["window_start"].dt.date
#   - bikes_to_move: bicis que esa política movió en la simulación, con el
#     presupuesto de referencia. Positivo = llevar bicis; negativo = retirar
#     bicis. OJO: cero con riesgo alto casi siempre quiere decir que el
#     presupuesto no alcanzó, no que la estación esté bien.
#   - start_inventory_target: bicis con las que la estación queda justo
#     después de ese traslado (las que tenía más bikes_to_move). No es el
#     inventario ideal.
#   - risk_level: se calcula con el pronóstico y con el inventario de ANTES
#     del traslado.
#   - feasible: si es False, según el pronóstico ningún inventario inicial
#     evita a la vez que la estación se vacíe y que se llene.
#   OJO: hay estaciones sin recomendaciones. No se simularon: no tienen
#   capacidad registrada o no tuvieron viajes en los días simulados.
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

# OJO, DECISIÓN 4 DEL EQUIPO: la prueba final se usa una sola vez.
# - Para cualquier gráfica, tabla o número que ponga lo observado (history)
#   junto al pronóstico, usa SOLO forecast_val: son las filas de validación
#   (split == "val").
# - Las filas de prueba final (split == "test") se pueden mostrar como
#   pronóstico, pero nunca junto a lo observado ni con su error, hasta la
#   evaluación final del proyecto.
forecast_val = forecast[forecast["split"] == "val"]

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
    f"{len(history)} de historia, {len(forecast)} de pronóstico "
    f"({len(forecast_val)} de validación) y {len(actions)} de recomendaciones."
)

# Recomendaciones de la estación elegida: hay una fila por día, franja horaria
# y política. Aquí solo se cuenta, para comprobar que se pueden filtrar.
# El día de cada fila es el día en que empieza su franja (window_start).
action_days = actions["window_start"].dt.normalize()
synthetic = datos.synthetic_tables()
if len(actions) > 0:
    st.write(
        f"Las recomendaciones de esta estación cubren {action_days.nunique()} "
        f"días y {actions['policy'].nunique()} políticas. Para mostrar una acción "
        "hay que elegir un día y una política."
    )
elif "stations" not in synthetic and "recommendations" not in synthetic:
    # Con tablas reales, una estación sin recomendaciones es una estación que
    # la simulación dejó fuera. Se dice, para que nadie lea "0" como "sin riesgo".
    not_simulated = stations[~stations["station_id"].isin(recommendations["station_id"])]
    st.info(
        "**Esta estación no se simuló: por eso no tiene recomendaciones.** La "
        "simulación deja fuera a las estaciones sin capacidad registrada y a las "
        "que no tuvieron viajes en los días simulados. No se simularon "
        f"{len(not_simulated)} de las {len(stations)} estaciones "
        f"({int(not_simulated['capacity'].isna().sum())} sin capacidad registrada)."
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
  del pronóstico) con una banda entre el rango bajo y el rango alto. Con el
  modelo `baseline`, el centro es el promedio histórico, no la mediana.
- Poner en la misma gráfica lo observado, para comparar, **solo en los días de
  validación** (`split` = `val`). En los días de prueba final (`split` =
  `test`) va el pronóstico solo, sin lo observado, hasta la evaluación final
  (decisión 4 del equipo).
- Agregar selectores de modelo (línea base o modelo) y de fechas.
- Agregar un selector de día y otro de política para la acción sugerida.
  `recommendations` trae una fila por día, franja horaria y política: sin
  elegirlos salen varias filas para la misma franja. El día sale de
  `window_start`. Solo hay recomendaciones en los días simulados.
- Acción sugerida en una frase por franja horaria, para el día y la política
  elegidos, por ejemplo "llevar N bicis antes de la franja de la mañana" o
  "retirar N bicis", con su nivel de riesgo. El número sale de
  `bikes_to_move`: positivo es llevar, negativo es retirar. Es lo que esa
  política movió en la simulación con el presupuesto de referencia.
- Si `bikes_to_move` es cero y el riesgo es alto, decirlo: casi siempre quiere
  decir que el presupuesto no alcanzó, no que la estación esté bien.
- `start_inventory_target` son las bicis con las que la estación queda justo
  después del traslado. No presentarlo como inventario ideal.
- Escribir junto a la acción que es el resultado de una simulación: riesgo
  simulado, no una medición.
- Si `feasible` es falso, decir con claridad que ningún inventario inicial evita
  a la vez que la estación se vacíe y que se llene.
- Mostrar la capacidad y el tipo de la estación. Pueden faltar: mostrar
  "Sin dato".
- Recordar junto a la gráfica que lo observado es solo demanda atendida: si la
  estación estuvo vacía pudo haber posible demanda insatisfecha.
"""
)
