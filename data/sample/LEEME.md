# DATOS SINTÉTICOS: NO SON RESULTADOS

Los archivos de esta carpeta son **datos inventados** por un programa. Sirven
para una sola cosa: construir y probar el dashboard mientras llegan los datos
reales.

- **No son resultados** del proyecto. No vienen de viajes reales de Citi Bike.
- **Ningún número de esta carpeta se puede citar en el pitch**, ni en láminas
  ni en conclusiones.
- Las estaciones no existen. Sus id empiezan con `SIM-` y sus nombres con
  `Sintética`.
- En `predictions.parquet`, las filas con `model = "lgbm"` son un **relleno
  simulado**. No salen de ningún modelo entrenado.
- Las recomendaciones y la curva de `policy_curve.parquet` son números
  inventados para acomodar las gráficas. No miden nada.
- `routes.parquet` son **viajes de usuarios entre estaciones**, también
  inventados. **No son rutas de camiones de rebalanceo**: de qué estación a
  cuál mover bicis no está en ningún archivo.

## Qué contiene cada archivo

| Archivo | Filas | Qué contiene |
|---|---|---|
| `stations.parquet` | 40 | Estaciones inventadas en Manhattan y Brooklyn. `capacity` y `station_type` traen vacíos a propósito. |
| `station_hour.parquet` | 26414 | Salidas y llegadas por estación y hora, 28 días. Tabla densa: una fila por estación y hora, también las que valen cero, solo entre el primer y el último viaje de cada estación. |
| `predictions.parquet` | 26468 | Pronóstico por estación y hora de los últimos 14 días, modelos `baseline` y `lgbm`, para las horas que tienen fila en `station_hour`. Las filas `lgbm` son relleno simulado, no un modelo. |
| `recommendations.parquet` | 1632 | Bicis a mover por estación, franja y política, para los 7 días de test. Números inventados. |
| `policy_curve.parquet` | 18 | Curva de riesgo contra presupuesto, por política. `budget` es el total de bicis movidas al día. Números inventados para acomodar la gráfica. |
| `routes.parquet` | 1296 | Viajes entre pares de estaciones, sumados sobre los 28 días. Son viajes de usuarios de una estación a otra, NO rutas de camiones de rebalanceo. |

Las columnas y sus tipos son los de los contratos de `src/contracts.py`. Los
archivos reales del flujo irán en `data/processed/` con los mismos nombres y
columnas.

## Detalles que el dashboard debe aguantar

- Periodo: del 13 de octubre al 9 de noviembre de 2025, hora local de Nueva
  York.
- El 2 de noviembre de 2025 termina el horario de verano: ese día tiene 25
  horas y la 01:00 aparece dos veces. Es a propósito.
- Estaciones sin `capacity`: SIM-007, SIM-012, SIM-035.
- Estaciones sin `station_type`: SIM-007, SIM-025, SIM-026.
- `station_hour.parquet` es densa: cada estación tiene una fila por cada hora
  entre su primer y su último viaje, también las horas en cero (4158
  filas sin ningún viaje). Antes del primer viaje y después del último no hay
  filas. Por eso no todas las estaciones empiezan ni terminan a la misma
  hora, y la tabla no es una malla completa de estaciones por horas.
- Estaciones que no cubren todo el periodo:
  SIM-004 (último viaje el 4 de noviembre), SIM-015 (primer viaje el 19 de octubre), SIM-029 (del 17 de octubre al 6 de noviembre).
  Fuera de esos días no tienen filas en `station_hour`, ni pronóstico, ni
  recomendaciones. Las demás empiezan el 13 de octubre y terminan el 9 de
  noviembre.
- Cada viaje inventado llega en la misma hora en que sale. Por eso, en cada
  hora, las llegadas de todas las estaciones suman lo mismo que las salidas.
- `routes.parquet`: una fila por par (estación de inicio, estación de fin) con
  al menos un viaje en el periodo. Sale de los mismos viajes inventados que
  `station_hour` y cuadra exacto por los dos lados: para cada estación, la
  suma de `trips` de las rutas que empiezan en ella es igual a sus
  `departures` de todo el periodo, y la suma de las rutas que terminan en ella
  es igual a sus `arrivals`. Hay más viajes entre estaciones cercanas y entre
  zonas residenciales y de oficinas. Hay 40 rutas de una estación a sí
  misma (viajes redondos). No todos los pares existen: tienen fila
  1296 de los 1600 pares posibles.
- `predictions.parquet`: `split = "val"` es del 27 de octubre al 2 de
  noviembre; `split = "test"` es del 3 al 9 de noviembre. Las filas
  `baseline` son el promedio de los primeros 14 días de estos mismos datos
  inventados. Las filas `lgbm` son relleno simulado.
- `recommendations.parquet`: una fila por día de test, franja de
  `config/operacion.yaml`, estación y política. Solo entran las estaciones
  que tienen fila en `station_hour` a la hora en que empieza la franja.
  - En estos datos inventados, las tres políticas empiezan cada franja con el
    mismo inventario: la `ocupacion_respaldo` de esa franja (según el tipo de
    la estación) por su capacidad, redondeado. Es una simplificación de la
    muestra. En la simulación real el inventario del primer día sale de la
    ocupación observada en la captura de GBFS (ese día es de calentamiento) y
    después cada día empieza con las bicis que dejó el anterior.
  - Con `policy = "none"` no se mueve nada: `start_inventory_target` es ese
    inventario de respaldo.
  - En las otras dos, lo que reciben unas estaciones es lo que entregan
    otras. La regla histórica (`historical`) no usa `ocupacion_respaldo`: el
    inventario que busca se inventa aparte.
  - Unas pocas filas tienen `feasible` en falso. En ellas las tres políticas
    se quedan con el inventario de respaldo y `bikes_to_move` es 0.
  - Donde falta `capacity` se supuso 20. Una estación sin `station_type` no
    tiene estrategia en `config/operacion.yaml`: aquí empieza con una
    ocupación inventada de 0.5.
  - Los traslados de la muestra no respetan ningún presupuesto ni el reparto
    por franja (`parte_del_presupuesto`): son inventados. Aquí se llevan
    entre 202 y 310 bicis al día, según el día y la política; el
    presupuesto de referencia serían 20. Los reales salen de la
    simulación con el presupuesto de referencia.
  - Hay filas con los tres niveles de riesgo.
- `policy_curve.parquet`: una fila por política y presupuesto. `budget` es el
  total de bicis que se pueden mover en un día. `config/operacion.yaml` da
  los presupuestos en bicis por estación por día
  (`presupuestos_por_estacion`: 0, 0.1, 0.25, 0.5, 1.0 y 2.0).
  Aquí cada uno se multiplica por las 40 estaciones de la muestra y se
  redondea: 0, 4, 10, 20, 40 y 80.
  Si dos valores dan el mismo total, queda una sola fila.
  El presupuesto de referencia (`presupuesto_de_referencia`: 0.5) es
  la fila con `budget` = 20.

## Cómo regenerarlos

    python scripts/make_sample_data.py

El resultado es siempre el mismo (la semilla es fija). Hay que volver a
correrlo si en `config/operacion.yaml` cambian las franjas (cuántas son, sus
nombres o sus horas), `presupuestos_por_estacion`,
`presupuesto_de_referencia` o la `ocupacion_respaldo`.
Cambiar solo `parte_del_presupuesto` no cambia la muestra.

No edites estos archivos a mano. Este LEEME lo escribe el mismo script.
