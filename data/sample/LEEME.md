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

## Qué contiene cada archivo

| Archivo | Filas | Qué contiene |
|---|---|---|
| `stations.parquet` | 40 | Estaciones inventadas en Manhattan y Brooklyn. `capacity` y `station_type` traen vacíos a propósito. |
| `station_hour.parquet` | 26920 | Salidas y llegadas por estación y hora, 28 días. Cada estación tiene todas las horas, también las que valen cero. |
| `predictions.parquet` | 26960 | Pronóstico por estación y hora de los últimos 14 días, modelos `baseline` y `lgbm`. Las filas `lgbm` son relleno simulado, no un modelo. |
| `recommendations.parquet` | 1680 | Bicis a mover por estación, franja y política, para los 7 días de test. Números inventados. |
| `policy_curve.parquet` | 18 | Curva de riesgo contra presupuesto diario, por política. Números inventados para acomodar la gráfica. |

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
- `predictions.parquet`: `split = "val"` es del 27 de octubre al 2 de
  noviembre; `split = "test"` es del 3 al 9 de noviembre. Las filas
  `baseline` son el promedio de los primeros 14 días de estos mismos datos
  inventados. Las filas `lgbm` son relleno simulado.
- `recommendations.parquet`: una fila por día de test, franja de
  `config/operacion.yaml`, estación y política. Con `policy = "none"` no se
  mueve nada. En las otras dos, lo que reciben unas estaciones es lo que
  entregan otras. Donde falta `capacity` se supuso 20. Hay
  filas con los tres niveles de riesgo y unas pocas con `feasible` en falso.
- `policy_curve.parquet`: una fila por política y presupuesto de
  `presupuestos_diarios`.

## Cómo regenerarlos

    python scripts/make_sample_data.py

El resultado es siempre el mismo (la semilla es fija). Si cambian las franjas o
los presupuestos en `config/operacion.yaml`, hay que volver a correrlo.

No edites estos archivos a mano. Este LEEME lo escribe el mismo script.
