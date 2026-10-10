## Experimento descartado: modelo con corrección de mediodía

Se probó una cuarta política, solo en validación: en la intervención de la
tarde, el pronóstico de cada estación se multiplicó por la razón entre lo
observado y lo pronosticado en la mañana de ese mismo día (suavizada con 5
viajes), por separado para salidas y llegadas. Con el presupuesto de
referencia (1,158 bicis al día) y la banda 0.5, horas de estación en riesgo
por día:

| Política | Riesgo por día | Mañana | Tarde | Noche (22 a 06) | Resto |
|---|---:|---:|---:|---:|---:|
| Sin intervención | 10,328.5 | 2,511.3 | 2,524.2 | 3,494.7 | 1,798.3 |
| Regla histórica | 4,757.7 | 1,002.0 | 1,100.6 | 1,826.2 | 828.8 |
| Modelo | 4,752.7 | 1,007.2 | 1,108.5 | 1,805.1 | 831.9 |
| Modelo con corrección de mediodía | 5,027.4 | 1,046.3 | 1,217.9 | 1,870.5 | 892.7 |

La corrección dio más riesgo que el modelo sin corregir en los 13 días
evaluados (274.7 horas al día en promedio) y se descartó. Son números de la
corrida del 10 de octubre de 2026, anteriores a cambios posteriores del
simulador; no se recalculan. El código queda como registro en
`notebooks/a/exp_mediodia.py`.
