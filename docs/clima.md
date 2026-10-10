# Clima observado y demanda

Descripción, no entrada del modelo. El clima usado aquí es **observado** (lo que
realmente pasó, según un reanálisis); la regla del equipo es que un modelo solo
puede usar clima **pronosticado**, así que nada de esto entra al modelo, al
simulador ni a la prueba final. Lo que se reporta es asociación, no causa.

Se reproduce con `python scripts/build_clima.py` (lógica en `src/data/clima.py`).
Las cifras están en `data/processed/reports/clima.json`.

## Fuente

- Endpoint: `https://archive-api.open-meteo.com/v1/archive` (API histórica de Open-Meteo).
- Coordenada pedida: 40.7549, -73.9840 (Midtown Manhattan). La respuesta viene de
  la celda de la malla más cercana, 40.738136, -74.04254 (elevación 56 m): es un
  solo punto para toda la ciudad, no una estación meteorológica por barrio.
- Variables por hora: `temperature_2m` (°C) y `precipitation` (mm), con
  `timezone=America/New_York`.
- Periodo: 2026-07-01 a 2026-09-29, 2,184 horas, ninguna sin dato.

## Lluvia contra seco

Una hora cuenta como con lluvia si su precipitación es de 0.1 mm o más. La demanda
es la suma de salidas de todas las estaciones en esa hora (hora de Nueva York).
Para no confundir lluvia con hora del día, se compara dentro de cada celda (hora
del día, laborable o fin de semana): demanda media de las horas con lluvia contra
la de las horas secas. Una celda se usa si tiene al menos 3 horas de cada lado. El
promedio pondera cada celda por su demanda en seco.

| | Diferencia con lluvia | Celdas | Horas con lluvia | Horas secas |
|---|---|---|---|---|
| Todo | -31.2 % | 47 de 48 | 514 | 1,644 |
| Laborable | -23.4 % | 24 de 24 | 323 | 1,237 |
| Fin de semana | -39.2 % | 23 de 24 | 191 | 407 |

En las 47 celdas comparables la demanda con lluvia es menor que en seco.

Cuidados: las horas de un mismo día lluvioso no son independientes (pocos días
muy lluviosos pesan mucho, sobre todo en fin de semana); el umbral de 0.1 mm en
un reanálisis marca como lluvia muchas horas de llovizna (casi 1 de cada 4); no
se controla por mes, temperatura ni días festivos.

Temperatura: la correlación simple (Pearson) entre temperatura y demanda por hora
es 0.43, pero casi toda viene de que las horas cálidas son las de la tarde; dentro
de la misma hora y tipo de día baja a 0.16.

## Fines de semana del tramo de prueba (2026-09-15 a 2026-09-29)

Revisión descriptiva con los pronósticos ya guardados; no se volvió a correr ni a
calificar nada. WAPE de salidas por día (error absoluto sumado entre salidas
sumadas, sobre todas las estaciones y horas del día).

| Día | Horas con lluvia | Precipitación (mm) | Salidas | WAPE promedio histórico | WAPE modelo |
|---|---|---|---|---|---|
| Sábado 19 | 0 | 0.0 | 199,077 | 0.47 | 0.42 |
| Domingo 20 | 18 | 10.9 | 87,346 | 1.08 | 0.94 |
| Sábado 26 | 24 | 24.0 | 63,077 | 1.88 | 1.77 |
| Domingo 27 | 24 | 51.2 | 33,303 | 3.78 | 2.90 |

Los tres días de fin de semana con error alto son exactamente los tres con lluvia;
el único seco (sábado 19) tiene un error parecido al de un día laborable seco
(0.37 a 0.42 con el modelo). El lunes 28, con 15 horas de lluvia y 13.0 mm, también
sube (0.73 y 0.61). El error crece porque las salidas reales caen mucho (33 mil el
domingo 27 contra 199 mil el sábado seco) y ningún modelo ve la lluvia. Son cuatro
días de fin de semana: coincide, pero no alcanza para medir cuánto del error se
debe a la lluvia.
