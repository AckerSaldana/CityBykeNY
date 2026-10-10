# CityBykeNY: cuántas bicis mover y en qué ronda

Proyecto para DataRush, un reto de ciencia de datos de 24 horas.

Las estaciones de Citi Bike en Nueva York se vacían o se llenan a lo largo del
día, y el operador mueve bicis en camiones de forma reactiva. Este proyecto es
para un **analista de operaciones** que no programa y apoya una decisión
concreta: **cuántas bicis mover a cada estación y en qué ronda** (la nocturna,
lista a las 05:00, o la de mediodía, lista a las 14:00). Une los viajes
históricos, pronostica salidas y llegadas por estación y hora, simula el
inventario con distintas políticas de traslado y lo presenta en un dashboard.

## Resultados principales

Todo es **riesgo simulado**: horas en que una estación pasa casi vacía o casi
llena al reproducir los viajes reales sobre un inventario supuesto. Validación
son 13 días (2 a 14 de septiembre de 2026) y la prueba final, usada una sola
vez, 14 días (16 a 29 de septiembre). El presupuesto de referencia es 1.4
bicis por estación al día (3,241 en validación y 3,273 en la prueba).

| Resultado | Validación | Prueba final |
|---|---|---|
| Riesgo simulado evitado, regla histórica, presupuesto de referencia | 56.9 % (52.8 a 60.9) | 64.0 % (59.5 a 68.8) |
| Riesgo simulado evitado, modelo, presupuesto de referencia | 57.0 % (53.9 a 60.1) | 63.3 % (58.9 a 68.0) |
| Modelo menos regla histórica, referencia (horas de estación al día) | −7.5 (−194.9 a +129.5) | +86.4 (+39.7 a +127.9) |
| Modelo menos regla histórica, mitad del presupuesto | −377.5 (−489.0 a −274.9) | −428.0 (−649.0 a −212.6) |
| Puntos extra al casi duplicar el presupuesto, regla histórica | +3.4 | +0.9 |
| Riesgo simulado evitado moviendo las mismas bicis al azar | 7.9 % | 8.6 % |
| Viajes bloqueados por día (simulados), sin intervención | 4,983.2 | 4,168.7 |
| Viajes bloqueados por día (simulados), regla histórica | 3,424.9 | 2,698.1 |
| Error del pronóstico (WAPE de salidas), promedio histórico | 49.1 % | 55.8 % |
| Error del pronóstico (WAPE de salidas), LightGBM | 45.8 % | 49.5 % |

Intervalos al 90 % por remuestreo de días. En las diferencias, negativo
significa que el modelo tiene menos riesgo.

Cómo leerlo:

- **Planear los traslados con presupuesto evita más de la mitad del riesgo
  simulado** en los dos periodos, unas siete veces lo que evita mover al azar.
- **Con el presupuesto de referencia, la regla histórica y el modelo
  empatan.** En validación la diferencia no se distingue de cero; en la prueba
  final el modelo queda algo peor (86.4 horas de estación al día).
- **Con la mitad del presupuesto el modelo ayuda** en los dos periodos.
- **Casi duplicar el presupuesto agrega menos de cuatro puntos.**
- LightGBM pronostica mejor que el promedio histórico, pero eso no se traduce
  en menos riesgo con el presupuesto de referencia.

Las cifras salen de `data/processed/reports/` (`validacion_y_prueba.json`,
`prueba_final.md`, `simulacion.md`) y están también en los notebooks.

## Cómo reproducir

Probado en macOS (Apple silicon) con Python 3.12. No se verificó en Windows ni
en Linux.

```bash
git clone https://github.com/AckerSaldana/CityBykeNY.git
cd CityBykeNY
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
pytest                         # no usa la red ni los datos reales
python scripts/build_app_tables.py --datos data/sample --salida data/processed/app   # tablas del dashboard con la muestra
streamlit run app/Inicio.py    # abre con la muestra de data/sample
```

**La muestra de `data/sample` es sintética**: sirve para correr las pruebas y
probar el flujo, y ningún número suyo es un resultado. **La página de Contexto
solo aparece con los datos del censo** (paso 8 de abajo).

Para reproducir con datos reales, en este orden:

```bash
python scripts/download_trips.py --desde 202607 --hasta 202609   # 1. viajes, unos 3 GB
python scripts/build_tables.py        # 2. limpieza: estaciones, estación-hora, rutas
python scripts/run_baseline.py        # 3. línea base y métricas de validación
python scripts/run_lgbm.py            # 4. LightGBM con cuantiles
python scripts/run_simulation.py      # 5. simulación en validación
python scripts/run_piso_wape.py       # 6. referencia del error del pronóstico
python scripts/build_clima.py         # 7. clima observado contra demanda
python scripts/build_station_context.py   # 8. empleos y residentes del censo
python scripts/build_app_tables.py    # 9. tablas pequeñas para el dashboard
streamlit run app/Inicio.py
```

La capacidad y el inventario inicial salen de `config/gbfs_congelado.csv`, que
ya viene en el repositorio (`python scripts/freeze_gbfs.py` solo si falta). La
prueba final ya se usó una vez; los scripts se niegan a repetirla salvo que se
pida de forma explícita.

Notebooks, en orden (ya ejecutados, con sus salidas guardadas):

1. `notebooks/01_limpieza.ipynb`
2. `notebooks/02_modelado.ipynb`
3. `notebooks/03_evaluacion_y_simulacion.ipynb`

Son delgados: leen `data/processed` y los reportes, y no recalculan lo pesado.

## Estructura

| Carpeta | Qué hay |
|---|---|
| `app/` | Dashboard de Streamlit: siete páginas en `app/vistas/` y el acceso a datos en `app/datos.py` |
| `config/` | Reglas de operación (`operacion.yaml`), periodo y tramos (`datos.yaml`) y capacidad congelada de GBFS |
| `data/raw/` | Descargas pesadas: viajes, GBFS, censo y clima. No va en Git |
| `data/processed/` | Tablas reales y reportes que genera el flujo. No va en Git |
| `data/sample/` | Muestra sintética pequeña con los seis contratos de datos. Sí va en Git |
| `docs/` | Reglas de operación en lenguaje llano, contexto de estaciones, clima y el experimento descartado |
| `notebooks/` | Los tres notebooks de entrega; `a/` y `b/` tienen el experimento y la regla de tipo de estación |
| `scripts/` | Puntos de entrada de cada paso del flujo |
| `src/` | Código: contratos de datos, limpieza, variables, modelos y simulador |
| `tests/` | Pruebas (contratos, fuga de información, simulador, dashboard) |

`CLAUDE.md` tiene el contexto completo: decisiones del equipo, contratos de
datos y hechos verificados.

## Checklist técnico

| Punto | Dónde se cumple |
|---|---|
| Integración de archivos | `01_limpieza`, sección 2 "Archivos crudos" (tres meses, varios CSV por mes unidos en una tabla) |
| Manejo de esquemas y faltantes | `01_limpieza`, secciones 3 "Qué se quitó y por qué", 4 "Estaciones escritas de dos formas", 5 "Estaciones excluidas" y 6 "Zona horaria" |
| Baseline | `02_modelado`, sección 2 "Línea base" |
| Comparación de dos modelos | `02_modelado`, secciones 3 "LightGBM global con cuantiles", 5 "Métricas en validación" y 6 "Piso del WAPE" |
| Validación temporal sin fuga | `02_modelado`, secciones 1 "Tres tramos ordenados en el tiempo", 4 "Sin fuga de información" y 7 "Prueba final, usada una sola vez" |
| Métricas | `02_modelado`, sección 5 (MAE, RMSE, WAPE y cobertura); `03_evaluacion_y_simulacion`, secciones 3 y 4 (curva de riesgo contra presupuesto) |

## Fuentes de datos

- [Citi Bike Trip Histories](https://citibikenyc.com/system-data): viajes de julio a septiembre de 2026 ([índice de archivos](https://s3.amazonaws.com/tripdata/index.html)).
- [GBFS de Citi Bike](https://gbfs.citibikenyc.com/gbfs/gbfs.json): capacidad e inventario de estaciones, capturado el 10 de octubre de 2026.
- [LODES (LEHD, Census Bureau)](https://lehd.ces.census.gov/data/): empleos y trabajadores residentes por manzana, 2023.
- [Open-Meteo](https://open-meteo.com/): temperatura y precipitación por hora, clima observado.

## Limitaciones

- **Demanda satisfecha.** Los datos solo registran viajes que sí ocurrieron.
  Los viajes bloqueados son un resultado de la simulación, no demanda
  insatisfecha medida.
- **Inventario inicial supuesto.** Sale de una captura de GBFS de octubre, no
  de los días simulados, y el resultado depende de él: el riesgo simulado
  evitado va de 40 % a 57 % según el inventario inicial que se suponga.
- **Sin límite de camiones: los resultados son un techo.** La simulación no
  limita camiones ni visitas y los traslados son instantáneos, así que reparte
  muchos traslados pequeños: en validación, una mediana de 853 estaciones por
  ronda y 2 bicis por visita. Un operador real priorizaría las de mayor
  movimiento y evitaría menos riesgo.
- La distancia entre estaciones es en línea recta: ignora ríos y puentes.
- Los días no son independientes, así que los intervalos son indicativos.
- La capacidad es la del GBFS de hoy, no la del periodo.
- La segunda versión del simulador no tiene una reimplementación
  independiente; sus invariantes (bicis conservadas, inventario dentro de la
  capacidad, presupuesto, distancia máxima y grupos aislados) se comprobaron
  aparte con cero violaciones.
- Los reportes de métricas del pronóstico desglosan con franjas anteriores;
  las actuales están en `data/processed/reports/piso_wape.md`.
- Un experimento se descartó: corregir el pronóstico de la tarde con lo
  observado en la mañana dio más riesgo en los 13 días de validación
  (`docs/experimento_mediodia.md`).

## Siguientes pasos

- Usar el **pronóstico del clima**: con lluvia se usan 31 % menos bicis y los
  días de mayor error del pronóstico fueron los lluviosos.
- **Inventario en tiempo real** desde GBFS, en lugar de un inventario inicial
  supuesto.
- **Rutas de camiones con capacidad**, para pasar del techo a un plan
  ejecutable.

## Equipo

- Acker Saldaña: programación (datos, modelos y simulador).
- Rodrigo Gutiérrez: logística (reglas de operación, tipos de estación y dashboard).
