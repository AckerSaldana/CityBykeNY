# CityBykeNY

Proyecto para DataRush, un reto de ciencia de datos de 24 horas.

## Qué es

Citi Bike mueve bicis en camiones para que las estaciones no se queden vacías
ni llenas, hoy de forma reactiva. Este proyecto ayuda a decidir **cuántas bicis
mover a cada estación y en qué franja horaria**, y lo presenta en un dashboard
para un analista de operaciones que no programa.

Flujo: viajes de Citi Bike de julio a septiembre de 2026 (Parquet, DuckDB) →
tabla por estación y hora → promedio histórico contra LightGBM con cuantiles →
simulación de inventario con tres políticas (sin intervención, regla histórica
y modelo) → recomendación de bicis a mover → dashboard en Streamlit.

Tramos: entrenamiento 1 jul a 31 ago, validación 1 a 14 sep, prueba final 15 a
29 sep. Todas las decisiones se tomaron con validación. **La prueba final se
usó una sola vez, el 10 de octubre de 2026**, con todo congelado.

El contexto completo, las decisiones y los contratos de datos están en
[`CLAUDE.md`](CLAUDE.md); las reglas de operación, en
[`docs/reglas_operacion.md`](docs/reglas_operacion.md) y
`config/operacion.yaml`.

## Cómo reproducir

Entorno (Python 3.12):

```bash
git clone https://github.com/AckerSaldana/CityBykeNY.git
cd CityBykeNY
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

Si en macOS LightGBM falla al importarse por `libomp`: `brew install libomp`.

Comandos, en este orden, desde la raíz del repositorio:

```bash
python scripts/download_trips.py --desde 202607 --hasta 202609   # unos 3 GB, a data/raw/trips
python scripts/freeze_gbfs.py            # SOLO si falta config/gbfs_congelado.csv (necesita una captura de GBFS)
python scripts/build_tables.py           # limpieza: stations, station_hour, routes y trips
python scripts/run_baseline.py           # promedio histórico: predictions y métricas de validación
python scripts/run_lgbm.py               # LightGBM con cuantiles: predictions y métricas de validación
python scripts/run_simulation.py         # simulación en validación: recommendations y policy_curve
python scripts/run_piso_wape.py          # piso del WAPE y métricas por franja, en validación
python scripts/build_station_context.py  # contexto de las estaciones (LODES)
python scripts/build_app_tables.py       # tablas agregadas del dashboard, en data/processed/app
streamlit run app/Inicio.py              # dashboard

pytest                                   # pruebas; no usan la red
```

`config/gbfs_congelado.csv` va en el repositorio: es la capacidad y el
inventario inicial con que se obtuvieron los números de abajo. Volver a
congelarlo con otra captura cambia los resultados.

La prueba final ya se usó. Estos dos comandos la recalculan con la banda
fijada en el valor elegido en validación, y escriben aparte, sin tocar las
salidas de validación (el segundo se niega si la prueba ya está calificada):

```bash
python scripts/run_simulation.py --tramo prueba --usar-prueba-final --banda 0
python scripts/run_prueba_final.py --usar-prueba-final
```

Cada paso deja un reporte en `data/processed/reports/` (`limpieza.md`,
`metricas_validacion.md`, `metricas_lgbm.md`, `simulacion.md`, `piso_wape.md`,
`prueba_final.md`). Los datos pesados (`data/raw`, `data/processed`) no van en
Git. Sin `data/processed`, el dashboard usa `data/sample`, que es sintético, y
lo avisa.

## Hallazgos: validación y prueba final

Presupuesto de referencia: 1.4 bicis por estación por día (3,241 bicis al día
en validación, con 2,315 estaciones; 3,273 en la prueba, con 2,338). Entre
paréntesis, intervalo del 90 % por remuestreo de días. Validación: 13 días
evaluados (2 a 14 sep). Prueba: 14 días (16 a 29 sep).

| | Validación | Prueba final |
|---|---|---|
| Horas de estación en riesgo por día, sin intervención | 12,477.5 | 12,360.1 |
| Riesgo simulado evitado, regla histórica (referencia) | 56.9 % (52.8 a 60.9) | 64.0 % (59.5 a 68.8) |
| Riesgo simulado evitado, modelo (referencia) | 57.0 % (53.9 a 60.1) | 63.3 % (58.9 a 68.0) |
| Horas en riesgo por día, modelo menos regla histórica (referencia) | −7.5 (−194.9 a +129.5); modelo con menos riesgo en 4 de 13 días | +86.4 (+39.7 a +127.9); 2 de 14 días |
| Lo mismo con la mitad del presupuesto (1,620 y 1,637 bicis al día) | −377.5 (−489.0 a −274.9); 12 de 13 días | −428.0 (−649.0 a −212.6); 10 de 14 días |
| Riesgo evitado adicional al pasar al presupuesto siguiente (6,019 y 6,079 bicis), regla histórica | +3.4 puntos (1.3 a 6.3) | +0.9 puntos (0.3 a 1.7) |
| Lo mismo, modelo | +2.3 puntos (0.9 a 3.9) | +0.7 puntos (0.1 a 1.4) |
| Riesgo simulado evitado moviendo las mismas bicis al azar | 7.9 % | 8.6 % |
| Viajes bloqueados por día (simulados), sin intervención | 4,983.2 | 4,168.7 |
| Viajes bloqueados por día (simulados), regla histórica | 3,424.9 | 2,698.1 |
| Viajes bloqueados por día (simulados), modelo | 3,476.7 | 2,676.4 |
| WAPE de salidas: promedio histórico / LightGBM / media global | 49.1 % / 45.8 % / 115.9 % | 55.8 % / 49.5 % / 125.3 % |
| MAE de salidas: promedio histórico / LightGBM | 1.518 / 1.414 | 1.573 / 1.395 |
| Cobertura del intervalo de salidas (nominal 80 %): promedio histórico / LightGBM | 80.0 % / 84.0 % | 83.0 % / 84.9 % |

Lo que dicen esos números:

- Mover bicis con cualquiera de las dos políticas evita más de la mitad del
  riesgo simulado con el presupuesto de referencia; moverlas al azar, 8 %.
- **Con el presupuesto de referencia el modelo no supera a la regla
  histórica.** En validación la diferencia no se distingue de cero; en la
  prueba el modelo queda con 86.4 horas de estación más en riesgo por día.
- Con la mitad del presupuesto el modelo tiene menos riesgo que la regla
  histórica en los dos tramos.
- Casi duplicar el presupuesto de referencia agrega 3.4 y 2.3 puntos de
  riesgo evitado en validación y menos de 1 punto en la prueba.
- LightGBM tiene menos error de pronóstico que el promedio histórico en los
  dos tramos (WAPE 45.8 % contra 49.1 %, y 49.5 % contra 55.8 %). Un
  pronóstico con menos error no dio menos riesgo simulado con el presupuesto
  de referencia.

La tabla completa está en `data/processed/reports/prueba_final.md` y en
`validacion_y_prueba.json` (la lee el dashboard). En la prueba no se eligió
nada: la banda de incertidumbre entró fijada en 0, el valor elegido en
validación, con las mismas reglas y el mismo archivo congelado de GBFS. Las
filas de prueba de LightGBM usan como rezago los conteos observados de días de
prueba anteriores.

## Limitaciones

- Los datos solo registran **demanda atendida**. Los viajes bloqueados son un
  resultado de la simulación (posible demanda insatisfecha), no demanda
  medida. Se habla de riesgo simulado evitado.
- La distancia entre estaciones es en línea recta: ignora ríos y puentes.
- Los traslados son instantáneos y no hay camiones: no se modela su número,
  su capacidad ni su ruta. El único límite es el presupuesto de bicis.
- Los días no son independientes (el inventario pasa de un día al siguiente y
  son 13 y 14 días): los intervalos son indicativos.
- El inventario inicial es un supuesto: sale de una captura de GBFS de octubre
  de 2026, no de los días simulados, y el resultado depende de él. En
  validación, el riesgo evitado va de 39.5 % a 56.9 % (regla histórica) según
  el inventario inicial que se suponga.
- La capacidad de las estaciones es la de hoy, no la del periodo.
- La segunda versión del simulador no tiene una reimplementación
  independiente; solo sus pruebas con redes pequeñas y la conservación del
  total de bicis.
- `metricas_validacion.md` y `metricas_lgbm.md` todavía desglosan por las
  franjas anteriores; las métricas por las franjas de hoy están en
  `piso_wape.md` (validación) y `prueba_final.md` (prueba).
- La regla de valet no cambia el reparto del presupuesto con las horas de
  intervención de hoy (05:00 y 14:00): ningún intervalo de valet las contiene,
  así que solo cambia la cuenta del riesgo.
- Probado solo en macOS (Apple silicon, Python 3.12). No se verificó en
  Windows ni en Linux.

## Experimento descartado: corrección de mediodía

Se probó una cuarta política, solo en validación: en la intervención de la
tarde, el pronóstico del modelo se multiplicó por la razón entre lo observado
y lo pronosticado esa misma mañana. Con el presupuesto de referencia de
entonces (1,158 bicis al día) y banda 0.5 dio 5,027.4 horas de estación en
riesgo por día, contra 4,752.7 del modelo sin corregir y 4,757.7 de la regla
histórica (sin intervención: 10,328.5), y fue peor que el modelo sin corregir
en los 13 días. Se descartó. Son números de una versión anterior del simulador
y no se recalcularon ([`docs/experimento_mediodia.md`](docs/experimento_mediodia.md)).

## Notebooks

Tres notebooks en `notebooks/`, guardados ya ejecutados (tablas y gráficas a
la vista). No recalculan nada pesado: leen `data/processed` y sus reportes,
así que antes hay que correr los comandos de "Cómo reproducir".

- [`01_limpieza.ipynb`](notebooks/01_limpieza.ipynb): de los archivos crudos a
  la tabla por estación y hora, con los hallazgos de la limpieza.
- [`02_modelado.ipynb`](notebooks/02_modelado.ipynb): promedio histórico,
  LightGBM, los tres tramos, la regla de no fuga y las métricas de pronóstico
  en validación y en la prueba final.
- [`03_evaluacion_y_simulacion.ipynb`](notebooks/03_evaluacion_y_simulacion.ipynb):
  reglas de operación, las tres políticas, la curva de riesgo simulado contra
  presupuesto, mover al azar, el experimento descartado y las limitaciones.

Para volver a ejecutar uno:
`python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_limpieza.ipynb`.

## Estructura

```
app/            Dashboard de Streamlit
config/         operacion.yaml (reglas de operación), datos.yaml (periodo, tramos, limpieza, simulación),
                gbfs_congelado.csv (capacidad e inventario inicial)
data/raw/       Descargas pesadas                     (no va en Git)
data/processed/ Salidas reales del flujo              (no va en Git)
data/sample/    Datos sintéticos pequeños             (sí va en Git)
docs/           Reglas de operación, contexto de estaciones, experimento descartado
notebooks/      a/ programación, b/ logística
scripts/        Puntos de entrada
src/            Contratos, datos, variables, modelos, simulador
tests/          Pruebas
```
