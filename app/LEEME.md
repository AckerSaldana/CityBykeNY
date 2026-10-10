# Dashboard (carpeta `app/`)

Esto es un **esqueleto**: la app abre, lee los datos y tiene tres vistas casi
vacías. Todavía no hay mapa, ranking ni gráfica de pronóstico. Cada vista
termina con una lista "Pendiente" que dice qué falta construir.

El usuario final es un analista de operaciones que no programa.

## Cómo abrir la app

En una terminal, desde la carpeta raíz del proyecto y con el entorno activado:

    source .venv/bin/activate        # En Windows: .venv\Scripts\activate
    streamlit run app/Inicio.py

Se abre el navegador con la app. La terminal muestra la dirección, que
normalmente es `http://localhost:8501`. Para cerrar la app, presiona
`Ctrl + C` en la terminal.

Con las tablas reales la app usa entre 2 y 3 GB de memoria: guarda las tablas
completas para no releerlas en cada clic. Si la computadora de la demo tiene
poca memoria, cierra otros programas antes de abrirla.

Después de cambiar un archivo, guárdalo. Streamlit avisa en el navegador que el
archivo cambió y ofrece volver a correr la app ("Rerun"). Si no ves el aviso,
recarga la página.

Para comprobar que la app sigue abriendo después de un cambio:

    pytest tests/test_app_smoke.py

## Mapa de archivos

| Archivo | Para qué sirve |
|---|---|
| `app/Inicio.py` | Página de inicio. Es la que se abre con `streamlit run`. |
| `app/pages/1_Mapa_de_demanda.py` | Vista 1: dónde y a qué hora salen y llegan más bicis. |
| `app/pages/2_Ranking_de_estaciones_y_rutas.py` | Vista 2: qué estaciones se atienden primero, cuántas bicis se mueven y entre qué pares de estaciones viajan más los usuarios. |
| `app/pages/3_Pronostico_por_estacion.py` | Vista 3: qué se esperaba en una estación en un día ya evaluado y, en los días simulados, cuántas bicis movió cada política en la simulación. |
| `app/datos.py` | Lee las tablas. Es el único archivo que sabe dónde están los datos. |
| `app/LEEME.md` | Este documento. |
| `app/.streamlit/config.toml` | Configuración mínima de Streamlit. Va junto a `Inicio.py` para que se lea desde cualquier carpeta. |
| `tests/test_app_smoke.py` | Prueba que la app abre, que avisa de los datos sintéticos y que dice qué tablas son resultados de una simulación. |

El menú de la izquierda lo arma Streamlit con los archivos de `app/pages/`. El
número al inicio del nombre da el orden. El resto del nombre es el texto del
menú, con espacios en lugar de guiones bajos. Por eso la vista 3 sale en el
menú sin acentos. Para cambiar el texto del menú se cambia el nombre del
archivo; el nombre puede llevar acentos.

Si agregas una vista (un archivo nuevo en `app/pages/`), la prueba la revisa
sola: comprueba que abre, que muestra el aviso de datos sintéticos siempre
que lea una tabla sintética y que muestra la nota de simulación siempre que
lea una tabla simulada real. Solo falta agregar su enlace en `app/Inicio.py`.

Si cambias el nombre de un archivo de `app/pages/`, actualiza dos lugares: su
enlace en `app/Inicio.py` y su nombre en la lista `PAGES` de
`tests/test_app_smoke.py`. Si se te olvida el enlace, Inicio muestra un mensaje
en rojo que dice cuál vista no encuentra.

## De dónde salen los datos

Son seis tablas. Para cada una, `app/datos.py` busca así:

1. Si el archivo está en `data/processed`, usa ese. Son los datos reales.
2. Si no, usa el de `data/sample`. Son **datos sintéticos**, inventados por un
   programa para poder construir la app.
3. Si no está en ninguna de las dos carpetas, la página muestra un mensaje que
   dice cómo generar la muestra: `python scripts/make_sample_data.py`.

Mientras alguna tabla venga de `data/sample`, cada página lo dice, en el
cuerpo y en el menú de la izquierda:

- **La página que lee al menos una tabla sintética** muestra el aviso
  **"DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"** con el nombre de
  las tablas sintéticas.
- **La página cuyas tablas son todas reales** muestra una nota corta: "Esta
  vista usa solo tablas reales", con el nombre de las tablas que siguen
  sintéticas en otras vistas.

Ni el aviso ni esa nota corta se quitan a mano. Desaparecen solos cuando los
archivos reales llegan a `data/processed`. No hay que cambiar nada en `app/`
para que eso pase.

Cada vista dice qué tablas lee, en una lista que le pasa al aviso y a la nota
de simulación (ver más abajo). Por ejemplo, en la vista 1:

    tables_used = ["stations", "station_hour"]
    datos.show_synthetic_warning(tables_used)
    datos.show_simulation_note(tables_used)

**Si una vista empieza a leer otra tabla, hay que agregarla a esa lista.** Si
falta, la prueba (`pytest tests/test_app_smoke.py`) falla y dice cuál.

**OJO con la mezcla.** Los archivos reales llegan de uno en uno. Mientras unas
tablas sean reales y otras sigan en la muestra, sus estaciones no coinciden:
las de la muestra se llaman `SIM-...` y no existen en los datos reales. Al
cruzar una tabla real con una sintética no coincide ninguna estación y el
resultado sale vacío (por ejemplo, la vista 3 dice "0 de recomendaciones"). No
es un error de la vista: se arregla solo cuando llegue el resto de los archivos
reales. El aviso lo dice cuando pasa.

Ningún número que salga de `data/sample` se puede citar en el pitch.

### Tablas que salen de la simulación

`recommendations` y `policy_curve` no salen de viajes observados. Salen de la
simulación de inventario (`python scripts/run_simulation.py`). Cuando sus
archivos reales están en `data/processed` ya no llevan el aviso de datos
sintéticos, pero **siguen sin ser mediciones**.

Por eso la página que lee alguna de las dos, ya real, muestra en el cuerpo y
en el menú de la izquierda la nota **"RESULTADOS DE UNA SIMULACIÓN: no son
mediciones"**. La nota dice:

- Qué tablas simuladas lee la vista.
- Que salen de simular días de validación y que la prueba final no se simula
  (decisión 4), y qué días cubre `recommendations`. Los días se cuentan de la
  tabla.
- Si las reglas de operación de esa corrida eran provisionales y qué
  pronóstico alimentó a la política `model`. Esos dos datos no vienen en las
  tablas: `app/datos.py` los lee del reporte que la simulación escribe junto
  a ellas, `data/processed/reports/simulacion.json`. Si el reporte no se
  puede leer, la nota dice que no se sabe.
- Que se habla de riesgo simulado y de riesgo simulado evitado, y que los
  viajes bloqueados de la simulación son simulados: no son demanda
  insatisfecha medida.

Si una tabla simulada viene de `data/sample`, manda el aviso de datos
sintéticos y la nota no la nombra.

La nota se pide con `datos.show_simulation_note(tables_used)`, justo después
del aviso y con la misma lista. Tampoco se quita a mano. Va en todas las
páginas: en una vista que no lee tablas simuladas no muestra nada.

La sección "Datos leídos" de cada vista dice además qué trae cada tabla
simulada: cuántas estaciones, días, franjas, políticas y presupuestos. Todo
se cuenta de la tabla. Si alguien vuelve a simular con otras reglas, esos
textos cambian solos.

### Datos reales de hoy (10 de octubre de 2026)

Las cifras de esta sección son las de ese día. Cambian cuando se vuelven a
generar las tablas. La app no trae ninguna escrita: lo que muestra en pantalla
lo cuenta de las tablas.

- **Reales:** las seis tablas. `stations`, `station_hour` y `routes` salen de
  la limpieza. `predictions` sale de la línea base. `recommendations` y
  `policy_curve` salen de la simulación de inventario
  (`python scripts/run_simulation.py`).
- Con los seis archivos en `data/processed`, ninguna vista muestra el aviso
  de datos sintéticos. Inicio y las vistas 2 y 3 muestran la nota de
  simulación. `data/processed` no va en Git: en una copia nueva del
  repositorio la app usa `data/sample` hasta que se corra el flujo.
- **Periodo:** del 1 de julio al 29 de septiembre de 2026, en hora local de
  Nueva York.
- **Pronóstico** (`predictions`): solo hay filas con `model = "baseline"`. Los
  días de validación (`split = "val"`) van del 1 al 14 de septiembre. Los de
  prueba final (`split = "test"`), del 15 al 29 de septiembre.
- **Simulación:** se corrió sobre los días de validación, del 1 al 14 de
  septiembre. El día 1 es de calentamiento y no cuenta. Hay resultados de
  **13 días: del 2 al 14 de septiembre de 2026**. La prueba final no se
  simuló (decisión 4).
- **Las reglas de operación de esa corrida son provisionales**
  (`estado: provisional` en `config/operacion.yaml`). Los números de la
  simulación cambian cuando logística las confirme.
- **La política `model` usa hoy el pronóstico de la línea base** (promedio
  histórico), porque LightGBM todavía no existe. En pantalla no se le llama
  "modelo predictivo" sin esa aclaración.
- **`recommendations`:** 180,570 filas = 2,315 estaciones simuladas × 13 días
  × 2 franjas × 3 políticas. Son 78 filas por estación. Salen de la corrida
  con el presupuesto de referencia: 1,158 bicis movidas al día en toda la red.
- `feasible` es falso en 12,599 de las 60,190 combinaciones de estación, día
  y franja (20.9 %).
- **Estaciones sin recomendaciones:** 83 de las 2,398 de `stations`. No se
  simularon: 42 no tienen capacidad registrada y 41 no tuvieron viajes en los
  días simulados. Un ejemplo es `2803.04`. La vista 3 lo dice al elegir una.
- **`policy_curve`:** 18 filas = 3 políticas × 6 presupuestos diarios: 0, 232,
  579, 1,158, 2,315 y 4,630 bicis al día en toda la red. En esa corrida una
  estación está en riesgo con menos de 2 bicis o menos de 2 lugares libres. El
  intervalo (`ci_low`, `ci_high`) es del 90 %.
- **Lo que dice la curva con 1,158 bicis al día:** 5,977.8 horas de estación
  en riesgo simulado por día sin intervención, 4,312.8 con la regla histórica
  y 5,875.1 con `model`. El riesgo simulado evitado por día es 1,665.0 con la
  regla histórica y 102.7 con `model`.
- **En la muestra sintética el orden es el contrario:** con su presupuesto de
  referencia, `model` da 260.7 y `historical` 338.3. Son números inventados.
  La muestra no sirve para saber qué política da menos riesgo.
- **Tamaño:** cerca de 2,400 estaciones, 5 millones de filas en `station_hour`,
  1.6 millones en `predictions` y 1.2 millones en `routes`. La cifra exacta
  sale en la sección "Datos leídos" de cada vista.
- **Regla del equipo (decisión 4): la prueba final se usa una sola vez.** Lo
  observado se compara con el pronóstico solo en los días de validación
  (`split == "val"`). Las filas de `split == "test"` se muestran como
  pronóstico, sin lo observado al lado y sin calcular su error, hasta la
  evaluación final del proyecto.

## Tablas y columnas disponibles

Cada tabla se pide con una función de `app/datos.py`.

### `stations` — `datos.load_stations()`

Una fila por estación.

| Columna | Qué es |
|---|---|
| `station_id` | Identificador de la estación. Es texto, no número. |
| `name` | Nombre de la estación. |
| `lat`, `lon` | Ubicación: latitud y longitud. |
| `capacity` | Cuántas bicis caben. **Puede venir vacío.** |
| `station_type` | Tipo: `oficinas`, `residencial`, `recreativa` o `mixta`. **Puede venir vacío**, incluso en todas las estaciones, mientras no exista la regla que lo asigna. |

### `station_hour` — `datos.load_station_hour()`

Una fila por estación y hora. Es lo que se observó.

**Cada estación tiene filas solo entre su primer y su último viaje.** Dentro de
ese tramo están todas las horas; las que no tuvieron viajes valen cero. Fuera de
ese tramo no hay filas.

| Columna | Qué es |
|---|---|
| `station_id` | Identificador de la estación. |
| `ts_hour` | Inicio de la hora, en hora local de Nueva York, con zona horaria. |
| `departures` | Salidas: viajes que empezaron en la estación en esa hora. |
| `arrivals` | Llegadas: viajes que terminaron en la estación en esa hora. |
| `net_flow` | Llegadas menos salidas. Negativo: la estación tiende a vaciarse. |
| `departures_member` | Salidas hechas por miembros. |
| `departures_casual` | Salidas hechas por usuarios casuales. |

### `predictions` — `datos.load_predictions()`

Una fila por estación, hora, modelo y periodo. Es lo que el modelo esperaba
para horas de periodos de evaluación que ya pasaron, no para horas futuras.

| Columna | Qué es |
|---|---|
| `station_id` | Identificador de la estación. |
| `ts_hour` | Inicio de la hora, en hora local de Nueva York, con zona horaria. |
| `model` | `baseline` (promedio histórico) o `lgbm` (modelo). |
| `split` | `val` (validación) o `test` (prueba final). |
| `dep_q10`, `dep_q50`, `dep_q90` | Salidas esperadas: rango bajo, centro y rango alto. |
| `arr_q10`, `arr_q50`, `arr_q90` | Llegadas esperadas: rango bajo, centro y rango alto. |

Con `model = "baseline"`, `q50` es el promedio histórico de la celda (la misma
estación, día de la semana y hora), no la mediana. En pantalla conviene
llamarlo "promedio histórico", no "mediana".

**Lo observado se compara solo con `split == "val"`.** Las filas de
`split == "test"` se muestran como pronóstico, sin lo observado al lado y sin
calcular su error, hasta la evaluación final (decisión 4 del equipo: la prueba
final se usa una sola vez).

### `recommendations` — `datos.load_recommendations()`

Una fila por estación, día, franja horaria y política. **Sale de la simulación
de inventario: son resultados simulados, no mediciones.** Dice lo que hizo
cada política en la simulación, con el presupuesto de referencia.

El día de cada fila sale de `window_start`: `tabla["window_start"].dt.date`.
**Siempre hay que elegir un día y una política antes de mostrar o sumar.** Si
no, la misma estación sale repetida y las sumas mezclan días y políticas.

En la tabla real solo hay filas de los días evaluados de la simulación, que
son días de validación. No hay filas del día de calentamiento ni de la prueba
final.

| Columna | Qué es |
|---|---|
| `station_id` | Identificador de la estación. |
| `window_start`, `window_end` | Inicio y fin de la franja, en hora local de Nueva York, con zona horaria. El día de la fila es el de `window_start`. |
| `policy` | `none` (no intervenir), `historical` (regla histórica) o `model` (decide con el pronóstico). **Hoy ese pronóstico es la línea base**, no un modelo entrenado. La nota de simulación dice cuál se usó. |
| `start_inventory_target` | Bicis con las que la estación queda en la simulación justo después del traslado de esa política (las que tenía más `bikes_to_move`). **No es el inventario ideal:** si el presupuesto no alcanzó, es el mismo que ya tenía y puede estar bajo el umbral de riesgo. |
| `bikes_to_move` | Bicis que la política movió con el presupuesto de referencia. **Positivo es llevar bicis a la estación. Negativo es retirarlas.** Cero con riesgo alto casi siempre quiere decir que el presupuesto no alcanzó, no que la estación esté bien. |
| `risk_level` | `alto`, `medio` o `bajo`. Se calcula con el pronóstico y con el inventario de ANTES del traslado. Puede cambiar de una política a otra. |
| `feasible` | Verdadero o falso. **Falso: según el pronóstico, ningún inventario inicial evita a la vez que la estación se vacíe y que se llene.** Puede diferir entre políticas, porque cada una usa su pronóstico. |

El traslado se hace a la hora de intervención de la franja
(`momento_intervencion` en `config/operacion.yaml`): antes de que empiece la
franja o justo al empezar.

**No todas las estaciones tienen recomendaciones.** La simulación deja fuera a
las estaciones sin capacidad registrada y a las que no tuvieron viajes en los
días simulados. De esas, la vista debe decir "esta estación no se simuló". No
debe decir "sin riesgo" ni "0 bicis". La vista 3 ya muestra ese mensaje.

### `policy_curve` — `datos.load_policy_curve()`

Una fila por política y presupuesto diario. **Sale de la misma simulación:
son resultados simulados, no mediciones.**

| Columna | Qué es |
|---|---|
| `policy` | `none`, `historical` o `model`. |
| `budget` | Presupuesto: total de bicis que se pueden trasladar en un día en toda la red. |
| `risk_station_hours` | Horas de estación en riesgo simulado **por día**: promedio de los días evaluados, contando las 24 horas. Una estación está en riesgo cuando tiene menos bicis o menos lugares libres que el umbral de `config/operacion.yaml`. |
| `ci_low`, `ci_high` | Límite bajo y alto de un intervalo que sale de sortear los días evaluados. **Los días no son independientes: es indicativo, no una garantía.** |

Con `policy = "none"` el valor es el mismo en todos los presupuestos, porque
no se mueve nada. La diferencia entre una política y `none` con el mismo
presupuesto es **riesgo simulado evitado**.

Los presupuestos, el umbral y el nivel del intervalo de hoy están en "Datos
reales de hoy".

### `routes` — `datos.load_routes()`

Una fila por par de estaciones: la de inicio y la de fin del viaje. Son viajes
de usuarios. **No son rutas de camiones de rebalanceo.**

| Columna | Qué es |
|---|---|
| `start_station_id` | Identificador de la estación donde empezó el viaje. Es texto. |
| `end_station_id` | Identificador de la estación donde terminó. Es texto. Puede ser la misma del inicio. |
| `trips` | Cuántos viajes hubo de la estación de inicio a la de fin en todo el periodo. Siempre es 1 o más. |

## Lo que hay que saber de los datos

- **Rutas: hay viajes entre estaciones, no rutas de camiones.** `routes` dice
  cuántos viajes de usuarios hubo de una estación a otra. Las rutas de
  rebalanceo quedan fuera por ahora: `recommendations` dice cuántas bicis
  recibe o entrega cada estación, pero ningún archivo dice de cuál estación
  salen ni a cuál van. No se inventa ese dato en la app.
- En `routes`, el viaje de A a B y el de B a A son dos filas distintas. Los
  pares sin ningún viaje no tienen fila. Hay filas con la misma estación de
  inicio y de fin: el usuario devolvió la bici donde la tomó.
- `trips` es un total de todo el periodo. `routes` no trae día, hora ni tipo de
  usuario, así que las rutas no se pueden filtrar por fecha ni por hora.
- Para mostrar nombres en lugar de identificadores, `routes` se une dos veces
  con `stations`: una por `start_station_id` y otra por `end_station_id`.
- **`station_hour` no trae todas las horas de todas las estaciones.** Cada
  estación tiene filas solo entre su primer y su último viaje. Por eso no
  todas empiezan ni terminan el mismo día. En la muestra hay estaciones que
  cubren menos días; `data/sample/LEEME.md` dice cuáles. Una estación así
  también puede tener menos filas de pronóstico y de recomendaciones.
- Si una estación no tiene fila en una hora, esa hora queda fuera de lo
  observado para esa estación. No se dibuja como cero: se deja sin dato.
- Un cero en `departures` o en `arrivals` dice que no se registró ningún viaje
  en esa hora. No dice que nadie buscó una bici: si la estación estaba vacía,
  ahí puede haber posible demanda insatisfecha.
- `capacity` y `station_type` pueden venir vacíos. La app debe mostrar esas
  estaciones con "Sin dato", no esconderlas ni fallar.
- `ts_hour`, `window_start` y `window_end` ya vienen en hora local de Nueva
  York, con zona horaria. La hora del día se saca con
  `tabla["ts_hour"].dt.hour`.
- **Horas en el eje de una gráfica.** En las tablas (`st.dataframe`) y en las
  gráficas de `plotly` esas columnas se usan tal cual. En `st.line_chart`,
  `st.bar_chart`, `st.area_chart` y `altair` no: el navegador convierte la hora
  a la zona de la computadora y la gráfica sale corrida (con la computadora en
  hora del centro de México, 1 o 2 horas). En esas gráficas usa en el eje
  `datos.hora_para_grafica(tabla["ts_hour"])`, o la hora del día como número:
  `tabla["ts_hour"].dt.hour`.
- El día que termina el horario de verano tiene 25 horas y la 01:00 aparece
  dos veces. En la muestra es el 2 de noviembre de 2025. Es normal. En una
  gráfica hecha con `datos.hora_para_grafica`, esas dos filas caen en el mismo
  punto del eje.
- `predictions` cubre periodos de evaluación que ya pasaron (`split` = `val` o
  `test`), no horas futuras. El tablero muestra lo que el modelo esperaba para
  esos días. No es un pronóstico en vivo.
- **Lo observado se compara solo con `split == "val"` (validación).** Las
  filas de `split == "test"` (prueba final) se muestran como pronóstico, sin
  lo observado al lado y sin calcular su error, hasta la evaluación final. Es
  la decisión 4 del equipo: la prueba final se usa una sola vez. `station_hour`
  sí trae lo observado de esos días; no se dibuja junto al pronóstico. En la
  vista 3 ya existe `forecast_val`, con las filas de validación de la estación
  elegida.
- Con `model = "baseline"`, `q50` es el promedio histórico, no la mediana.
- En la muestra, las filas con `model = "lgbm"` son un relleno simulado. No
  salen de ningún modelo entrenado.
- Con `policy = "none"` nunca se mueven bicis.
- **`recommendations` y `policy_curve` son resultados de una simulación.** No
  dicen lo que pasó en las estaciones: dicen lo que pasa en una simulación
  hecha con los viajes de los días de validación y con las reglas de
  `config/operacion.yaml`.
- **`recommendations` trae una fila por día.** Sin un selector de día y otro
  de política, una tabla o una suma mezcla todos los días y las tres
  políticas.
- **Una estación con "0 de recomendaciones" no es una estación sin riesgo.**
  Con las seis tablas reales quiere decir que no se simuló (sin capacidad
  registrada o sin viajes en los días simulados). Con tablas reales y
  sintéticas mezcladas, quiere decir que las estaciones no coinciden.
- **La muestra no se parece a la simulación real en dos cosas.** En
  `data/sample` las recomendaciones caen en los días de `split = "test"` de
  la muestra; las reales solo existen en días de validación. Y en la muestra
  `model` da menos riesgo que `historical`; con los datos reales de hoy es al
  revés. Una vista hecha solo con la muestra hay que revisarla con los datos
  reales.

## Lenguaje honesto

Los datos solo registran **demanda atendida**: viajes que sí ocurrieron. Si una
estación estaba vacía, no queda registro de quien no encontró bici.

- Sí se dice: "posible demanda insatisfecha" y "riesgo simulado evitado".
- Lo que sale de `recommendations` y de `policy_curve` es "riesgo simulado".
  La diferencia contra no intervenir es "riesgo simulado evitado". No es una
  medición.
- Los viajes bloqueados de la simulación son simulados. Son posible demanda
  insatisfecha, no demanda insatisfecha medida.
- No se afirma cuántos viajes más habría con el rebalanceo. Eso no se puede
  medir con estos datos.
- No se muestra ninguna cifra inventada. Si algo no se pudo calcular, se dice.

## Cómo pedir cambios a un asistente

Di en qué vista, con qué tabla y qué quieres ver. Ejemplos:

1. "En la vista 1 dibuja un mapa con un punto por estación, con el tamaño según
   las salidas promedio de la hora elegida. Agrega un selector de hora del día."
2. "En la vista 1 pinta cada punto según el flujo neto promedio de esa hora: un
   color para las estaciones que se vacían y otro para las que se llenan. Pon
   una leyenda que lo explique."
3. "En la vista 2 agrega un selector de día, otro de franja y otro de política.
   El día sale de `window_start`. Muestra una tabla de estaciones ordenada de
   mayor a menor riesgo, para el día, la franja y la política que yo elija.
   Muestra el nombre de la estación, no solo el identificador."
4. "En la vista 2 separa en dos tablas las estaciones a las que hay que llevar
   bicis y las estaciones de las que hay que retirar bicis."
5. "En la vista 2 dibuja la curva de riesgo contra presupuesto, con una línea
   por política y una banda entre `ci_low` y `ci_high`. Titula el eje vertical
   'Horas de estación en riesgo simulado por día'. Debajo explica en una frase
   que la diferencia contra no intervenir es riesgo simulado evitado y que la
   banda es indicativa."
6. "En la vista 2 muestra una tabla con los pares de estaciones con más viajes
   (tabla `routes`), de mayor a menor, con el nombre de la estación de inicio y
   el de la estación de fin. Agrega un selector de cuántos pares mostrar.
   Escribe debajo que son viajes de usuarios que sí ocurrieron, no rutas de
   camiones."
7. "En la vista 2 dibuja en un mapa una línea por cada uno de los pares con más
   viajes, más gruesa mientras más viajes tenga. Deja fuera los viajes que
   empiezan y terminan en la misma estación y di cuántos son."
8. "En la vista 3, para la estación elegida, dibuja las salidas esperadas por
   hora con una banda entre `dep_q10` y `dep_q90`. Encima pon las salidas
   observadas solo en los días de validación (`split == "val"`, usa
   `forecast_val`). En los días de prueba final (`split == "test"`) deja el
   pronóstico solo, sin lo observado. Agrega un selector de modelo y otro de
   fechas."
9. "En la vista 3 agrega un selector de día y otro de política para las
   recomendaciones de la estación elegida. Escribe la acción en una frase por
   cada franja del día y la política elegidos: llevó bicis si `bikes_to_move`
   es positivo, retiró bicis si es negativo y no movió nada si es cero. Si es
   cero y `risk_level` es `alto`, agrega que casi siempre quiere decir que el
   presupuesto no alcanzó. Si `feasible` es falso, agrega una advertencia."
10. "En todas las vistas, cuando falte la capacidad o el tipo de una estación,
    muestra 'Sin dato' en lugar de dejar el espacio vacío."

Al final de cada instrucción conviene agregar: "Deja los comentarios en
español, no quites el aviso de datos sintéticos ni la nota de simulación, no
compares lo observado con el pronóstico de prueba final (`split == "test"`) y
corre `pytest tests/test_app_smoke.py` al terminar."

## Reglas de esta carpeta

- `app/` no importa nada de `src/`. Hoy solo usa `streamlit` y `pandas`.
- Para mapas y gráficas ya están instalados `pydeck`, `altair` y `plotly`. No
  hace falta instalar nada más.
- Las vistas no abren archivos. Piden las tablas a `app/datos.py`.
- El aviso de datos sintéticos va en todas las páginas, también en las nuevas:
  `datos.show_synthetic_warning(tables_used)`, con la lista de las tablas que lee
  la vista. Sin lista, la página muestra el aviso completo mientras quede alguna
  tabla sintética. La prueba revisa todas las vistas de `app/pages/` y falla
  si una vista lee una tabla sintética y no muestra el aviso completo.
- La nota de simulación también va en todas las páginas, justo después del
  aviso y con la misma lista: `datos.show_simulation_note(tables_used)`. La
  prueba falla si una vista lee `recommendations` o `policy_curve` reales y no
  muestra la nota.
- Lo que sale de `recommendations` o de `policy_curve` se presenta como
  resultado de una simulación: "riesgo simulado" y "riesgo simulado evitado".
- Lo observado se compara con el pronóstico solo en los días de validación
  (`split == "val"`). La prueba final (`split == "test"`) se usa una sola vez
  (decisión 4 del equipo). Las recomendaciones y la curva cubren solo días de
  validación: la prueba final no se simula.
- No se escriben en el código horas de franjas, umbrales ni presupuestos. Las
  franjas y los presupuestos salen de las tablas (`window_start`, `window_end`,
  `budget`). Tampoco se escribe cuántos días o cuántas estaciones tiene la
  simulación: se cuentan de las tablas. Las reglas de operación viven en
  `config/operacion.yaml`.
