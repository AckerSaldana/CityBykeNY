# CityBykeNY — DataRush (reto de ciencia de datos de 24 horas)

Este archivo es el punto de partida de cualquier sesión, de cualquiera de los
dos integrantes. Si algo de aquí cambia, se actualiza aquí en el mismo commit.

## Qué construimos

Citi Bike es el sistema de bicis compartidas de Nueva York. Las estaciones se
vacían o se llenan y el operador mueve bicis en camiones (rebalanceo), hoy de
forma reactiva. Construimos un sistema que ayuda a **prevenir estaciones vacías
o llenas**, no solo un modelo de demanda.

La decisión que apoyamos: **cuántas bicis mover a cada estación y en qué franja
horaria**. El usuario es un analista de operaciones que no programa.

Flujo: viajes de Citi Bike en Parquet consultados con DuckDB → tabla por
estación y hora → línea base contra LightGBM con cuantiles → simulación de
inventario → recomendación de bicis a mover → dashboard en Streamlit.

Rúbrica: pitch 35, dashboard en vivo 35, impacto y accionabilidad 15, checklist
técnico 15. **Prioridad: que siempre exista un dashboard que funcione de punta
a punta**, aunque sea con la línea base, y mejorar desde ahí.

## Decisiones ya tomadas (no reabrir sin preguntar al usuario)

1. **Unidad de análisis:** estación y hora, en hora local de Nueva York.
2. **Datos:** solo formato nuevo (desde febrero de 2021). Para el MVP, pocos
   meses recientes; el script de descarga recibe los meses como parámetro.
3. **Línea base:** promedio histórico por estación, hora y día de la semana.
   **Modelo:** LightGBM global con cuantiles 10, 50 y 90 para salidas y llegadas.
4. **Validación** en tres tramos ordenados en el tiempo: entrenamiento,
   validación y prueba final. Las decisiones se toman con validación. La prueba
   final se usa una sola vez.
5. **Sin fuga de información:** toda variable usa solo datos disponibles al
   momento de decidir. Clima de pronóstico, no observado. Rezagos de días
   anteriores, no del mismo día.
6. **Métricas de pronóstico:** MAE, RMSE, WAPE y cobertura de intervalos,
   desagregadas por estación, hora y tipo de día. No usar MAPE como métrica
   principal (explota con demanda cercana a cero).
7. **Criterio de éxito operativo:** minimizar las horas de estación en riesgo
   (vacía o llena en la simulación) con un presupuesto fijo de bicis trasladadas
   por día. Se comparan tres políticas —sin intervención, regla histórica y
   modelo predictivo— con varios presupuestos.
8. **Reglas del simulador:** las tres políticas se califican con las mismas
   secuencias reales de viajes y el mismo inventario inicial; cada política
   decide solo con información previa al periodo evaluado; los movimientos
   respetan disponibilidad en origen y capacidad en destino; el total de bicis
   del sistema se conserva.
9. **Lenguaje honesto:** los datos solo registran demanda atendida. Decimos
   "posible demanda insatisfecha" y "riesgo simulado evitado", nunca "viajes
   recuperados". No inventar resultados ni cifras; si algo no se pudo correr o
   descargar, se dice.
10. **Variables externas por capas**, cada una se queda solo si mejora en
    validación: clima y festivos, capacidad de GBFS, empleos y residentes
    (LODES), uso de predios (PLUTO), lugares frecuentados (OpenStreetMap con
    OSMnx), afluencia del metro (MTA).
11. **Rutas:** contrato `routes.parquet` con los viajes entre pares de
    estaciones, sumados sobre el periodo. Es para la vista de ranking de
    estaciones y rutas. Las rutas de rebalanceo (camiones) quedan fuera por
    ahora.
12. **`station_hour` es densa:** una fila por estación y hora, con ceros, solo
    entre el primer y el último viaje observado de cada estación.
13. **`station_type`** se asigna con una regla de comportamiento calculada solo
    con el tramo de entrenamiento: flujo neto de la mañana entre semana,
    proporción de fin de semana y proporción de usuarios ocasionales. Puede ser
    nulo mientras no exista la regla.
14. **`ocupacion_respaldo`** (antes `ocupacion_objetivo`) es el inventario
    inicial común de las tres políticas y el valor que se usa cuando no hay
    solución factible. La regla histórica no lo usa.
15. **Ids de estación:** son texto. Las variantes como `2231.1` y `2231.10` se
    unen solo si coinciden nombre y coordenadas. Se reporta cuántas se unieron
    y cuántas no.
16. **GBFS:** la captura no se traslapa con los meses de viajes. Se usa solo
    para capacidad, modo en vivo y perfil de ocupación por hora. No se usa para
    validar demanda insatisfecha.

## Contratos de datos

Son seis archivos que conectan el trabajo de los dos integrantes. **Nombres y
tipos exactos.** La
definición ejecutable está en `src/contracts.py`; si un contrato cambia, se
cambia ahí, aquí y en el diccionario `SPEC` de `tests/test_contracts.py`,
avisando al otro integrante.

`data/processed/stations.parquet`
station_id (str), name (str), lat (float), lon (float), capacity (int, puede
ser nulo), station_type (str, puede ser nulo)

`data/processed/station_hour.parquet`
station_id (str), ts_hour (timestamp en America/New_York), departures (int),
arrivals (int), net_flow (int, llegadas menos salidas), departures_member
(int), departures_casual (int)

`data/processed/predictions.parquet`
station_id, ts_hour, model ("baseline" o "lgbm"), split ("val" o "test"),
dep_q10, dep_q50, dep_q90, arr_q10, arr_q50, arr_q90

`data/processed/recommendations.parquet`
station_id, window_start, window_end, policy ("none", "historical" o "model"),
start_inventory_target (int), bikes_to_move (int, positivo es llevar y negativo
es retirar), risk_level ("alto", "medio" o "bajo"), feasible (bool, falso
cuando ningún inventario inicial evita a la vez vaciarse y llenarse)

`data/processed/policy_curve.parquet`
policy, budget (int), risk_station_hours (float), ci_low (float), ci_high (float)

`data/processed/routes.parquet`
start_station_id (str), end_station_id (str), trips (int, viajes de la estación
de inicio a la de fin en todo el periodo; al menos 1)

Cómo se cumplen en la práctica:

- Tipos en el Parquet: str = `string`, int = `int64`, float = `float64`,
  fechas = `timestamp[us, tz=America/New_York]`, bool = `bool`.
- **Escribir siempre con `src.contracts.write_contract(df, "nombre", ruta)`**:
  convierte a los tipos exactos, valida y no escribe nada si no cumple. DuckDB
  por sí solo guarda las fechas como UTC y pandas guarda el texto como
  `large_string`; `write_contract` corrige ambos.
- Una fecha sin zona horaria se rechaza (se guardaría desplazada 4 o 5 horas).
- `write_contract` recibe un DataFrame de pandas, una tabla de Arrow o lo que
  devuelve `.arrow()` en DuckDB. Ajusta el tipo dentro de la misma familia
  (por ejemplo `int32` a `int64`), pero no convierte entre familias: un número
  donde va texto, o texto donde va un número, se rechaza.
- Reglas de contenido que valida el contrato: al menos una fila, llaves sin
  duplicados, sin negativos, `net_flow = arrivals - departures`, miembros +
  casuales no supera el total, fechas en hora exacta, cuantiles ordenados
  (q10 ≤ q50 ≤ q90), `policy "none"` no mueve bicis, `window_end >
  window_start`, el intervalo de confianza contiene al valor, sin NaN ni
  infinitos en las columnas decimales, y lat/lon dentro de rango.
- Antes de escribir `predictions`: recortar los cuantiles en 0, ordenar
  q10 ≤ q50 ≤ q90 por fila y rellenar de forma explícita las combinaciones sin
  historia de la línea base. LightGBM con cuantiles puede dar negativos y
  cuantiles cruzados; las métricas de cobertura se calculan sobre los valores
  ya corregidos.
- `pytest` valida `data/sample` siempre y `data/processed` en cuanto los
  archivos existen. También comprueba que toda estación usada en las otras
  tablas exista en `stations`: esa tabla debe incluir todas las estaciones que
  aparecen en los viajes, no solo las del feed GBFS de hoy.
- `station_id` es el identificador de los archivos de viajes (por ejemplo
  `6289.06`). Es **texto**: los ceros finales importan. En GBFS ese valor se
  llama `short_name`; el `station_id` de GBFS es otro (un id propio de GBFS:
  un UUID o un número largo). `write_contract` rechaza un id que llegue como
  número, porque ya perdió los ceros.
- `station_type` usa las categorías de la estrategia de operación: oficinas,
  residencial, recreativa, mixta. No es el `station_type` de GBFS ("classic").
  Queda nulo hasta que exista la regla de la decisión 13.
- `station_hour` densa (decisión 12): el validador exige una fila por hora
  entre la primera y la última de cada estación, que esas dos tengan al menos
  un viaje y que no haya estaciones sin viajes. Se cuenta en instantes: el día
  de 25 horas lleva 25 filas.
- `routes` son viajes de usuarios entre estaciones, no rutas de camión. Inicio
  y fin pueden ser la misma estación.

## Reparto del equipo

| | Acker (programación) | Compañero (logística) |
|---|---|---|
| Carpetas | `src/data`, `src/models`, `src/sim`, `notebooks/a` | `docs/reglas_operacion.md`, `config/operacion.yaml`, `app`, `src/features`, `notebooks/b` |
| Trabajo | Descarga, limpieza, tabla por estación y hora, modelos, validación y simulador | Reglas de operación que el simulador debe implementar, dashboard y pitch |

- Cada quien trabaja en su rama y en sus carpetas. Commits pequeños con
  mensajes claros. Los datos pesados no van en Git (`data/raw` y
  `data/processed` están en `.gitignore`).
- Archivos compartidos, se cambian avisando al otro: `CLAUDE.md`,
  `src/contracts.py`, `src/paths.py`, `requirements.txt`, `constraints.txt`,
  `tests/`.
- **El compañero programa poco.** El código de `app/` debe ser simple, estar
  bien comentado en español y ser fácil de modificar con instrucciones en
  lenguaje normal. `app/` no importa nada de `src/`.

## Reglas de operación

El simulador (todavía no existe) **no llevará números de operación escritos en
el código**. Los leerá de `config/operacion.yaml` con
`src.sim.operacion.load_operacion()`. Hoy solo lo usan los datos de muestra:
si cambian las franjas o los presupuestos del YAML hay que regenerar
`data/sample` (`pytest` avisa). Los ajusta el
compañero con su criterio de logística, respondiendo `docs/reglas_operacion.md`.

Parámetros: franjas horarias de rebalanceo y momento de intervención; umbral de
riesgo (bicis o lugares libres mínimos); lista de presupuestos diarios para la
curva; distancia máxima entre la estación que dona y la que recibe; estrategia
por tipo de estación (oficinas, residencial, recreativa, mixta).

**Hoy todos los valores son provisionales** (`estado: provisional`). No se
citan como reglas de operación hasta que el compañero los confirme.

## Comandos

Todo se corre desde la raíz del repositorio, con el entorno activado.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt

pytest                                    # contratos, fuga, config, scripts y app
python scripts/make_sample_data.py        # regenera data/sample (sintético)

python scripts/build_tables.py            # limpieza: stations, station_hour y routes reales
python scripts/run_baseline.py            # línea base: predictions y métricas de validación
streamlit run app/Inicio.py               # dashboard

python scripts/download_trips.py --listar 202609          # qué hay en el índice
python scripts/download_trips.py 202607 202608 202609     # descarga a data/raw/trips
python scripts/download_trips.py --desde 202607 --hasta 202609
python scripts/capture_gbfs.py            # captura GBFS cada 5 minutos, sin parar
python scripts/capture_gbfs.py --una-vez  # una sola captura (para cron)
```

## Mapa del repositorio

```
app/            Dashboard de Streamlit (Inicio.py, pages/, datos.py, LEEME.md)
config/         operacion.yaml: reglas de operación (logística)
                datos.yaml: periodo, tramos y reglas de limpieza (programación)
data/raw/       Descargas pesadas: trips/, zips/, gbfs/      (no va en Git)
data/processed/ Salidas reales del flujo, los seis contratos (no va en Git)
data/sample/    Datos SINTÉTICOS pequeños con los seis contratos (sí va en Git)
docs/           reglas_operacion.md
notebooks/a/    Notebooks de Acker
notebooks/b/    Notebooks del compañero
scripts/        Puntos de entrada: python scripts/<nombre>.py
src/contracts.py  Contratos de datos: esquemas, validación, lectura y escritura
src/paths.py      Rutas y zona horaria
src/data/       Descarga de viajes, captura de GBFS, limpieza y tablas, datos de muestra
src/features/   Variables: registro de constructores y revisión de fuga
src/models/     Línea base y métricas; LightGBM pendiente
src/sim/        Simulador (pendiente) y lector de operacion.yaml
tests/          Pruebas
```

## Hechos verificados sobre los datos (9 de octubre de 2026)

Viajes (`https://s3.amazonaws.com/tripdata`):

- `index.html` es una página con JavaScript; el índice real es el XML del
  bucket. El script resuelve cada mes contra ese índice, nunca arma el nombre.
- Meses 202401 a 202609: un zip por mes, `AAAAMM-citibike-tripdata.zip`, de
  165 MB a 1.05 GB, con **varios CSV** dentro (cerca de un millón de filas cada
  uno). Casi todos se llaman `..._1.csv`, `..._2.csv`; 202604 usa
  `...-part1.csv`. Por eso se toman todos los `.csv`, sin suponer nombres.
- Años 2021 a 2023: solo existe el zip anual (1.0 a 1.6 GB) con zips mensuales
  anidados y archivos basura de macOS. El script baja solo el mes pedido con
  peticiones parciales.
- Jersey City (`JC-...`) pesa de 0.1 a 4.4 MB por mes y su extensión varía
  (`.csv.zip` o `.zip`). Sirve para pruebas rápidas con `--dataset jc`.
- Columnas: ride_id, rideable_type, started_at, ended_at, start_station_name,
  start_station_id, end_station_name, end_station_id, start_lat, start_lng,
  end_lat, end_lng, member_casual.
- `started_at` y `ended_at` son hora local de Nueva York **sin zona horaria**.
  Al construir `ts_hour` hay que asignarles la zona. En el cambio de horario de
  otoño la hora 01:00 se repite y el dato no dice cuál de las dos es.
- Las filas de un CSV no vienen ordenadas por fecha.
- El script deja la capa cruda en `data/raw/trips/AAAAMM.parquet` con esas 13
  columnas, sin limpiar: ids y nombres como texto, fechas sin zona, coordenadas
  decimales. En la máquina de Acker hay cuatro meses reales: 202604
  (3,860,371 filas) y 202607 a 202609, que son los que usa `config/datos.yaml`.
  No van en Git; se regeneran con `python scripts/download_trips.py`.
- **Los ids de estación vienen inconsistentes en el origen** (visto en 202604,
  está así en el CSV): 60 ids aparecen con un solo decimal (`2231.1`) y 57 de
  ellos también con el cero final (`2231.10`) y el mismo nombre de estación;
  4 llevan un guion bajo al final (`5303.06_`); hay ids `SYS...`, `HB...`,
  `JC...` y uno con espacio final. Agrupar por el id crudo parte estaciones en
  dos: la limpieza necesita una regla de normalización, decidida junto con el
  cruce contra `short_name` de GBFS. No se verificó que cada par sea la misma
  estación física, solo que el nombre coincide.
- En 202604: 1,939 viajes sin estación de inicio y 10,498 sin estación de fin;
  ningún `member_casual` vacío; `ride_id` único; 512 viajes terminan en mayo.
  En 202102 hay algún `started_at` de 2019.
- **Un archivo mensual no es el mes calendario de `started_at`, y no todos
  cortan igual.** 202604 trae los viajes que EMPIEZAN en abril. 202102 completo
  (173 viajes empiezan antes de febrero, ninguno termina fuera) y las muestras
  revisadas de 202603, 202605 y 202609 traen los viajes que TERMINAN en el mes.
  Por eso meses seguidos pueden repetir viajes: en una muestra de 128,494 filas
  de 202605, los 36 viajes que empiezan en abril están también en
  `202604.parquet`. También pueden faltar: en las muestras de 202603 ningún
  viaje termina en abril y 202604 no trae ninguno que empiece en marzo (no se
  revisó el archivo completo). La limpieza debe unir los meses, quitar
  duplicados por `ride_id` y después filtrar la ventana por `started_at`.

Julio a septiembre de 2026 (medido el 10 de octubre sobre los tres archivos,
del 1 de julio al 30 de septiembre y antes de excluir estaciones):

- 15,222,228 viajes; son exactamente los que terminan entre el 1 de julio y
  el 30 de septiembre. Ningún `ride_id` repetido. 869 empiezan el 30 de junio.
- No hay cambio de horario dentro del periodo: Nueva York va 4 horas detrás de
  UTC todo el tiempo.
- Viajes sin estación de inicio: 8,814. Sin estación de fin: 44,469 (3,080
  traen nombre pero no id). De los 3,430 viajes de más de 24 horas, 3,345 no
  tienen estación de fin.
- 2,572 ids crudos. 98 pares de variantes: 94 son un decimal contra dos
  (`2231.1` / `2231.10`) y 4 son con guion bajo final. En los 98 coincide el
  nombre; en 95 la coordenada es idéntica y en 3 difiere entre 12 y 33 metros
  (`5303.06_`, `6517.08_`, `6569.09_`). `5303.06_` es una estación aparte en
  GBFS. `3593.1` y `6131.1` son ids oficiales de un decimal, sin gemelo.
- La escritura de un decimal aparece casi solo en las llegadas (23.7 % de las
  llegadas de esas estaciones contra 0.035 % de las salidas). Sin unir, el
  flujo neto de esas 94 estaciones sale muy mal.
- La coordenada es fija por id (no hay ruido de GPS; la única excepción es una
  salida de `5914.08` a 362 m de su punto) y el nombre también, salvo 4 ids con
  dos formas de escribirlo.
- Una estación reubicada recibe un id nuevo a pocos metros y su historia se
  parte (`3022.01` → `3022.05`, `7363.03` → `7363.04`, `5329.08` → `5329.13`).
- **Ceros que no son demanda cero:** una estación cerrada varios días sigue
  teniendo filas en cero dentro de su tramo. En la tabla final, 40,323
  horas-estación caen en rachas de 72 horas o más sin ningún viaje, también en
  estaciones grandes. No se determinó la causa. Conviene tenerlo presente al
  leer errores de pronóstico y al simular.
- Hay días con mucho menos uso que los de su misma semana. No se contrastó con
  clima ni con otra fuente.
- La exploración miró todo el periodo, también el tramo de prueba final. Lo
  que se vio ahí (qué estaciones y qué días) queda fuera de este archivo a
  propósito: ninguna decisión de modelado se ajusta con eso (decisión 4).
- Contra el GBFS de hoy: 27 estaciones con viajes en el periodo tienen
  capacidad 0 (ya no están instaladas) y 21 ids de los viajes no están, sin
  contar las 95 escrituras que se unen a su gemela: 15 estaciones de las tablas
  y 6 excluidas (`1234.56`, `HB106`, `SYS016`, `SYS033`, `SYS038`,
  `Shop Morgan`).
- Viaje conocido para la zona horaria: `ride_id` `0005122465C6A4A8`, inicio
  `2026-07-15 08:20:16` en la estación `4327.09`. Su hora es
  `2026-07-15 08:00` de Nueva York, que es `12:00` UTC. Con un `CAST` a secas
  en esta máquina cae dos horas tarde.

GBFS (`https://gbfs.citibikenyc.com/gbfs/gbfs.json`):

- No guarda historial: lo que no se captura se pierde. Por eso corre
  `scripts/capture_gbfs.py`.
- La captura empezó el 10 de octubre de 2026: no se traslapa con los meses de
  viajes. Sirve para capacidad, modo en vivo y perfil de ocupación por hora.
  No sirve para validar demanda insatisfecha (decisión 16).
- `station_information` trae `capacity` y `short_name`. Es el puente entre el
  id de GBFS y el `station_id` de los viajes. Hoy son 2,519 estaciones, todas
  con `short_name` único.
- El feed se renueva cada 60 segundos (`ttl`).
- La captura guarda un Parquet por foto en
  `data/raw/gbfs/station_status/date=AAAA-MM-DD/` y la información de
  estaciones (solo cuando cambia) en `data/raw/gbfs/station_information/`.
  Todo en UTC. El eje de tiempo para analizar es `feed_last_updated`.
- Unas 60 estaciones por foto no están instaladas: filtrar con `is_installed`.
  Hay estaciones con capacidad 0.
- El `station_type` de GBFS se guarda como `gbfs_station_type`.
- Revisar la captura: `ps -o pid=,command= -p "$(cat data/raw/gbfs/captura.pid)"` y
  `tail -3 data/raw/gbfs/captura.log`. No iniciar una segunda captura sobre la
  misma carpeta mientras ese proceso viva: nada lo impide y duplica las fotos.
  Si la computadora se duerme o se apaga, la captura se detiene y ese periodo
  no se recupera. Los pasos para iniciarla y detenerla están al inicio de
  `src/data/gbfs.py`.

Herramientas:

- DuckDB necesita `pytz` para devolver fechas con zona a Python (está en
  `requirements.txt`).
- DuckDB convierte Ctrl+C en `RuntimeError("Query interrupted")` y el proceso
  se queda colgado al salir si no se cierra la conexión: cerrarla en `finally`.
- En DuckDB usar `SET TimeZone='America/New_York'` antes de extraer hora o día
  de `ts_hour`.
- **La zona de sesión de DuckDB es la de la computadora** (en la de Acker,
  America/Monterrey), no Nueva York. Para construir `ts_hour` usar solo
  `date_trunc('hour', started_at) AT TIME ZONE 'America/New_York'` (o el `SET
  TimeZone` anterior). Nunca un `::TIMESTAMPTZ` o `CAST(... AS TIMESTAMPTZ)` a
  secas: desplaza cada hora 1 o 2 horas, el resultado trae zona y
  `write_contract` no lo nota. Con `AT TIME ZONE`, la 01:xx repetida del cambio
  de horario de otoño queda toda en la segunda 01:00 (-05:00).
- Un viaje que empieza en la primera 01:xx del cambio de horario de otoño y
  termina en la repetida trae `ended_at` menor que `started_at`, y la limpieza
  lo quita como fecha imposible (queda contado en "terminan antes de empezar").
  El periodo actual no tiene cambio de horario; si se agrega noviembre, hay que
  arreglarlo antes.
- pandas 3: el texto es tipo `str` y un entero con nulos se lee como decimal
  salvo que se use `src.contracts.read_contract`.

## Periodo, limpieza y línea base

El periodo, los tres tramos y las reglas de limpieza están en
`config/datos.yaml` (lo lee `src.data.datos_config.load_datos()`), no en el
código. Todo mes que toque el periodo debe estar en `meses_crudos`.

- **Periodo:** 1 de julio a 29 de septiembre de 2026. Tramos: entrenamiento
  1 jul a 31 ago, validación 1 a 14 sep, prueba final 15 a 29 sep.
- **El 30 de septiembre queda fuera a propósito.** Los archivos de julio a
  septiembre traen los viajes que TERMINAN en cada mes y el de octubre no
  existe todavía: faltan los viajes que salieron el 30 por la noche y
  terminaron el 1 de octubre (estimado: unas 750 salidas, cerca de 1 de cada 6
  de las 23:00). Hasta el 29, salidas y llegadas están completas.

Reglas de la limpieza (`src/data/build_tables.py`), cada una con su conteo en
`data/processed/reports/limpieza.md`:

1. Un viaje cuenta una sola vez (`ride_id`), aunque venga en dos archivos.
2. Los ids son texto; solo se les quitan espacios sobrantes.
3. **Unión de variantes (decisión 15):** dos escrituras del mismo id se unen
   solo si tienen el mismo nombre y la misma coordenada (tolerancia en el
   YAML). Queda la escritura con más viajes. Un id sin gemelo no se toca.
4. **Estaciones excluidas** (lista en el YAML): Nueva Jersey (`HB`, `JC`),
   porque aquí solo aparecen como destino y su flujo quedaría incompleto, y
   talleres o bodegas (`SYS...`, `Shop Morgan`, `1234.56`). El otro extremo del
   viaje sigue contando.
5. Salidas: viajes que empiezan en el periodo, en la hora de `started_at`.
   Llegadas: viajes que terminan en el periodo, en la hora de `ended_at`.
6. Un viaje sin estación de inicio o de fin cuenta solo del lado que sí tiene.
   No se adivina la estación a partir del nombre.
7. `capacity` es la del feed GBFS de hoy, no la del periodo; hace falta al
   menos una captura (`python scripts/capture_gbfs.py --una-vez`) o queda nula
   en todas. Una capacidad 0 (estación que hoy no está instalada) se guarda
   como nulo. Al revés no vale: 11 estaciones de las tablas no están
   instaladas hoy y conservan su capacidad. Para saber cuáles ya no existen se
   usa `is_installed` de `station_status`.
8. `station_type` queda nulo (decisión 13).

*Las reglas 4, 6 y 7 y el corte del 30 de septiembre son decisiones de
programación: si se cambian, se cambian en el YAML o aquí y se avisa.*

Resultado de la limpieza del 10 de octubre de 2026 (todo está en
`data/processed/reports/limpieza.md`):

- 15,222,228 viajes leídos, ninguno repetido ni con fechas imposibles.
- 98 grupos de ids parecidos: **95 unidos y 3 sin unir** (`5303.06_`,
  `6517.08_` y `6569.09_` quedan como estaciones aparte, a 16, 12 y 33 metros
  de su gemela). Cambiaron de id 826 salidas y 205,130 llegadas.
- `stations`: 2,398 estaciones; 2,356 con capacidad y 42 sin ella (27 con
  capacidad 0 en el GBFS de hoy y 15 que no están en GBFS). `station_type`
  nulo en todas.
- `station_hour`: 5,029,327 filas, 15,017,726 salidas y 14,981,442 llegadas;
  32.1 % de las filas están en cero.
- `routes`: 1,163,401 pares con 14,972,447 viajes.
- Las salidas y las llegadas no cuadran entre sí ni tienen por qué: 44,003
  viajes no traen estación de fin y 8,706 no traen estación de inicio.
- Zona horaria comprobada con viajes reales: el viaje `0005122465C6A4A8`
  (inicio `2026-07-15 08:20:16`, estación `4327.09`) cae en la fila de las
  `08:00-04:00` (12:00 UTC), y las 4 salidas de esa fila coinciden con el
  conteo hecho aparte sobre el archivo crudo.
- Una segunda construcción hecha solo con pandas, sin DuckDB, dio tablas
  idénticas.

Línea base (`src/models/baseline.py`): promedio histórico por estación, día de
la semana y hora local. En `predictions`, para `model = "baseline"`, **`q50`
guarda ese promedio (no es una mediana)** y `q10`/`q90` son los cuantiles
empíricos de la misma celda. Para validación la historia es el entrenamiento;
para la prueba final, entrenamiento más validación. Las filas de la prueba
final se escriben, pero **no se calculan sus métricas**: la prueba final se
usa una sola vez (decisión 4): `src.models.metrics.evaluate` se niega a
evaluarla si no se le pide de forma explícita. Las métricas de validación
quedan en `data/processed/reports/metricas_validacion.md`.

Línea base en validación (1 a 14 de septiembre de 2026, 784,929 filas):

| | MAE | RMSE | WAPE | Cobertura del intervalo |
|---|---|---|---|---|
| Salidas | 1.518 | 2.962 | 49.1 % | 80.0 % |
| Llegadas | 1.516 | 2.957 | 49.2 % | 80.0 % |

Cómo leerlo sin engañarse:

- El 80 % de cobertura es un promedio: de noche sube hasta 92 % (casi todo es
  cero y el intervalo lo cubre sin mérito) y entre las 07:00 y las 19:00 baja a
  73–77 % en salidas y 73–79 % en llegadas. No es un intervalo calibrado: en
  salidas, 6.1 % de las filas quedan debajo de `q10` y 13.8 % arriba de `q90`
  (lo nominal es 10 % de cada lado), y la cobertura es 93.9 % cuando el valor
  real es 0 contra 69.4 % cuando es mayor que 0.
- `q50` es un promedio y el promedio no minimiza el error absoluto (eso lo hace
  la mediana). Con la mediana de la misma celda, en salidas, el MAE bajaría de
  1.518 a 1.476 y el RMSE subiría de 2.962 a 3.009. Al comparar MAE o WAPE con
  el `q50` de LightGBM, parte de la diferencia viene de ahí: comparar también
  RMSE.
- **Filas de respaldo:** si una estación no tiene historia, su pronóstico es
  el promedio del sistema para ese día y hora (nivel 3). No es un pronóstico de
  esa estación, es un marcador. En validación son 4,579 filas de 26 estaciones
  (0.6 %) y ahí el pronóstico es unas seis veces lo observado (WAPE 617 % en
  salidas). El tramo de prueba tiene 2,485 filas así. El dashboard y el
  simulador no deben leerlas como pronósticos normales.
- El WAPE pasa de 100 % de madrugada, cuando casi no hay viajes. Siempre se
  cita con su hora.
- "Festivo" es un solo día (Labor Day, 7 de septiembre): no generaliza.
- Solo mide demanda atendida, y las horas de una estación cerrada cuentan como
  error.
- Si se reconstruye `station_hour` o cambian las fechas de
  `config/datos.yaml`, hay que volver a correr
  `python scripts/run_baseline.py`; una prueba falla a propósito hasta
  hacerlo. Compara las filas y la huella de la historia que la línea base deja
  en `data/processed/reports/linea_base_entrada.json` (entrenamiento y
  validación; de la prueba final, solo las fechas).
- Para comparar con LightGBM se usa `evaluate(...)` sobre las mismas filas:
  falla si a alguna fila real del tramo le falta predicción, si hay valores
  vacíos o si `q10` es mayor que `q90`. Quien escriba las predicciones de
  LightGBM debe conservar las filas de la línea base en `predictions.parquet`.

## Sin fuga de información, en concreto

Regla que se prueba: las variables de una fila cuyo día local es D solo pueden
depender de filas de `station_hour` de días anteriores a D.

Toda variable nueva se escribe como un constructor: una función que recibe
`station_hour` y devuelve una fila por cada fila de entrada, con `station_id`,
`ts_hour` y las variables. Se registra con `@register("nombre")`
(`src/features/registry.py`) en cualquier archivo de `src/features/`, también
en subcarpetas. `tests/test_no_leakage.py` revisa automáticamente a todos los
registrados con dos experimentos desde un día de corte: cambia los valores, y
aparte quita cerca de la mitad de las filas. Las variables de ese día y de los
anteriores no deben cambiar en ninguno. Por eso un rezago por posición
(`shift(24)`) se marca como fuga: los rezagos van por fecha y hora local.

`src/features/lags.py` es la plantilla (rezagos de 1 y 7 días y media de 7
días, por fecha y hora local). Es un ejemplo de referencia, no el conjunto de
variables del proyecto. Las variables externas todavía no existen; cuando
lleguen, cada fuente necesita su propia regla de disponibilidad (por ejemplo,
el pronóstico de clima emitido antes de decidir).

## Cómo trabajar en este repositorio

- Correr lo que se escribe y decir con claridad qué no se pudo verificar.
- Antes de cada commit: `pytest`.
- Nunca presentar como resultado algo que salió de `data/sample`. Son datos
  sintéticos; el dashboard lo avisa mientras los use.
- No escribir números de operación en el código: van en `config/operacion.yaml`.
- La prueba final se usa una sola vez. Cualquier ajuste se decide con validación.

## Pendientes y decisiones abiertas

- Confirmar los valores de `config/operacion.yaml` (logística).
- Fijar los umbrales de la regla de `station_type` (decisión 13). Hasta
  entonces la columna queda nula.
- `ocupacion_respaldo` tiene un valor por franja y la decisión 14 no dice cuál
  da el inventario inicial de la simulación. Programación usará el de la
  primera franja del día salvo que logística diga otra cosa.
- El dashboard ya lee datos reales en `stations`, `station_hour`, `routes` y
  `predictions`. `recommendations` y `policy_curve` siguen siendo sintéticas
  hasta que exista el simulador.
- Aún no existen: LightGBM, simulador, regla de `station_type` y variables
  externas.
- El simulador tiene que decidir qué hace con las 42 estaciones sin capacidad
  y con las estaciones sin tipo (hoy todas): `ocupacion_respaldo` se define por
  tipo.
- Los 3,039 viajes con nombre de estación de fin pero sin id no cuentan como
  llegada. En 3,038 el nombre corresponde a una sola estación de las tablas,
  así que se podrían asignar si el equipo lo decide; sería una regla nueva. El
  otro (`Clinton St & Grand St`) no se puede: los tres pares sin unir comparten
  nombre.
