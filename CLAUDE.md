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
config/         operacion.yaml: reglas de operación
data/raw/       Descargas pesadas: trips/, zips/, gbfs/      (no va en Git)
data/processed/ Salidas reales del flujo, los seis contratos (no va en Git)
data/sample/    Datos SINTÉTICOS pequeños con los seis contratos (sí va en Git)
docs/           reglas_operacion.md
notebooks/a/    Notebooks de Acker
notebooks/b/    Notebooks del compañero
scripts/        Puntos de entrada: python scripts/<nombre>.py
src/contracts.py  Contratos de datos: esquemas, validación, lectura y escritura
src/paths.py      Rutas y zona horaria
src/data/       Descarga de viajes, captura de GBFS, datos de muestra
src/features/   Variables: registro de constructores y revisión de fuga
src/models/     Línea base y LightGBM (pendiente)
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
  decimales. En la máquina de Acker ya existe un mes real, `202604.parquet`
  (3,860,371 filas); no va en Git y se regenera con
  `python scripts/download_trips.py 202604`.
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
  de horario de otoño queda toda en la segunda 01:00 (-05:00). Al escribir la
  limpieza conviene una prueba con un viaje conocido.
- pandas 3: el texto es tipo `str` y un entero con nulos se lee como decimal
  salvo que se use `src.contracts.read_contract`.

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
- Aún no existen: limpieza y tabla por estación y hora, línea base, LightGBM,
  simulador y variables externas.
