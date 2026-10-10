# CityBykeNY

Proyecto para DataRush, un reto de ciencia de datos de 24 horas.

Citi Bike mueve bicis en camiones para que las estaciones no se queden vacías
ni llenas, hoy de forma reactiva. Este proyecto ayuda a decidir **cuántas bicis
mover a cada estación y en qué franja horaria**, y lo presenta en un dashboard
para un analista de operaciones que no programa.

Flujo: viajes históricos en Parquet (DuckDB) → tabla por estación y hora →
línea base contra LightGBM con cuantiles → simulación de inventario →
recomendación de bicis a mover → dashboard en Streamlit.

> **Estado actual:** existen la limpieza, la tabla por estación y hora, la
> línea base y un simulador mínimo, con viajes reales de julio a septiembre de
> 2026. Las seis tablas del dashboard ya salen de datos reales. Las
> recomendaciones y la curva de riesgo son resultados **simulados** sobre los
> días de validación, con reglas de operación todavía provisionales. Todavía
> no hay LightGBM: la política "modelo" usa hoy el pronóstico de la línea base.

El contexto completo, las decisiones del equipo y los contratos de datos están
en [`CLAUDE.md`](CLAUDE.md).

## Cómo reproducir

### 1. Preparar el entorno

Se necesita Python 3.12.

```bash
git clone https://github.com/AckerSaldana/CityBykeNY.git
cd CityBykeNY

python3.12 -m venv .venv
source .venv/bin/activate            # En Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
```

`requirements.txt` fija la versión de cada paquete que usamos y
`constraints.txt` fija también las dependencias indirectas.

Si en macOS LightGBM falla al importarse por `libomp`, se instala con
`brew install libomp`.

### 2. Comprobar que todo funciona

```bash
pytest
```

Valida los seis contratos de datos sobre `data/sample`, la revisión contra
fuga de información, `config/operacion.yaml`, los scripts y que el dashboard
abra. Las pruebas no usan la red.

Los archivos reales de `data/processed` se validan con el mismo comando en
cuanto existen; mientras no existan, esas pruebas se saltan.

### 3. Abrir el dashboard

```bash
streamlit run app/Inicio.py
```

Usa los archivos de `data/processed` si existen y, si no, los de `data/sample`.
Mientras use la muestra, cada vista avisa que los datos son sintéticos.

### 4. Descargar viajes reales

```bash
python scripts/download_trips.py --listar 202609           # ver qué hay, sin descargar
python scripts/download_trips.py 202607 202608 202609      # meses sueltos
python scripts/download_trips.py --desde 202607 --hasta 202609
```

Deja un Parquet por mes en `data/raw/trips/AAAAMM.parquet`, con los viajes tal
como vienen (sin limpiar). Un mes reciente pesa cerca de 1 GB de descarga.
Para una prueba rápida de unos megabytes: `--dataset jc` (Jersey City).

Solo se aceptan meses desde febrero de 2021, que es el formato que usamos.

Para consultar con DuckDB desde la terminal (o la misma consulta con
`duckdb.sql(...)` en un notebook):

```bash
python -c "import duckdb; print(duckdb.sql(\"SELECT count(*) FROM 'data/raw/trips/*.parquet'\"))"
```

Ese conteo es de filas, no de viajes: meses seguidos pueden repetir viajes
(ver `CLAUDE.md`). Para contar viajes se usa `count(DISTINCT ride_id)`.

### 5. Construir las tablas reales y la línea base

```bash
python scripts/download_trips.py --desde 202607 --hasta 202609   # unos 3 GB
python scripts/capture_gbfs.py --una-vez   # una foto de GBFS: de ahí sale la capacidad
python scripts/build_tables.py     # limpieza: stations, station_hour y routes
python scripts/run_baseline.py     # línea base: predictions y métricas de validación
python scripts/run_simulation.py   # simulador: recommendations y policy_curve
```

El periodo, los tres tramos (entrenamiento, validación y prueba final) y las
reglas de limpieza están en `config/datos.yaml`. Las tablas quedan en
`data/processed` y cada paso deja un reporte legible en
`data/processed/reports/` (`limpieza.md`, `metricas_validacion.md` y
`simulacion.md`). Las reglas del simulador (franjas, umbral de riesgo,
presupuestos, distancia máxima) están en `config/operacion.yaml`.

La limpieza tarda unos 20 segundos y necesita cerca de 2 GB libres mientras
corre; además deja `trips.parquet`, una tabla interna de unos 300 MB con los
viajes limpios que usa el simulador. La línea base tarda menos de 10 segundos
y la simulación unos 15. Los tres pasos van en ese orden: si se reconstruyen
las tablas hay que volver a correr los dos siguientes, y `pytest` avisa si
algo quedó desfasado.

Las métricas y la simulación se calculan solo en validación. La prueba final
se usa una sola vez, al cierre.

El simulador reproduce los viajes reales uno por uno sobre un inventario
supuesto. Sus resultados son riesgo simulado, no mediciones: dependen del
inventario inicial, la distancia es en línea recta (ignora ríos y puentes) y
los días no son independientes. `simulacion.md` lista las limitaciones.

### 6. Capturar el estado de las estaciones (GBFS)

El feed en vivo no guarda historial: lo que no se captura se pierde. Conviene
dejarlo corriendo desde el inicio del reto.

```bash
python scripts/capture_gbfs.py              # una captura cada 5 minutos, sin parar
python scripts/capture_gbfs.py --una-vez    # una sola captura
```

Guarda en `data/raw/gbfs/`. Para dejarlo en segundo plano, aunque se cierre la
terminal:

```bash
# ¿Ya hay una captura corriendo? Si muestra una línea con capture_gbfs.py, NO inicies otra.
ps -o pid=,command= -p "$(cat data/raw/gbfs/captura.pid 2>/dev/null)" 2>/dev/null

mkdir -p data/raw/gbfs
nohup .venv/bin/python scripts/capture_gbfs.py >> data/raw/gbfs/captura.log 2>&1 &
echo $! > data/raw/gbfs/captura.pid
nohup caffeinate -i -w $(cat data/raw/gbfs/captura.pid) > /dev/null 2>&1 &   # solo macOS

tail -3 data/raw/gbfs/captura.log          # ver cómo va
kill $(cat data/raw/gbfs/captura.pid)      # detenerla
```

La línea de `caffeinate` evita que la Mac se duerma por inactividad mientras
la captura viva. No alcanza con la tapa cerrada ni sin corriente: conviene
dejar la laptop abierta y conectada. Lo que pase mientras la computadora
duerme o está apagada no se recupera. Más detalle al inicio de
`src/data/gbfs.py`.

### 7. Regenerar los datos de muestra

```bash
python scripts/make_sample_data.py
```

Siempre produce los mismos datos. Son sintéticos: sirven para construir el
dashboard, no para sacar conclusiones.

## Estructura

```
app/            Dashboard de Streamlit
config/         operacion.yaml: reglas de operación (hoy provisionales)
                datos.yaml: periodo, tramos y reglas de limpieza
data/raw/       Descargas pesadas                     (no va en Git)
data/processed/ Salidas reales del flujo              (no va en Git)
data/sample/    Datos sintéticos pequeños             (sí va en Git)
docs/           reglas_operacion.md: las cinco preguntas de operación
notebooks/a/    Notebooks de programación
notebooks/b/    Notebooks de logística
scripts/        Puntos de entrada
src/            Código: contratos, datos, variables, modelos, simulador
tests/          Pruebas
```

## Cómo trabajamos

- Dos integrantes en paralelo, cada quien en su rama y en sus carpetas (ver el
  reparto en `CLAUDE.md`). Commits pequeños.
- Los seis archivos de `data/processed` son el punto de encuentro. Se escriben
  con `src.contracts.write_contract`, que valida nombres y tipos.
- Las reglas de operación viven en `config/operacion.yaml`, no en el código. Se
  responden en lenguaje normal en `docs/reglas_operacion.md`.
- Los datos pesados no van en Git.

## Lo que este proyecto no afirma

Los datos solo registran la demanda que sí se atendió: un viaje que no ocurrió
porque la estación estaba vacía no aparece. Por eso hablamos de **posible
demanda insatisfecha** y de **riesgo simulado evitado**, no de viajes
recuperados. Si algo no se pudo correr o descargar, se dice.
