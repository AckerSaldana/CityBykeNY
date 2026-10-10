# Tipo de estación contra empleos y residentes cercanos

Pregunta: el tipo de estación se deduce del patrón de uso (decisión 13).
¿Coincide con los empleos y los residentes que hay alrededor de cada estación?

Se reproduce con `python scripts/build_station_context.py` (unos 12 a 22
segundos; con `--osm` intenta además la consulta a OpenStreetMap).

## Datos usados

Fuente: LODES 8 de la Oficina del Censo de Estados Unidos (programa LEHD),
estado de Nueva York. Los nombres se leyeron del listado del servidor
(`https://lehd.ces.census.gov/data/lodes/LODES8/ny/`, `wac/` y `rac/`); el año
más reciente presente en ambas carpetas es **2023**.

| Archivo | Qué trae | Tamaño |
|---|---|---|
| `ny_wac_S000_JT00_2023.csv.gz` | Empleos por manzana donde está el trabajo | 2.8 MB |
| `ny_rac_S000_JT00_2023.csv.gz` | Trabajadores por manzana donde viven | 6.3 MB |
| `ny_xwalk.csv.gz` | Latitud y longitud del punto interior de cada manzana de 2020 | 5.2 MB |

Todos los empleos (JT00), todos los segmentos (S000), columna C000. Quedan en
`data/raw/lodes/` (no van en Git).

Advertencias:

- "Residentes" son **trabajadores residentes**: personas con empleo, contadas
  en la manzana donde viven. No es población total: no incluye niños,
  jubilados, estudiantes sin empleo ni personas sin trabajo.
- La Oficina del Censo **agrega ruido a propósito** a estos conteos para
  proteger la confidencialidad. Las cifras por manzana no son exactas.
- Los datos son de **2023**; los viajes son de julio y agosto de **2026**. Hay
  tres años de diferencia.
- LODES cubre empleos con seguro de desempleo y empleo federal civil; no
  cubre a quienes trabajan por cuenta propia.
- Solo se usó el archivo de Nueva York. Las 2,398 estaciones tienen al menos
  una manzana de Nueva York a 300 metros.

## Método

- Para cada estación de `stations.parquet` se suman los empleos y los
  trabajadores residentes de las manzanas cuyo punto interior queda a 300
  metros o menos (distancia haversine). Antes se recortan las manzanas a la
  caja de las estaciones más 0.01 grados: quedan 24,420 manzanas, con
  4,134,324 empleos y 3,079,160 trabajadores residentes.
- `proporcion_empleos` = empleos / (empleos + residentes); nula si ambos son 0.
- Una manzana cuenta completa si su punto interior cae dentro del radio y no
  cuenta si cae fuera; no se reparte por área.
- El patrón de uso (índice de la mañana, tipo) sale de correr
  `notebooks/b/tipo_estacion.sql` tal cual, que usa solo el tramo de
  entrenamiento (1 de julio al 31 de agosto de 2026, decisión 4). El tipo que
  entrega coincide en las 2,398 estaciones con el de `stations.parquet`.
- Índice de la mañana (`am_indice`): (llegadas − salidas) / (llegadas +
  salidas), entre semana de 7 a 10. Positivo: llegan más bicis de las que salen.

Resultado: `data/processed/station_context.parquet` (2,398 filas; 2,396 con
proporción, 2 con cero empleos y cero residentes).

## Resultados

### Proporción de empleos contra índice de la mañana

| Estaciones | n | Pearson | Spearman |
|---|---|---|---|
| Todas las que tienen ambos datos | 2,350 | 0.57 | 0.53 |
| Sin las de poca actividad (menos de 5 movimientos por día) | 2,296 | 0.60 | 0.55 |

La relación es positiva: donde hay más empleos que trabajadores residentes,
en la mañana tienden a llegar más bicis de las que salen. Una correlación de
0.57 corresponde a un 32% de la variación del índice asociada linealmente con
la proporción (0.57²); el resto no lo explica esta variable.

Dentro de cada tipo la relación se mantiene, con menor valor (Spearman):
oficinas 0.39 (n 297), residencial 0.36 (n 548), mixta 0.45 (n 1,465),
recreativa 0.44 (n 40).

### Medianas por tipo de estación

| Tipo | Estaciones | Empleos a 300 m (mediana) | Trabajadores residentes a 300 m (mediana) | Proporción de empleos (mediana) |
|---|---|---|---|---|
| Oficinas | 297 | 10,432 | 2,418 | 0.87 |
| Residencial | 548 | 1,357.5 | 3,545 | 0.29 |
| Recreativa | 42 (40 con proporción) | 1,139.5 | 2,207.5 | 0.42 |
| Mixta | 1,469 | 1,216 | 2,837 | 0.32 |

42 estaciones no tienen tipo (sin viajes en el tramo de entrenamiento) y no
entran en la tabla.

Rango entre cuartiles de la proporción: oficinas 0.59 a 0.94; residencial 0.19
a 0.42; mixta 0.20 a 0.54; recreativa 0.21 a 0.73.

Coincidencia con un corte simple de 0.5:

- 79% de las estaciones de oficinas (236 de 297) tienen más empleos que
  trabajadores residentes; 21% no.
- 87% de las residenciales (476 de 548) tienen más trabajadores residentes
  que empleos; 13% no.
- 29% de las mixtas tienen más empleos que trabajadores residentes.

### Lectura

- El tipo "oficinas" coincide con el entorno: su mediana de empleos es 7.7
  veces la de las residenciales y su proporción mediana es 0.87 contra 0.29.
- "Residencial" y "mixta" casi no se distinguen por el entorno: proporción
  mediana 0.29 contra 0.32 y rangos entre cuartiles que se traslapan. La
  diferencia entre esos dos tipos está en el patrón de uso, no en estos
  conteos.
- "Recreativa" (42 estaciones) queda en medio y con un rango amplio; empleos y
  trabajadores residentes no son la variable que la define.
- Una de cada cinco estaciones de oficinas está en un entorno con más
  trabajadores residentes que empleos.
- Es una asociación entre dos mediciones; no prueba que los empleos causen el
  flujo.

## OpenStreetMap (paso opcional)

**No se hizo.** La consulta de parques, atracciones turísticas y universidades
se intentó el 10 de octubre de 2026 y los servidores de Overpass no
respondieron: `overpass-api.de` devolvió 504 (dos intentos) y el espejo
`maps.mail.ru` devolvió 504. No hay conteos ni correlaciones con la proporción
de usuarios ocasionales ni con la de fin de semana. El código queda listo
(`--osm`): si falla, el script lo dice y termina lo demás.

## Qué no se verificó

- No se comparó con población total (censo decenal o ACS); solo con
  trabajadores residentes.
- No se probó otro radio (150 o 500 metros) ni repartir manzanas por área.
- No se midió el efecto del ruido del Censo sobre las sumas a 300 metros.
- No se revisó si el entorno cambió entre 2023 y 2026 (edificios nuevos,
  oficinas vacías).
- Los percentiles y cortes del tipo de estación no se revisaron aquí; se
  tomaron de la consulta como están.

## Archivos

- `src/data/station_context.py`, `scripts/build_station_context.py`
- `data/processed/station_context.parquet`
- `data/processed/app/contexto_estaciones.parquet` y `app/seccion_contexto.py`
  (función `mostrar()`)
- `data/processed/reports/contexto_estaciones.json` (los números de este
  documento)
- `tests/test_station_context.py`
