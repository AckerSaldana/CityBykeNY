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
| `app/pages/2_Ranking_de_estaciones_y_rutas.py` | Vista 2: qué estaciones se atienden primero y cuántas bicis se mueven. |
| `app/pages/3_Pronostico_por_estacion.py` | Vista 3: qué se esperaba en una estación en un día ya evaluado y qué convenía hacer. |
| `app/datos.py` | Lee las tablas. Es el único archivo que sabe dónde están los datos. |
| `app/LEEME.md` | Este documento. |
| `app/.streamlit/config.toml` | Configuración mínima de Streamlit. Va junto a `Inicio.py` para que se lea desde cualquier carpeta. |
| `tests/test_app_smoke.py` | Prueba que la app abre y que avisa de los datos sintéticos. |

El menú de la izquierda lo arma Streamlit con los archivos de `app/pages/`. El
número al inicio del nombre da el orden. El resto del nombre es el texto del
menú, con espacios en lugar de guiones bajos. Por eso la vista 3 sale en el
menú sin acentos. Para cambiar el texto del menú se cambia el nombre del
archivo; el nombre puede llevar acentos.

Si agregas una vista (un archivo nuevo en `app/pages/`), la prueba la revisa
sola: comprueba que abre y que muestra el aviso de datos sintéticos. Solo
falta agregar su enlace en `app/Inicio.py`.

Si cambias el nombre de un archivo de `app/pages/`, actualiza dos lugares: su
enlace en `app/Inicio.py` y su nombre en la lista `PAGES` de
`tests/test_app_smoke.py`. Si se te olvida el enlace, Inicio muestra un mensaje
en rojo que dice cuál vista no encuentra.

## De dónde salen los datos

Para cada tabla, `app/datos.py` busca así:

1. Si el archivo está en `data/processed`, usa ese. Son los datos reales.
2. Si no, usa el de `data/sample`. Son **datos sintéticos**, inventados por un
   programa para poder construir la app.
3. Si no está en ninguna de las dos carpetas, la página muestra un mensaje que
   dice cómo generar la muestra: `python scripts/make_sample_data.py`.

Mientras alguna tabla venga de `data/sample`, todas las páginas muestran el
aviso **"DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"** con el nombre
de esas tablas. El aviso no se quita a mano. Desaparece solo cuando los
archivos reales llegan a `data/processed`. No hay que cambiar nada en `app/`
para que eso pase.

**OJO con la mezcla.** Los archivos reales llegan de uno en uno. Mientras unas
tablas sean reales y otras sigan en la muestra, sus estaciones no coinciden:
las de la muestra se llaman `SIM-...` y no existen en los datos reales. Al
cruzar una tabla real con una sintética no coincide ninguna estación y el
resultado sale vacío (por ejemplo, la vista 3 dice "0 de pronóstico"). No es un
error de la vista: se arregla solo cuando llegue el resto de los archivos
reales. El aviso lo dice cuando pasa.

Ningún número que salga de `data/sample` se puede citar en el pitch.

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
| `station_type` | Tipo: `oficinas`, `residencial`, `recreativa` o `mixta`. **Puede venir vacío.** |

### `station_hour` — `datos.load_station_hour()`

Una fila por estación y hora. Es lo que se observó.

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

### `recommendations` — `datos.load_recommendations()`

Una fila por estación, franja horaria y política.

| Columna | Qué es |
|---|---|
| `station_id` | Identificador de la estación. |
| `window_start`, `window_end` | Inicio y fin de la franja, en hora local de Nueva York, con zona horaria. |
| `policy` | `none` (no intervenir), `historical` (regla histórica) o `model` (modelo predictivo). |
| `start_inventory_target` | Bicis con las que conviene que la estación empiece la franja. |
| `bikes_to_move` | Bicis a mover. **Positivo es llevar bicis a la estación. Negativo es retirarlas.** |
| `risk_level` | `alto`, `medio` o `bajo`. |
| `feasible` | Verdadero o falso. **Falso: ningún inventario inicial evita a la vez que la estación se vacíe y que se llene.** |

### `policy_curve` — `datos.load_policy_curve()`

Una fila por política y presupuesto diario.

| Columna | Qué es |
|---|---|
| `policy` | `none`, `historical` o `model`. |
| `budget` | Presupuesto: bicis que se pueden trasladar en un día. |
| `risk_station_hours` | Horas de estación en riesgo (vacía o llena en la simulación). |
| `ci_low`, `ci_high` | Límite bajo y alto del intervalo de confianza de ese valor. |

## Lo que hay que saber de los datos

- **No hay datos de rutas.** `recommendations` dice cuántas bicis recibe o
  entrega cada estación. No dice de cuál estación salen ni a cuál van. Ningún
  archivo lo dice todavía. Es una decisión pendiente del equipo. No se inventa
  un archivo de rutas en la app.
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
  esos días, para compararlo con lo observado. No es un pronóstico en vivo.
- En la muestra, las filas con `model = "lgbm"` son un relleno simulado. No
  salen de ningún modelo entrenado.
- Con `policy = "none"` nunca se mueven bicis.

## Lenguaje honesto

Los datos solo registran **demanda atendida**: viajes que sí ocurrieron. Si una
estación estaba vacía, no queda registro de quien no encontró bici.

- Sí se dice: "posible demanda insatisfecha" y "riesgo simulado evitado".
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
3. "En la vista 2 muestra una tabla de estaciones ordenada de mayor a menor
   riesgo, para el día, la franja y la política que yo elija. Muestra el nombre
   de la estación, no solo el identificador."
4. "En la vista 2 separa en dos tablas las estaciones a las que hay que llevar
   bicis y las estaciones de las que hay que retirar bicis."
5. "En la vista 2 dibuja la curva de riesgo contra presupuesto, con una línea
   por política y una banda entre `ci_low` y `ci_high`. Debajo explica en una
   frase que la diferencia contra no intervenir es riesgo simulado evitado."
6. "En la vista 3, para la estación elegida, dibuja las salidas esperadas por
   hora con una banda entre `dep_q10` y `dep_q90`, y encima las salidas
   observadas. Agrega un selector de modelo y otro de fechas."
7. "En la vista 3 escribe la acción sugerida en una frase por cada franja del
   día elegido: llevar bicis si `bikes_to_move` es positivo, retirar bicis si
   es negativo y no mover si es cero. Si `feasible` es falso, agrega una
   advertencia."
8. "En todas las vistas, cuando falte la capacidad o el tipo de una estación,
   muestra 'Sin dato' en lugar de dejar el espacio vacío."

Al final de cada instrucción conviene agregar: "Deja los comentarios en
español, no quites el aviso de datos sintéticos y corre
`pytest tests/test_app_smoke.py` al terminar."

## Reglas de esta carpeta

- `app/` no importa nada de `src/`. Hoy solo usa `streamlit` y `pandas`.
- Para mapas y gráficas ya están instalados `pydeck`, `altair` y `plotly`. No
  hace falta instalar nada más.
- Las vistas no abren archivos. Piden las tablas a `app/datos.py`.
- El aviso de datos sintéticos va en todas las páginas, también en las nuevas:
  `datos.show_synthetic_warning()`. La prueba revisa todas las vistas de
  `app/pages/` y falla si a alguna le falta.
- No se escriben en el código horas de franjas, umbrales ni presupuestos. Las
  franjas y los presupuestos salen de las tablas (`window_start`, `window_end`,
  `budget`). Las reglas de operación viven en `config/operacion.yaml`.
