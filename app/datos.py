"""Lectura de datos para el dashboard.

Este es el ÚNICO archivo que sabe dónde viven los datos. Las vistas nunca abren
archivos por su cuenta: piden las tablas a las funciones de aquí.

Regla para cada tabla:

1. Si su archivo existe en data/processed (datos reales del flujo), se usa ese.
2. Si no, se usa el de data/sample (datos SINTÉTICOS, inventados por un programa).
3. Si no está en ninguna de las dos carpetas, la página muestra un mensaje que
   dice cómo generar la muestra y se detiene. No aparece un error de Python.

Mientras ALGUNA tabla venga de data/sample, cada página lo dice:

- La página que lee al menos una tabla sintética muestra el aviso completo.
- La página cuyas tablas son todas reales muestra una nota corta: dice que
  usa solo tablas reales y cuáles siguen sintéticas en otras vistas.

Ni el aviso ni la nota se quitan a mano: desaparecen solos cuando los
archivos reales llegan a data/processed.

OJO con la mezcla: si unas tablas ya son reales y otras siguen en la muestra,
sus estaciones no coinciden (las de la muestra se llaman SIM-...). Al cruzar
una tabla real con una sintética el resultado sale vacío. El aviso lo explica.

Dos tablas no salen de viajes observados: `recommendations` y `policy_curve`
salen de la SIMULACIÓN de inventario (python scripts/run_simulation.py).
Cuando sus archivos reales están en data/processed ya no hay aviso de datos
sintéticos, pero siguen sin ser mediciones. Por eso la página que las lee
muestra una nota: "RESULTADOS DE UNA SIMULACIÓN: no son mediciones". Tampoco
se quita a mano. Mira show_simulation_note(), más abajo.

Este archivo no importa nada de src/. Solo usa streamlit y pandas, para que la
app siga abriendo aunque cambie el resto del repositorio.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Las carpetas se ubican a partir de este archivo, no desde la carpeta en la
# que se abrió la terminal. Así la app funciona igual desde cualquier lugar.
ROOT = Path(__file__).resolve().parents[1]    # carpeta raíz del repositorio
DATA_PROCESSED = ROOT / "data" / "processed"  # datos reales
DATA_SAMPLE = ROOT / "data" / "sample"        # datos SINTÉTICOS de muestra

# Las seis tablas del proyecto: nombre -> qué contiene.
# El archivo de cada una se llama <nombre>.parquet.
TABLES = {
    "stations": "Estaciones: nombre, ubicación, capacidad y tipo",
    "station_hour": "Salidas y llegadas observadas, por estación y hora",
    "predictions": "Pronóstico de salidas y llegadas, por estación y hora",
    "recommendations": "Bicis que movió cada política en la simulación, por estación, día y franja horaria",
    "policy_curve": "Horas de estación en riesgo simulado por día, por política y presupuesto diario",
    "routes": "Viajes de usuarios entre pares de estaciones, sumados sobre el periodo",
}

SYNTHETIC_WARNING = "DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"
REAL_ONLY_NOTE = "Esta vista usa solo tablas reales"
MAKE_SAMPLE_COMMAND = "python scripts/make_sample_data.py"

# Las dos tablas que salen de la simulación de inventario. Las otras cuatro
# salen de viajes que sí ocurrieron o del pronóstico.
SIMULATED_TABLES = ["recommendations", "policy_curve"]
SIMULATION_NOTE = "RESULTADOS DE UNA SIMULACIÓN: no son mediciones"
# Reporte que la simulación escribe junto a sus tablas, dentro de data/processed.
SIMULATION_REPORT = "reports/simulacion.json"

# Frases de la nota de simulación. Cada diccionario va de lo que dice el
# reporte de la corrida a la frase que sale en pantalla.
RULES_TEXT = {
    "provisional": (
        "Las reglas de operación de esa corrida son **provisionales**: estos "
        "números cambian cuando logística las confirme."
    ),
    "confirmado": "Las reglas de operación de esa corrida ya estaban confirmadas por logística.",
}
FORECAST_TEXT = {
    "baseline": (
        "La política `model` se alimentó del pronóstico de la línea base "
        "(promedio histórico): no es un modelo entrenado."
    ),
    "lgbm": "La política `model` se alimentó del pronóstico de LightGBM.",
}

MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


# --------------------------------------------------------------------------- #
# De dónde sale cada tabla
# --------------------------------------------------------------------------- #

def find_file(table: str) -> Path | None:
    """Archivo de una tabla: primero data/processed, después data/sample.

    Devuelve None si no está en ninguna de las dos carpetas.
    """
    for folder in (DATA_PROCESSED, DATA_SAMPLE):
        path = folder / f"{table}.parquet"
        if path.is_file():
            return path
    return None


def synthetic_tables() -> list[str]:
    """Nombres de las tablas que hoy se leen de data/sample (sintéticas)."""
    return [table for table in TABLES if find_file(table) == DATA_SAMPLE / f"{table}.parquet"]


def missing_tables() -> list[str]:
    """Nombres de las tablas cuyo archivo no está en ninguna de las dos carpetas."""
    return [table for table in TABLES if find_file(table) is None]


def show_synthetic_warning(used: list[str] | None = None) -> None:
    """Aviso visible mientras alguna tabla venga de data/sample.

    Hay que llamarla al inicio de TODAS las páginas.

    `used` es la lista de las tablas que lee la vista, por ejemplo
    ["stations", "station_hour"]. Según esa lista, la página muestra:

    - El aviso completo, si la vista lee al menos una tabla sintética.
    - Una nota corta, si las tablas de la vista son todas reales pero quedan
      tablas sintéticas en otras vistas.
    - Nada, si las seis tablas ya son reales.

    Sin lista se supone que la vista lee las seis tablas: sale el aviso
    completo mientras quede alguna sintética. Es lo que usa Inicio.

    OJO: si la vista empieza a leer otra tabla, hay que agregarla a su lista.
    La prueba (pytest tests/test_app_smoke.py) falla si falta.
    """
    tables = synthetic_tables()
    if not tables:
        return
    names = ", ".join(f"`{table}`" for table in tables)
    # ¿La vista lee alguna tabla sintética? Sin lista se supone que sí. Un
    # nombre mal escrito también cuenta como sí: ante la duda, aviso completo.
    uses_synthetic = not used or any(
        table in tables or table not in TABLES for table in used
    )
    if not uses_synthetic:
        # Todas las tablas de esta vista son reales: se dice eso, sin alarmar,
        # y se nombran las que siguen sintéticas en las demás vistas.
        note = f"{REAL_ONLY_NOTE}. Siguen sintéticas en otras vistas: {names}."
        st.info(note)
        st.sidebar.info(note)
        return
    text = (
        f"**{SYNTHETIC_WARNING}.**\n\n"
        f"Tablas que vienen de `data/sample`: {names}.\n\n"
        "Ningún número de estas tablas se puede citar en el pitch. Este aviso "
        "desaparece solo cuando los archivos reales estén en `data/processed`."
    )
    # Si además ya llegó alguna tabla real, hay mezcla. Se explica por qué los
    # cruces entre tablas salen vacíos, para que nadie crea que la vista falla.
    if len(tables) + len(missing_tables()) < len(TABLES):
        text += (
            "\n\n**OJO: hay tablas reales y tablas sintéticas mezcladas.** Las "
            "estaciones de la muestra (`SIM-...`) no existen en los datos reales. "
            "Al cruzar una tabla real con una sintética no coincide ninguna "
            "estación y el resultado sale vacío. No es un error de la vista: se "
            "arregla solo cuando llegue el resto de los archivos reales."
        )
    st.warning(text)
    # También en el menú de la izquierda, que sigue a la vista al bajar la
    # página. Ahí también se nombran las tablas sintéticas.
    st.sidebar.warning(f"{SYNTHETIC_WARNING}. Tablas: {names}.")


# --------------------------------------------------------------------------- #
# Nota de las tablas que salen de la simulación
# --------------------------------------------------------------------------- #

def fecha_en_texto(day) -> str:
    """Una fecha escrita en español, por ejemplo "2 de septiembre de 2026"."""
    return f"{day.day} de {MONTHS[day.month - 1]} de {day.year}"


def periodo_en_texto(first, last) -> str:
    """Un periodo escrito en español, por ejemplo "del 2 al 14 de septiembre de 2026"."""
    if (first.year, first.month, first.day) == (last.year, last.month, last.day):
        return f"el {fecha_en_texto(first)}"  # un solo día
    if (first.year, first.month) == (last.year, last.month):
        return f"del {first.day} al {fecha_en_texto(last)}"
    return f"del {fecha_en_texto(first)} al {fecha_en_texto(last)}"


def simulation_run() -> dict:
    """Dos datos de la corrida de la simulación que NO vienen en las tablas.

    Salen del reporte que la simulación escribe junto a sus tablas
    (data/processed/reports/simulacion.json):

    - "rules": "provisional" o "confirmado". Es el estado que tenían las
      reglas de operación (config/operacion.yaml) cuando se simuló.
    - "forecast": "baseline" o "lgbm". Es el pronóstico que alimentó a la
      política "model".

    Si el reporte no está o no se puede leer, los dos valen None y la nota
    dice que no se sabe. Nunca se supone.
    """
    try:
        report = json.loads((DATA_PROCESSED / SIMULATION_REPORT).read_text(encoding="utf-8"))
        return {
            "rules": report["reglas"]["estado"],
            "forecast": report["pronostico"]["modelo_que_alimenta_model"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {"rules": None, "forecast": None}


def show_simulation_note(used: list[str] | None = None) -> None:
    """Nota visible cuando la vista lee una tabla real que sale de la simulación.

    Hay que llamarla en TODAS las páginas, justo después del aviso de datos
    sintéticos y con la misma lista de tablas.

    Las tablas simuladas son `recommendations` y `policy_curve`. La página
    muestra:

    - La nota, si la vista lee alguna de esas dos tablas y su archivo está en
      data/processed. La nota nombra solo esas.
    - Nada, si la vista no lee ninguna de las dos.
    - Nada, si las que lee vienen de data/sample: ahí manda el aviso de datos
      sintéticos, que ya dice que no son resultados reales.

    Sin lista se supone que la vista lee las seis tablas. Es lo que usa Inicio.

    Los días, los cuenta de la tabla. El estado de las reglas y el pronóstico
    que alimentó a la política "model" los lee del reporte de la corrida. Así
    la nota sigue diciendo la verdad cuando alguien vuelve a simular.
    """
    # Tablas simuladas cuyo archivo real ya está en data/processed.
    real = [table for table in SIMULATED_TABLES if find_file(table) == DATA_PROCESSED / f"{table}.parquet"]
    # ¿Cuáles lee la vista? Sin lista se supone que las seis tablas. Un nombre
    # mal escrito también cuenta como las seis: ante la duda, se muestra la nota.
    reads_all = not used or any(table not in TABLES for table in used)
    tables = [table for table in real if reads_all or table in used]
    if not tables:
        return
    names = ", ".join(f"`{table}`" for table in tables)
    text = (
        f"**{SIMULATION_NOTE}.**\n\n"
        f"Tablas simuladas que lee esta vista: {names}. Salen de simular el "
        "inventario de las estaciones en días de validación; la prueba final no "
        "se simula (decisión 4)."
    )
    if "recommendations" in tables:
        # El día de cada recomendación es el día en que empieza su franja.
        days = load_recommendations()["window_start"].dt.normalize()
        if len(days) > 0:
            text += (
                f" `recommendations` cubre {days.nunique()} días: "
                f"{periodo_en_texto(days.min(), days.max())}."
            )
    # Lo que las tablas no traen se lee del reporte de la corrida.
    run = simulation_run()
    if run["rules"] is None:
        text += (
            f"\n\nNo se pudo leer el reporte de la corrida (`data/processed/{SIMULATION_REPORT}`): "
            "no se sabe si sus reglas de operación eran provisionales ni qué pronóstico "
            "alimentó a la política `model`."
        )
    else:
        # Si el reporte trae un valor que no está en los diccionarios, se
        # muestra tal cual.
        rules = RULES_TEXT.get(run["rules"], f"Estado de las reglas de operación de esa corrida: `{run['rules']}`.")
        forecast = FORECAST_TEXT.get(
            run["forecast"], f"La política `model` se alimentó del pronóstico `{run['forecast']}`."
        )
        text += f"\n\n{rules} {forecast}"
    # Lenguaje honesto (decisión 9 del equipo).
    text += (
        "\n\nEstas tablas hablan de *riesgo simulado* y, contra no intervenir, de "
        "*riesgo simulado evitado*. Los viajes bloqueados de la simulación son "
        "simulados: no son demanda insatisfecha medida."
    )
    st.info(text)
    # También en el menú de la izquierda, que sigue a la vista al bajar la página.
    st.sidebar.info(f"{SIMULATION_NOTE}. Tablas: {names}.")


# --------------------------------------------------------------------------- #
# Enlaces a las vistas
# --------------------------------------------------------------------------- #

def page_link(file: str, label: str) -> None:
    """Enlace a una vista. Si el archivo ya no existe, avisa en lugar de fallar.

    `file` es la ruta de la vista desde la carpeta app/, por ejemplo
    "pages/1_Mapa_de_demanda.py".
    """
    if (ROOT / "app" / file).is_file():
        st.page_link(file, label=label)
    else:
        st.error(
            f"No encuentro la vista `app/{file}`. Si le cambiaste el nombre al "
            "archivo, actualiza este enlace en `app/Inicio.py`."
        )


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Leyendo datos...")
def _read_parquet(path: str, modified: float) -> pd.DataFrame:
    """Lee un archivo Parquet. Streamlit guarda el resultado y no lo relee en cada clic.

    `modified` es la fecha de modificación del archivo. No se usa aquí adentro:
    sirve para que el archivo se vuelva a leer si alguien lo reemplaza. Como la
    ruta también cuenta, al llegar el archivo real se deja de usar la muestra.
    """
    return pd.read_parquet(path)


def _load(table: str) -> pd.DataFrame:
    """Devuelve una tabla. Si su archivo no existe, avisa y detiene la página."""
    path = find_file(table)
    if path is None:
        missing = ", ".join(f"`{name}.parquet`" for name in missing_tables())
        message = (
            f"No se encontró el archivo de la tabla `{table}`.\n\n"
            f"Archivos que faltan en `data/processed` y en `data/sample`: {missing}.\n\n"
            "Para generar los datos sintéticos de muestra, corre este comando en "
            f"la terminal, desde la carpeta del proyecto: `{MAKE_SAMPLE_COMMAND}`\n\n"
            "Después recarga esta página."
        )
        st.error(message)
        st.stop()
        # Dentro de Streamlit nunca se llega aquí: st.stop() detiene la página.
        # Fuera de Streamlit (un notebook, una prueba) st.stop() no hace nada.
        raise FileNotFoundError(message)
    # Cada llamada entrega una copia: una vista puede modificar su tabla sin
    # afectar a las demás vistas.
    return _read_parquet(str(path), path.stat().st_mtime)


# Una función por tabla. Las columnas de cada una están en app/LEEME.md.

def load_stations() -> pd.DataFrame:
    """Estaciones: una fila por estación.

    OJO: `capacity` y `station_type` pueden venir vacíos.
    """
    stations = _load("stations")
    # capacity es un número entero con vacíos. pandas lo leería como decimal
    # (59.0). Con "Int64" queda como entero y los vacíos se ven como <NA>.
    stations["capacity"] = stations["capacity"].astype("Int64")
    return stations


def load_station_hour() -> pd.DataFrame:
    """Salidas y llegadas observadas: una fila por estación y hora.

    `ts_hour` es el inicio de la hora, en hora local de Nueva York, con zona
    horaria. Son viajes que sí ocurrieron: solo demanda atendida.
    Para el eje de una gráfica, mira hora_para_grafica() más abajo.

    OJO: cada estación tiene filas solo entre su primer y su último viaje.
    Dentro de ese tramo están todas las horas; las que no tuvieron viajes
    valen cero. Fuera de ese tramo no hay filas. Por eso no todas las
    estaciones empiezan ni terminan el mismo día.
    """
    return _load("station_hour")


def load_predictions() -> pd.DataFrame:
    """Pronóstico: una fila por estación, hora, modelo y periodo (val o test).

    `ts_hour` es hora local de Nueva York, con zona horaria.
    OJO: son periodos de evaluación que ya pasaron, no horas futuras. Es lo
    que el modelo esperaba para esos días.

    OJO, decisión 4 del equipo (la prueba final se usa una sola vez):
    lo observado se compara SOLO con las filas de `split == "val"`
    (validación). Las filas de `split == "test"` (prueba final) se muestran
    como pronóstico, sin lo observado al lado y sin calcular su error, hasta
    la evaluación final del proyecto.

    Con `model == "baseline"`, `q50` es el promedio histórico de esa estación,
    día de la semana y hora. No es la mediana.
    """
    return _load("predictions")


def load_recommendations() -> pd.DataFrame:
    """Recomendaciones: una fila por estación, día, franja horaria y política.

    Salen de la simulación de inventario, corrida con el presupuesto de
    referencia. No son mediciones. El día de cada fila es el día de
    `window_start`. OJO: para mostrar o sumar hay que elegir antes un día y
    una política; si no, la misma estación sale repetida.

    `bikes_to_move` es lo que esa política movió en la simulación: positivo
    es llevar bicis a la estación; negativo es retirarlas. Cero con riesgo
    alto casi siempre quiere decir que el presupuesto no alcanzó, no que la
    estación esté bien.

    `start_inventory_target` son las bicis con las que la estación queda en
    la simulación justo después de ese traslado (las que tenía más
    `bikes_to_move`). No es el inventario ideal.

    `risk_level` se calcula con el pronóstico y con el inventario de ANTES
    del traslado. `feasible` en falso significa que, según el pronóstico,
    ningún inventario inicial evita a la vez que la estación se vacíe y que
    se llene.

    OJO: no todas las estaciones tienen filas. La simulación deja fuera a las
    que no tienen capacidad registrada y a las que no tuvieron viajes en los
    días simulados.
    """
    return _load("recommendations")


def load_policy_curve() -> pd.DataFrame:
    """Curva de riesgo contra presupuesto: una fila por política y presupuesto diario.

    Sale de la simulación de inventario. No son mediciones.

    `budget` es el total de bicis que se pueden trasladar en un día en toda
    la red. `risk_station_hours` son las horas de estación en riesgo simulado
    POR DÍA: el promedio de los días evaluados, contando las 24 horas.
    `ci_low` y `ci_high` son un intervalo que sale de sortear esos mismos
    días. Los días no son independientes: es indicativo, no una garantía.

    La diferencia entre una política y `none` con el mismo presupuesto es
    riesgo simulado evitado.
    """
    return _load("policy_curve")


def load_routes() -> pd.DataFrame:
    """Viajes entre pares de estaciones: una fila por estación de inicio y estación de fin.

    `trips` es cuántos viajes hubo de la primera a la segunda en todo el
    periodo. De A a B y de B a A son dos filas distintas.
    OJO: son viajes de usuarios, NO rutas de camiones de rebalanceo. Ningún
    archivo dice de qué estación a cuál conviene mover bicis.
    El inicio y el fin pueden ser la misma estación (el usuario devolvió la
    bici donde la tomó). Los pares sin ningún viaje no tienen fila.
    Son viajes que sí ocurrieron: solo demanda atendida.
    """
    return _load("routes")


# --------------------------------------------------------------------------- #
# Horas en el eje de una gráfica
# --------------------------------------------------------------------------- #

def hora_para_grafica(columna: pd.Series) -> pd.Series:
    """La misma hora de Nueva York, sin zona horaria. SOLO para el eje de una gráfica.

    Úsala con st.line_chart, st.bar_chart, st.area_chart y altair. Esas
    gráficas se dibujan en el navegador: si la hora trae zona, el navegador la
    convierte a la hora de la computadora y la gráfica sale corrida.
    Las tablas (st.dataframe) y plotly no la necesitan.

        station_hour["hora"] = datos.hora_para_grafica(station_hour["ts_hour"])
        st.line_chart(station_hour, x="hora", y="departures")

    OJO: el día de 25 horas, las dos filas de la 01:00 caen en el mismo punto.
    """
    return columna.dt.tz_localize(None)


# --------------------------------------------------------------------------- #
# Prueba de lectura (ayuda temporal del esqueleto)
# --------------------------------------------------------------------------- #

def describe_simulated(name: str, table: pd.DataFrame) -> str:
    """Una frase que dice qué trae una tabla simulada. Todo se cuenta de la tabla.

    Para las otras tablas devuelve un texto vacío. No lleva ninguna cifra
    escrita a mano: si alguien vuelve a simular con otras reglas, la frase
    cambia sola.
    """
    if len(table) == 0:
        return ""  # sin filas no hay nada que contar
    if name == "recommendations":
        start = table["window_start"]
        days = start.dt.normalize()  # el día en que empieza la franja
        # Franjas por día: cuántas horas de inicio distintas hay.
        windows = (start.dt.hour * 60 + start.dt.minute).nunique()
        return (
            " Una fila por estación, día, franja horaria y política: "
            f"{table['station_id'].nunique()} estaciones, {days.nunique()} días "
            f"({periodo_en_texto(days.min(), days.max())}), "
            f"{windows} franjas por día y {table['policy'].nunique()} políticas."
        )
    if name == "policy_curve":
        budgets = ", ".join(str(budget) for budget in sorted(table["budget"].unique()))
        return (
            " Una fila por política y presupuesto diario: "
            f"{table['policy'].nunique()} políticas y {table['budget'].nunique()} "
            f"presupuestos ({budgets} bicis al día en toda la red)."
        )
    return ""


def show_data_check(tables: dict[str, pd.DataFrame]) -> None:
    """Muestra cuántas filas se leyeron de cada tabla y una vista previa plegada.

    `tables` es un diccionario nombre -> tabla ya leída. Solo sirve para
    comprobar que los datos llegan a la vista. Se puede quitar de una vista
    cuando ya tenga sus gráficas.

    En las dos tablas simuladas dice además qué trae cada una (cuántas
    estaciones, días, franjas, políticas y presupuestos).
    """
    synthetic = synthetic_tables()
    lines = []
    for name, table in tables.items():
        origin = "SINTÉTICA, de `data/sample`" if name in synthetic else "real, de `data/processed`"
        lines.append(f"- `{name}`: {len(table)} filas leídas ({origin}).{describe_simulated(name, table)}")
    st.markdown("\n".join(lines))
    for name, table in tables.items():
        with st.expander(f"Ver las primeras filas de {name}"):
            st.dataframe(table.head(5), hide_index=True)
