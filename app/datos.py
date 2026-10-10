"""Lectura de datos para el dashboard.

Este es el ÚNICO archivo que sabe dónde viven los datos. Las vistas nunca abren
archivos por su cuenta: piden las tablas a las funciones de aquí.

Regla para cada tabla:

1. Si su archivo existe en data/processed (datos reales del flujo), se usa ese.
2. Si no, se usa el de data/sample (datos SINTÉTICOS, inventados por un programa).
3. Si no está en ninguna de las dos carpetas, la página muestra un mensaje que
   dice cómo generar la muestra y se detiene. No aparece un error de Python.

Mientras ALGUNA tabla venga de data/sample, cada página muestra un aviso
visible. Ese aviso no se quita a mano: desaparece solo cuando los archivos
reales llegan a data/processed.

OJO con la mezcla: si unas tablas ya son reales y otras siguen en la muestra,
sus estaciones no coinciden (las de la muestra se llaman SIM-...). Al cruzar
una tabla real con una sintética el resultado sale vacío. El aviso lo explica.

Este archivo no importa nada de src/. Solo usa streamlit y pandas, para que la
app siga abriendo aunque cambie el resto del repositorio.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# Las rutas se arman desde la ubicación de este archivo, no desde la carpeta en
# la que se abrió la terminal. Así la app funciona igual desde cualquier lugar.
ROOT = Path(__file__).resolve().parents[1]    # carpeta raíz del repositorio
DATA_PROCESSED = ROOT / "data" / "processed"  # datos reales
DATA_SAMPLE = ROOT / "data" / "sample"        # datos SINTÉTICOS de muestra

# Las cinco tablas del proyecto: nombre -> qué contiene.
# El archivo de cada una se llama <nombre>.parquet.
TABLES = {
    "stations": "Estaciones: nombre, ubicación, capacidad y tipo",
    "station_hour": "Salidas y llegadas observadas, por estación y hora",
    "predictions": "Pronóstico de salidas y llegadas, por estación y hora",
    "recommendations": "Bicis a mover, por estación y franja horaria",
    "policy_curve": "Estaciones-hora en riesgo según el presupuesto diario, por política",
}

SYNTHETIC_WARNING = "DATOS SINTÉTICOS DE MUESTRA: no son resultados reales"
MAKE_SAMPLE_COMMAND = "python scripts/make_sample_data.py"


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


def show_synthetic_warning() -> None:
    """Aviso visible mientras alguna tabla venga de data/sample.

    Hay que llamarla al inicio de TODAS las páginas. Si todas las tablas son
    reales no muestra nada.
    """
    tables = synthetic_tables()
    if not tables:
        return
    names = ", ".join(f"`{table}`" for table in tables)
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
    # También en el menú de la izquierda, que sigue a la vista al bajar la página.
    st.sidebar.warning(f"{SYNTHETIC_WARNING}.")


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
    """
    return _load("station_hour")


def load_predictions() -> pd.DataFrame:
    """Pronóstico: una fila por estación, hora, modelo y periodo (val o test).

    `ts_hour` es hora local de Nueva York, con zona horaria.
    OJO: son periodos de evaluación que ya pasaron, no horas futuras. Es lo
    que el modelo esperaba para esos días, para compararlo con lo observado.
    """
    return _load("predictions")


def load_recommendations() -> pd.DataFrame:
    """Recomendaciones: una fila por estación, franja horaria y política.

    `bikes_to_move` positivo es llevar bicis a la estación; negativo es
    retirarlas. `feasible` en falso significa que ningún inventario inicial
    evita a la vez que la estación se vacíe y que se llene.
    """
    return _load("recommendations")


def load_policy_curve() -> pd.DataFrame:
    """Curva de riesgo contra presupuesto: una fila por política y presupuesto diario."""
    return _load("policy_curve")


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

def show_data_check(tables: dict[str, pd.DataFrame]) -> None:
    """Muestra cuántas filas se leyeron de cada tabla y una vista previa plegada.

    `tables` es un diccionario nombre -> tabla ya leída. Solo sirve para
    comprobar que los datos llegan a la vista. Se puede quitar de una vista
    cuando ya tenga sus gráficas.
    """
    synthetic = synthetic_tables()
    lines = []
    for name, table in tables.items():
        origin = "SINTÉTICA, de `data/sample`" if name in synthetic else "real, de `data/processed`"
        lines.append(f"- `{name}`: {len(table)} filas leídas ({origin}).")
    st.markdown("\n".join(lines))
    for name, table in tables.items():
        with st.expander(f"Ver las primeras filas de {name}"):
            st.dataframe(table.head(5), hide_index=True)
