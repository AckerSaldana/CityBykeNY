"""Registro de constructores de variables (features).

Un "constructor" es una función que recibe la tabla station_hour y devuelve
variables para el modelo. Reglas de forma:

- Devuelve un DataFrame con EXACTAMENTE una fila por cada fila que recibió.
- Conserva las dos columnas llave: station_id y ts_hour.
- Las demás columnas son las variables que calcula.

Cómo agregar uno nuevo (copia src/features/lags.py como plantilla):

    from src.features.registry import register

    @register("mi_variable")
    def mi_variable(station_hour):
        ...
        return tabla_con_llaves_y_variables

Todo constructor registrado queda cubierto por tests/test_no_leakage.py sin
hacer nada más: esa prueba recorre BUILDERS y revisa que ninguno use
información del mismo día ni del futuro (decisión 5 de CLAUDE.md).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import pandas as pd

# Columnas que identifican una fila. Todo constructor las devuelve tal cual.
KEY_COLUMNS = ["station_id", "ts_hour"]

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]

# nombre -> función. Se llena solo, cuando se importa cada archivo de src/features.
BUILDERS: dict[str, FeatureBuilder] = {}


def register(name: str) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Decorador: anota la función en BUILDERS con el nombre dado."""

    def decorator(builder: FeatureBuilder) -> FeatureBuilder:
        previous = BUILDERS.get(name)
        # Volver a ejecutar el mismo archivo o la misma celda de un notebook no
        # es un error. Dos constructores distintos con el mismo nombre sí.
        same = previous is not None and (previous.__module__, previous.__qualname__) == (
            builder.__module__,
            builder.__qualname__,
        )
        if previous is not None and not same:
            raise ValueError(
                f"Ya hay un constructor registrado como {name!r} "
                f"({previous.__module__}.{previous.__qualname__}). Usa otro nombre."
            )
        BUILDERS[name] = builder
        return builder

    return decorator


def load_builders() -> dict[str, FeatureBuilder]:
    """Importa todos los archivos de src/features y devuelve BUILDERS completo.

    Un constructor solo se registra cuando alguien importa su archivo. Esta
    función los importa todos, así ninguno se queda sin revisar por olvido.
    Recorre también las subcarpetas (por ejemplo src/features/externas/),
    tengan o no un archivo __init__.py.
    """
    import src.features as package

    for folder in package.__path__:
        for path in sorted(Path(folder).rglob("*.py")):
            parts = path.relative_to(folder).with_suffix("").parts
            # Carpetas ocultas (.ipynb_checkpoints): copias que deja Jupyter, no son código del proyecto.
            if any(part.startswith(".") for part in parts):
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]  # el __init__.py de una subcarpeta se importa como la subcarpeta
            if parts:
                importlib.import_module(".".join([package.__name__, *parts]))
    return BUILDERS
