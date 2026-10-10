"""Lectura y validación de config/datos.yaml: periodo, tramos y reglas de limpieza.

La limpieza y los modelos leen de aquí las fechas. Así el periodo y los tres
tramos (entrenamiento, validación y prueba) están en un solo lugar y no
escritos en el código.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.paths import DATOS_YAML, TZ

# Nombre del tramo en el YAML -> nombre en el contrato de predictions.
SPLIT_NAMES = {"validacion": "val", "prueba": "test"}
TRAMOS = ("entrenamiento", "validacion", "prueba")


class DatosError(ValueError):
    """config/datos.yaml tiene un valor faltante o imposible."""


@dataclass(frozen=True)
class Tramo:
    nombre: str
    inicio: date   # primer día, incluido
    fin: date      # último día, incluido

    @property
    def start(self) -> pd.Timestamp:
        """Primer instante del tramo (00:00 del primer día, hora de Nueva York)."""
        return pd.Timestamp(self.inicio, tz=TZ)

    @property
    def end(self) -> pd.Timestamp:
        """Primer instante DESPUÉS del tramo (00:00 del día siguiente al último)."""
        return pd.Timestamp(self.fin + timedelta(days=1), tz=TZ)

    def contains(self, ts_hour: pd.Series) -> pd.Series:
        """Qué filas de una columna de fechas con zona caen en el tramo."""
        return (ts_hour >= self.start) & (ts_hour < self.end)


@dataclass(frozen=True)
class Datos:
    meses_crudos: tuple[str, ...]
    periodo: Tramo
    tramos: dict[str, Tramo]
    distancia_maxima_m: float
    prefijos_excluidos: tuple[str, ...]
    ids_excluidos: tuple[str, ...]


def _tramo(nombre: str, raw, problems: list[str], donde: str = "") -> Tramo | None:
    """`donde` es el prefijo con el que se nombra en los mensajes (p. ej. "tramos.")."""
    etiqueta = f"{donde}{nombre}"
    if not isinstance(raw, dict) or not all(type(raw.get(k)) is date for k in ("inicio", "fin")):
        problems.append(f"{etiqueta}: debe tener inicio y fin como fechas AAAA-MM-DD, sin comillas")
        return None
    if raw["inicio"] > raw["fin"]:
        problems.append(f"{etiqueta}: inicio no puede ser posterior a fin")
        return None
    return Tramo(nombre, raw["inicio"], raw["fin"])


def load_datos(path: str | Path = DATOS_YAML) -> Datos:
    """Lee config/datos.yaml. Lanza DatosError si no es válido."""
    try:
        with open(path, encoding="utf-8") as archivo:
            cfg = yaml.safe_load(archivo)
    except OSError as error:
        raise DatosError(f"No se pudo abrir {path}: {error}") from None
    except yaml.YAMLError as error:
        raise DatosError(f"{path} está mal escrito y no se pudo leer: {error}") from None
    if not isinstance(cfg, dict):
        raise DatosError(f"{path} debe ser un mapa de parámetros")
    problems: list[str] = []

    meses = cfg.get("meses_crudos")
    if not isinstance(meses, list) or not meses or not all(
        isinstance(m, int) and not isinstance(m, bool) and 202102 <= m <= 999912 and 1 <= m % 100 <= 12
        for m in meses
    ):
        problems.append("meses_crudos: debe ser una lista de meses AAAAMM, por ejemplo [202607, 202608]")
        meses = []

    periodo = _tramo("periodo", cfg.get("periodo"), problems)
    tramos: dict[str, Tramo] = {}
    raw_tramos = cfg.get("tramos")
    if not isinstance(raw_tramos, dict) or set(raw_tramos) != set(TRAMOS):
        problems.append(f"tramos: deben ser exactamente {list(TRAMOS)}")
    else:
        for nombre in TRAMOS:
            tramo = _tramo(nombre, raw_tramos[nombre], problems, donde="tramos.")
            if tramo:
                tramos[nombre] = tramo
    if periodo and len(tramos) == len(TRAMOS):
        ordered = [tramos[nombre] for nombre in TRAMOS]
        if ordered[0].inicio != periodo.inicio or ordered[-1].fin != periodo.fin:
            problems.append("tramos: deben empezar y terminar igual que el periodo")
        for before, after in zip(ordered, ordered[1:]):
            if after.inicio != before.fin + timedelta(days=1):
                problems.append(
                    f"tramos: '{after.nombre}' debe empezar el día siguiente al fin de '{before.nombre}' "
                    "(ordenados en el tiempo, sin huecos ni traslapes)"
                )

    # Un mes del periodo que no se lee saldría como un mes entero de ceros en station_hour.
    if periodo and meses:
        en_lista = {str(m) for m in meses}
        tocados = pd.period_range(periodo.inicio, periodo.fin, freq="M").strftime("%Y%m")
        faltan = [mes for mes in tocados if mes not in en_lista]
        if faltan:
            problems.append(f"meses_crudos: el periodo toca meses que no están en la lista: {', '.join(faltan)}")

    union = cfg.get("union_de_ids")
    distancia = union.get("distancia_maxima_m") if isinstance(union, dict) else None
    if not isinstance(distancia, (int, float)) or isinstance(distancia, bool) or not 0 <= distancia < 1000:
        problems.append(
            "union_de_ids.distancia_maxima_m: debe ser un número de metros mayor o igual a 0 y menor que 1000"
        )

    excluidas = cfg.get("estaciones_excluidas")
    prefijos = excluidas.get("prefijos") if isinstance(excluidas, dict) else None
    ids = excluidas.get("ids") if isinstance(excluidas, dict) else None
    for nombre, valor in (("prefijos", prefijos), ("ids", ids)):
        if not isinstance(valor, list) or not all(isinstance(item, str) and item for item in valor):
            problems.append(f"estaciones_excluidas.{nombre}: debe ser una lista de textos entre comillas (puede ir vacía)")

    if problems:
        raise DatosError(f"{path} no es válido:\n- " + "\n- ".join(problems))
    return Datos(
        meses_crudos=tuple(str(m) for m in meses),
        periodo=periodo,
        tramos=tramos,
        distancia_maxima_m=float(distancia),
        prefijos_excluidos=tuple(prefijos),
        ids_excluidos=tuple(ids),
    )
