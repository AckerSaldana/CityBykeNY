"""Lectura y validación de config/datos.yaml: periodo, tramos, reglas de limpieza y ajustes de la simulación.

La limpieza y los modelos leen de aquí las fechas. Así el periodo y los tres
tramos (entrenamiento, validación y prueba) están en un solo lugar y no
escritos en el código.

El bloque `simulacion` trae los ajustes del simulador que no son reglas de
operación (calentamiento, remuestreo, bandas de incertidumbre candidatas,
franja de noche del reporte y ocupaciones de la prueba de sensibilidad). Solo lo usa el simulador: un archivo
sin ese bloque es válido para la limpieza y la línea base, y entonces
`datos.simulacion` es None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import math
import re

import pandas as pd
import yaml

from src.paths import DATOS_YAML, TZ

# Nombre del tramo en el YAML -> nombre en el contrato de predictions.
SPLIT_NAMES = {"validacion": "val", "prueba": "test"}
TRAMOS = ("entrenamiento", "validacion", "prueba")


_HORA = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


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
class Simulacion:
    """Bloque `simulacion`: ajustes del simulador que no son reglas de operación."""

    dias_de_calentamiento: int   # primeros días simulados; se simulan pero no cuentan en las métricas
    remuestreos: int             # veces que se sortean los días para calcular el intervalo
    nivel_del_intervalo: float   # 0.90 es un intervalo del 90 %
    semilla: int                 # semilla del sorteo: misma semilla, mismo intervalo
    # Multiplicadores candidatos de la banda de incertidumbre (0 = sin banda, 1 = q10 a q90). Se elige uno
    # con validación y se aplica igual a todas las políticas con pronóstico.
    bandas: tuple[float, ...]
    # Franja de noche del reporte, ("HH:MM", "HH:MM"); puede cruzar la medianoche. No es una franja de operación.
    franja_noche: tuple[str, str]
    # Ocupaciones planas (fracción de la capacidad) con las que se repite la corrida de referencia para ver
    # cuánto depende el resultado del inventario inicial. Puede ir vacía.
    sensibilidad_ocupacion_inicial: tuple[float, ...]


@dataclass(frozen=True)
class Datos:
    meses_crudos: tuple[str, ...]
    periodo: Tramo
    tramos: dict[str, Tramo]
    distancia_maxima_m: float
    prefijos_excluidos: tuple[str, ...]
    ids_excluidos: tuple[str, ...]
    # None si el archivo no trae el bloque `simulacion` (la limpieza y la línea base no lo usan).
    simulacion: Simulacion | None = None


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


def _es_entero(valor) -> bool:
    """Entero de verdad: ni 1.0, ni "1", ni true (que en Python también es un entero)."""
    return type(valor) is int


def _simulacion(raw, validacion: Tramo | None, problems: list[str]) -> Simulacion | None:
    """Valida el bloque `simulacion`. `validacion` es None si los tramos ya traen algún problema."""
    if not isinstance(raw, dict):
        problems.append(
            "simulacion: debe ser un bloque con dias_de_calentamiento, remuestreos, nivel_del_intervalo, semilla, "
            "bandas, franja_noche y sensibilidad_ocupacion_inicial"
        )
        return None
    antes = len(problems)

    # El calentamiento no cuenta en las métricas: debe quedar al menos un día de validación que sí cuente.
    calentamiento = raw.get("dias_de_calentamiento")
    dias_de_validacion = None if validacion is None else (validacion.fin - validacion.inicio).days + 1
    tope = "" if validacion is None else f" y menor que los {dias_de_validacion} días del tramo de validación"
    if (
        not _es_entero(calentamiento)
        or calentamiento < 0
        or (dias_de_validacion is not None and calentamiento >= dias_de_validacion)
    ):
        problems.append(f"simulacion.dias_de_calentamiento: debe ser un número entero mayor o igual a 0{tope}")

    remuestreos = raw.get("remuestreos")
    if not _es_entero(remuestreos) or remuestreos < 100:
        problems.append("simulacion.remuestreos: debe ser un número entero mayor o igual a 100")

    nivel = raw.get("nivel_del_intervalo")
    if not isinstance(nivel, (int, float)) or isinstance(nivel, bool) or not 0.5 < nivel < 1:
        problems.append(
            "simulacion.nivel_del_intervalo: debe ser un número mayor que 0.5 y menor que 1 "
            "(0.90 es un intervalo del 90 %)"
        )

    # Sin negativos: el generador de números al azar de numpy no los acepta como semilla.
    semilla = raw.get("semilla")
    if not _es_entero(semilla) or semilla < 0:
        problems.append("simulacion.semilla: debe ser un número entero mayor o igual a 0")

    def numeros(valor, minimo: float, maximo: float) -> bool:
        return isinstance(valor, list) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and minimo <= v <= maximo
            for v in valor
        ) and len(set(valor)) == len(valor)

    bandas = raw.get("bandas")
    if not numeros(bandas, 0, 10) or not bandas:
        problems.append(
            "simulacion.bandas: debe ser una lista con al menos un multiplicador de la banda de incertidumbre, "
            "entre 0 y 10 y sin repetir (por ejemplo [0, 0.5, 1]; 0 es sin banda y 1 es de q10 a q90)"
        )

    noche = raw.get("franja_noche")
    if (
        not isinstance(noche, dict)
        or not all(isinstance(noche.get(k), str) and _HORA.match(noche[k]) for k in ("inicio", "fin"))
        or noche["inicio"] == noche["fin"]
    ):
        problems.append(
            'simulacion.franja_noche: debe tener inicio y fin como horas "HH:MM" entre comillas y distintas '
            '(puede cruzar la medianoche: inicio "22:00", fin "06:00")'
        )

    sensibilidad = raw.get("sensibilidad_ocupacion_inicial")
    if not numeros(sensibilidad, 0, 1):
        problems.append(
            "simulacion.sensibilidad_ocupacion_inicial: debe ser una lista de fracciones de la capacidad entre "
            "0 y 1, sin repetir (por ejemplo [0.4, 0.5, 0.6]; puede ir vacía: [])"
        )

    if len(problems) > antes:
        return None
    return Simulacion(
        dias_de_calentamiento=calentamiento,
        remuestreos=remuestreos,
        nivel_del_intervalo=float(nivel),
        semilla=semilla,
        bandas=tuple(float(v) for v in bandas),
        franja_noche=(noche["inicio"], noche["fin"]),
        sensibilidad_ocupacion_inicial=tuple(float(v) for v in sensibilidad),
    )


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

    # El bloque `simulacion` solo lo usa el simulador. Si no está, no es un error aquí.
    simulacion = None
    if "simulacion" in cfg:
        validacion = tramos.get("validacion") if len(tramos) == len(TRAMOS) else None
        simulacion = _simulacion(cfg["simulacion"], validacion, problems)

    if problems:
        raise DatosError(f"{path} no es válido:\n- " + "\n- ".join(problems))
    return Datos(
        meses_crudos=tuple(str(m) for m in meses),
        periodo=periodo,
        tramos=tramos,
        distancia_maxima_m=float(distancia),
        prefijos_excluidos=tuple(prefijos),
        ids_excluidos=tuple(ids),
        simulacion=simulacion,
    )
