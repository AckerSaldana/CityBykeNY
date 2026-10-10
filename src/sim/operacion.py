"""Lectura y validación de config/operacion.yaml.

El simulador no lleva números de operación escritos en el código: todo sale de
ese archivo, que ajusta el responsable de logística. Aquí solo se lee y se
comprueba que esté bien formado, para fallar con un mensaje claro si alguien
lo deja mal escrito o con un valor imposible.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from src.contracts import STATION_TYPES
from src.paths import OPERACION_YAML

_HORA = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DECIMAL_CON_COMA = re.compile(r"^\d+,\d+$")
_PISTA_MILES = "Si usaste separador de miles, quítalo: 1,500 se lee como dos valores (1 y 500); escribe 1500"


class OperacionError(ValueError):
    """config/operacion.yaml está mal escrito o tiene un valor faltante o imposible."""


class _LectorSinRepetidas(yaml.SafeLoader):
    """Lector de YAML que rechaza una clave escrita dos veces en el mismo bloque.

    PyYAML se queda con la última sin avisar: una segunda línea
    `distancia_maxima_km: 300` al final del archivo ganaría en silencio.
    """

    def construct_mapping(self, node, deep=False):
        pares = node.value if isinstance(node, yaml.MappingNode) else []
        vistas = set()
        for clave, _ in pares:
            if not isinstance(clave, yaml.ScalarNode):
                continue
            if (clave.tag, clave.value) in vistas:
                raise OperacionError(
                    f"'{clave.value}' está escrito dos veces (la segunda, en la línea "
                    f"{clave.start_mark.line + 1}): deja solo una"
                )
            vistas.add((clave.tag, clave.value))
        return super().construct_mapping(node, deep=deep)


def _minutos(hora: str) -> int:
    horas, minutos = hora.split(":")
    return int(horas) * 60 + int(minutos)


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _pista_coma(valor) -> str:
    """Pista para quien escribió un decimal con coma: YAML lo lee como texto, no como número."""
    if isinstance(valor, str) and _DECIMAL_CON_COMA.match(valor.strip()):
        return f" (usa punto decimal: {valor.strip().replace(',', '.')})"
    return ""


def validate_operacion(cfg: dict) -> list[str]:
    """Devuelve la lista de problemas. Lista vacía = archivo válido."""
    if not isinstance(cfg, dict):
        return ["el archivo debe ser un mapa de parámetros"]
    problems: list[str] = []

    if cfg.get("estado") not in ("provisional", "confirmado"):
        problems.append("estado: debe ser 'provisional' o 'confirmado'")
    elif cfg["estado"] == "confirmado" and not (cfg.get("confirmado_por") and cfg.get("fecha_confirmacion")):
        problems.append("estado 'confirmado' requiere confirmado_por y fecha_confirmacion")

    # --- Pregunta 1: franjas ---
    nombres: list[str] = []
    franjas = cfg.get("franjas")
    if not isinstance(franjas, list) or not franjas:
        problems.append("franjas: debe ser una lista con al menos una franja")
        franjas = []
    intervalos = []
    for i, franja in enumerate(franjas):
        etiqueta = f"franjas[{i}]"
        if not isinstance(franja, dict) or not isinstance(franja.get("nombre"), str) or not franja["nombre"]:
            problems.append(f"{etiqueta}: falta nombre")
            continue
        etiqueta = f"franja '{franja['nombre']}'"
        nombres.append(franja["nombre"])
        horas = {campo: franja.get(campo) for campo in ("momento_intervencion", "inicio", "fin")}
        malas = [campo for campo, valor in horas.items() if not isinstance(valor, str) or not _HORA.match(valor)]
        if malas:
            problems.append(f"{etiqueta}: {malas} deben ser horas \"HH:MM\" entre comillas")
            continue
        intervencion, inicio, fin = (_minutos(horas[campo]) for campo in ("momento_intervencion", "inicio", "fin"))
        if not intervencion <= inicio < fin:
            problems.append(f"{etiqueta}: debe cumplirse momento_intervencion <= inicio < fin")
            continue
        intervalos.append((inicio, fin, franja["nombre"]))
    if len(set(nombres)) != len(nombres):
        problems.append("franjas: hay nombres repetidos")
    intervalos.sort()
    for (_, fin_a, nombre_a), (inicio_b, _, nombre_b) in zip(intervalos, intervalos[1:]):
        if inicio_b < fin_a:
            problems.append(f"franjas '{nombre_a}' y '{nombre_b}' se enciman")

    # --- Pregunta 2: umbral de riesgo ---
    umbral = cfg.get("umbral_riesgo")
    if not isinstance(umbral, dict):
        problems.append("umbral_riesgo: falta")
    else:
        for campo in ("bicis_minimas", "lugares_libres_minimos"):
            if not _es_entero(umbral.get(campo)) or umbral[campo] < 0:
                problems.append(f"umbral_riesgo.{campo}: debe ser un entero mayor o igual a 0")

    # --- Pregunta 3: presupuestos ---
    presupuestos = cfg.get("presupuestos_diarios")
    if not isinstance(presupuestos, list) or not presupuestos:
        problems.append("presupuestos_diarios: debe ser una lista con al menos un valor")
    elif not all(_es_entero(valor) and valor >= 0 for valor in presupuestos):
        problems.append("presupuestos_diarios: todos deben ser enteros mayores o iguales a 0")
    elif len(set(presupuestos)) != len(presupuestos):
        problems.append(f"presupuestos_diarios: hay valores repetidos. {_PISTA_MILES}")
    elif presupuestos != sorted(presupuestos):
        problems.append(f"presupuestos_diarios: deben ir de menor a mayor. {_PISTA_MILES}")

    # --- Pregunta 4: distancia ---
    distancia = cfg.get("distancia_maxima_km")
    if not _es_numero(distancia) or not math.isfinite(distancia) or distancia <= 0:
        problems.append(f"distancia_maxima_km: debe ser un número mayor que 0{_pista_coma(distancia)}")

    # --- Pregunta 5: estrategia por tipo ---
    estrategia = cfg.get("estrategia_por_tipo")
    if not isinstance(estrategia, dict):
        problems.append("estrategia_por_tipo: falta")
    else:
        if set(estrategia) != set(STATION_TYPES):
            problems.append(f"estrategia_por_tipo: los tipos deben ser exactamente {list(STATION_TYPES)}")
        for tipo, regla in estrategia.items():
            if not isinstance(regla, dict):
                problems.append(f"estrategia_por_tipo.{tipo}: falta prioridad y ocupacion_respaldo")
                continue
            if regla.get("prioridad") not in (1, 2, 3) or isinstance(regla.get("prioridad"), bool):
                problems.append(f"estrategia_por_tipo.{tipo}.prioridad: debe ser 1, 2 o 3")
            ocupacion = regla.get("ocupacion_respaldo")
            if not isinstance(ocupacion, dict) or set(ocupacion) != set(nombres):
                problems.append(
                    f"estrategia_por_tipo.{tipo}.ocupacion_respaldo: debe tener un valor por franja {nombres}"
                )
                continue
            for franja, valor in ocupacion.items():
                if not _es_numero(valor) or not 0 <= valor <= 1:
                    problems.append(
                        f"estrategia_por_tipo.{tipo}.ocupacion_respaldo.{franja}: debe estar entre 0 y 1"
                        f"{_pista_coma(valor)}"
                    )
    return problems


def load_operacion(path: str | Path = OPERACION_YAML) -> dict:
    """Lee las reglas de operación. Lanza OperacionError si el archivo no es válido."""
    with open(path, encoding="utf-8") as archivo:
        try:
            cfg = yaml.load(archivo, Loader=_LectorSinRepetidas)
        except OperacionError as error:  # una clave repetida
            raise OperacionError(f"{path} no es válido:\n- {error}") from None
        except yaml.YAMLError as error:
            # PyYAML da hasta dos marcas: dónde empezó lo que leía y dónde se detuvo.
            # El error suele estar en una de las dos, a veces en medio; con una sola
            # marca puede estar una línea antes. Por eso se dicen ambas y "cerca de".
            marcas = (getattr(error, "context_mark", None), getattr(error, "problem_mark", None))
            lineas = sorted({marca.line + 1 for marca in marcas if marca is not None})
            if len(lineas) == 2:
                donde = f" Revisa de la línea {lineas[0]} a la {lineas[1]}, empezando por esas dos."
            else:
                donde = f" Revisa cerca de la línea {lineas[0]}." if lineas else ""
            raise OperacionError(
                f"{path} está mal escrito y no se pudo leer.{donde} Causas comunes: un tabulador "
                "(usa espacios), una sangría distinta a la de las líneas vecinas, falta el espacio "
                "después de los dos puntos, o una comilla o corchete sin cerrar."
            ) from None
    problems = validate_operacion(cfg)
    if problems:
        raise OperacionError(f"{path} no es válido:\n- " + "\n- ".join(problems))
    return cfg
