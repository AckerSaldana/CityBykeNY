"""Lectura y validación de config/operacion.yaml.

El simulador no lleva números de operación escritos en el código: todo sale de
ese archivo, que ajusta el responsable de logística. Aquí solo se lee y se
comprueba que esté bien formado, para fallar con un mensaje claro si alguien
lo deja mal escrito o con un valor imposible.

Unidades que conviene tener presentes al leer el resultado:

- `presupuestos_por_estacion` y `presupuesto_de_referencia` van en bicis
  movidas por estación por día. El total del día se obtiene con
  `presupuesto_en_bicis`.
- `parte_del_presupuesto` de cada franja es una fracción del presupuesto del
  día; las de todas las franjas suman 1.
- Una estación sin tipo no tiene estrategia: no usa ninguna `ocupacion_respaldo`
  y en el reparto va después de las que sí tienen tipo.
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
_PORCENTAJE = re.compile(r"^(\d+(?:\.\d+)?)\s*%$")
# Holgura al sumar las partes del presupuesto. Solo perdona el redondeo de los
# decimales de la computadora (0.29 + 0.7 + 0.01 da 0.9999999999999999); 0.99
# no pasa.
_HOLGURA_DE_LA_SUMA = 1e-6
# En una lista escrita entre corchetes la coma separa valores: un decimal con
# coma o un separador de miles se leen como dos valores, sin ningún aviso.
_PISTA_LISTA = (
    "Dentro de la lista la coma separa valores. Si escribiste un decimal con coma, usa punto: 0,5 se lee "
    "como dos valores (0 y 5); escribe 0.5. Si usaste separador de miles, quítalo: 1,500 se lee como dos "
    "valores (1 y 500); además aquí van bicis por estación por día, no el total del día"
)


# Clave del YAML con la parte del presupuesto del día que le toca a cada franja. Es el único lugar donde
# está escrito su nombre para LEERLA: el validador y el simulador (`Reglas.from_operacion`) usan esta constante.
CLAVE_PARTE = "parte_del_presupuesto"


class OperacionError(ValueError):
    """config/operacion.yaml está mal escrito o tiene un valor faltante o imposible."""


def _es_entero_escrito(nodo) -> bool:
    """¿El valor está escrito solo con dígitos y sin comillas (0, 5, 500)?"""
    return isinstance(nodo, yaml.ScalarNode) and nodo.style is None and nodo.value.isdigit()


def _revisar_comas_pegadas(clave: str, lista: yaml.SequenceNode) -> None:
    """Rechaza `0,5` o `1,500` dentro de una lista: YAML los lee como dos valores.

    Es el único error de coma que podría pasar sin aviso: `[0, 0.5, 1,5]` se
    lee como 0, 0.5, 1 y 5, que es una lista válida. Dos enteros separados
    solo por una coma, sin espacio, casi siempre son un decimal con coma o un
    separador de miles.
    """
    for antes, despues in zip(lista.value, lista.value[1:]):
        pegados = despues.start_mark.index - antes.end_mark.index == 1
        if pegados and _es_entero_escrito(antes) and _es_entero_escrito(despues):
            a, b = antes.value, despues.value
            raise OperacionError(
                f"{clave}: '{a},{b}' (línea {antes.start_mark.line + 1}) se lee como dos valores ({a} y {b}), "
                f"no como un número. Si es un decimal, usa punto: {a}.{b}. Si es un separador de miles, "
                f"quítalo. Si de verdad son dos valores, sepáralos con coma y espacio: {a}, {b}"
            )


class _LectorSinRepetidas(yaml.SafeLoader):
    """Lector de YAML que rechaza dos descuidos que PyYAML deja pasar sin avisar.

    - Una clave escrita dos veces en el mismo bloque. PyYAML se queda con la
      última: una segunda línea `distancia_maxima_km: 300` al final del
      archivo ganaría en silencio.
    - Un número con coma dentro de una lista (ver `_revisar_comas_pegadas`).
    """

    def construct_mapping(self, node, deep=False):
        pares = node.value if isinstance(node, yaml.MappingNode) else []
        vistas = set()
        for clave, valor in pares:
            if not isinstance(clave, yaml.ScalarNode):
                continue
            if (clave.tag, clave.value) in vistas:
                raise OperacionError(
                    f"'{clave.value}' está escrito dos veces (la segunda, en la línea "
                    f"{clave.start_mark.line + 1}): deja solo una"
                )
            vistas.add((clave.tag, clave.value))
            if isinstance(valor, yaml.SequenceNode):
                _revisar_comas_pegadas(clave.value, valor)
        return super().construct_mapping(node, deep=deep)


def _minutos(hora: str) -> int:
    horas, minutos = hora.split(":")
    return int(horas) * 60 + int(minutos)


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _texto(numero: float) -> str:
    """El número sin la cola de decimales de la computadora: 0.9000000000000001 se muestra 0.9."""
    return f"{numero:.10g}"


def _pista_coma(valor) -> str:
    """Pista para quien escribió un decimal con coma: YAML lo lee como texto, no como número."""
    if isinstance(valor, str) and _DECIMAL_CON_COMA.match(valor.strip()):
        return f" (usa punto decimal: {valor.strip().replace(',', '.')})"
    return ""


def _pista_porcentaje(valor) -> str:
    """Pista para quien escribió un porcentaje (50 o 50 %) donde va una fracción (0.5)."""
    if isinstance(valor, str) and _PORCENTAJE.match(valor.strip()):
        valor = float(_PORCENTAJE.match(valor.strip()).group(1))
    if _es_numero(valor) and 1 < valor <= 100:
        return f" (si es un porcentaje, escríbelo como fracción: {_texto(valor)} % es {_texto(valor / 100)})"
    return ""


def presupuesto_en_bicis(por_estacion: float, estaciones: int) -> int:
    """Total de bicis que se pueden mover en un día con un presupuesto por estación.

    El YAML da los presupuestos en bicis movidas por estación por día. Aquí se
    multiplica por el número de estaciones y se redondea al entero más cercano
    (un medio exacto va al entero par, como redondea Python). Ese total es lo
    que va en la columna `budget` de policy_curve.

    Ejemplo: 0.5 con 2,315 estaciones da 1,157.5, que queda en 1,158 bicis
    movidas al día.
    """
    return round(por_estacion * estaciones)


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
    partes: list[tuple[str, float]] = []
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

        # Parte del presupuesto del día que le toca a la franja.
        parte = franja.get(CLAVE_PARTE)
        if _es_numero(parte) and 0 < parte <= 1:
            partes.append((franja["nombre"], parte))
        elif parte is None:
            problems.append(
                f"{etiqueta}: falta parte_del_presupuesto (qué parte del presupuesto del día le toca: "
                "un número mayor que 0 y hasta 1; 0.5 es la mitad)"
            )
        else:
            problems.append(
                f"{etiqueta}: parte_del_presupuesto debe ser un número mayor que 0 y hasta 1 "
                f"(0.5 es la mitad del presupuesto del día){_pista_coma(parte)}{_pista_porcentaje(parte)}"
            )

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
    # Las partes se suman solo si todas son válidas: si falta una, ya se dijo arriba.
    if franjas and len(partes) == len(franjas):
        suma = math.fsum(parte for _, parte in partes)
        if abs(suma - 1) > _HOLGURA_DE_LA_SUMA:
            detalle = " + ".join(f"{nombre} {_texto(parte)}" for nombre, parte in partes)
            problems.append(
                f"franjas: las parte_del_presupuesto deben sumar 1 y suman {_texto(suma)} ({detalle}). "
                "Ajusta una para que entre todas repartan el presupuesto completo del día"
            )

    # --- Pregunta 2: umbral de riesgo ---
    umbral = cfg.get("umbral_riesgo")
    if not isinstance(umbral, dict):
        problems.append("umbral_riesgo: falta")
    else:
        for campo in ("bicis_minimas", "lugares_libres_minimos"):
            if not _es_entero(umbral.get(campo)) or umbral[campo] < 1:
                problems.append(
                    f"umbral_riesgo.{campo}: debe ser un entero mayor o igual a 1. Está en riesgo la estación con "
                    "MENOS que este valor: con 0 nunca lo estaría. Para \"vacía\" o \"llena\" escribe 1"
                )

    # --- Pregunta 3: presupuestos, en bicis por estación por día ---
    clave_anterior = "presupuestos_diarios" in cfg
    if clave_anterior:
        problems.append(
            "presupuestos_diarios: esta clave ya no se usa. Ahora se llama presupuestos_por_estacion y cambió "
            "la unidad: son bicis movidas por estación por día (por ejemplo 0.5), no el total de bicis del "
            "día. El simulador multiplica cada valor por el número de estaciones que simula"
        )
    presupuestos = cfg.get("presupuestos_por_estacion")
    lista_valida = False
    if presupuestos is None and clave_anterior:
        pass  # el mensaje de arriba ya dice qué clave escribir
    elif not isinstance(presupuestos, list) or not presupuestos:
        problems.append(
            "presupuestos_por_estacion: debe ser una lista con al menos un valor, entre corchetes "
            f"(por ejemplo [0, 0.5, 1.0]){_pista_coma(presupuestos)}"
        )
    elif not all(_es_numero(valor) and math.isfinite(valor) and valor >= 0 for valor in presupuestos):
        pista = "".join(_pista_coma(valor) for valor in presupuestos)
        problems.append(
            "presupuestos_por_estacion: todos deben ser números mayores o iguales a 0, "
            f"en bicis por estación por día{pista}"
        )
    elif len(set(presupuestos)) != len(presupuestos):
        problems.append(f"presupuestos_por_estacion: hay valores repetidos. {_PISTA_LISTA}")
    elif presupuestos != sorted(presupuestos):
        problems.append(f"presupuestos_por_estacion: deben ir de menor a mayor. {_PISTA_LISTA}")
    else:
        lista_valida = True

    # El presupuesto de referencia es el de las recomendaciones del dashboard.
    referencia = cfg.get("presupuesto_de_referencia")
    if not _es_numero(referencia) or not math.isfinite(referencia) or referencia <= 0:
        problems.append(
            "presupuesto_de_referencia: debe ser un número mayor que 0, en bicis por estación por día "
            f"(con 0 no se mueve ninguna bici y no habría recomendaciones){_pista_coma(referencia)}"
        )
    elif lista_valida and referencia not in presupuestos:
        problems.append(
            f"presupuesto_de_referencia: {_texto(referencia)} no está en presupuestos_por_estacion "
            f"{presupuestos}. Debe ser uno de los valores de esa lista"
        )

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

    if "tipo_si_falta" in cfg:
        problems.append(
            "tipo_si_falta: esta clave ya no se usa, bórrala. Una estación sin tipo ya no toma la estrategia de "
            "otro tipo: en las franjas sin solución no se toca, y el inventario inicial sale de la captura de GBFS"
        )
    return problems


def load_operacion(path: str | Path = OPERACION_YAML) -> dict:
    """Lee las reglas de operación. Lanza OperacionError si el archivo no es válido."""
    try:
        archivo = open(path, encoding="utf-8")
    except OSError as error:
        raise OperacionError(f"No se pudo abrir {path}: {error}") from None
    with archivo:
        try:
            cfg = yaml.load(archivo, Loader=_LectorSinRepetidas)
        except OperacionError as error:  # una clave repetida o un número con coma en una lista
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
