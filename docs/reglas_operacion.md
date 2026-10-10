# Reglas de operación del rebalanceo

> **Estado: PLANTILLA SIN CONFIRMAR.**
> Los valores que aparecen como "provisional" los puso programación solo para
> que el flujo corra. No son criterio de logística ni salen de los datos.
> Quien lleva logística debe revisarlos, cambiarlos y confirmarlos.

## Para qué sirve este documento

El simulador todavía no existe. Cuando exista, reproducirá cómo se vacían y se
llenan las estaciones, y probará qué pasaría si moviéramos bicis antes de cada
franja del día. Para eso necesitará cinco reglas de operación. Esas reglas
**no están escritas en el código**: se leen de `config/operacion.yaml` (hoy
solo las usan los datos de muestra).

Este documento es la versión en lenguaje normal de ese archivo. Aquí se
responde y se justifica cada regla; en el YAML se anota el número.

## Cómo llenarlo

1. Responde las cinco preguntas de abajo, con tu razón en una o dos líneas.
2. Pasa cada respuesta a `config/operacion.yaml` (cada pregunta dice qué
   parámetro tocar).
3. Corre `pytest tests/test_operacion_config.py` para comprobar que el archivo
   quedó bien escrito.
4. En el YAML cambia `estado: provisional` por `estado: confirmado` y anota tu
   nombre y la fecha.
5. Avisa a programación: hay que regenerar los datos de muestra y, cuando
   exista el simulador, volver a correrlo, porque cambiar una regla cambiará
   sus resultados.

Regla de honestidad: si un número es una suposición, dilo en la justificación.
En el pitch solo se presentan como reglas de operación las que estén
confirmadas aquí.

---

## Pregunta 1. ¿En qué franjas del día rebalanceamos y a qué hora se interviene?

**Qué significa.** Una franja es un periodo del día que queremos proteger (por
ejemplo, la mañana de entrada a oficinas). El momento de intervención es la
hora a la que los camiones ya dejaron las estaciones listas para esa franja.

**Por qué importa.** El sistema recomienda con cuántas bicis debe empezar cada
estación en cada franja. Además, solo puede usar la información disponible
antes del momento de intervención.

**Para pensarlo.** ¿Cuántas rondas de camiones son realistas en un día? ¿A qué
hora empiezan los picos? ¿Cuánto tarda una ronda?

| Franja | Momento de intervención | Inicio | Fin | Estado |
|---|---|---|---|---|
| manana | 05:00 | 06:00 | 12:00 | provisional |
| tarde | 14:00 | 15:00 | 21:00 | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `franjas` (horas en formato "HH:MM", hora de Nueva
York). Todo va en el mismo día: la intervención se anota a la hora del inicio
o antes (no se puede anotar una intervención de la noche anterior) y una
franja que termina a medianoche se escribe con fin "23:59". Si cambias los
nombres de las franjas, cámbialos también en la pregunta 5.

---

## Pregunta 2. ¿Cuándo decimos que una estación está en riesgo?

**Qué significa.** El umbral de riesgo es el mínimo de bicis disponibles, o de
lugares libres, por debajo del cual consideramos que la estación ya está
fallando o a punto de fallar.

**Por qué importa.** Es la medida de éxito del proyecto: contaremos las horas
de estación en riesgo en la simulación y buscaremos reducirlas. Un umbral más
alto marca más horas como riesgo.

**Para pensarlo.** ¿Con cuántas bicis una persona todavía encuentra una que
sirva? ¿Con cuántos lugares libres todavía se puede devolver sin dar vueltas?

| Parámetro | Valor | Estado |
|---|---|---|
| Bicis mínimas | 2 | provisional |
| Lugares libres mínimos | 2 | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `umbral_riesgo.bicis_minimas` y
`umbral_riesgo.lugares_libres_minimos`.

---

## Pregunta 3. ¿Cuántas bicis puede mover la operación en un día?

**Qué significa.** El presupuesto diario es el total de bicis trasladadas en
camión en un día, sumando todas las estaciones.

**Por qué importa.** Compararemos tres formas de decidir (no intervenir, regla
histórica y modelo predictivo) con el mismo presupuesto. Probaremos varios
presupuestos para dibujar una curva: cuánto riesgo simulado se evita por cada
nivel de esfuerzo.

**Para pensarlo.** ¿Cuántos camiones hay, cuántas bicis caben en cada uno y
cuántos viajes hace cada camión por turno? Conviene incluir un valor realista,
uno austero y uno ambicioso.

| Parámetro | Valor | Estado |
|---|---|---|
| Presupuestos a simular (bicis por día) | 0, 50, 100, 200, 400, 800 | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `presupuestos_diarios`.

---

## Pregunta 4. ¿Qué tan lejos puede estar la estación que dona de la que recibe?

**Qué significa.** Es la distancia máxima, en kilómetros y en línea recta,
entre la estación de donde se sacan bicis y la estación a la que se llevan.

**Por qué importa.** Evita recomendaciones imposibles, como cruzar la ciudad
para mover tres bicis.

**Para pensarlo.** ¿Cuánto recorre un camión entre paradas en un turno normal?
¿Se permite cruzar un puente o un río?

| Parámetro | Valor | Estado |
|---|---|---|
| Distancia máxima (km) | 3.0 | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `distancia_maxima_km`.

---

## Pregunta 5. ¿Qué estrategia seguimos según el tipo de estación?

**Qué significa.** Clasificamos las estaciones en cuatro tipos: oficinas,
residencial, recreativa y mixta. Para cada tipo se define a quién atender
primero cuando el presupuesto no alcanza (prioridad) y qué tan llena conviene
dejar la estación al iniciar cada franja (ocupación objetivo, de 0 % vacía a
100 % llena).

Interpretación de programación, pendiente de confirmar: se usaría como
respaldo cuando ningún inventario inicial evite a la vez que la estación se
vacíe y se llene; falta decidir si es eso o la meta de la regla histórica.

**Por qué importa.** Una estación de oficinas se llena en la mañana y se vacía
en la tarde; una residencial hace lo contrario. La misma regla no sirve para
todas.

**Para pensarlo.** ¿Qué tipo de estación duele más cuando falla? ¿En qué
franja conviene dejar bicis y en cuál conviene dejar lugares libres?

| Tipo | Prioridad (3 alta, 1 baja) | Ocupación al iniciar la mañana | Ocupación al iniciar la tarde | Estado |
|---|---|---|---|---|
| oficinas | 3 | 30 % | 70 % | provisional |
| residencial | 3 | 70 % | 30 % | provisional |
| recreativa | 1 | 50 % | 50 % | provisional |
| mixta | 2 | 50 % | 50 % | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `estrategia_por_tipo` (la ocupación se escribe como
fracción: 30 % es `0.30`).

---

## Pendientes que también decide logística

- **Cómo se clasifica cada estación en un tipo.** Todavía no existe la regla.
  Mientras no exista, `station_type` queda vacío en los datos reales.
- **Rutas de camión.** Hoy las recomendaciones dicen cuántas bicis llevar o
  retirar por estación, pero no de qué estación a cuál. Si el dashboard debe
  mostrar rutas, hay que acordar ese dato con programación.

## Registro de cambios

| Fecha | Quién | Qué cambió y por qué |
|---|---|---|
| _pendiente_ | _pendiente_ | Versión inicial con valores provisionales |
