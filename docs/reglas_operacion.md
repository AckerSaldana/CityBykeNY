# Reglas de operación del rebalanceo

> **Estado: PLANTILLA SIN CONFIRMAR.**
> Los valores que aparecen como "provisional" los puso programación solo para
> que el flujo corra. No son criterio de logística ni salen de los datos.
> Quien lleva logística debe revisarlos, cambiarlos y confirmarlos.

## Para qué sirve este documento

El simulador reproduce cómo se vacían y se llenan las estaciones, y prueba qué
pasaría si moviéramos bicis antes de cada franja del día. Para eso necesita
cinco reglas de operación. Esas reglas **no están escritas en el código**: se
leen de `config/operacion.yaml`.

Este documento es la versión en lenguaje normal de ese archivo. Aquí se
responde y se justifica cada regla; en el YAML se anota el número.

## Cómo llenarlo

1. Responde las cinco preguntas de abajo, con tu razón en una o dos líneas.
2. Pasa cada respuesta a `config/operacion.yaml` (cada pregunta dice qué
   parámetro tocar). Los decimales van con punto: `0.5`, no `0,5`.
3. Corre `pytest tests/test_operacion_config.py` para comprobar que el archivo
   quedó bien escrito. Si algo está mal, el mensaje dice qué es y cómo
   arreglarlo.
4. En el YAML cambia `estado: provisional` por `estado: confirmado` y anota tu
   nombre y la fecha.
5. Avisa a programación: hay que regenerar los datos de muestra y volver a
   correr el simulador, porque cambiar una regla cambia sus resultados.

Regla de honestidad: si un número es una suposición, dilo en la justificación.
En el pitch solo se presentan como reglas de operación las que estén
confirmadas aquí.

---

## Pregunta 1. ¿En qué franjas del día rebalanceamos, a qué hora se interviene y qué parte del presupuesto le toca a cada una?

**Qué significa.** Una franja es un periodo del día que queremos proteger (por
ejemplo, la mañana de entrada a oficinas). El momento de intervención es la
hora a la que los camiones ya dejaron las estaciones listas para esa franja.
La parte del presupuesto es qué porción de las bicis que se pueden mover en el
día (pregunta 3) le toca a esa franja.

**Por qué importa.** El sistema recomienda con cuántas bicis debe empezar cada
estación en cada franja. Además, solo puede usar la información disponible
antes del momento de intervención. El reparto decide cuánto esfuerzo va a cada
franja; la pregunta 3 explica cómo funciona.

**Para pensarlo.** ¿Cuántas rondas de camiones son realistas en un día? ¿A qué
hora empiezan los picos? ¿Cuánto tarda una ronda? ¿Hay los mismos camiones y
la misma gente en la mañana que en la tarde?

| Franja | Momento de intervención | Inicio | Fin | Parte del presupuesto del día | Estado |
|---|---|---|---|---|---|
| manana | 05:00 | 06:00 | 12:00 | 50 % | provisional |
| tarde | 14:00 | 15:00 | 21:00 | 50 % | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `franjas` (horas en formato "HH:MM", hora de Nueva
York). Todo va en el mismo día: la intervención se anota a la hora del inicio
o antes (no se puede anotar una intervención de la noche anterior) y una
franja que termina a medianoche se escribe con fin "23:59". Si cambias los
nombres de las franjas, cámbialos también en la pregunta 5.

La parte del presupuesto va en `parte_del_presupuesto` de cada franja y se
escribe como fracción: 50 % es `0.5`. Cada una debe ser mayor que 0 y entre
todas deben sumar 1 (100 %). Si agregas o quitas una franja, vuelve a
repartir.

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
`umbral_riesgo.lugares_libres_minimos`. Son enteros, de 1 en adelante. Con 0
ninguna estación contaría como en riesgo, porque nunca hay menos de 0. Para
contar solo las estaciones vacías o llenas, escribe 1.

---

## Pregunta 3. ¿Cuántas bicis puede mover la operación en un día, por cada estación?

**Qué significa.** El presupuesto es cuántas bicis se trasladan en camión en
un día, **repartidas entre el número de estaciones**. La unidad es bicis
movidas por estación por día. Una bici que se saca de una estación y se deja
en otra cuenta como una bici movida.

Ejemplo: 0.5 quiere decir media bici por estación al día. Hoy se simulan
2,315 estaciones: 0.5 × 2,315 da 1,157.5, que queda en 1,158 bicis movidas al
día.

Redondeo: el total del día se redondea a la bici entera más cercana (si queda
justo en medio, va al número par), y cuando ese total es impar y se reparte
mitad y mitad, la bici que sobra va a la franja que interviene primero (579
bicis se reparten en 290 y 289).

**Por qué se pide por estación y no el total del día.** El número de
estaciones que entran a la simulación puede cambiar: quedan fuera las que no
tienen capacidad conocida y las inactivas, y esa lista se puede ajustar. El
simulador multiplica tu número por las estaciones que sí simula. Así el
esfuerzo sigue siendo comparable aunque cambie cuántas estaciones entran.

Si tú piensas en el total del día (camiones × bicis por camión × viajes),
divídelo entre el número de estaciones simuladas. Las 2,315 de los ejemplos
son las de la corrida del 10 de octubre de 2026. Ese número puede cambiar
cuando se reconstruyen las tablas; el vigente te lo da programación.

**Por qué importa.** Compararemos tres formas de decidir (no intervenir, regla
histórica y modelo predictivo) con el mismo presupuesto. Probaremos varios
presupuestos para dibujar una curva: cuánto riesgo simulado se evita por cada
nivel de esfuerzo.

**Presupuesto de referencia.** De la lista, uno es el de referencia: con ese
se calculan las recomendaciones que muestra el dashboard (cuántas bicis llevar
o retirar en cada estación). Los demás solo sirven para la curva. Conviene que
sea el más realista. Debe ser uno de la lista y mayor que 0.

**Cómo se reparte en el día.** El presupuesto del día se divide entre las
franjas según la columna "Parte del presupuesto del día" de la pregunta 1.
Lo que no se usa en la mañana pasa a la tarde del mismo día. Lo que sobra al
terminar el día se pierde: nunca pasa al día siguiente.

Ejemplo con 1,158 bicis al día, mitad en la mañana y mitad en la tarde: la
mañana puede mover hasta 579. Si mueve 500, la tarde puede mover hasta 658
(sus 579 más las 79 que sobraron).

**Para pensarlo.** ¿Cuántos camiones hay, cuántas bicis caben en cada uno y
cuántos viajes hace cada camión por turno? Eso da un total de bicis al día:
divídelo entre el número de estaciones. Conviene incluir un valor realista,
uno austero y uno ambicioso.

| Parámetro | Valor | Con las 2,315 estaciones que se simulan hoy son | Estado |
|---|---|---|---|
| Presupuestos a simular (bicis por estación por día) | 0, 0.1, 0.25, 0.5, 1.0, 2.0 | 0, 232, 579, 1,158, 2,315 y 4,630 bicis al día | provisional |
| Presupuesto de referencia (el de las recomendaciones del dashboard) | 0.5 | 1,158 bicis al día | provisional |
| Reparto entre franjas | manana 50 %, tarde 50 % | 579 y 579 bicis con el de referencia | provisional (se anota en la pregunta 1) |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `presupuestos_por_estacion` (la lista, de menor a
mayor y sin repetir; el 0 equivale a no intervenir), `presupuesto_de_referencia`
y `parte_del_presupuesto` en cada franja.

Dos cuidados al escribir la lista:

- Los decimales van con punto. Dentro de la lista la coma separa valores:
  `0,5` se leería como dos valores (0 y 5).
- El nombre anterior, `presupuestos_diarios`, ya no se acepta: era el total de
  bicis del día. Si tienes un total, divídelo entre el número de estaciones.

---

## Pregunta 4. ¿Qué tan lejos puede estar la estación que dona de la que recibe?

**Qué significa.** Es la distancia máxima, en kilómetros y en línea recta,
entre la estación de donde se sacan bicis y la estación a la que se llevan.

**Por qué importa.** Evita recomendaciones imposibles, como cruzar la ciudad
para mover tres bicis.

**Limitación.** El simulador mide en línea recta y no sabe de ríos ni de
puentes. No puede prohibir un traslado que cruza un río: solo conoce esta
distancia. Una distancia más corta reduce ese problema, pero no lo quita.

**Para pensarlo.** ¿Cuánto recorre un camión entre paradas en un turno normal?
¿Qué tan seguido una distancia corta en el mapa obliga a cruzar un puente?

| Parámetro | Valor | Estado |
|---|---|---|
| Distancia máxima (km) | 3.0 | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `distancia_maxima_km`.

---

## Pregunta 5. ¿Qué estrategia seguimos según el tipo de estación?

**Qué significa.** Clasificamos las estaciones en cuatro tipos: oficinas,
residencial, recreativa y mixta. Para cada tipo se definen dos cosas: a quién
atender primero cuando el presupuesto no alcanza (prioridad) y qué tan llena
empieza cada estación, como fracción de su capacidad (ocupación de respaldo,
de 0 % vacía a 100 % llena).

**Ojo: hoy ninguna estación tiene tipo.** Faltan los umbrales para
clasificarlas (ver los pendientes de abajo). Mientras falten, a todas las
estaciones se les aplica una sola fila de la tabla: la del tipo que diga
`tipo_si_falta`. El valor provisional es mixta, así que hoy solo cuenta la
fila "mixta" y las otras tres no se usan todavía.

La ocupación de respaldo tiene dos usos:

1. Es el inventario inicial común de las tres políticas (sin intervención,
   regla histórica y modelo predictivo): las tres empiezan con las mismas
   bicis en cada estación, para compararlas desde el mismo punto. Solo el
   primer día de la simulación empieza así (ver "Ya decidido").
2. Es el valor que se usa cuando ningún inventario inicial evita a la vez que
   la estación se vacíe y se llene.

La regla histórica no usa este valor para decidir.

El significado ya está decidido; los valores siguen siendo provisionales.

Queda un detalle abierto: el parámetro tiene un valor por franja y la decisión
no dice cuál de esos valores da el inventario inicial de la simulación;
programación usará el de la primera franja del día, salvo que logística diga
otra cosa.

**Por qué importa.** Una estación de oficinas se llena en la mañana y se vacía
en la tarde; una residencial hace lo contrario. La misma regla no sirve para
todas. Además, como las tres políticas empiezan desde este valor, cambiarlo
cambiará el riesgo simulado de las tres, también el de no intervenir.

**Para pensarlo.** ¿Qué tipo de estación duele más cuando falla? ¿En qué
franja conviene que una estación empiece con bicis y en cuál con lugares
libres? Mientras no haya tipos, ¿qué fila representa mejor a una estación
cualquiera?

| Tipo | Prioridad (3 alta, 1 baja) | Ocupación de respaldo en la mañana | Ocupación de respaldo en la tarde | Estado |
|---|---|---|---|---|
| oficinas | 3 | 30 % | 70 % | provisional |
| residencial | 3 | 70 % | 30 % | provisional |
| recreativa | 1 | 50 % | 50 % | provisional |
| mixta | 2 | 50 % | 50 % | provisional |

| Parámetro | Valor | Estado |
|---|---|---|
| Tipo que se aplica a una estación sin tipo (hoy, todas) | mixta | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `estrategia_por_tipo`, con `prioridad` y
`ocupacion_respaldo` para cada tipo (la ocupación se escribe como fracción:
30 % es `0.30`), y `tipo_si_falta` (uno de los cuatro tipos, en minúsculas).

---

## Pendientes que también decide logística

- **Umbrales para clasificar cada estación en un tipo.** Ya está decidido cómo
  se asigna el tipo: con una regla de comportamiento calculada solo con el
  tramo de entrenamiento (el primero de los tres tramos de datos, el que se
  usa para ajustar el modelo), a partir de tres indicadores: el flujo neto de
  la mañana entre semana (llegadas menos salidas), la proporción de fin de
  semana y la proporción de usuarios ocasionales. Lo que sigue pendiente son
  los umbrales: desde qué valor de cada indicador una estación cuenta como
  oficinas, residencial, recreativa o mixta. Los fija logística. Mientras no
  existan, `station_type` queda vacío en los datos reales.
- **Qué estrategia usa una estación sin tipo (`tipo_si_falta`).** El simulador
  necesita una prioridad y una ocupación de respaldo para cada estación, y hoy
  ninguna tiene tipo. Por eso usa las del tipo que diga `tipo_si_falta`
  (provisional: mixta). Es punto abierto: lo confirma o lo cambia logística en
  la pregunta 5. Cuando existan los umbrales, este valor solo se aplicará a
  las estaciones que sigan sin tipo.

Ya decidido (se anota para tenerlo presente; no hay nada que responder):

- **Rutas.** El dashboard muestra los pares de estaciones más transitados:
  los viajes de una estación a otra, sumados en todo el periodo. Las rutas de
  camión quedan fuera por ahora: las recomendaciones dicen cuántas bicis
  llevar o retirar por estación, no de qué estación a cuál.
- **Captura de GBFS.** No se traslapa con los meses de viajes. Se usa solo
  para la capacidad de las estaciones, el modo en vivo y el perfil de
  ocupación por hora. No se usa para validar la posible demanda insatisfecha.
- **Arranque de la simulación.** El simulador empieza el 1 de septiembre de
  2026 con cada estación en su ocupación de respaldo (pregunta 5). Ese primer
  día es de calentamiento y no cuenta en los resultados. Después la simulación
  sigue de corrido, sin reinicios: cada día empieza con las bicis que dejó el
  anterior, y toda bici que se mueve gasta presupuesto.
- **Qué limita un movimiento.** Un traslado de una estación a otra solo se
  hace si se cumplen cuatro cosas: la estación de origen tiene esas bicis, la
  de destino tiene lugar, las dos están dentro de la distancia máxima en línea
  recta (pregunta 4) y queda presupuesto (pregunta 3). El total de bicis del
  sistema no cambia: lo que sale de una estación llega a otra.
- **Limitación: la línea recta ignora ríos y puentes.** Dos estaciones con un
  río en medio pueden quedar dentro de la distancia máxima aunque el camión
  tenga que rodear por un puente. El simulador no lo corrige. Hay que decirlo
  al presentar los resultados y tenerlo en cuenta al contestar la pregunta 4.
- **Hechos de la corrida actual (10 de octubre de 2026).** El simulador deja
  fuera 42 estaciones sin capacidad y 41 que tienen capacidad pero ningún
  viaje en los días simulados; simula las otras 2,315. Todas las estaciones
  usan la estrategia "mixta" porque ninguna tiene tipo todavía: la ocupación
  de respaldo es 50 % en todas y la prioridad no cambia nada hoy, porque todas
  tienen la misma. Los resultados dependen de la ocupación de respaldo, porque
  es el inventario inicial supuesto y el día de calentamiento no quita esa
  dependencia; por eso hace falta confirmar ese valor (pregunta 5).

## Registro de cambios

| Fecha | Quién | Qué cambió y por qué |
|---|---|---|
| _pendiente_ | _pendiente_ | Versión inicial con valores provisionales |
| 2026-10-10 | Acker (programación) | En la pregunta 5, `ocupacion_objetivo` pasa a llamarse `ocupacion_respaldo` y queda decidido su significado. Se anota cómo se asigna el tipo de estación, qué rutas muestra el dashboard y para qué se usa la captura de GBFS. Ningún valor cambió. |
| 2026-10-10 | Acker (programación) | La pregunta 3 cambia de unidad: el presupuesto ya no es el total de bicis del día sino bicis movidas por estación por día, porque el número de estaciones simuladas puede cambiar. `presupuestos_diarios` (0, 50, 100, 200, 400, 800) se sustituye por `presupuestos_por_estacion` (0, 0.1, 0.25, 0.5, 1.0, 2.0). Se agregan tres parámetros, todos provisionales: `presupuesto_de_referencia` (0.5), `parte_del_presupuesto` por franja (50 % y 50 %) y `tipo_si_falta` (mixta). Se anota cómo arranca la simulación, qué limita un movimiento y que la línea recta ignora ríos y puentes. Los demás valores no cambiaron. |
| 2026-10-10 | Acker (programación) | Los ejemplos de la pregunta 3 pasan de 2,300 estaciones (un número redondo) a las 2,315 que se simulan hoy: la lista provisional da 0, 232, 579, 1,158, 2,315 y 4,630 bicis al día. Se anota cómo se redondea. El umbral de riesgo (pregunta 2) ya no acepta 0: va de 1 en adelante. Se anotan tres hechos de la corrida actual en "Ya decidido". Ningún valor cambió. |
