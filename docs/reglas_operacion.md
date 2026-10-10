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
clasificarlas (ver los pendientes de abajo). Una estación sin tipo no tiene
estrategia: no toma la de ningún otro tipo. Mientras falten los umbrales, esta
tabla no cambia ningún resultado.

La ocupación de respaldo tiene un solo uso: es el valor al que se lleva una
estación CON tipo cuando ningún inventario evita a la vez que se vacíe y se
llene en la franja ("franja sin solución"). El resultado se ajusta para que
no quede bajo el umbral de riesgo (pregunta 2). Una estación sin tipo no se
toca en una franja sin solución: ni recibe ni entrega bicis.

Ya no es el inventario inicial de la simulación. Ese sale de la ocupación
observada en la captura de GBFS (ver "Ya decidido").

La prioridad ordena a quién se atiende primero cuando el presupuesto no
alcanza. Dentro de la misma prioridad se atiende primero a la estación a la
que le faltan o le sobran menos bicis. Las estaciones sin tipo van después de
las que sí tienen.

El significado ya está decidido; los valores siguen siendo provisionales.

**Por qué importa.** Una estación de oficinas se llena en la mañana y se vacía
en la tarde; una residencial hace lo contrario. La misma regla no sirve para
todas.

**Para pensarlo.** ¿Qué tipo de estación duele más cuando falla? ¿En qué
franja conviene dejar una estación con bicis y en cuál con lugares libres,
cuando no se puede evitar todo?

| Tipo | Prioridad (3 alta, 1 baja) | Ocupación de respaldo en la mañana | Ocupación de respaldo en la tarde | Estado |
|---|---|---|---|---|
| oficinas | 3 | 30 % | 70 % | provisional |
| residencial | 3 | 70 % | 30 % | provisional |
| recreativa | 1 | 50 % | 50 % | provisional |
| mixta | 2 | 50 % | 50 % | provisional |

**Respuesta de logística:** _pendiente_

**Justificación:** _pendiente_

**Parámetro en el YAML:** `estrategia_por_tipo`, con `prioridad` y
`ocupacion_respaldo` para cada tipo (la ocupación se escribe como fracción:
30 % es `0.30`).

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
- **Qué se hace con una estación sin tipo.** Ya no existe `tipo_si_falta`.
  Una estación sin tipo no usa ninguna ocupación de respaldo (en una franja
  sin solución no se toca) y en el reparto va después de las que sí tienen
  tipo. Cuando existan los umbrales, esto solo aplicará a las estaciones que
  sigan sin tipo.

Ya decidido (se anota para tenerlo presente; no hay nada que responder):

- **Rutas.** El dashboard muestra los pares de estaciones más transitados:
  los viajes de una estación a otra, sumados en todo el periodo. Las rutas de
  camión quedan fuera por ahora: las recomendaciones dicen cuántas bicis
  llevar o retirar por estación, no de qué estación a cuál.
- **Captura de GBFS.** No se traslapa con los meses de viajes (es de octubre
  de 2026). Se usa para la capacidad de las estaciones, el inventario inicial
  de la simulación, el modo en vivo y el perfil de ocupación por hora. No se
  usa para validar la posible demanda insatisfecha. La capacidad y el
  inventario inicial quedan congelados en `config/gbfs_congelado.csv`, para
  que los resultados no cambien cada vez que la captura crece.
- **Arranque de la simulación.** El simulador empieza el 1 de septiembre de
  2026 con cada estación en la ocupación que tenía en la captura de GBFS entre
  las 00:00 y las 00:59 (una madrugada de octubre de 2026, no la de ese día:
  es un supuesto). Ese primer día es de calentamiento y no cuenta en los
  resultados. Después la simulación sigue de corrido, sin reinicios: cada día
  empieza con las bicis que dejó el anterior, y toda bici que se mueve gasta
  presupuesto.
- **El total de bicis no cambia.** Una bici que sale en un viaje vuelve
  siempre a una estación simulada: a la de destino o, si el viaje termina
  fuera de las estaciones simuladas, a la simulada más cercana al punto donde
  terminó (si el viaje no trae ese punto, a la estación de la que salió).
- **Banda de incertidumbre.** El rango de inventario que se acepta para una
  estación se calcula con el pronóstico y una banda alrededor. El ancho de
  la banda se elige con los datos de validación (config/datos.yaml) y es el
  mismo para la regla histórica y para el modelo. Si la estación ya está
  dentro del rango no se mueve nada; si no, se lleva solo hasta el borde más
  cercano.
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
  viaje en los días simulados; simula las otras 2,315. Ninguna tiene tipo
  todavía: la prioridad y la ocupación de respaldo de la pregunta 5 no cambian
  nada hoy. La red arranca con 33,889 bicis en 70,221 lugares (la ocupación
  de la captura de GBFS) y ese total no cambia en toda la corrida. Los
  resultados dependen mucho de ese inventario inicial, que es un supuesto: el
  reporte de la simulación trae una tabla de sensibilidad con otros
  inventarios. Las cifras están en `data/processed/reports/simulacion.md`.

## Registro de cambios

| Fecha | Quién | Qué cambió y por qué |
|---|---|---|
| _pendiente_ | _pendiente_ | Versión inicial con valores provisionales |
| 2026-10-10 | Acker (programación) | En la pregunta 5, `ocupacion_objetivo` pasa a llamarse `ocupacion_respaldo` y queda decidido su significado. Se anota cómo se asigna el tipo de estación, qué rutas muestra el dashboard y para qué se usa la captura de GBFS. Ningún valor cambió. |
| 2026-10-10 | Acker (programación) | La pregunta 3 cambia de unidad: el presupuesto ya no es el total de bicis del día sino bicis movidas por estación por día, porque el número de estaciones simuladas puede cambiar. `presupuestos_diarios` (0, 50, 100, 200, 400, 800) se sustituye por `presupuestos_por_estacion` (0, 0.1, 0.25, 0.5, 1.0, 2.0). Se agregan tres parámetros, todos provisionales: `presupuesto_de_referencia` (0.5), `parte_del_presupuesto` por franja (50 % y 50 %) y `tipo_si_falta` (mixta). Se anota cómo arranca la simulación, qué limita un movimiento y que la línea recta ignora ríos y puentes. Los demás valores no cambiaron. |
| 2026-10-10 | Acker (programación) | Los ejemplos de la pregunta 3 pasan de 2,300 estaciones (un número redondo) a las 2,315 que se simulan hoy: la lista provisional da 0, 232, 579, 1,158, 2,315 y 4,630 bicis al día. Se anota cómo se redondea. El umbral de riesgo (pregunta 2) ya no acepta 0: va de 1 en adelante. Se anotan tres hechos de la corrida actual en "Ya decidido". Ningún valor cambió. |
| 2026-10-10 | Acker (programación) | Simulador v2. Se quita `tipo_si_falta`: una estación sin tipo no toma la estrategia de otro tipo. `ocupacion_respaldo` ya solo se usa en las franjas sin solución de las estaciones con tipo, y ya no es el inventario inicial: ese sale de la captura de GBFS congelada en `config/gbfs_congelado.csv`. Dentro de la misma prioridad se atiende primero la diferencia más pequeña. El total de bicis ya no cambia durante la simulación. Ningún valor de la tabla cambió. |
