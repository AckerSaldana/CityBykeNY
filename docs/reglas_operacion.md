# Reglas de operación del rebalanceo

> **Estado: CONFIRMADO por logística (Rodrigo Gutiérrez, 2026-10-10).**
> Cada respuesta dice de dónde sale cada número con una de cuatro etiquetas:
> **[Dato]** viene de una fuente publicada (al final están los enlaces);
> **[Cálculo propio]** lo calculamos nosotros con los viajes de octubre de
> 2025 o con el feed GBFS del 10 de octubre de 2026; **[Supuesto]** es una
> decisión razonada que ningún dato fija; **[Falta validar]** se tiene que
> revisar con el tramo de entrenamiento.
> En el pitch, los supuestos se presentan como supuestos.
>
> Una **estación-día** es una estación en un día (en octubre de 2025, unas
> 2,100 estaciones por 31 días, unas 66,000).

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

## Cómo usa estas reglas el simulador (acordado con programación)

- Se simula la red limpia de Nueva York: 2,315 estaciones en la corrida del 10
  de octubre de 2026 (Jersey City y Hoboken quedaron fuera en la limpieza).
- El inventario pasa de un día al otro. Cada tramo se simula por separado y
  su primer día arranca con la ocupación de respaldo de la primera franja y
  no cuenta (es de calentamiento): en prueba, el 15 de septiembre no cuenta y
  se reportan del 16 al 29. La ronda nocturna gasta presupuesto.
- El riesgo se cuenta las 24 horas, para que ninguna política pueda empujar el
  problema fuera de las franjas. El reporte lleva tres renglones que suman las
  24 horas (mañana de 06:00 a 14:00, tarde de 14:00 a 22:00 y fuera de franja
  de 22:00 a 06:00) y también se reporta por día.
- En el momento de intervenir, todas las políticas ven el mismo inventario
  simulado, como un operador real; ninguna ve viajes futuros. La regla
  histórica y el modelo usan la misma maquinaria y la misma ocupación de
  respaldo: la única diferencia entre ellas es el pronóstico.
- Bicis clásicas y eléctricas van juntas en el inventario (limitación
  declarada en la pregunta 5).
- Tramos: entrenamiento del 1 de julio al 31 de agosto de 2026, validación
  del 1 al 14 de septiembre y prueba del 15 al 29 de septiembre.

## De dónde salen los números

- **Reportes mensuales de Citi Bike a NYC DOT** (2019 y 2024 a 2026): viajes,
  estaciones, flota, bicis rebalanceadas y el texto de los niveles de servicio
  del contrato.
- **Viajes de octubre de 2025** (archivo oficial `202510-citibike-tripdata.zip`):
  4,731,261 viajes que empiezan en octubre; 2,132 estaciones con al menos 5
  llegadas más salidas por día. Es un mes anterior a los tres tramos, así que
  no los contamina. Los cálculos con este mes son exploratorios y en
  retrospectiva: los definitivos se repiten con el tramo de entrenamiento con
  `notebooks/b/calibrar_reglas.py`. En verano hay más usuarios ocasionales,
  así que algunos umbrales se van a mover.
- **Feed GBFS en vivo** del 10 de octubre de 2026: capacidad de cada estación
  y una sola foto del estado de madrugada (sábado, 03:16).
- **Literatura** sobre rebalanceo de Citi Bike (Cornell, 2015 a 2019) y
  contratos de otros sistemas.

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
| manana | 05:00 | 06:00 | 14:00 | 60 % | confirmado |
| tarde | 14:00 | 14:00 | 22:00 | 40 % | confirmado |

**Respuesta de logística:** Dos rondas de camiones al día. La **ronda
nocturna** trabaja de madrugada y deja las estaciones listas a las 05:00 para
la franja de 06:00 a 14:00, que contiene el pico de la mañana; le toca el 60%
del presupuesto. La **ronda de mediodía** trabaja entre el pico de la mañana y
el de la tarde y deja las estaciones listas a las 14:00 para la franja de 14:00
a 22:00, que contiene el pico de la tarde; le toca el 40%, más lo que la
nocturna no haya usado.

**Justificación:**
- [Cálculo propio] Con octubre de 2025, dentro de las franjas ocurre el 92% de
  los viajes entre semana y el 87% en fin de semana. Las provisionales (06:00
  a 12:00 y 15:00 a 21:00) cubrían 74% y 61%: dejaban fuera de 12:00 a 15:00,
  donde ocurre el 15% de los viajes entre semana y el 23% en fin de semana
  (tres de las cinco horas más concurridas del sábado y el domingo).
- [Cálculo propio] Los picos entre semana caen a las 08:00 (7.8% de los viajes
  del día) y a las 17:00 (10.5%), dentro de cada franja. Entre los dos picos,
  las horas más tranquilas son de 10:00 a 12:00 (4% cada una): el hueco
  natural para la ronda de mediodía. [Dato] Coincide con las horas pico de la
  literatura sobre Citi Bike (6 a 10 y 16 a 20) y del metro (7 a 10 y 16 a 19).
- [Dato] En los reportes mensuales de 2024 a 2026, el nivel de servicio 11 del
  contrato dice que ninguna estación debe estar fuera de servicio más de 4
  horas y que esa multa no corre entre las 22:00 y las 06:00: las franjas
  cubren el horario en que ese nivel cobra. El nivel 10 vigila las estaciones
  "Never-Die" (cerca del 10%, lista no pública) hasta la medianoche, y el
  Contralor de NYC (2023) describe otra versión del contrato con metas también
  fuera de la hora pico. Por eso el riesgo se cuenta las 24 horas. El contrato
  de 2015 de Bay Wheels y el del antiguo Hubway usaban la ventana de 06:00 a
  22:00.
- [Dato] La ronda nocturna es la práctica documentada: en los estudios de
  Cornell con el operador salían de 3 a 5 camiones cada noche (rutas piloto de
  unas 8 horas; datos de 2016). Hoy Citi Bike dice que rebalancea a toda hora,
  y tiene valet en 6 estaciones de Nueva York, casi todas entre semana de 7:00
  a 11:00 y de 16:00 a 20:00.
- Reparto 60 y 40 [Supuesto]: de día los camiones van lento (Midtown a 4 a 6
  millas por hora según NYC DOT; Lyft dice que sus camionetas tardan hasta 40%
  más por el tráfico), el operador prefiere mover camiones llenos de noche
  (O'Mahony y Shmoys, 2015) y de noche hay que deshacer el desbalance del día
  anterior. Si da tiempo, una sensibilidad con 50 y 50.
- [Supuesto] La ronda de mediodía trabaja de unas 10:00 a las 14:00. El
  simulador la representa como un movimiento instantáneo a las 14:00, con el
  inventario simulado de esa hora. Los camiones reales salen hacia las 10:00
  sin saber qué pasará de 10:00 a 14:00 (17% de los viajes entre semana), así
  que la tarde es un poco optimista para la regla histórica y para el modelo,
  por igual. Si da tiempo, una sensibilidad en la que la tarde decide con el
  inventario de las 10:00.
- [Supuesto] Las mismas franjas aplican al fin de semana, porque el YAML no
  distingue tipo de día.
- Costo de la decisión [Cálculo propio]: con franjas de 8 horas, más
  estaciones no tienen ningún inventario inicial que aguante toda la franja
  (umbral de la pregunta 2): 3.7% de las estación-día en la mañana y 3.1% en
  la tarde, contra 3.0% y 1.8% con las franjas provisionales.

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
| Bicis mínimas | 2 | confirmado |
| Lugares libres mínimos | 3 | confirmado |

**Respuesta de logística:** Una estación está en riesgo en una hora si le
quedan 0 o 1 bicis, o 0, 1 o 2 lugares libres. Se cuenta las 24 horas.
Excepción: las 6 estaciones con valet no cuentan como riesgo en sus horas de
valet (ver abajo).

**Justificación:**
- [Cálculo propio] El simulador mira una foto del inventario en cada hora en
  punto, pero las bicis entran y salen dentro de la hora. En octubre de 2025,
  en el 32% de las estación-hora con movimiento, el inventario bajó dentro de
  la hora al menos 1 bici por debajo de lo que muestran las fotos; en el 12%,
  al menos 2. En estaciones concurridas (10 o más movimientos en la hora), 60%
  y 33%. Hacia el lado lleno pasa lo mismo. Ejemplo: si en la foto la estación
  tiene 2 bicis y dentro de la hora bajó 1 más, llegó a 1 pero nunca a 0, que
  es el caso más común. Si bajó 2, sí se vació.
- [Dato] Los estudios de Cornell con Citi Bike usan una "falla suave": la
  estación cuenta como fallando antes de llegar a cero, porque hay bicis y
  anclajes descompuestos; dicen que el valor correcto está entre 2 y 5.
- Por qué un lugar más del lado lleno [Cálculo propio y supuesto]: en la foto
  GBFS de las 03:16 del 10 de octubre de 2026 había 5,693 bicis descompuestas
  ocupando anclajes, unas 2.3 por estación. El simulador no las ve, así que
  cree que hay más lugares libres de los que hay. Además, quien no puede
  devolver tiene la bici en la mano y el tiempo sigue corriendo; quien no
  puede sacar una bici solo no empieza el viaje.
- [Cálculo propio] 2 y 3 son el 9% y el 13% de la capacidad de la estación
  mediana (23 anclajes).
- Costo [Cálculo propio]: con 2 y 3, el 3.7% de las estación-día no tienen
  ningún inventario inicial que aguante la mañana; con 2 y 2 sería el 2.6% y
  con 3 y 3 el 5.0%. El umbral es el mismo para las tres políticas, así que la
  comparación entre ellas es justa.
- Excepción de las estaciones con valet [Dato, cálculo propio y supuesto]: en
  9 Ave & W 33 St llegan en promedio 126 bicis netas por mañana hábil y solo
  tiene 94 anclajes; en la calle el valet recibe las bicis fuera de los
  anclajes. El simulador las vería llenas toda la mañana con cualquier
  inventario, y ningún camión lo arregla. Por eso, en sus horas de valet, no
  cuentan como riesgo ni reciben presupuesto. Son 9 Ave & W 33 St (6492.08),
  E 47 St & Park Ave (6584.12), Dock 72 Way & Market St (4804.02), 1 Ave & E
  68 St (6822.09) y Park Pl & Church St (5288.08), de lunes a viernes de 7:00
  a 11:00 y de 16:00 a 20:00; y North Moore St & Greenwich St (5470.12), de
  6:00 a 10:00 y de 16:00 a 20:00 (la página de Citi Bike no dice los días;
  [Supuesto] de lunes a viernes, como las demás).

**Parámetro en el YAML:** `umbral_riesgo.bicis_minimas` y
`umbral_riesgo.lugares_libres_minimos`. Son enteros, de 1 en adelante. Con 0
ninguna estación contaría como en riesgo, porque nunca hay menos de 0. Para
contar solo las estaciones vacías o llenas, escribe 1. La excepción va en
`excepciones_valet` (aceptada por programación; el validador todavía no la
revisa).

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
| Presupuestos a simular (bicis por estación por día) | 0, 0.35, 0.7, 1.4, 2.6, 4.3 | 0, 810, 1,620, 3,241, 6,019 y 9,954 bicis al día | confirmado |
| Presupuesto de referencia (el de las recomendaciones del dashboard) | 1.4 | 3,241 bicis al día | confirmado |
| Reparto entre franjas | manana 60 %, tarde 40 % | unas 1,945 y 1,296 bicis con el de referencia | confirmado (se anota en la pregunta 1) |

**Respuesta de logística:** Presupuestos de 0, 0.35, 0.7, 1.4, 2.6 y 4.3 bicis
por estación por día: sin intervención, mitad del austero, austero, realista,
ambicioso y un techo de referencia. El de referencia es el realista, 1.4.

**Justificación:**
- [Dato] Citi Bike reportó 100,909 bicis rebalanceadas en octubre de 2025, en
  promedio 3,255 al día, con 2,287 estaciones; en julio de 2026, 3,379 al día
  con 2,424. En 2024 eran de 850 a 1,430 al día hasta julio y de 1,960 a
  2,640 después. En julio de 2019 eran 6,177 al día con 778 estaciones.
- Realista, 1.4 [Dato y supuesto]: lo que Citi Bike reporta mover hoy por
  estación (3,255 / 2,287 = 1.42 en octubre de 2025; 3,379 / 2,424 = 1.39 en
  julio de 2026). [Dato] El reporte no dice si la cifra son bicis o
  "acciones" (en 2016 se contaba por separado cada vez que se recogía y cada
  vez que se dejaba una bici), ni si incluye valets y viajes de Bike Angels:
  lo atribuye a su personal y menciona esos medios sin separarlos. Por eso es
  un techo para lo que mueven solo los camiones.
- Austero, 0.7 [Dato y supuesto]: la mitad del realista. Es el nivel de los
  meses de invierno y de 2024, y también la lectura de "acciones" de la cifra
  actual: si fueran acciones, todos los medios juntos moverían la mitad, y
  los camiones solos, menos.
- Ambicioso, 2.6 [Dato y supuesto]: el volumen de julio de 2019 (6,177 al día)
  repartido entre la red de hoy, cuando el rebalanceo equivalía al 8.6% de
  los viajes. Se toma el volumen, no su intensidad por estación de entonces
  (7.9), que hoy serían más de 18,000 bicis. Tiene la misma duda de bicis o
  acciones.
- Techo, 4.3 [Cálculo propio]: en una mañana hábil de octubre de 2025, de
  06:00 a 14:00, se acumulan unas 10,000 bicis netas en las estaciones que
  ganan (en la tarde, unas 10,500). No es un presupuesto operativo: sirve
  para ver dónde se aplana la curva.
- 0.35 [Supuesto, con cálculo propio]: el desbalance está concentrado (el 10%
  de las estaciones explica el 28% del desbalance diario y el 20%, el 45%),
  así que la curva debería subir rápido con poco presupuesto; este punto lo
  deja ver.
- [Cálculo propio] Como el inventario pasa de un día al otro, la ronda
  nocturna tiene que deshacer también el desbalance del día anterior, igual
  que la operación real. Al cierre de un día hábil de octubre de 2025, las
  estaciones que ganan bicis acumulan unas 4,900 netas y las que pierden se
  quedan sin unas 5,300. Si la cifra del reporte son bicis, Citi Bike mueve
  unas dos terceras partes de ese desbalance; si son acciones, un tercio. Con
  presupuestos menores que ese desbalance, "sin intervención" y los
  presupuestos bajos van a empeorar día tras día: por eso se reporta también
  por día.
- [Supuesto] El presupuesto es solo de camiones. Los viajes de Bike Angels ya
  están dentro de los datos de viajes (no vienen marcados), así que su efecto
  está en los flujos que se reproducen y no se suma aparte.

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
| Distancia máxima (km) | 2.0 | confirmado |

**Respuesta de logística:** 2.0 km en línea recta, corta a propósito porque la
línea recta no ve los ríos. Lo que todavía cruza el agua se declara como
limitación. Las 3 estaciones de Governors Island solo intercambian bicis
entre ellas.

**Justificación:**
- [Cálculo propio, en retrospectiva] Emparejando, en cada día hábil de octubre
  de 2025, las estaciones que terminaron con bicis de más con las que
  terminaron con bicis de menos, de la más cercana a la más lejana: con
  donantes a 2 km o menos se cubre el 63% de lo que falta en la franja de la
  mañana (el máximo posible es 88%, porque faltan más de las que sobran), el
  70% de lo que falta en la tarde y el 78% de lo que falta en el día completo.
  Con 3 km serían 72%, 82% y 82%: la distancia corta cuesta unos 10 puntos de
  cobertura.
- Ríos [Cálculo propio]: con 2 km, el 3.5% de las bicis emparejadas en la
  mañana cruzan el borde de Manhattan (East River o Harlem); con 3 km, el
  4.6%. Pasar de 3 a 2 km quita cerca de un tercio de las bicis que cruzan,
  pero no las elimina (con 1 km todavía cruza el 2.3%), así que se declara
  como limitación. [Dato] El puente de Brooklyn no admite camiones; el
  Williamsburg y el Queensboro, solo en carriles específicos.
- [Dato y cálculo propio] Con un recorrido real 1.3 a 1.4 veces la línea
  recta, 2 km son unos 14 a 26 minutos de manejo de día (Midtown va a 4 a 6
  millas por hora según NYC DOT; la zona de cobro de congestión iba a 7 antes
  del cobro, según MTA). Un traslado de 20 bicis a 2 km toma en Midtown de 56
  a 66 minutos (16 a 26 de manejo más unos 40 de carga y descarga): queda en
  el límite de una hora. A 3 km toma de 64 a 79 minutos.
- [Cálculo propio] En la red de Nueva York, la estación mediana tiene 110
  estaciones a 2 km o menos (la del percentil 10, 63; la que menos, 8, en
  Riverdale): ninguna queda sin donantes posibles.
- [Dato] El "tren de bicis" de Citi Bike repartía en un radio de unas 2 millas
  (3.2 km) en 2017. No encontramos ningún operador ni artículo que publique un
  tope en km.
- [Dato] Governors Island no tiene puente; sus estaciones (Soissons Landing
  4631.02, Yankee Ferry Terminal 4440.02 y Picnic Point 4374.01) quedan a
  cerca de 1 km de Manhattan y Brooklyn en línea recta, así que los 2 km no
  bastan para aislarlas. [Supuesto] Solo intercambian bicis entre ellas.
  Revisar también Roosevelt Island, cuyo puente vehicular va solo a Queens.
- Si da tiempo, una sensibilidad con 3.0 km.

**Parámetro en el YAML:** `distancia_maxima_km`. Governors Island va en
`grupos_aislados` (aceptado por programación; el validador todavía no lo
revisa).

---

## Pregunta 5. ¿Qué estrategia seguimos según el tipo de estación?

**Qué significa.** Clasificamos las estaciones en cuatro tipos: oficinas,
residencial, recreativa y mixta. Para cada tipo se definen dos cosas: a quién
atender primero cuando el presupuesto no alcanza (prioridad) y qué tan llena
empieza cada estación, como fracción de su capacidad (ocupación de respaldo,
de 0 % vacía a 100 % llena).

**Tipos de estación.** La regla para clasificarlas ya está (ver los
pendientes de abajo y `notebooks/b/tipo_estacion.sql`). Programación ya la
aplicó al tramo de entrenamiento y llenó `station_type`. Una estación que
sigue sin tipo (no tiene historia) no toma la estrategia de otro tipo: en una
franja sin solución factible no se toca.

La ocupación de respaldo tiene dos usos:

1. Es el inventario inicial común de las tres políticas (sin intervención,
   regla histórica y modelo predictivo): las tres empiezan con las mismas
   bicis en cada estación, para compararlas desde el mismo punto. Solo el
   primer día de cada tramo empieza así (ver "Ya decidido"), con el valor de
   la primera franja (manana).
2. Es el valor que se usa cuando ningún inventario inicial evita a la vez que
   la estación se vacíe y se llene.

La regla histórica y el modelo lo usan igual: la única diferencia entre ellas
es el pronóstico.

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
| oficinas | 3 | 10 % | 90 % | confirmado |
| residencial | 3 | 90 % | 15 % | confirmado |
| recreativa | 1 | 50 % | 50 % | confirmado |
| mixta | 2 | 50 % | 50 % | confirmado |

| Parámetro | Valor | Estado |
|---|---|---|
| Tipo que se aplica a una estación sin tipo | mixta | confirmado |

**Respuesta de logística:** El tipo no fija cuántas bicis se mueven: eso lo
calcula cada política para cada estación con su pronóstico. El tipo cambia dos
cosas: el orden de atención cuando el presupuesto no alcanza (prioridad) y el
inventario de respaldo en las estación-día que no tienen solución. En
promedio esperamos ver esto:
- **Oficinas** (prioridad alta): la ronda nocturna les retira bicis para dejar
  lugares a quienes llegan a trabajar; si la ola de la mañana no cabe de
  ninguna forma, se dejan casi vacías (10%) para que el desborde llegue lo
  más tarde posible. La de mediodía las ajusta para la salida de la tarde; si
  no hay solución, casi llenas (90%).
- **Residencial** (prioridad alta): al revés. La ronda nocturna les lleva
  bicis para quienes salen de casa; si no alcanzan, casi llenas (90%). La de
  mediodía les abre lugares para quienes regresan; si no hay solución, casi
  vacías (15%).
- **Mixta** (prioridad media): se atienden después de oficinas y residencial.
- **Recreativa** (prioridad baja todos los días, porque el YAML no distingue
  fin de semana): se atienden al final.

Una estación sin tipo se trata como mixta, igual que la regla de
clasificación hace con las de poca actividad.

**Justificación:**
- [Cálculo propio, exploratorio] El respaldo se usa a diario en las
  estación-día sin solución. Sin contar las estaciones con valet, en octubre
  de 2025 fueron, en la mañana, el 7.4% de las de oficinas y el 7.5% de las
  residenciales; en la tarde, el 6.3% y el 5.7%; en mixtas y recreativas, 2% o
  menos. En esas estación-día, las horas en riesgo dentro de la franja son:
  oficinas en la mañana, 3.1 al 10% contra 4.4 a la mitad; residenciales en la
  mañana, 2.5 al 90% contra 2.9 al 80% y 4.2 a la mitad; oficinas en la tarde,
  2.5 al 90% contra 2.8 al 80% y 3.6 a la mitad; residenciales en la tarde,
  1.7 al 15% contra 2.9 a la mitad. En las mixtas la mitad queda a menos de
  0.2 horas del mejor valor, y las recreativas sin solución son muy pocas (10
  a 14), así que se dejan en 50%.
- La idea operativa [Supuesto]: si una estación de oficinas va a recibir en la
  mañana más bicis de las que caben, cada lugar libre de más retrasa el
  desborde; por eso se deja casi vacía. Con las residenciales es lo
  contrario.
- Por qué no 0% ni 100% (criterio de programación): el riesgo cuenta el tiempo
  bajo el umbral, y una estación vacía o llena arranca ya en riesgo.
  [Cálculo propio] Con 3 lugares libres de umbral, 0.90 todavía deja 2 lugares
  en la estación mediana (23 anclajes), y el 54% de las estaciones arrancaría
  en riesgo; aun así, en las estación-día sin solución deja menos horas en
  riesgo que 0.80, porque las primeras salidas abren lugares enseguida. Se
  pide a programación acotar el respaldo entre `bicis_minimas` y capacidad −
  `lugares_libres_minimos`: con la cota, las horas bajan a 2.2 en
  residenciales de mañana y a 2.4 en oficinas de tarde, y ninguna estación
  arranca en riesgo.
- El valor inicial también se arrastra [Supuesto]: el primer día arranca a la
  medida de cada tipo, también para "sin intervención". Ese día no cuenta,
  pero sin intervención o con poco presupuesto cada día arranca donde terminó
  el anterior, así que el arranque sigue pesando hasta que la estación se
  vacía o se llena. Se declara.
- [Cálculo propio, una sola foto] La calle ya amanece acomodada en esa
  dirección: en la madrugada del sábado 10 de octubre de 2026 la mediana de
  llenado era 0.24 en oficinas, 0.68 en residenciales, 0.47 en mixtas y 0.39
  en recreativas.
- [Supuesto] "Sin intervención" quiere decir sin camiones en todo el periodo.
  Como los datos no traen el rebalanceo real, su riesgo va a crecer día tras
  día. Es la referencia de no hacer nada, no una foto de cómo está hoy la
  calle.
- [Dato, verificado en parte] O'Mahony y Shmoys (2015) agrupan las estaciones
  de Citi Bike por su uso en la hora pico y le ponen a cada grupo un nivel de
  llenado deseado; en una figura de resultados aparecen niveles de 10%, 50% y
  90%. Qué nivel va con qué grupo está en una tesis que no se pudo abrir.
- Prioridad [Cálculo propio y supuesto]: oficinas y residencial concentran el
  53% de las llegadas y salidas y son las que cambian de sentido entre mañana
  y tarde; empezando a la mitad, solo el 56% y el 57% de sus estación-día
  pasan la mañana sin riesgo. Las mixtas (45% de los movimientos) se
  equilibran solas casi siempre: el 91%. Las recreativas fallan más que las
  mixtas (85%), pero son el 1.7% de los movimientos y su pico es el fin de
  semana, que el YAML no distingue. La prioridad se aplica igual a la regla
  histórica y al modelo.
- Hallazgo [Dato y cálculo propio]: ordenando las 2,132 estaciones por
  llegadas netas en la mañana, cinco de las seis estaciones con valet en Nueva
  York quedan en los lugares 1, 2, 4, 7 y 14 (9 Ave & W 33 St y E 47 St & Park
  Ave, con unas 126 bicis netas por mañana hábil; Dock 72 Way & Market St;
  North Moore St & Greenwich St; 1 Ave & E 68 St). La sexta, Park Pl & Church
  St, queda en el lugar 154. Las seis salen como oficinas con la regla, con
  el cambio de signo de la tarde muy marcado. El dato de la mañana encuentra
  los mismos puntos que la operación real ya atiende con personal.
- Limitación [Dato]: el simulador junta bicis clásicas y eléctricas, pero las
  eléctricas hicieron el 70% de los viajes de 2025 y Lyft dice que en general
  no las mueve en camioneta. Se declara.
- [Supuesto] Las recreativas tienen prioridad baja todos los días. En fin de
  semana su demanda pesa más: distinguir el tipo de día es el siguiente paso.
- [Falta validar] Todo esto se repite con julio y agosto de 2026 con
  `notebooks/b/calibrar_reglas.py`.

**Parámetro en el YAML:** `estrategia_por_tipo`, con `prioridad` y
`ocupacion_respaldo` para cada tipo (la ocupación se escribe como fracción:
30 % es `0.30`).

---

## Pendientes que también decide logística

### Umbrales para clasificar cada estación en un tipo (confirmado)

Se calcula solo con el tramo de entrenamiento (del 1 de julio al 31 de agosto
de 2026), con estos indicadores por estación:

- **Índice de la mañana**: (llegadas − salidas) / (llegadas + salidas) entre
  semana de 7:00 a 10:00, sin festivos. Va de −1 a +1. Se usa normalizado y no
  el neto crudo para que no dependa del tamaño de la estación.
- **Índice de la tarde**: lo mismo de 16:00 a 19:00. Sirve para comprobar que
  la estación cambia de sentido.
- **Proporción de fin de semana**: llegadas y salidas en sábado y domingo
  entre el total.
- **Proporción de ocasionales**: salidas de usuarios ocasionales entre el
  total de salidas (`station_hour` solo trae `departures_casual`).

Regla, en este orden:

1. Menos de 5 llegadas más salidas por día en promedio: **mixta** (poca
   actividad, sin tipo propio).
2. **Oficinas**: índice de la mañana ≥ +0.20 (llegan al menos 1.5 veces las
   bicis que salen), índice de la tarde negativo, y al menos 5 movimientos
   por día hábil tanto de 7:00 a 10:00 como de 16:00 a 19:00.
3. **Residencial**: índice de la mañana ≤ −0.20, índice de la tarde positivo y
   los mismos mínimos.
4. **Recreativa**: sin patrón de ir al trabajo (índice de la mañana entre
   −0.20 y +0.20), al menos 20 llegadas más salidas por día, ocasionales ≥
   percentil 80 y fin de semana ≥ percentil 75 de las estaciones activas.
5. El resto: **mixta**.

Por qué así:
- Mínimo de 5 movimientos [Cálculo propio]: con pocos viajes el índice
  brinca. Su error típico es como mucho 1 / √(movimientos). Con 5 por día
  hábil y los 43 días hábiles de julio y agosto son unos 215 movimientos y un
  error típico de 0.07, así que una estación en ±0.20 está a unos 3 errores
  de cero. Es exigente: en octubre de 2025, 470 estaciones con índice fuerte
  no llegaron al volumen y quedaron como mixtas.
- Cambio de signo en la tarde (criterio de programación): una estación de
  oficinas o residencial se invierte en la tarde; si no, va a mixta.
  [Cálculo propio] En octubre de 2025 fueron 55 estaciones: 36 quedan casi
  parejas en la tarde (por ejemplo, LaGuardia Pl & W 3 St: +0.33 en la mañana
  y +0.01 en la tarde) y 5 van en el mismo sentido todo el día (por ejemplo,
  Fort Washington Ave & W 173 St, en Washington Heights: −0.33 y −0.15).
  [Supuesto] Las que van en el mismo sentido todo el día suelen deberlo a una
  pendiente.
- Recreativa sin patrón de ir al trabajo: así una estación de barrio con
  patrón residencial que no llega al volumen mínimo no termina con la
  prioridad más baja; queda mixta (prioridad media).
- Los cortes de ±0.20, de 5 y de 20 movimientos son fijos porque tienen
  lectura operativa. Los de ocasionales y fin de semana son percentiles
  porque esas proporciones cambian con la temporada (en octubre de 2025, 17%
  de los viajes fueron de ocasionales; en julio de 2026, 21%).

Con octubre de 2025 [Cálculo propio, exploratorio]: cortes de 0.240
(ocasionales) y 0.279 (fin de semana). De las 2,132 estaciones activas: 306
oficinas, 572 residenciales, 31 recreativas y 1,223 mixtas; otras 89 de poca
actividad quedan como mixtas. Por movimientos: 21%, 32%, 1.7% y 45%. El
índice de la mañana tiene correlación −0.73 con el de la tarde. Las
recreativas quedaron en Central Park, Prospect Park, Brooklyn Bridge Park, el
Seaport, la terminal del ferry de Staten Island, Governors Island, Randall's
Island, Astoria Park, Forest Park, Red Hook e Inwood, más unas ocho
estaciones de Bushwick y Ridgewood y algunas sueltas de barrio.

Sensibilidad del corte de ±0.20 (octubre de 2025, regla definitiva): con
±0.15, 359 oficinas y 640 residenciales; con ±0.20, 306 y 572; con ±0.25, 267
y 492; con ±0.30, 223 y 423. [Falta validar] Repetirla con julio y agosto de
2026: en verano hay más ocasionales y los percentiles se mueven.

`notebooks/b/tipo_estacion.sql` aplica la regla sobre `station_hour`, ya con
las fechas del tramo de entrenamiento y el festivo del 3 de julio de 2026. Se
probó contra el cálculo en Python con octubre de 2025: coinciden las 2,132
estaciones. `notebooks/b/calibrar_reglas.py` corre esa consulta y, con
`station_hour` y `stations`, recalcula qué parte de las estación-día no tiene
solución y qué ocupación de respaldo deja menos horas en riesgo en cada tipo y
franja. Se corre desde la raíz del repositorio con
`python notebooks/b/calibrar_reglas.py`; no escribe nada.

### Qué pasa con una estación sin tipo (confirmado)

No toma la estrategia de ningún tipo. En una franja sin solución factible la
estación sin tipo no se toca; en las demás franjas se trata como cualquier
otra.

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

## Fuentes

- Reportes mensuales de Citi Bike a NYC DOT (índice):
  https://citibikenyc.com/system-data/operating-reports
  - Octubre 2025: https://mot-marketing-whitelabel-prod.s3.amazonaws.com/nyc/October-2025-Citi-Bike-Monthly-Report.pdf
  - Julio 2026 (texto de los niveles de servicio 10 y 11): https://mot-marketing-whitelabel-prod.s3.us-east-1.amazonaws.com/nyc/July-2026-Citi-Bike-Monthly-Report.pdf
- Contralor de NYC, "Riding Forward" (2023):
  https://comptroller.nyc.gov/reports/riding-forward-overhauling-citi-bikes-contract-for-better-more-equitable-service/
- O'Mahony y Shmoys (2015), AAAI, rebalanceo de Citi Bike:
  https://ojs.aaai.org/index.php/AAAI/article/view/9245
- Freund, Henderson, O'Mahony y Shmoys, rebalanceo nocturno de Citi Bike:
  https://people.orie.cornell.edu/shane/pubs/BSOvernight.pdf
- Freund, Henderson y Shmoys, conteo de acciones de 2016:
  https://arxiv.org/pdf/1611.09304
- Freund et al. (2019), Interfaces, Bike Angels y valets:
  https://courses.ie.bilkent.edu.tr/ie102/wp-content/uploads/sites/16/2020/03/Analytics-and-Bikes-Riding-Tandem-with-Motivate-to-Improve-Mobility.pdf
- Estaciones con valet (actualizado el 9 de octubre de 2026):
  https://citibikenyc.com/service
- Lyft (2023), cómo rebalancea Citi Bike:
  https://www.lyft.com/blog/posts/the-math-that-ensures-you-always-get-a-citi-bike-when-you-want-it
- Lyft (2026), operación de Citi Bike y 70% de viajes en e-bike:
  https://www.lyfturbansolutions.com/blog/2026/02/a-behind-the-scenes-look-at-citi-bike-operations
- Datos de viajes de Citi Bike (qué se excluye, sin marca de Bike Angels):
  https://citibikenyc.com/system-data
- Contrato de Bay Wheels (2015):
  https://sanjose.granicus.com/MetaViewer.php?view_id=&event_id=1475&meta_id=544267
- Contrato del antiguo Hubway (Harvard GPL, 2019):
  https://govlab.hks.harvard.edu/wp-content/uploads/2019/02/boston_ma_bike_share_system_project_feature.pdf
- Horas pico del metro (MTA):
  https://catalog.data.gov/dataset/mta-subway-service-delivered-beginning-2025
- Velocidades: NYC DOT (2025) https://www.nyc.gov/html/dot/html/pr2025/nyc-dot-announces-gridlock-alert-days.shtml
  y MTA https://mta.info/document/131571
- Restricciones de camiones en puentes, NYC DOT (2024):
  https://www.nyc.gov/html/dot/downloads/pdf/dot_bridgereport24.pdf
- Tren de bicis, radio de 2 millas (amNY, 2017):
  https://amny.com/transit/citi-bike-expansion-making-it-harder-to-keep-stations-stocked-1-13730056
- Feed GBFS: https://gbfs.citibikenyc.com/gbfs/gbfs.json
- Viajes: https://s3.amazonaws.com/tripdata (archivo 202510)

## Registro de cambios

| Fecha | Quién | Qué cambió y por qué |
|---|---|---|
| 2026-10-10 | Acker (programación) | Versión inicial con valores provisionales |
| 2026-10-10 | Acker (programación) | En la pregunta 5, `ocupacion_objetivo` pasa a llamarse `ocupacion_respaldo` y queda decidido su significado. Se anota cómo se asigna el tipo de estación, qué rutas muestra el dashboard y para qué se usa la captura de GBFS. Ningún valor cambió. |
| 2026-10-10 | Acker (programación) | La pregunta 3 cambia de unidad: el presupuesto ya no es el total de bicis del día sino bicis movidas por estación por día, porque el número de estaciones simuladas puede cambiar. `presupuestos_diarios` (0, 50, 100, 200, 400, 800) se sustituye por `presupuestos_por_estacion` (0, 0.1, 0.25, 0.5, 1.0, 2.0). Se agregan parámetros provisionales: `presupuesto_de_referencia` (0.5) y `parte_del_presupuesto` por franja (50 % y 50 %); el tercero, que daba un tipo a las estaciones sin tipo, se quitó después. Se anota cómo arranca la simulación, qué limita un movimiento y que la línea recta ignora ríos y puentes. Los demás valores no cambiaron. |
| 2026-10-10 | Acker (programación) | Los ejemplos de la pregunta 3 pasan de 2,300 estaciones (un número redondo) a las 2,315 que se simulan hoy: la lista provisional da 0, 232, 579, 1,158, 2,315 y 4,630 bicis al día. Se anota cómo se redondea. El umbral de riesgo (pregunta 2) ya no acepta 0: va de 1 en adelante. Se anotan tres hechos de la corrida actual en "Ya decidido". Ningún valor cambió. |
| 2026-10-10 | Rodrigo Gutiérrez (logística) | Se confirman las cinco reglas y los umbrales de tipo de estación, con lo acordado con programación (inventario que pasa de un día al otro, cada tramo con su día de calentamiento, riesgo contado las 24 horas y reportado por franja y por día, la regla histórica y el modelo con la misma maquinaria y el mismo respaldo). Franjas 06:00-14:00 (intervención 05:00, 60 % del presupuesto) y 14:00-22:00 (intervención 14:00, 40 %); umbral 2 bicis y 3 lugares libres, sin contar las 6 estaciones con valet en sus horas; presupuestos de 0, 0.35, 0.7, 1.4, 2.6 y 4.3 bicis por estación por día, con 1.4 de referencia; 2.0 km, con Governors Island aislada; ocupación de respaldo 0.10/0.90 oficinas, 0.90/0.15 residencial y 0.50/0.50 recreativa y mixta. Tipos: índice de la mañana ±0.20 con cambio de signo de 16 a 19 y al menos 5 movimientos por día hábil en cada ventana; recreativa sin patrón de ir al trabajo, con 20 movimientos por día y percentiles 80 y 75 (`notebooks/b/tipo_estacion.sql`). Claves nuevas aceptadas por programación: `excepciones_valet` y `grupos_aislados`. Base: reportes mensuales de Citi Bike a NYC DOT, literatura y cálculo propio con los viajes de octubre de 2025. Los supuestos se declaran en cada justificación. |
| 2026-10-10 | Acker (programación) | Se quitan las menciones a la clave que daba un tipo a las estaciones sin tipo: ya no existe en `config/operacion.yaml`. Una estación sin tipo no se toca en las franjas sin solución factible. |
