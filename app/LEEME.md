# Tablero (carpeta `app/`)

El usuario final es un analista de operaciones que no programa.

## Cómo abrirlo

Desde la carpeta raíz del proyecto, con el entorno activado:

    python scripts/build_app_tables.py     # resume los datos (unos segundos)
    streamlit run app/Inicio.py

El primer comando deja en `data/processed/app/` unas tablas pequeñas, que son
lo único que lee el tablero. **Hay que repetirlo cada vez que cambien los
datos o se corra una simulación nueva**; el tablero toma las tablas nuevas sin
reiniciarlo. Si faltan, cada página muestra un mensaje que dice qué correr.

Para comprobar que todo abre después de un cambio:

    pytest tests/test_app_smoke.py tests/test_app_tables.py

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `app/Inicio.py` | Entrada. Solo arma el menú: orden y títulos de las páginas. |
| `app/vistas/inicio.py` | Página 1, Inicio: el problema, cuatro indicadores, la curva de riesgo contra presupuesto y tres hallazgos. |
| `app/vistas/mapa_de_demanda.py` | Página 2, Mapa de demanda. |
| `app/vistas/estaciones_y_rutas.py` | Página 3, Estaciones y rutas. |
| `app/vistas/pronostico_por_estacion.py` | Página 4, Pronóstico por estación. |
| `app/vistas/confiabilidad.py` | Página 5, Confiabilidad. |
| `app/datos.py` | Lee las tablas, da formato a los números y guarda las etiquetas. Es el único archivo que sabe dónde están los datos. |
| `src/app_tables.py` | El constructor de las tablas pequeñas (lo corre `scripts/build_app_tables.py`). Ahí están las definiciones. |

## Reglas de las páginas

- Cada página empieza con `datos.encabezado("Título")`: pone el título y una
  sola línea discreta, "Resultados de simulación con datos de validación".
- En pantalla no van nombres de columnas ni de archivos, tablas crudas ni
  listas de pendientes.
- Números con `datos.num()` y `datos.pct()` (separador de miles y un decimal
  como mucho). Lo que falta se escribe con `datos.texto()`: "Sin dato".
- Políticas, franjas, niveles de riesgo, tipos de día y de estación se
  traducen con los diccionarios de `app/datos.py` (`POLITICAS`, `FRANJAS`...).
- Ninguna cifra se escribe a mano: salen de `datos.indicadores()`,
  `datos.hallazgos()` y las tablas.
- `app/` no importa nada de `src/`.

## Cómo leer lo que se muestra

- Los datos solo registran demanda atendida: un viaje que no se pudo hacer
  porque la estación estaba vacía no aparece.
- Lo observado se compara con el pronóstico solo en los días de validación. La
  prueba final no se muestra ni se califica.
- Recomendaciones: una fila por estación, día, franja horaria y política, con
  el presupuesto de referencia. Las bicis son las que esa política movió en la
  simulación (positivo es llevar, negativo es retirar). El inventario es el
  simulado justo después del traslado. No es el inventario ideal.
- Si una franja queda en riesgo alto y no se movió ninguna bici,
  el presupuesto no alcanzó para esa estación.
- Una estación sin recomendaciones no se simuló (no tiene capacidad conocida o
  no tuvo viajes en esos días).
- En el mapa, el filtro de tipo de usuario solo cambia las salidas: los datos
  no separan las llegadas por tipo de usuario.
- Capacidad o tipo de estación desconocidos se muestran como "Sin dato".
