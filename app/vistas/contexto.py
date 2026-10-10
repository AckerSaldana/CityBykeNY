"""Página 6: ¿el tipo de estación coincide con los empleos y residentes de alrededor?

El contenido vive en app/seccion_contexto.py (datos de empleo del Censo, LODES).
"""
import datos
import seccion_contexto

datos.encabezado("Contexto de las estaciones", "viajes")
seccion_contexto.mostrar()

import seccion_clima  # demanda en horas con lluvia contra horas secas (clima observado)

seccion_clima.mostrar()
