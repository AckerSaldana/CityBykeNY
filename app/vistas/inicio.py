"""Página "Inicio". Esqueleto: título y línea discreta; el contenido va debajo."""

import streamlit as st

import datos  # app/datos.py: el único archivo que sabe dónde están los datos

datos.encabezado("Rebalanceo de Citi Bike en Nueva York")

ind = datos.indicadores()  # cifras ya calculadas; si faltan las tablas, avisa y se detiene
