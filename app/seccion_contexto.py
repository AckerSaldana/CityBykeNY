"""Sección del dashboard: ¿el tipo de estación coincide con los empleos y residentes cercanos?

Expone una sola función, mostrar(). Lee, por medio de datos.py, una tabla
pequeña que escribe scripts/build_station_context.py. No importa nada de src.
"""

from __future__ import annotations

from pathlib import Path

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TIPOS_ESTACION = {"oficinas": "Oficinas", "residencial": "Residencial", "recreativa": "Recreativa",
                  "mixta": "Mixta"}
COLORES_TIPO = {"oficinas": "#0072B2", "residencial": "#D55E00", "recreativa": "#009E73", "mixta": "#8A8F98"}
FALTA = "Esta sección todavía no tiene datos: falta preparar la información de empleos y residentes."


def _correlacion(x: pd.Series, y: pd.Series) -> float:
    """Correlación de orden (Spearman: Pearson sobre los rangos). No la mueven los valores extremos."""
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def mostrar(ruta: str | Path | None = None) -> None:
    """Dibuja la sección. `ruta` solo se usa en pruebas."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # para encontrar datos.py también en pruebas
    import datos  # el único archivo de la app que lee tablas

    ruta = Path(ruta) if ruta else datos.ruta_de_contexto()
    st.subheader("¿El tipo de estación coincide con lo que hay alrededor?")
    if not ruta.exists():
        st.info(FALTA)
        return
    df = datos.tabla_pequena(ruta)
    anio = int(df["anio_datos"].iloc[0]) if "anio_datos" in df.columns and len(df) else None
    df = df[df["proporcion_empleos"].notna() & df["am_indice"].notna() & df["tipo"].notna()].copy()
    if len(df) < 3:
        st.info(FALTA)
        return
    for col in ["proporcion_empleos", "am_indice", "empleos", "residentes"]:
        df[col] = df[col].astype(float)
    df["tipo_texto"] = df["tipo"].map(TIPOS_ESTACION).fillna("Mixta")

    orden = _correlacion(df["proporcion_empleos"], df["am_indice"])
    if math.isnan(orden) or abs(orden) < 0.1:
        lectura = "saber cuántos empleos hay cerca casi no dice hacia dónde se mueven las bicis en la mañana"
    elif orden > 0:
        lectura = ("donde hay más empleos que trabajadores residentes, en la mañana entre semana "
                   "tienden a llegar más bicis de las que salen")
    else:
        lectura = ("donde hay más empleos que trabajadores residentes, en la mañana entre semana "
                   "tienden a salir más bicis de las que llegan")
    st.markdown(
        f"En {datos.num(len(df))} estaciones, la proporción de empleos a 300 metros y el flujo neto de la "
        f"mañana tienen una correlación de {datos.num(orden, 1)} (de -1 a 1; 0 es sin relación): {lectura}."
    )
    st.markdown(
        "El tipo de cada estación sale solo de su patrón de uso (a qué horas salen y llegan bicis). Los datos "
        "de empleo del Censo no se usaron para asignarlo: lo confirman por separado."
    )

    orden_tipos = [TIPOS_ESTACION[t] for t in TIPOS_ESTACION]
    grafica = (
        alt.Chart(df)
        .mark_circle(size=28, opacity=0.55)
        .encode(
            x=alt.X("proporcion_empleos:Q", title="Proporción de empleos a 300 metros",
                    axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                  labelExpr="format(datum.value * 100, '.0f') + ' %'"),
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("am_indice:Q", title="Flujo neto de la mañana entre semana (de -1 a 1)",
                    scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("tipo_texto:N", title="Tipo de estación", sort=orden_tipos,
                            scale=alt.Scale(domain=orden_tipos, range=[COLORES_TIPO[t] for t in TIPOS_ESTACION])),
            tooltip=[
                alt.Tooltip("estacion:N", title="Estación"),
                alt.Tooltip("tipo_texto:N", title="Tipo de estación"),
                alt.Tooltip("proporcion_empleos:Q", title="Proporción de empleos", format=".1%"),
                alt.Tooltip("am_indice:Q", title="Flujo neto de la mañana", format=".1f"),
                alt.Tooltip("empleos:Q", title="Empleos a 300 metros", format=",.0f"),
                alt.Tooltip("residentes:Q", title="Trabajadores residentes a 300 metros", format=",.0f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(grafica, width="stretch")

    medianas = df.groupby("tipo")["proporcion_empleos"].median()
    partes = [f"{TIPOS_ESTACION[t]} {datos.pct(medianas[t])}" for t in TIPOS_ESTACION if t in medianas.index]
    st.markdown("Proporción de empleos en la estación típica de cada tipo (mediana): " + "; ".join(partes) + ".")
    meses = datos.meses_de_historia()
    cuando = "antes de los días de validación" if meses == datos.SIN_DATO else f"de {meses}"
    st.caption(
        "Flujo neto de la mañana: llegadas menos salidas entre llegadas más salidas, de 7 a 10 entre semana, "
        f"{cuando}. Positivo: llegan más bicis de las que salen. "
        "Proporción de empleos: empleos entre empleos más trabajadores residentes. "
        "Los residentes son trabajadores (personas con empleo, contadas donde viven), no población total. "
        f"Fuente: Oficina del Censo de Estados Unidos (LODES), datos de {anio if anio else 'un año anterior'}, "
        "anteriores a los viajes de 2026; el Censo agrega ruido a propósito para proteger la confidencialidad."
    )
