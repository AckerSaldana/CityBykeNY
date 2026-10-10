"""Sección del dashboard: ¿el tipo de estación coincide con los empleos y residentes cercanos?

Expone una sola función, mostrar(). Lee una tabla pequeña que escribe
scripts/build_station_context.py. No importa nada de src.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TABLA = Path(__file__).resolve().parents[1] / "data" / "processed" / "app" / "contexto_estaciones.parquet"
TIPOS_ESTACION = {"oficinas": "Oficinas", "residencial": "Residencial", "recreativa": "Recreativa",
                  "mixta": "Mixta"}
COLORES_TIPO = {"oficinas": "#0072B2", "residencial": "#D55E00", "recreativa": "#009E73", "mixta": "#8A8F98"}
FALTA = "Esta sección todavía no tiene datos: falta preparar la información de empleos y residentes."


def _num(x, decimales: int = 0) -> str:
    return "Sin dato" if x is None or pd.isna(x) else f"{x:,.{decimales}f}"


def _correlaciones(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Correlación lineal (Pearson) y de orden (Spearman, Pearson sobre rangos)."""
    lineal = float(np.corrcoef(x, y)[0, 1])
    orden = float(np.corrcoef(x.rank(), y.rank())[0, 1])
    return lineal, orden


def mostrar(ruta: str | Path | None = None) -> None:
    """Dibuja la sección. `ruta` solo se usa en pruebas."""
    ruta = Path(ruta) if ruta else TABLA
    st.subheader("¿El tipo de estación coincide con lo que hay alrededor?")
    if not ruta.exists():
        st.info(FALTA)
        return
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # para encontrar datos.py también en pruebas
    import datos  # el único archivo de la app que lee tablas

    df = datos.tabla_pequena(ruta)
    anio = int(df["anio_datos"].iloc[0]) if "anio_datos" in df.columns and len(df) else None
    df = df[df["proporcion_empleos"].notna() & df["am_indice"].notna() & df["tipo"].notna()].copy()
    if len(df) < 3:
        st.info(FALTA)
        return
    for col in ["proporcion_empleos", "am_indice", "empleos", "residentes"]:
        df[col] = df[col].astype(float)
    df["tipo_texto"] = df["tipo"].map(TIPOS_ESTACION).fillna("Mixta")

    lineal, orden = _correlaciones(df["proporcion_empleos"], df["am_indice"])
    if abs(orden) < 0.1:
        lectura = "saber cuántos empleos hay cerca casi no dice hacia dónde se mueven las bicis en la mañana"
    elif orden > 0:
        lectura = ("donde hay más empleos que trabajadores residentes, en la mañana entre semana "
                   "tienden a llegar más bicis de las que salen")
    else:
        lectura = ("donde hay más empleos que trabajadores residentes, en la mañana entre semana "
                   "tienden a salir más bicis de las que llegan")
    st.markdown(
        f"En {_num(len(df))} estaciones, la relación entre la proporción de empleos a 300 metros y el flujo "
        f"neto de la mañana tiene correlación de orden {orden:.2f} y lineal {lineal:.2f} (escala de -1 a 1; "
        f"0 es sin relación). En palabras simples: {lectura}."
    )

    orden_tipos = [TIPOS_ESTACION[t] for t in TIPOS_ESTACION]
    grafica = (
        alt.Chart(df)
        .mark_circle(size=28, opacity=0.55)
        .encode(
            x=alt.X("proporcion_empleos:Q", title="Proporción de empleos a 300 metros",
                    axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
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
    partes = [f"{TIPOS_ESTACION[t]} {medianas[t] * 100:.1f}%" for t in TIPOS_ESTACION if t in medianas.index]
    st.markdown("Proporción de empleos en la estación típica de cada tipo (mediana): " + "; ".join(partes) + ".")
    st.caption(
        "Flujo neto de la mañana: llegadas menos salidas entre llegadas más salidas, de 7 a 10 entre semana, "
        "en julio y agosto de 2026. Positivo: llegan más bicis de las que salen. "
        "Proporción de empleos: empleos entre empleos más trabajadores residentes. "
        "Los residentes son trabajadores (personas con empleo, contadas donde viven), no población total. "
        f"Fuente: Oficina del Censo de Estados Unidos (LODES), datos de {anio if anio else 'un año anterior'}, "
        "anteriores a los viajes de 2026; el Censo agrega ruido a propósito para proteger la confidencialidad."
    )
