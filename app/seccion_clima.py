"""Sección del dashboard: ¿se mueve menos el sistema en las horas con lluvia?

Expone una sola función, mostrar(). Lee, por medio de datos.py, una tabla
pequeña que escribe scripts/build_clima.py. No importa nada de src.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

TIPOS_DIA = {"laborable": "Laborable", "fin_de_semana": "Fin de semana"}
CONDICIONES = {"demanda_seco": "Horas secas", "demanda_lluvia": "Horas con lluvia"}
COLORES = {"demanda_seco": "#E69F00", "demanda_lluvia": "#0072B2"}
FALTA = "Esta sección todavía no tiene datos: falta preparar la información del clima."


def resumen(df: pd.DataFrame) -> dict:
    """Diferencia relativa promedio (lluvia contra seco), ponderada por la demanda en seco."""
    c = df[df["comparable"]]
    if c.empty or c["demanda_seco"].sum() <= 0:
        return {"diferencia": None, "celdas": 0, "horas_lluvia": 0}
    peso = c["demanda_seco"]
    return {"diferencia": float((c["diferencia"] * peso).sum() / peso.sum()), "celdas": int(len(c)),
            "horas_lluvia": int(c["horas_lluvia"].sum()), "horas_seco": int(c["horas_seco"].sum())}


def mostrar(ruta: str | Path | None = None) -> None:
    """Dibuja la sección. `ruta` solo se usa en pruebas."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # para encontrar datos.py también en pruebas
    import datos  # el único archivo de la app que lee tablas

    ruta = Path(ruta) if ruta else datos.CARPETA / "clima_demanda.parquet"
    st.subheader("¿Se usan menos las bicis cuando llueve?")
    if not ruta.exists():
        st.info(FALTA)
        return
    df = datos.tabla_pequena(ruta).copy()
    r = resumen(df) if len(df) else {"diferencia": None}
    if r["diferencia"] is None:
        st.info(FALTA)
        return
    umbral = float(df["umbral_lluvia_mm"].iloc[0]) if "umbral_lluvia_mm" in df.columns else 0.1

    d = r["diferencia"]
    if abs(d) < 0.02:
        lectura = "salieron prácticamente las mismas bicis que en las horas secas"
    else:
        lectura = f"salieron en promedio {datos.pct(abs(d))} {'menos' if d < 0 else 'más'} bicis que en las horas secas"
    st.markdown(
        f"A la misma hora del día y el mismo tipo de día, en las horas con lluvia {lectura} "
        f"(comparación sobre {datos.num(r['horas_lluvia'])} horas con lluvia y {datos.num(r['horas_seco'])} "
        f"horas secas, en {datos.num(r['celdas'])} combinaciones de hora y tipo de día)."
    )

    largo = df[df["comparable"]].melt(id_vars=["tipo_dia", "hora"], value_vars=list(CONDICIONES),
                                      var_name="condicion", value_name="salidas")
    largo["condicion_texto"] = largo["condicion"].map(CONDICIONES)
    largo["tipo_texto"] = largo["tipo_dia"].map(TIPOS_DIA).fillna(largo["tipo_dia"])
    orden = list(CONDICIONES.values())
    grafica = (
        alt.Chart(largo)
        .mark_line(point=True)
        .encode(
            x=alt.X("hora:Q", title="Hora del día", scale=alt.Scale(domain=[0, 23]),
                    axis=alt.Axis(values=list(range(0, 24, 3)), format="d")),
            y=alt.Y("salidas:Q", title="Salidas por hora en todo el sistema (promedio)",
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color("condicion_texto:N", title=None, sort=orden,
                            scale=alt.Scale(domain=orden, range=[COLORES[c] for c in CONDICIONES])),
            tooltip=[alt.Tooltip("tipo_texto:N", title="Tipo de día"),
                     alt.Tooltip("hora:Q", title="Hora del día", format="d"),
                     alt.Tooltip("condicion_texto:N", title="Condición"),
                     alt.Tooltip("salidas:Q", title="Salidas por hora", format=",.0f")],
        )
        .properties(height=300)
        .facet(column=alt.Column("tipo_texto:N", title=None, sort=list(TIPOS_DIA.values())))
        .resolve_scale(y="shared")
    )
    st.altair_chart(grafica)
    st.caption(
        f"Clima observado (lo que realmente pasó), no clima pronosticado: es una descripción y no una entrada "
        f"del modelo, que solo podría usar clima pronosticado. Una hora cuenta como con lluvia si cayeron al "
        f"menos {datos.num(umbral, 1)} mm en esa hora. Es una asociación, no una prueba de causa. "
        f"Fuente del clima: Open-Meteo."
    )
