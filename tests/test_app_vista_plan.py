"""Página "Plan de la ronda": abre, tiene sus tres selectores, tres indicadores, tabla y mapa.

Igual que tests/test_app_smoke.py: AppTest de Streamlit, con las tablas pequeñas
construidas con data/sample en una carpeta temporal. Si existen las tablas
reales (data/processed/app), la página también se abre con ellas.

    pytest tests/test_app_vista_plan.py
"""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REALES = ROOT / "data" / "processed" / "app"
PAGINA = "vistas/plan_de_ronda.py"
BUILD = "python scripts/build_app_tables.py"
INDICADORES = ["Estaciones a visitar", "Bicis a llevar", "Bicis a retirar"]
COLUMNAS = ["Estación", "Tipo", "Acción", "Bicis antes", "Bicis después", "Riesgo simulado"]
# Los mismos nombres internos que vigila la prueba de humo.
PROHIBIDOS = [
    "station_id", "ts_hour", "departures", "arrivals", "net_flow", "station_type", "capacity",
    "dep_q", "arr_q", "window_start", "window_end", "start_inventory_target", "bikes_to_move",
    "risk_level", "feasible", "risk_station_hours", "ci_low", "ci_high", "start_station",
    "end_station", "station_hour", "policy_curve", "baseline", "lgbm", "historical",
    "estacion_id", "nivel_riesgo", "pct_evitado", "flujo_neto", "tipo_dia", "sin_dato",
    "inventario_antes", "hora_inicio", "hora_fin", "tipo_estacion", "pct_min", "media_global",
    "politica", "manana", "factible", "traceback",
]


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_plan")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch, tablas_de_muestra):
    """app/datos.py leyendo de la carpeta temporal (muestra) o de las tablas reales si existen."""
    if request.param == "reales":
        if not (REALES / "indicadores.json").exists():
            pytest.skip("No están construidas las tablas reales (data/processed/app)")
        carpeta = REALES
    else:
        carpeta = tablas_de_muestra
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    return at.switch_page(PAGINA).run()


def sin_errores(at: AppTest) -> None:
    assert [str(e.value) for e in at.exception] == []
    assert not at.error
    assert not at.main.warning
    assert len(at.title) == 1 and at.title[0].value == "Plan de la ronda"


def radio(at: AppTest, etiqueta: str):
    return next(r for r in at.radio if r.label == etiqueta)


def indicadores(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def entero(texto: str) -> int:
    return int(texto.replace(",", ""))


def todo_el_texto(at: AppTest) -> str:
    """Lo que se ve: títulos, textos, selectores, indicadores, desplegable y la tabla completa."""
    partes = []
    for clase in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error",
                  "metric", "selectbox", "radio", "expander"):
        for elemento in getattr(at, clase, []):
            for campo in ("label", "value", "body", "options"):
                valor = getattr(elemento, campo, None)
                if valor is not None:
                    partes.append(str(valor))
    for tabla in at.dataframe:
        partes.append(" ".join(map(str, tabla.value.columns)))
        partes.append(" ".join(tabla.value.astype(str).to_numpy().ravel()))
    return " ".join(partes)


def revisar_contenido(at: AppTest) -> None:
    """Titular, indicadores, tabla y mapa cuentan lo mismo."""
    sin_errores(at)
    valores = indicadores(at)
    assert list(valores) == INDICADORES
    n, llevar, retirar = (entero(valores[k]) for k in INDICADORES)
    titular = next(s.value for s in at.subheader if "a visitar en" in s.value)
    assert n > 0 and (f"{n:,} estaciones" in titular or (n == 1 and "la estación" in titular))

    assert len(at.dataframe) == 1
    tabla = at.dataframe[0].value
    assert list(tabla.columns) == COLUMNAS
    assert len(tabla) == min(n, 200)
    assert tabla["Acción"].str.fullmatch(r"(Llevar|Retirar) [\d,]+").all()  # solo estaciones con traslado
    assert set(tabla["Riesgo simulado"]) <= {"Alto", "Medio", "Bajo"}
    # Riesgo alto primero y, dentro de cada nivel, de más a menos bicis movidas.
    orden = tabla["Riesgo simulado"].map({"Alto": 0, "Medio": 1, "Bajo": 2})
    assert orden.is_monotonic_increasing
    movidas = tabla["Acción"].str.split().str[1].map(entero)
    for _, grupo in movidas.groupby(orden):
        assert grupo.is_monotonic_decreasing
    if n <= 200:  # la tabla está completa: sus sumas son los indicadores
        assert movidas[tabla["Acción"].str.startswith("Llevar")].sum() == llevar
        assert movidas[tabla["Acción"].str.startswith("Retirar")].sum() == retirar
        assert not any("más." in c.value for c in at.caption)
    else:
        assert any(f"hay {n - 200:,} estaciones más" in c.value or f"hay {n - 200:,} estación más" in c.value
                   for c in at.caption)

    assert len(at.get("deck_gl_json_chart")) == 1
    leyenda = next(m.value for m in at.markdown if "Llevar" in m.value and "Retirar" in m.value)
    assert "●" in leyenda


def test_abre_con_tres_selectores_y_tres_indicadores(datos):
    at = abrir()

    assert [s.label for s in at.selectbox] == ["Día"]
    assert [r.label for r in at.radio] == ["Ronda", "Política"]
    assert re.fullmatch(r"\w+ \d{1,2} de \w+ de \d{4}", at.selectbox[0].options[0])  # fecha en español
    assert radio(at, "Política").options == ["Regla histórica", "Modelo"]  # sin "Sin intervención"
    assert radio(at, "Política").value == "Regla histórica"
    assert len(radio(at, "Ronda").options) == 2
    revisar_contenido(at)
    assert any(datos.LINEA in c.value for c in at.caption)
    assert [e.label for e in at.expander] == ["Cómo se calculó"]

    # La línea del presupuesto sale de los indicadores.
    presupuesto = datos.indicadores()["referencia"]["presupuesto"]
    linea = next(m.value for m in at.markdown if "presupuesto de referencia" in m.value)
    assert linea.startswith("Resultado de la simulación") and f"({datos.num(presupuesto)} bicis al día" in linea
    assert "validación" in linea


def test_la_ronda_y_su_hora_salen_de_las_tablas(datos):
    at = abrir()
    opciones = radio(at, "Ronda").options
    rondas = datos.rondas()
    if rondas:  # la simulación reportó el momento de intervención
        esperadas = [f"{r['nombre'][0].upper()}{r['nombre'][1:]} (lista a las {datos.hora(r['hora'])})"
                     for r in rondas.values()]
        assert sorted(opciones) == sorted(esperadas)
        assert opciones[0].startswith("Ronda nocturna") and opciones[1].startswith("Ronda de mediodía")
    else:  # datos de muestra: se nombra la franja que se prepara
        assert all(re.fullmatch(r"Antes de la franja de la \w+ \(\d\d:00 a \d\d:00\)", o) for o in opciones)


def test_cambiar_cada_selector_vuelve_a_calcular_sin_errores(datos):
    at = abrir()
    al_abrir = indicadores(at)

    at.selectbox[0].select_index(1).run()
    revisar_contenido(at)
    otro_dia = indicadores(at)

    radio(at, "Ronda").set_value(radio(at, "Ronda").options[1]).run()
    revisar_contenido(at)
    otra_ronda = indicadores(at)

    radio(at, "Política").set_value("Modelo").run()
    revisar_contenido(at)
    assert "**Modelo**" in next(m.value for m in at.markdown if "presupuesto de referencia" in m.value)
    # Algo cambió en el camino: los indicadores se calculan con la selección.
    assert len({tuple(d.values()) for d in (al_abrir, otro_dia, otra_ronda, indicadores(at))}) > 1


def test_los_indicadores_salen_de_la_tabla_de_recomendaciones(datos):
    at = abrir()
    rec = datos.recomendaciones()
    dia = sorted(rec["fecha"].unique())[0]
    franja = rec.sort_values("hora_inicio")["franja"].iloc[0]
    plan = rec[(rec["fecha"] == dia) & (rec["franja"] == franja) & (rec["politica"] == "historical")
               & (rec["bicis"] != 0)]
    assert indicadores(at) == {
        "Estaciones a visitar": datos.num(len(plan)),
        "Bicis a llevar": datos.num(plan.loc[plan["bicis"] > 0, "bicis"].sum()),
        "Bicis a retirar": datos.num(-plan.loc[plan["bicis"] < 0, "bicis"].sum()),
    }


def test_no_se_ven_nombres_internos_ni_valores_vacios(datos):
    texto = todo_el_texto(abrir()).lower()
    for nombre in PROHIBIDOS:
        assert not re.search(rf"\b{re.escape(nombre)}", texto), f"se ve '{nombre}'"
    assert not re.search(r"\bnone\b|\bnan\b|\bmodel\b|<na>", texto)
    assert ".parquet" not in texto and ".json" not in texto
    sin_coordenadas = re.sub(r"<[^>]+>", " ", texto)
    assert not re.search(r"\d\.\d{2,}", sin_coordenadas), "número con más de un decimal"


def test_si_faltan_las_tablas_hay_mensaje_amable(datos, monkeypatch, tmp_path):
    monkeypatch.setattr(datos, "CARPETA", tmp_path / "no_existe")
    at = abrir()

    assert not at.exception
    assert any(BUILD in e.value for e in at.info)


def test_sin_traslados_hay_mensaje_amable(datos, monkeypatch):
    vacia = datos.recomendaciones().iloc[0:0]
    monkeypatch.setattr(datos, "recomendaciones", lambda: vacia)
    at = abrir()

    assert not at.exception and not at.error
    assert any("todavía no tiene traslados" in e.value for e in at.info)
    assert not at.metric and not at.dataframe


def test_el_menu_la_pone_despues_de_pronostico():
    codigo = (APP / "Inicio.py").read_text(encoding="utf-8")
    rutas = re.findall(r'st\.Page\("([^"]+)"', codigo)
    assert rutas.index(PAGINA) == rutas.index("vistas/pronostico_por_estacion.py") + 1
    assert 'title="Plan de la ronda"' in codigo
