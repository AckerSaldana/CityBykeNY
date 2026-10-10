"""La página "Confiabilidad" abre y muestra lo que promete.

Se prueba con las tablas de data/sample (en una carpeta temporal) y, si
existen, con las tablas reales de data/processed/app (solo se leen).

    pytest tests/test_app_vista_confiabilidad.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REAL = ROOT / "data" / "processed" / "app"
PAGE = "vistas/confiabilidad.py"
LINE = "Resultados de simulación con datos de validación"


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_confiabilidad")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch, tablas_de_muestra):
    """app/datos.py leyendo de las tablas de muestra o de las reales."""
    if request.param == "reales":
        if not (REAL / "indicadores.json").exists():
            pytest.skip("no están construidas las tablas reales")
        carpeta = REAL
    else:
        carpeta = tablas_de_muestra
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    at = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()
    return at.switch_page(PAGE).run()


def graficas(at: AppTest) -> list[str]:
    """Cada gráfica como texto: su especificación y los datos que dibuja."""
    salida = []
    for g in at.get("vega_lite_chart"):
        partes = [json.dumps(json.loads(g.proto.spec), ensure_ascii=False)]
        for conjunto in g.proto.datasets:
            tabla = pa.ipc.open_stream(conjunto.data.data).read_all().to_pandas()
            partes.append(tabla.to_json(orient="records", force_ascii=False))
        salida.append(" ".join(partes))
    return salida


def texto(at: AppTest) -> str:
    return "\n".join(str(e.value) for clase in ("title", "subheader", "markdown", "caption")
                     for e in getattr(at, clase))


def test_abre_con_un_titulo_la_linea_discreta_y_cuatro_graficas(datos):
    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert not at.error and not at.main.warning
    assert [t.value for t in at.title] == ["Confiabilidad"]
    assert len([c for c in at.caption if LINE in c.value]) == 1
    assert len(at.dataframe) == 0  # no hay tablas crudas
    # Error por bloque y por volumen; riesgo por franja y por tipo de estación.
    assert len(graficas(at)) == 4


def test_la_comparacion_usa_las_cifras_de_las_tablas(datos):
    pantalla = texto(abrir())
    general = datos.metricas("general")

    assert not general.empty
    for _, fila in general.iterrows():
        assert datos.pct(fila["wape"]) in pantalla
        assert datos.num(fila["mae"], 1) in pantalla
        assert datos.pct(fila["cobertura"]) in pantalla
    for nombre in (*datos.MODELOS.values(), *datos.OBJETIVOS.values(), "WAPE", "MAE", "Cobertura"):
        assert nombre in pantalla
    assert "riesgo simulado" in pantalla.lower()
    assert "demanda atendida" in pantalla


def test_las_graficas_no_llevan_claves_internas(datos):
    at = abrir()
    # "baseline" también es una propiedad de Vega para alinear texto: esa no cuenta.
    spec = " ".join(graficas(at)).lower().replace('"baseline": "bottom"', "")

    for clave in ("baseline", "lgbm", "historical", "pct_alto", "ic_bajo", "tipo_estacion", "sin_dato"):
        assert clave not in spec, clave
    assert not re.search(r"\b(nan|none)\b", texto(at).lower())
    for etiqueta in (*datos.MODELOS.values(), *datos.POLITICAS.values()):
        assert etiqueta.lower() in spec


def test_cambiar_de_medida_y_de_viajes_no_rompe_la_pagina(datos):
    at = abrir()
    for viajes in at.radio[0].options:
        for medida in at.radio[1].options:
            at.radio[0].set_value(viajes)
            at.radio[1].set_value(medida)
            at = at.run()
            assert [str(e.value) for e in at.exception] == []
            assert len(graficas(at)) == 4


def test_las_limitaciones_empiezan_por_el_inventario_inicial_y_la_demanda_atendida(datos):
    at = abrir()
    lineas = datos.limitaciones()
    listas = [m.value for m in at.markdown if lineas and lineas[0] in m.value]

    # Siempre que hubo simulación van primero estas dos; después, las que escribió la simulación.
    assert len(listas) == 1
    assert [renglon[2:] for renglon in listas[0].splitlines()] == lineas
    assert lineas[0].startswith("El resultado depende del inventario inicial supuesto")
    assert "demanda atendida" in lineas[1]
    assert sum("demanda atendida" in linea for linea in lineas) == 1  # no se repite la del reporte
    ind = datos.indicadores()
    if ind.get("limitaciones"):
        assert len(lineas) == 6
        rango = ind["sensibilidad"]["rango"]["historical"]
        assert datos.rango_pct(rango["pct_min"], rango["pct_max"]) in lineas[0]
    else:
        assert len(lineas) == 2


def test_sin_simulacion_no_hay_limitaciones(datos, monkeypatch):
    ind = datos.indicadores()
    vacio = {**ind, "limitaciones": [], "referencia": {"presupuesto": None, "politicas": {}}}
    monkeypatch.setattr(datos, "indicadores", lambda: vacio)
    assert datos.limitaciones() == []
    assert any(c.value == datos.SIN_DATO for c in abrir().caption)


def test_resto_del_dia_no_se_dibuja_si_no_tiene_riesgo(datos, monkeypatch):
    franja = datos.riesgo_horas("franja")
    if franja.empty:  # los datos de muestra no traen horas por franja: se inventan
        franja = pd.DataFrame({"corte": "franja", "grupo": ["manana"] * 3, "politica": ["none", "historical", "model"],
                               "riesgo": [9.0, 5.0, 4.0], "ic_bajo": [8.0, 4.0, 3.0], "ic_alto": [10.0, 6.0, 5.0]})
    resto = franja[franja["grupo"] == franja["grupo"].iloc[0]].assign(grupo="resto", riesgo=0.0, ic_bajo=0.0, ic_alto=0.0)
    original = datos.riesgo_horas
    monkeypatch.setattr(datos, "riesgo_horas", lambda corte=None: (
        pd.concat([franja, resto], ignore_index=True) if corte == "franja" else original(corte)))
    at = abrir()
    assert not at.exception
    assert not any(datos.FRANJAS["resto"] in g for g in graficas(at))
    assert any(datos.FRANJAS[franja["grupo"].iloc[0]] in g for g in graficas(at))


def test_el_riesgo_por_tipo_de_estacion_dice_que_se_muestra(datos):
    """Con horas por tipo de estación se dibujan horas; sin ellas, el porcentaje en riesgo alto."""
    pantalla = texto(abrir())
    aviso = "porcentaje de franjas de estación que quedaron en riesgo alto"

    hay_horas = [not datos.riesgo_horas(corte).empty for corte in ("franja", "tipo_estacion")]
    assert pantalla.count(aviso) == hay_horas.count(False)
    assert pantalla.count("Horas de estación en riesgo por día.") == hay_horas.count(True)


def test_sin_horas_ni_niveles_de_riesgo_la_pagina_dice_sin_dato(datos, monkeypatch):
    sin_niveles = datos.riesgo_niveles().iloc[0:0]
    sin_horas = datos.riesgo_horas().iloc[0:0]
    monkeypatch.setattr(datos, "riesgo_horas", lambda corte=None: sin_horas)
    monkeypatch.setattr(datos, "riesgo_niveles", lambda corte=None: sin_niveles)
    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert len(graficas(at)) == 2  # quedan las dos del pronóstico
    assert [c.value for c in at.caption].count(datos.SIN_DATO) >= 2


def test_cuando_la_simulacion_trae_horas_por_tipo_de_estacion_se_dibujan_las_horas(datos, monkeypatch):
    """Hoy ese desglose viene vacío; la simulación final puede llenarlo."""
    niveles = datos.riesgo_niveles("tipo_estacion")
    horas = niveles.assign(riesgo=niveles["alto"] * 1.5, ic_bajo=niveles["alto"] * 1.2,
                           ic_alto=niveles["alto"] * 1.8)[["corte", "grupo", "politica", "riesgo", "ic_bajo", "ic_alto"]]
    original = datos.riesgo_horas
    monkeypatch.setattr(datos, "riesgo_horas",
                        lambda corte=None: horas if corte == "tipo_estacion" else original(corte))
    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert len(graficas(at)) == 4
    spec = graficas(at)[3]
    assert "Horas de estación en riesgo por día" in spec
    assert datos.num(horas["riesgo"].max()) in spec
