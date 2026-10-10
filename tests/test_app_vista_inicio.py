"""La página Inicio: cuatro indicadores, la curva de riesgo contra presupuesto y cuatro hallazgos.

Se abre igual que en tests/test_app_smoke.py, con las tablas hechas de
data/sample en una carpeta temporal y, si existen, con las reales de
data/processed/app (solo se leen).

    pytest tests/test_app_vista_inicio.py
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import app_tables

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SAMPLE = ROOT / "data" / "sample"
REALES = ROOT / "data" / "processed" / "app"


@pytest.fixture(scope="module")
def tablas_de_muestra(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("app_tablas_inicio")
    app_tables.build(SAMPLE, carpeta)
    return carpeta


@pytest.fixture(params=["muestra", "reales"])
def datos(request, monkeypatch, tablas_de_muestra):
    """app/datos.py leyendo de las tablas de muestra o de las reales."""
    if request.param == "reales":
        if not (REALES / "indicadores.json").exists():
            pytest.skip("no hay tablas reales en data/processed/app")
        carpeta = REALES
    else:
        carpeta = tablas_de_muestra
    monkeypatch.syspath_prepend(str(APP))
    import datos as module

    assert Path(module.__file__).resolve() == APP / "datos.py"
    monkeypatch.setattr(module, "CARPETA", carpeta)
    return module


def abrir() -> AppTest:
    return AppTest.from_file(str(APP / "Inicio.py"), default_timeout=30).run()


def test_inicio_muestra_indicadores_curva_y_hallazgos(datos):
    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert not at.error and not at.main.warning
    assert len(at.title) == 1

    # Cuatro indicadores, con las cifras de los datos y sin valores vacíos de Python.
    ind = datos.indicadores()
    assert [m.label for m in at.metric] == [
        "Viajes analizados", "Estaciones", "Riesgo simulado evitado", "Error del pronóstico"]
    valores = [m.value for m in at.metric]
    # Bajo cada indicador va una línea; el detalle está en el desplegable "Cómo se calculó".
    notas = " ".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert valores[0] == datos.redondo(ind["viajes_analizados"])
    assert valores[1] == datos.redondo(ind["estaciones"])
    assert f"{datos.num(ind['viajes_analizados'])} viajes" in notas
    assert f"{datos.num(ind['estaciones'])} en total" in notas
    assert datos.redondo(15017726) == "15 millones" and datos.redondo(2398) == "2,400"
    assert datos.redondo(784929) == "785 mil" and datos.redondo(None) == "Sin dato"
    # El riesgo evitado es el de la regla histórica: el rango entre los inventarios iniciales supuestos
    # si la simulación lo reportó; si no, el valor con el presupuesto de referencia.
    rango = ind["sensibilidad"]["rango"].get("historical")
    if rango:
        assert valores[2] == datos.rango_pct(rango["pct_min"], rango["pct_max"])
        assert "depende del inventario inicial que se suponga" in notas
    else:
        assert valores[2] == datos.pct(ind["referencia"]["politicas"]["historical"]["pct_evitado"], 0)
    assert "Regla histórica, con" in notas
    if ind["comparacion"]["modelo_evita"] == "igual":
        assert "El modelo da prácticamente el mismo resultado" in notas
    # La frase bajo la curva es la que arman los datos (solo dice lo que las cifras sostienen).
    frase = datos.frase_de_la_curva()
    assert frase and notas.count(frase) == 1
    # El orden: los hallazgos van antes de la curva, y la página explica qué es una hora en riesgo.
    titulos = [s.value for s in at.subheader]
    assert len(titulos) >= 2 and "None" not in " ".join(titulos)
    assert notas.count(f"Hora de estación en riesgo: {datos.QUE_ES_RIESGO}.") == 1
    veredictos = datos.modelo_contra_regla()
    assert ("con la mitad, el modelo evita más" in frase) == (
        veredictos["referencia"].get("validacion") == "igual" and veredictos["mitad"].get("validacion") == "mas")
    # La línea discreta de Inicio nombra la prueba final solo cuando ya está.
    lineas = [c.value for c in at.caption if c.value.startswith("Resultados de simulación")]
    assert len(lineas) == 1
    assert lineas[0].startswith("Resultados de simulación: validación y prueba final") == datos.hay_prueba_final()
    if ind["piso_de_azar"]["pct_evitado"] is not None:
        assert f"Mover las mismas bicis al azar evita {datos.pct(ind['piso_de_azar']['pct_evitado'])}." in notas
    # El error es el del modelo, con el del promedio histórico y el de una sola media como contexto.
    assert valores[3] == datos.pct(ind["pronostico"]["lgbm"]["salidas"]["wape"], 0)
    assert f"Salidas, con el modelo: {datos.pct(ind['pronostico']['lgbm']['salidas']['wape'])}" in notas
    assert f"promedio histórico: {datos.pct(ind['pronostico']['baseline']['salidas']['wape'])}" in notas
    media_global = (ind["pronostico_piso"].get("media_global") or {}).get("salidas")
    if media_global:
        assert f"una sola media para todo el sistema: {datos.pct(media_global['wape'])}" in notas

    # Una gráfica: la curva con las tres políticas, sus nombres en pantalla y el presupuesto de referencia.
    graficas = at.get("vega_lite_chart")
    assert len(graficas) == 1
    spec = json.loads(graficas[0].proto.spec)
    texto_spec = json.dumps(spec, ensure_ascii=False)
    for nombre in ("Sin intervención", "Regla histórica", "Modelo",
                   "Bicis movidas por día", "Horas de estación en riesgo por día"):
        assert nombre in texto_spec
    assert f"Presupuesto de referencia: {datos.num(ind['referencia']['presupuesto'])} bicis" in texto_spec
    # Mover al azar: un punto con su nombre, solo si la simulación lo reportó.
    azar = ind["piso_de_azar"]
    assert ("Mover al azar" in texto_spec) == (azar["riesgo"] is not None)
    if azar["riesgo"] is not None:
        assert azar["presupuesto"] == ind["referencia"]["presupuesto"]
        assert f'"riesgo": {float(azar["riesgo"])}' in texto_spec

    # Cuatro hallazgos, los mismos que arma datos.hallazgos().
    frases = datos.hallazgos()
    textos = [m.value for m in at.markdown]
    assert len(frases) == 4
    for frase in frases:
        assert sum(frase in t for t in textos) == 1

    # La comparación con la prueba final solo aparece cuando el constructor copió ese archivo.
    hay_comparacion = any(s.value == "Validación y prueba final" for s in at.subheader)
    assert hay_comparacion == bool(ind.get("validacion_y_prueba"))


def test_validacion_y_prueba_final_lado_a_lado(datos, monkeypatch):
    ind = datos.indicadores()
    # El archivo trae más filas y en otro orden; la página muestra solo las suyas, con su propio texto.
    falso = {"presupuesto_de_referencia": 3241, "filas": [
        {"clave": "mae_modelo", "afirmacion": "MAE de salidas", "unidad": "numero", "validacion": 1.5177, "prueba": 2.0},
        {"clave": "bloqueados_historica", "afirmacion": "Viajes bloqueados", "unidad": "viajes", "validacion": 3424.9,
         "prueba": 2698.2, "validacion_ic": None, "prueba_ic": [2084.0, 3267.4]},
        {"clave": "pct_evitado_historica", "afirmacion": "Riesgo simulado evitado", "unidad": "pct",
         "validacion": 0.5692, "prueba": 0.6401, "validacion_ic": [0.52, 0.61], "prueba_ic": None},
        {"clave": "ganancia_pct_al_duplicar_historica", "afirmacion": "Ganancia", "unidad": "pct",
         "validacion": 0.0341, "prueba": None, "validacion_ic": [0.0133, 0.0634], "prueba_ic": None},
        {"clave": "wape_modelo", "afirmacion": "WAPE de salidas, modelo (LightGBM)", "unidad": "pct",
         "validacion": 0.4577, "prueba": 0.495},
        {"clave": "diferencia_modelo_menos_historica_referencia", "afirmacion": "Diferencia", "unidad": "horas",
         "validacion": -7.5, "prueba": 86.4}]}
    monkeypatch.setattr(datos, "indicadores", lambda: {**ind, "validacion_y_prueba": falso})

    subir = datos._subir_el_presupuesto(ind)
    assert subir in ("al casi duplicar el presupuesto", "al pasar al siguiente presupuesto")
    assert datos.validacion_y_prueba()["filas"] == [
        {"afirmacion": "Riesgo simulado evitado con la regla histórica", "validacion": "56.9 % (52.0 % a 61.0 %)",
         "prueba": "64.0 %"},
        {"afirmacion": f"Puntos extra de riesgo simulado evitado {subir}, regla histórica",
         "validacion": "3.4 puntos (1.3 a 6.3)", "prueba": "Sin dato"},
        {"afirmacion": "Viajes bloqueados por día en la simulación, con la regla histórica",
         "validacion": "3,425 viajes", "prueba": "2,698 viajes (2,084 a 3,267)"},
        {"afirmacion": "Error del pronóstico de salidas, modelo (LightGBM)", "validacion": "45.8 %", "prueba": "49.5 %"}]
    assert len(datos.FILAS_DE_COMPARACION) == 8
    at = abrir()
    assert [str(e.value) for e in at.exception] == []
    tabla = next(m.value for m in at.markdown if "| Validación | Prueba final |" in m.value)
    assert len(tabla.splitlines()) == 2 + 4
    assert "56.9&nbsp;%" in tabla and "None" not in tabla and "nan" not in tabla and "WAPE" not in tabla
    assert any("3,241 bicis movidas al día" in m.value for m in at.markdown)
    assert any(c.value.startswith("Resultados de simulación: validación y prueba final") for c in at.caption)

    # Sin el archivo de la prueba final no hay ni título ni tabla vacía.
    monkeypatch.setattr(datos, "indicadores", lambda: {**ind, "validacion_y_prueba": None})
    at = abrir()
    assert [str(e.value) for e in at.exception] == []
    assert not any("Prueba final" in m.value for m in at.markdown)
    assert not any("prueba final" in s.value.lower() for s in at.subheader)


def test_inicio_no_se_rompe_si_falta_la_simulacion(datos, monkeypatch):
    """Sin resultados de simulación, la página dice "Sin dato" en lugar de fallar."""
    ind = datos.indicadores()
    vacio = {**ind, "referencia": {"presupuesto": None, "politicas": {}}, "pronostico": {},
             "sensibilidad": {"filas": [], "rango": {}},
             "comparacion": {"modelo_pronostica_mejor": None, "modelo_evita": None},
             "presupuesto_umbral": {"politicas": {}}}
    monkeypatch.setattr(datos, "indicadores", lambda: vacio)
    monkeypatch.setattr(datos, "curva", lambda: datos._leer("curva").iloc[0:0])

    at = abrir()

    assert [str(e.value) for e in at.exception] == []
    assert len(at.metric) == 4
    assert [m.value for m in at.metric][2:] == ["Sin dato", "Sin dato"]
    assert len(at.get("vega_lite_chart")) == 0
    assert any("Sin dato" in e.value for e in at.info)
