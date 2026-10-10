"""Pruebas del generador de datos SINTÉTICOS de muestra (src/data/sample_data.py).

Qué revisan:

1. El generador da siempre las mismas tablas (semilla fija).
2. data/sample es lo que el generador produce HOY. Si alguien cambia el
   generador, o las franjas o los presupuestos de config/operacion.yaml, y no
   vuelve a correr el script, la prueba FALLA y dice qué correr.
3. Los datos se distinguen como sintéticos: ids "SIM-..." y nombres
   "Sintética ...".
4. `feasible` es de la estación y la franja: vale lo mismo en las tres
   políticas.
5. El generador se niega a escribir en data/processed y en data/raw, que son
   solo para datos reales.

Si falla el punto 2, corre:  python scripts/make_sample_data.py
"""

import copy
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.contracts import CONTRACTS, POLICIES
from src.data import sample_data
from src.data.sample_data import build_sample, check_output_folder, main, write_sample
from src.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE
from src.sim.operacion import load_operacion

COMMAND = "python scripts/make_sample_data.py"
REFUSAL = "No se escriben datos SINTÉTICOS"


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return build_sample()


# --------------------------------------------------------------------------- #
# 1) Determinista
# --------------------------------------------------------------------------- #

def test_el_generador_da_siempre_las_mismas_tablas(tables):
    again = build_sample()
    assert list(tables) == list(again) == list(CONTRACTS)
    for name in CONTRACTS:
        pd.testing.assert_frame_equal(tables[name], again[name])


# --------------------------------------------------------------------------- #
# 2) data/sample está al día
# --------------------------------------------------------------------------- #

def _outdated(folder: Path, fresh: Path) -> list[str]:
    """Archivos de `folder` que no coinciden con los que el generador acaba de escribir en `fresh`."""
    names = []
    for path in sorted(fresh.iterdir()):
        current = folder / path.name
        if not current.exists():
            names.append(path.name)
        elif path.suffix == ".parquet":
            # Se comparan las tablas (columnas, tipos y valores), no los bytes del archivo.
            if not pq.read_table(path).equals(pq.read_table(current)):
                names.append(path.name)
        elif path.read_text(encoding="utf-8") != current.read_text(encoding="utf-8"):
            names.append(path.name)
    return names


def _fail_if_outdated(folder: Path, tmp_path: Path) -> None:
    """Genera la muestra en una carpeta temporal y la compara con `folder`."""
    write_sample(tmp_path)
    outdated = _outdated(folder, tmp_path)
    if outdated:
        pytest.fail(
            f"data/sample está desactualizado. Ya no coinciden con lo que produce el generador: {outdated}.\n"
            "Cambió src/data/sample_data.py, o cambiaron las franjas o los presupuestos de config/operacion.yaml.\n"
            f"Para regenerarlos: {COMMAND}",
            pytrace=False,
        )


def test_data_sample_es_lo_que_genera_el_script_hoy(tmp_path):
    _fail_if_outdated(DATA_SAMPLE, tmp_path)


# --- La prueba de arriba sí muerde ---

def _changed_config(change) -> dict:
    """Copia de config/operacion.yaml ya leído, con un cambio hecho por `change`."""
    cfg = copy.deepcopy(load_operacion())
    change(cfg)
    return cfg


def _one_more_budget(cfg: dict) -> None:
    cfg["presupuestos_diarios"] = [*cfg["presupuestos_diarios"], max(cfg["presupuestos_diarios"]) + 123]


def _one_window_less(cfg: dict) -> None:
    assert len(cfg["franjas"]) >= 2, "la prueba necesita al menos dos franjas en config/operacion.yaml"
    cfg["franjas"] = cfg["franjas"][:-1]


@pytest.mark.parametrize(
    "change, stale_file",
    [
        pytest.param(_one_more_budget, "policy_curve.parquet", id="un presupuesto más"),
        pytest.param(_one_window_less, "recommendations.parquet", id="una franja menos"),
    ],
)
def test_si_cambia_operacion_yaml_y_no_se_regenera_la_prueba_falla(change, stale_file, tmp_path, monkeypatch):
    """Simula que alguien edita config/operacion.yaml y no vuelve a correr el script."""
    cfg = _changed_config(change)
    monkeypatch.setattr(sample_data, "load_operacion", lambda: cfg)
    with pytest.raises(pytest.fail.Exception) as failure:
        _fail_if_outdated(DATA_SAMPLE, tmp_path)
    message = str(failure.value)
    assert stale_file in message
    assert f"Para regenerarlos: {COMMAND}" in message


def test_si_cambia_el_generador_y_no_se_regenera_la_prueba_falla(tmp_path, monkeypatch):
    monkeypatch.setattr(sample_data, "SEED", sample_data.SEED + 1)  # cualquier cambio que mueva los números
    with pytest.raises(pytest.fail.Exception) as failure:
        _fail_if_outdated(DATA_SAMPLE, tmp_path)
    message = str(failure.value)
    for name in ("stations", "station_hour", "predictions", "recommendations"):
        assert f"{name}.parquet" in message
    assert f"Para regenerarlos: {COMMAND}" in message


def test_si_falta_un_archivo_de_la_muestra_la_prueba_falla(tmp_path):
    empty = tmp_path / "muestra_vacia"
    empty.mkdir()
    with pytest.raises(pytest.fail.Exception, match="LEEME.md"):
        _fail_if_outdated(empty, tmp_path / "recien_generada")


# --------------------------------------------------------------------------- #
# 3) Se distinguen como sintéticos
# --------------------------------------------------------------------------- #

def test_las_estaciones_se_distinguen_como_sinteticas(tables):
    assert tables["stations"]["name"].str.startswith("Sintética").all()
    for name in ("stations", "station_hour", "predictions", "recommendations"):
        assert tables[name]["station_id"].str.startswith("SIM-").all(), name


# --------------------------------------------------------------------------- #
# 4) feasible es de la estación y la franja
# --------------------------------------------------------------------------- #

def test_feasible_vale_lo_mismo_en_las_tres_politicas(tables):
    """El contrato lo define así: falso cuando ningún inventario inicial evita a la vez vaciarse y llenarse.

    Eso depende de la estación y de la franja, no de la política ni de si se movieron bicis.
    """
    recommendations = tables["recommendations"]
    per_window = recommendations.groupby(["station_id", "window_start"])["feasible"]
    assert (per_window.size() == len(POLICIES)).all()
    assert (per_window.nunique() == 1).all(), "feasible cambia entre políticas en la misma estación y franja"

    infeasible = recommendations[~recommendations["feasible"]]
    assert 0 < len(infeasible) < 0.2 * len(recommendations)  # "unas pocas", como dice el LEEME
    assert set(infeasible["policy"]) == set(POLICIES)        # también con "none", que no mueve nada
    assert (infeasible["bikes_to_move"] == 0).any()


# --------------------------------------------------------------------------- #
# 5) No escribe donde van los datos reales
# --------------------------------------------------------------------------- #

@pytest.fixture
def no_writing(monkeypatch):
    """Por si la negativa dejara de funcionar: que la prueba no pueda escribir nada."""

    def forbidden(*args, **kwargs):
        raise AssertionError("el generador siguió adelante en vez de negarse")

    for step in ("build_sample", "write_contract", "write_leeme"):
        monkeypatch.setattr(sample_data, step, forbidden)


def _files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


@pytest.mark.parametrize("folder", [DATA_PROCESSED, DATA_RAW], ids=["data/processed", "data/raw"])
def test_el_generador_se_niega_a_escribir_en_las_carpetas_de_datos_reales(folder, no_writing):
    assert folder.is_dir()  # ya existe: aunque la negativa fallara, la prueba no crearía nada
    with pytest.raises(ValueError, match=REFUSAL):
        write_sample(folder)
    # Desde la línea de comandos: termina con el mensaje y código de salida distinto de cero.
    with pytest.raises(SystemExit) as exit_info:
        main(["--salida", str(folder)])
    assert isinstance(exit_info.value.code, str) and REFUSAL in exit_info.value.code


def test_tampoco_escribe_dentro_de_esas_carpetas_ni_llegando_por_otra_ruta(tmp_path, monkeypatch, no_writing):
    """Se prueba con carpetas de mentira, para no acercarse a las reales."""
    processed, raw = tmp_path / "data" / "processed", tmp_path / "data" / "raw"
    processed.mkdir(parents=True)
    raw.mkdir()
    monkeypatch.setattr(sample_data, "DATA_PROCESSED", processed)
    monkeypatch.setattr(sample_data, "DATA_RAW", raw)
    monkeypatch.chdir(tmp_path)  # para probar rutas relativas, como las de --salida

    refused = [
        processed / "sub" / "carpeta",
        raw / "gbfs",
        Path("data/processed"),
        Path("data/raw/x"),
        tmp_path / "app" / ".." / "data" / "processed",
    ]
    shouting = tmp_path / "data" / "PROCESSED"
    if shouting.exists():  # macOS y Windows: es la misma carpeta con otra capitalización
        refused.append(shouting)
    for out_dir in refused:
        with pytest.raises(ValueError, match=REFUSAL):
            check_output_folder(out_dir)
    with pytest.raises(ValueError, match=REFUSAL):
        write_sample(processed / "sub" / "carpeta")
    with pytest.raises(SystemExit, match=REFUSAL):
        main(["--salida", "data/raw/x"])
    assert not (processed / "sub").exists() and not (raw / "x").exists()  # se niega antes de crear nada
    assert _files(tmp_path) == []

    # Cualquier otra carpeta se acepta, también una vecina de nombre parecido.
    for out_dir in (tmp_path / "salida", tmp_path / "data" / "sample", tmp_path / "data" / "processed_copia"):
        assert check_output_folder(out_dir) == out_dir
