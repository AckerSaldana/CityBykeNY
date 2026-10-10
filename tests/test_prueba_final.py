"""Prueba final: el candado de "una sola vez" y las cuentas de la tabla validación contra prueba.

Todo con datos inventados aquí: no toca data/processed ni califica nada real.
"""

import numpy as np

from src.models.prueba_final import FLAG, METRICS_FILE, main, mean_ci, ratio_ci
from src.sim.simulate import FINAL_FOLDER, resample_days


def test_no_califica_dos_veces_la_prueba_final(tmp_path, capsys):
    scored = tmp_path / FINAL_FOLDER / "reports" / METRICS_FILE
    scored.parent.mkdir(parents=True)
    scored.write_text("{}", encoding="utf-8")
    assert main([str(tmp_path), FLAG]) == 1
    assert "ya se calificó y se usa una sola vez" in capsys.readouterr().out
    assert scored.read_text(encoding="utf-8") == "{}" and not (tmp_path / "reports").exists()


def test_sin_los_reportes_dice_cuales_faltan_y_no_escribe(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "Faltan" in out and "simulacion.json" in out and not (tmp_path / "reports").exists()


def test_los_intervalos_remuestrean_dias_y_contienen_a_la_media():
    draws = resample_days(4, 500, 7)
    part, whole = np.array([1.0, 2.0, 3.0, 6.0]), np.array([10.0, 10.0, 20.0, 20.0])
    mean, (low, high) = mean_ci(part, draws, 0.9)
    assert mean == 3.0 and low <= mean <= high and low < high
    ratio, (low, high) = ratio_ci(part, whole, draws, 0.9)
    assert ratio == 12.0 / 60.0 and low <= ratio <= high
    # Con días iguales no hay dispersión: el intervalo es un punto.
    assert ratio_ci(np.ones(4), np.full(4, 2.0), draws, 0.9) == (0.5, [0.5, 0.5])
