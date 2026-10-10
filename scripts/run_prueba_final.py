"""Prueba final (decisión 4): métricas de pronóstico del tramo de prueba y tabla validación contra prueba.

    python scripts/run_prueba_final.py --usar-prueba-final   # califica el pronóstico (UNA vez) y arma la tabla
    python scripts/run_prueba_final.py                       # solo rearma la tabla con lo ya escrito

Antes hay que simular el tramo de prueba con la banda elegida en validación:

    python scripts/run_simulation.py --tramo prueba --usar-prueba-final --banda 0

No entrena nada. Escribe reports/validacion_y_prueba.json y reports/prueba_final.md.
La lógica está en src/models/prueba_final.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.prueba_final import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
