"""Piso de azar del WAPE y métricas de pronóstico por franja, en validación.

    python scripts/run_piso_wape.py

No entrena nada: lee predictions.parquet y station_hour.parquet y escribe
reports/piso_wape.md y reports/piso_wape.json. La prueba final no se califica.

La lógica y la definición de los dos pronósticos ingenuos están en
src/models/piso_wape.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.piso_wape import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
