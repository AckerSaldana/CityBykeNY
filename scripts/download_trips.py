"""Descarga viajes de Citi Bike por mes y los deja en Parquet.

Se puede ejecutar desde cualquier carpeta:

    python scripts/download_trips.py 202604
    python scripts/download_trips.py --desde 202601 --hasta 202609
    python scripts/download_trips.py --listar 202609 202102
    python scripts/download_trips.py --help

La lógica está en src/data/download_trips.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download_trips import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
