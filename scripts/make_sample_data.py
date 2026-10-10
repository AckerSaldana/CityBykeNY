"""Genera los datos SINTÉTICOS de muestra en data/sample.

Son datos inventados para construir el dashboard. No son resultados.

Uso (desde cualquier carpeta):
    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --salida otra/carpeta

La carpeta de salida no puede ser data/processed ni data/raw: ahí van solo
datos reales. Si se pide, el script termina con un error y no escribe nada.

La lógica está en src/data/sample_data.py.
"""

import sys
from pathlib import Path

# Agrega la raíz del repositorio para poder importar `src` desde cualquier carpeta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sample_data import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
