"""Captura el feed GBFS en vivo de Citi Bike y lo acumula en Parquet.

Uso, desde cualquier carpeta:

    python scripts/capture_gbfs.py                  # captura cada 5 minutos, sin parar
    python scripts/capture_gbfs.py --intervalo 120  # cada 2 minutos (mínimo 60)
    python scripts/capture_gbfs.py --una-vez        # una sola captura (para cron)
    python scripts/capture_gbfs.py --salida DIR     # guardar en otra carpeta

Cómo dejarlo corriendo, cómo detenerlo y cómo leer los datos: ver el texto al
inicio de src/data/gbfs.py. Toda la lógica está ahí.
"""

import sys
from pathlib import Path

# Permite `import src...` sin importar desde qué carpeta se ejecute.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gbfs import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
