"""Congela la capacidad y el inventario inicial de la captura de GBFS.

Se puede ejecutar desde cualquier carpeta:

    python scripts/freeze_gbfs.py
    python scripts/freeze_gbfs.py --reemplazar
    python scripts/freeze_gbfs.py --help

Lee la captura de data/raw/gbfs (solo lee: no la toca) y escribe
config/gbfs_congelado.csv, un archivo pequeño que sí va en Git, con una fila
por estación: la capacidad más frecuente entre las fotos, la ocupación media
en la hora local en que empieza la simulación (00:00 a 00:59 de Nueva York) y
cuántas fotos entraron en ese promedio.

La limpieza (scripts/build_tables.py) toma de ahí la capacidad y el simulador
el inventario inicial. Como el archivo está congelado, reconstruir las tablas
ya no cambia los resultados aunque la captura siga creciendo. Por eso el
script se niega a pisar el archivo si ya existe, salvo con --reemplazar.

La captura es de octubre de 2026, no de los días que se simulan.

La lógica está en src/data/gbfs_frozen.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.gbfs_frozen import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
