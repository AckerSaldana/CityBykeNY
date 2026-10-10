"""Limpia los viajes crudos y escribe las tablas reales en data/processed.

Se puede ejecutar desde cualquier carpeta:

    python scripts/build_tables.py
    python scripts/build_tables.py --help

Escribe stations.parquet, station_hour.parquet y routes.parquet, y el reporte
de la limpieza en data/processed/reports/limpieza.md y limpieza.json.

También escribe trips.parquet: una fila por viaje limpio, para el simulador.
Es una tabla interna, no uno de los seis contratos.

El periodo, los meses y las reglas salen de config/datos.yaml. Los meses deben
estar ya descargados (scripts/download_trips.py).

La capacidad de las estaciones sale de la última foto de GBFS
(scripts/capture_gbfs.py --una-vez). Sin ninguna foto la capacidad queda vacía
y el script lo avisa al terminar (AVISO).

La lógica está en src/data/build_tables.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_tables import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
