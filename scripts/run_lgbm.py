"""LightGBM con cuantiles: predicciones y métricas de validación, comparadas con la línea base.

Se puede ejecutar desde cualquier carpeta:

    python scripts/run_lgbm.py
    python scripts/run_lgbm.py --datos otra/carpeta --salida otra/salida
    python scripts/run_lgbm.py --config otro/datos.yaml --hilos 4
    python scripts/run_lgbm.py --help

Antes hay que correr build_tables.py y run_baseline.py: lee station_hour.parquet,
stations.parquet y la línea base de predictions.parquet.

Escribe sus filas (model = "lgbm") en predictions.parquet sin tocar las de la
línea base, agrega sus métricas a reports/metricas_validacion.parquet y escribe
reports/metricas_lgbm.md y reports/lgbm_entrada.json.
La prueba final no se califica.

La lógica está en src/models/lgbm.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lgbm import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
