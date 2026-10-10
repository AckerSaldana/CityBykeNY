"""Línea base: predicciones y métricas de validación.

Se puede ejecutar desde cualquier carpeta:

    python scripts/run_baseline.py
    python scripts/run_baseline.py --datos otra/carpeta --salida otra/salida
    python scripts/run_baseline.py --config otro/datos.yaml
    python scripts/run_baseline.py --help

Lee station_hour.parquet y escribe predictions.parquet,
reports/metricas_validacion.parquet, reports/metricas_validacion.md y
reports/linea_base_entrada.json (la huella de la entrada: con ella una prueba
nota si station_hour o el config cambiaron después de esta corrida).
La prueba final no se califica.

No corre si station_hour.parquet no cumple su contrato, ni si
reports/limpieza.json dice que las tablas se construyeron con otro periodo.

La lógica está en src/models/baseline.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
