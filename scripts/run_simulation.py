"""Simulación de inventario: tres políticas, varios presupuestos, tramo de validación.

Se puede ejecutar desde cualquier carpeta:

    python scripts/run_simulation.py
    python scripts/run_simulation.py --datos otra/carpeta --salida otra/salida
    python scripts/run_simulation.py --operacion otro/operacion.yaml --config otro/datos.yaml
    python scripts/run_simulation.py --help

Lee stations.parquet, trips.parquet, station_hour.parquet y predictions.parquet,
y las reglas de config/operacion.yaml. Escribe recommendations.parquet,
policy_curve.parquet, reports/simulacion.json y reports/simulacion.md.

Hay que volver a correrlo cuando cambia cualquiera de esas tablas, las reglas o
el bloque `simulacion` de config/datos.yaml. simulacion.json guarda una huella
de lo que se leyó y `pytest` avisa cuando las salidas ya no corresponden.

Compara sin intervención, regla histórica y modelo. Las tres parten del mismo
inventario y enfrentan los mismos viajes. El primer día es de calentamiento y
no cuenta.

La prueba final no se simula (decisión 4): `--tramo prueba` se rechaza salvo
que se agregue `--usar-prueba-final`.

Los viajes bloqueados son un resultado de la simulación (posible demanda
insatisfecha), no demanda medida.

La lógica está en src/sim/simulate.py, src/sim/engine.py y src/sim/policies.py.
"""

import sys
from pathlib import Path

# La raíz del repositorio entra al camino de importación para poder usar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.simulate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
