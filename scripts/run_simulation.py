"""Simulación de inventario: políticas, varios presupuestos, tramo de validación.

Se puede ejecutar desde cualquier carpeta:

    python scripts/run_simulation.py
    python scripts/run_simulation.py --datos otra/carpeta --salida otra/salida
    python scripts/run_simulation.py --operacion otro/operacion.yaml --config otro/datos.yaml
    python scripts/run_simulation.py --gbfs otro/gbfs_congelado.csv
    python scripts/run_simulation.py --help

Lee stations.parquet, trips.parquet, station_hour.parquet y predictions.parquet,
las reglas de config/operacion.yaml, el bloque `simulacion` de config/datos.yaml
y config/gbfs_congelado.csv (el inventario inicial; se crea con
scripts/freeze_gbfs.py). Escribe recommendations.parquet, policy_curve.parquet,
reports/simulacion.json y reports/simulacion.md.

Hay que volver a correrlo cuando cambia cualquiera de esas entradas.
simulacion.json guarda una huella de lo que se leyó y `pytest` avisa cuando
las salidas ya no corresponden.

Compara sin intervención, regla histórica y modelo. La regla histórica y el
modelo son la misma maquinaria: solo cambia su pronóstico (filas "baseline" o
filas "lgbm" de predictions). Si no hay filas "lgbm", la política "model" no
se simula y las tablas salen sin ella. Todas parten del mismo inventario (la
ocupación observada en la captura de GBFS, que es de octubre de 2026 y no de
los días simulados) y enfrentan los mismos viajes. El primer día es de
calentamiento y no cuenta. El total de bicis no cambia durante la corrida.

La banda de incertidumbre se elige en la misma corrida entre las candidatas
de config/datos.yaml: la de menor riesgo simulado en validación.

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
