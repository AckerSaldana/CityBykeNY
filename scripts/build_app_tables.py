"""Construye las tablas pequeñas que lee el dashboard (data/processed/app).

    python scripts/build_app_tables.py
    python scripts/build_app_tables.py --datos data/sample --salida /tmp/app

Hay que volver a correrlo cada vez que cambien las tablas de data/processed
(por ejemplo, después de una simulación nueva). El dashboard toma los archivos
nuevos solo, sin reiniciarlo.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import app_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, default=app_tables.DATOS, help="carpeta con los seis contratos")
    parser.add_argument("--salida", type=Path, default=None, help="carpeta de salida (por omisión, <datos>/app)")
    args = parser.parse_args()
    salida = args.salida or args.datos / "app"

    inicio = time.time()
    indicadores = app_tables.build(args.datos, salida)
    print(f"Tablas de la app en {salida} ({time.time() - inicio:.1f} s)")
    for ruta in sorted(salida.iterdir()):
        print(f"  {ruta.name:28s} {ruta.stat().st_size / 1024:10.1f} KB")
    ref = indicadores["referencia"]
    print(f"Viajes analizados: {indicadores['viajes_analizados']:,}; estaciones: {indicadores['estaciones']:,}; "
          f"presupuesto de referencia: {ref['presupuesto']}")


if __name__ == "__main__":
    main()
