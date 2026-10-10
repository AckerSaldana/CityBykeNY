"""Rutas del proyecto en un solo lugar.

Todo el código debe construir rutas a partir de estas constantes, para que
funcione igual sin importar desde qué carpeta se ejecute.
"""

from pathlib import Path

# Carpeta raíz del repositorio (la que contiene src/, app/, data/, ...).
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"              # Datos pesados descargados. NO va en Git.
DATA_PROCESSED = DATA / "processed"  # Salidas reales del flujo. NO va en Git.
DATA_SAMPLE = DATA / "sample"        # Datos SINTÉTICOS pequeños. Sí va en Git.

CONFIG = ROOT / "config"
OPERACION_YAML = CONFIG / "operacion.yaml"

# Zona horaria de todo el análisis (decisión 1: hora local de Nueva York).
TZ = "America/New_York"
