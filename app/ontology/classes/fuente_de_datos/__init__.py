"""
Módulo de gestión de fuentes de datos (raster, vectorial, censal, etc.)
correspondiente a la clase 'fuente_de_datos' en la ontología proyecto islas de calor.
"""

# backend/app/ontology/classes/fuente_de_datos/__init__.py
from .fuente_de_datos import FuenteDeDatos
from .sentinel2 import Sentinel2Fuente
from .reproyectar import Reproyectar

__all__ = ["FuenteDeDatos", "Sentinel2Fuente", "Reproyectar"]
