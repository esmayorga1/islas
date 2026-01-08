"""
Módulo de gestión de fuentes de datos (raster, vectorial, censal, etc.)
correspondiente a la clase 'fuente_de_datos' en la ontología proyecto islas de calor.
"""

# backend/app/ontology/classes/fuente_de_datos/__init__.py

from .reproyectar import Reproyectar

__all__ = ["Reproyectar"]
