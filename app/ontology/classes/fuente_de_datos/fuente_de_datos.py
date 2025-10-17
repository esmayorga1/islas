# backend/app/ontology/classes/fuente_de_datos/base_fuente.py

from pathlib import Path

class FuenteDeDatos:
    """
    Clase base para todas las fuentes de datos geoespaciales.
    """
    def __init__(self, ruta: Path, nombre: str):
        self.ruta = ruta
        self.nombre = nombre

    def __repr__(self):
        return f"FuenteDeDatos(nombre={self.nombre}, ruta={self.ruta})"
