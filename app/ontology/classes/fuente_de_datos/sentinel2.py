# backend/app/ontology/classes/fuente_de_datos/sentinel2_fuente.py

import rasterio
from pathlib import Path
from .fuente_de_datos import FuenteDeDatos

class Sentinel2Fuente(FuenteDeDatos):
    """
    Clase para manejar imágenes Sentinel-2 unificadas (una por mes).
    """
    def __init__(self, ruta: Path, anio: int, zona: str):
        super().__init__(ruta, f"Sentinel2_{anio}_{zona}")
        self.anio = anio
        self.zona = zona

    def listar_imagenes(self):
        return sorted(self.ruta.glob(f"S2_{self.anio}_*_*.tif"))

    def leer_imagen(self, indice: int):
        imagenes = self.listar_imagenes()
        if not imagenes:
            raise FileNotFoundError("No se encontraron imágenes Sentinel-2 en la ruta especificada.")
        if indice < 1 or indice > len(imagenes):
            raise IndexError("Índice de imagen fuera de rango.")
        return rasterio.open(imagenes[indice - 1])
