from .urbana import Urbana
import numpy as np
import rasterio

class ClasificacionSuperficie(Urbana):
    """
    Clasificación de la superficie en cuatro clases:
    1 = Vegetación
    2 = Agua
    3 = Impermeable
    4 = Suelo desnudo
    """

    def calculate(self):
        # Detectar prefijos según archivos NDBI
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_NDBI.tif"))

        for prefix in prefixes:
            # Leer índices existentes
            ndbi, profile = self.read_raster(prefix, "NDBI")
            ndvi, _ = self.read_raster(prefix, "NDVI")
            mndwi, _ = self.read_raster(prefix, "MNDWI")

            # Inicializar raster de clases
            classes = np.zeros(ndbi.shape, dtype=np.uint8)

            # Reglas de clasificación
            classes[(ndvi > 0.3) & (ndbi < 0.2)] = 1  # Vegetación
            classes[mndwi > 0] = 2                    # Agua
            classes[(ndbi > 0) & (ndvi < 0.2)] = 3    # Impermeable
            classes[(ndvi < 0.2) & (ndbi < 0) & (mndwi < 0)] = 4  # Suelo desnudo

            # Guardar raster de clasificación
            self.save_raster(classes, profile, prefix, "ClasificacionSuperficie")