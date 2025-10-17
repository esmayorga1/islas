# Anterior

# from .biofisica_base import BiofisicaBase
# import numpy as np

# class NDVI(BiofisicaBase):
#     def calculate(self):
#         # Agrupar prefijos de imágenes encontradas en la carpeta
#         prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B4.tif"))

#         for prefix in prefixes:
#             red, profile = self.read_band(prefix, "B4")
#             nir, _ = self.read_band(prefix, "B8")
#             ndvi = (nir - red) / (nir + red + 1e-10)
#             self.save_raster(ndvi, profile, prefix, "NDVI")

# Nuevo
from .biofisica_base import BiofisicaBase
import numpy as np

class NDVI(BiofisicaBase):
    """
    Calcula el índice NDVI (Normalized Difference Vegetation Index)
    a partir de las bandas Sentinel-2 B8 (NIR) y B4 (Rojo).
    Los resultados se normalizan entre 0 y 1 para su uso directo
    en la generación del índice de islas de calor (UHI).
    """

    def calculate(self):
        # Buscar prefijos únicos según archivos *_B4.tif (por mes y área)
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B4.tif"))

        for prefix in prefixes:
            # Leer bandas necesarias
            red, profile = self.read_band(prefix, "B4")
            nir, _ = self.read_band(prefix, "B8")

            # Validar existencia de ambas bandas
            if red is None or nir is None:
                print(f"⚠️  Faltan bandas B4 o B8 para {prefix}, se omite cálculo de NDVI.")
                continue

            # Calcular NDVI
            ndvi = (nir - red) / (nir + red + 1e-10)  # evitar división por cero

            # Reemplazar valores no válidos
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalización entre 0 y 1
            min_val = np.nanmin(ndvi)
            max_val = np.nanmax(ndvi)
            if max_val > min_val:
                ndvi_norm = (ndvi - min_val) / (max_val - min_val)
            else:
                ndvi_norm = np.zeros_like(ndvi)

            # Guardar raster normalizado
            self.save_raster(ndvi_norm, profile, prefix, "NDVI")

            print(f"✅ NDVI normalizado generado: {prefix}_NDVI.tif")
