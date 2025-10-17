# from .biofisica_base import BiofisicaBase
# import numpy as np

# class NDBI(BiofisicaBase):
#     def calculate(self):
#         prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))

#         for prefix in prefixes:
#             swir, profile = self.read_band(prefix, "B11")
#             nir, _ = self.read_band(prefix, "B8")
#             ndbi = (swir - nir) / (swir + nir + 1e-10)
#             self.save_raster(ndbi, profile, prefix, "NDBI")

from .biofisica_base import BiofisicaBase
import numpy as np

class NDBI(BiofisicaBase):
    """
    Calcula el índice NDBI (Normalized Difference Built-up Index)
    a partir de las bandas Sentinel-2 B11 (SWIR1) y B8 (NIR).
    Los resultados se normalizan entre 0 y 1 para su uso directo
    en la generación del índice de islas de calor (UHI).
    """

    def calculate(self):
        # Detectar prefijos según archivos *_B11.tif (por mes y zona)
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))

        for prefix in prefixes:
            # Leer bandas necesarias
            swir, profile = self.read_band(prefix, "B11")
            nir, _ = self.read_band(prefix, "B8")

            # Validar que ambas existan
            if swir is None or nir is None:
                print(f"⚠️  Faltan bandas B11 o B8 para {prefix}, se omite cálculo de NDBI.")
                continue

            # Calcular NDBI
            ndbi = (swir - nir) / (swir + nir + 1e-10)  # evitar división por cero

            # Reemplazar valores inválidos
            ndbi = np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalizar entre 0 y 1
            min_val = np.nanmin(ndbi)
            max_val = np.nanmax(ndbi)
            if max_val > min_val:
                ndbi_norm = (ndbi - min_val) / (max_val - min_val)
            else:
                ndbi_norm = np.zeros_like(ndbi)

            # Guardar raster normalizado
            self.save_raster(ndbi_norm, profile, prefix, "NDBI")

            print(f"✅ NDBI normalizado generado: {prefix}_NDBI.tif")
