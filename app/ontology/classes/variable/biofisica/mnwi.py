# from .biofisica_base import BiofisicaBase
# import numpy as np

# class MNDWI(BiofisicaBase):
#     def calculate(self):
#         """
#         Calcula el índice MNDWI a partir de bandas Sentinel-2 B3 (verde) y B11 (SWIR1).
#         """
#         # Detectar prefijos según archivos *_B11.tif
#         prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))

#         for prefix in prefixes:
#             # Leer bandas
#             green, profile = self.read_band(prefix, "B3")
#             swir, _ = self.read_band(prefix, "B11")

#             # Calcular MNDWI
#             mndwi = (green - swir) / (green + swir + 1e-10)  # evitar división por cero

#             # Guardar raster
#             self.save_raster(mndwi, profile, prefix, "MNDWI")


# Despues

from .biofisica_base import BiofisicaBase
import numpy as np

class MNDWI(BiofisicaBase):
    """
    Calcula el índice MNDWI (Modified Normalized Difference Water Index)
    a partir de las bandas Sentinel-2 B3 (verde) y B11 (SWIR1).
    Los resultados se normalizan entre 0 y 1 para su uso directo
    en la generación del índice de islas de calor (UHI).
    """

    def calculate(self):
        # Detectar prefijos según archivos *_B11.tif (por mes y área)
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))

        for prefix in prefixes:
            # Leer bandas requeridas
            green, profile = self.read_band(prefix, "B3")
            swir, _ = self.read_band(prefix, "B11")

            # Validar que ambas bandas existan
            if green is None or swir is None:
                print(f"⚠️  Faltan bandas para {prefix}, se omite cálculo de MNDWI.")
                continue

            # Calcular índice MNDWI
            mndwi = (green - swir) / (green + swir + 1e-10)  # evitar división por cero

            # Reemplazar valores no válidos
            mndwi = np.nan_to_num(mndwi, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalización entre 0 y 1
            min_val = np.nanmin(mndwi)
            max_val = np.nanmax(mndwi)
            if max_val > min_val:
                mndwi_norm = (mndwi - min_val) / (max_val - min_val)
            else:
                mndwi_norm = np.zeros_like(mndwi)

            # Guardar raster normalizado
            self.save_raster(mndwi_norm, profile, prefix, "MNDWI")

            print(f"✅ MNDWI normalizado generado: {prefix}_MNDWI.tif")
