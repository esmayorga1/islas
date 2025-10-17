# Anterior
# from .biofisica_base import BiofisicaBase
# import numpy as np
# 
# class LST(BiofisicaBase):
    # def calculate(self):
        # prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))
# 
        # for prefix in prefixes:
            # swir, profile = self.read_band(prefix, "B11")
            # lst = 0.1 * swir
            # self.save_raster(lst, profile, prefix, "LST")

# Despues
from .biofisica_base import BiofisicaBase
import numpy as np

class LST(BiofisicaBase):
    """
    Calcula la temperatura de superficie (LST) a partir de la banda B11.
    El resultado se normaliza entre 0 y 1 para su uso en el índice de islas de calor (UHI).
    """

    def calculate(self):
        # Buscar prefijos únicos (ej. S2_2021_1_PatioBonito)
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B11.tif"))

        for prefix in prefixes:
            # Leer banda B11 (SWIR)
            swir, profile = self.read_band(prefix, "B11")

            if swir is None:
                print(f"⚠️ No se encontró la banda B11 para {prefix}, se omite.")
                continue

            # Calcular LST (proporcionalidad simple, se puede reemplazar por una fórmula más avanzada)
            lst = 0.1 * swir

            # Reemplazar valores no válidos
            lst = np.nan_to_num(lst, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalizar entre 0 y 1
            min_val = np.nanmin(lst)
            max_val = np.nanmax(lst)
            if max_val > min_val:
                lst_norm = (lst - min_val) / (max_val - min_val)
            else:
                lst_norm = np.zeros_like(lst)

            # Guardar raster normalizado
            self.save_raster(lst_norm, profile, prefix, "LST")

            print(f"✅ LST normalizado generado: {prefix}_LST.tif")