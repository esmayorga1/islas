# Anterior

# from .biofisica_base import BiofisicaBase
# import numpy as np

# class Albedo(BiofisicaBase):
#     def calculate(self):
#         prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B2.tif"))

#         for prefix in prefixes:
#             bands = ["B2", "B3", "B4", "B8"]
#             data_list = []
#             profile = None
#             for b in bands:
#                 data, profile = self.read_band(prefix, b)
#                 data_list.append(data)
#             albedo = np.mean(np.array(data_list), axis=0)
#             self.save_raster(albedo, profile, prefix, "Albedo")

# nuevo

from .biofisica_base import BiofisicaBase
import numpy as np

class Albedo(BiofisicaBase):
    """
    Calcula el albedo promedio a partir de las bandas B2, B3, B4 y B8.
    Los valores resultantes se normalizan entre 0 y 1 para su uso directo
    en la generación del índice de islas de calor (UHI).
    """

    def calculate(self):
        # Identificar prefijos únicos (ej. S2_2021_1_PatioBonito)
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B2.tif"))

        for prefix in prefixes:
            bands = ["B2", "B3", "B4", "B8"]
            data_list = []
            profile = None

            # Leer las bandas necesarias
            for b in bands:
                data, profile = self.read_band(prefix, b)
                if data is not None:
                    data_list.append(data)

            if not data_list:
                print(f"⚠️ No se encontraron bandas para {prefix}, se omite.")
                continue

            # Calcular el albedo como promedio de las 4 bandas
            albedo = np.mean(np.array(data_list), axis=0)

            # Reemplazar valores inválidos (NaN, infinitos)
            albedo = np.nan_to_num(albedo, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalizar entre 0 y 1 (Min-Max)
            min_val = np.nanmin(albedo)
            max_val = np.nanmax(albedo)
            if max_val > min_val:
                albedo_norm = (albedo - min_val) / (max_val - min_val)
            else:
                albedo_norm = np.zeros_like(albedo)

            # Guardar raster normalizado
            self.save_raster(albedo_norm, profile, prefix, "Albedo")

            print(f"✅ Albedo normalizado generado: {prefix}_Albedo.tif")
