from .biofisica_base import BiofisicaBase
import numpy as np

class LST(BiofisicaBase):
    """
    Calcula un índice LST indirecto (proxy térmico) usando SOLO Sentinel-2,
    basado en índices espectrales normalizados:

        LST = 0.40*NDBI + 0.30*ALBEDO - 0.20*NDVI - 0.10*NDWI

    El resultado se normaliza entre 0 y 1 para su uso en el índice UHI.
    """

    # Pesos del proxy térmico
    W_NDBI = 0.40
    W_ALB  = 0.30
    W_NDVI = 0.20
    W_NDWI = 0.10

    def calculate(self):
        # 🔹 MISMA lógica que NDVI: buscar prefijos a partir de *_B4.tif
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_B4.tif"))

        for prefix in prefixes:
            try:
                # Leer bandas
                b2, profile = self.read_band(prefix, "B2")   # Azul
                b3, _       = self.read_band(prefix, "B3")   # Verde
                b4, _       = self.read_band(prefix, "B4")   # Rojo
                b8, _       = self.read_band(prefix, "B8")   # NIR
                b11, _      = self.read_band(prefix, "B11")  # SWIR1
                b12, _      = self.read_band(prefix, "B12")  # SWIR2
            except FileNotFoundError as e:
                print(f"⚠️  {prefix}: {e}. Se omite LST.")
                continue

            # Convertir a float
            b2  = b2.astype(np.float32)
            b3  = b3.astype(np.float32)
            b4  = b4.astype(np.float32)
            b8  = b8.astype(np.float32)
            b11 = b11.astype(np.float32)
            b12 = b12.astype(np.float32)

            # 🔹 Índices espectrales
            ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
            ndbi = (b11 - b8) / (b11 + b8 + 1e-10)
            ndwi = (b3 - b8) / (b3 + b8 + 1e-10)

            # 🔹 Albedo simplificado
            albedo = (b2 + b3 + b4 + b8 + b11 + b12) / 6.0

            # Limpiar valores inválidos
            ndvi   = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
            ndbi   = np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0)
            ndwi   = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)
            albedo = np.nan_to_num(albedo, nan=0.0, posinf=0.0, neginf=0.0)

            # 🔹 Normalización 0–1 (MISMO enfoque que NDVI)
            def norm01(x):
                min_val = np.nanmin(x)
                max_val = np.nanmax(x)
                if max_val > min_val:
                    return (x - min_val) / (max_val - min_val)
                return np.zeros_like(x)

            ndvi_n = norm01(ndvi)
            ndbi_n = norm01(ndbi)
            ndwi_n = norm01(ndwi)
            alb_n  = norm01(albedo)

            # 🔥 LST indirecto (proxy térmico)
            lst = (
                self.W_NDBI * ndbi_n +
                self.W_ALB  * alb_n  -
                self.W_NDVI * ndvi_n -
                self.W_NDWI * ndwi_n
            )

            lst = np.nan_to_num(lst, nan=0.0, posinf=0.0, neginf=0.0)

            # 🔹 Normalizar resultado final
            lst_norm = norm01(lst)

            # Guardar raster
            self.save_raster(lst_norm, profile, prefix, "LST")

            print(f"✅ LST indirecto normalizado generado: {prefix}_LST.tif")
