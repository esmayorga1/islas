import os
import numpy as np
import rasterio

class UHIGenerator:
    """
    Clase para generar el índice compuesto de Islas de Calor (UHI)
    combinando variables biofísicas normalizadas (Albedo, LST, NDVI, NDBI, MNDWI).
    """

    def __init__(self, ruta_indices: str, anio: int, salida: str):
        self.ruta_indices = ruta_indices
        self.anio = anio
        self.salida = salida
        os.makedirs(salida, exist_ok=True)

        # Variables esperadas
        self.vars_pos = ["LST", "NDBI", "Albedo"]   # aportan calor
        self.vars_neg = ["NDVI", "MNDWI"]           # reducen calor

    # ----------------------------------------------------------
    def generar(self):
        """Genera el UHI mensual para cada mes disponible."""
        print(f"🌡️ Generando índice UHI para el año {self.anio}\n")

        for mes in range(1, 13):
            prefijo = f"S2_{self.anio}_{mes}_PatioBonito"
            capas = {}

            # Leer todas las variables
            for var in self.vars_pos + self.vars_neg:
                ruta = os.path.join(self.ruta_indices, f"{prefijo}_{var}.tif")
                if not os.path.exists(ruta):
                    print(f"⚠️ No se encontró {ruta}, se omite el mes {mes}.")
                    capas = None
                    break

                with rasterio.open(ruta) as src:
                    data = src.read(1).astype(np.float32)
                    capas[var] = data
                    profile = src.profile

            if capas is None:
                continue

            # Calcular índice compuesto
            suma_pos = sum(capas[v] for v in self.vars_pos)
            suma_neg = sum(capas[v] for v in self.vars_neg)
            uhi = suma_pos - suma_neg

            # Reemplazar NaN y normalizar
            uhi = np.nan_to_num(uhi, nan=0.0, posinf=0.0, neginf=0.0)
            min_val, max_val = np.nanmin(uhi), np.nanmax(uhi)
            if max_val > min_val:
                uhi_norm = (uhi - min_val) / (max_val - min_val)
            else:
                uhi_norm = np.zeros_like(uhi)

            # Guardar resultado
            nombre_salida = os.path.join(self.salida, f"UHI_{self.anio}_{mes:02d}.tif")
            with rasterio.open(nombre_salida, "w", **profile) as dst:
                dst.write(uhi_norm, 1)

            print(f"✅ UHI mensual generado: {nombre_salida}")

        print("\n🏁 Finalizó la generación de índices UHI.")
