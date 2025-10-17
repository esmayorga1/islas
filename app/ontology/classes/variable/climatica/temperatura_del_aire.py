from .climatica import Climatica
import numpy as np
import rasterio

class TemperaturaAire(Climatica):
    """
    Calcula la temperatura promedio anual del aire a partir de los raster mensuales
    (TempAire_YYYY_MM.tif) generados a partir de MODIS MOD11A2.
    """

    def calculate(self):
        files = sorted(self.input_dir.glob("TempAire_*.tif"))
        if not files:
            print("⚠️ No se encontraron archivos con el patrón TempAire_YYYY_MM.tif")
            return

        print(f"📂 Se encontraron {len(files)} imágenes de temperatura mensual.")

        arrays = []
        profile = None

        for f in files:
            with rasterio.open(f) as src:
                data = src.read(1).astype(float)
                profile = src.profile
                data[data == src.nodata] = np.nan
                arrays.append(data)

        temp_anual = np.nanmean(np.stack(arrays), axis=0)

        output_name = "TemperaturaAire_Anual"
        self.save_raster(temp_anual, profile, "MODIS", output_name)

        print(f"✅ Raster de temperatura anual guardado: {output_name}.tif")
        return temp_anual
