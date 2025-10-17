import rasterio
import numpy as np
from pathlib import Path

class Climatica:
    """
    Clase base para variables climatológicas.
    Proporciona métodos para leer, guardar y manejar imágenes raster
    como temperatura, humedad, etc.
    """

    def __init__(self, input_dir, output_dir):
        """
        Parámetros:
        - input_dir: carpeta donde están las imágenes de entrada (.tif)
        - output_dir: carpeta donde se guardarán los resultados (.tif)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    # === Funciones base comunes ===

    def read_raster(self, prefix, suffix):
        """
        Lee un raster a partir del prefijo y sufijo.
        Ejemplo: prefix='TempAire_2021_01', suffix='TempAire'
        """
        raster_path = self.input_dir / f"{prefix}_{suffix}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {raster_path}")

        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(float)
            profile = src.profile
            data[data == src.nodata] = np.nan

        return data, profile

    def save_raster(self, array, profile, prefix, suffix):
        """
        Guarda un raster con el nombre prefix_suffix.tif
        """
        output_path = self.output_dir / f"{prefix}_{suffix}.tif"
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(array.astype(np.float32), 1)

        print(f"💾 Guardado: {output_path}")

    def list_rasters(self, pattern):
        """
        Lista los archivos .tif que cumplan un patrón dentro de la carpeta de entrada.
        Ejemplo: list_rasters('TempAire_*.tif')
        """
        return sorted(self.input_dir.glob(pattern))