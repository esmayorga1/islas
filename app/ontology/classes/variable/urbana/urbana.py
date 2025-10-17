import rasterio
from pathlib import Path
import numpy as np

class Urbana:
    def __init__(self, input_dir, output_dir):
        """
        Clase base para índices urbanos.
        input_dir: carpeta donde están los rasters ya calculados (NDVI, NDBI, MNDWI, etc.)
        output_dir: carpeta donde se guardarán los nuevos índices
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_raster(self, prefix, name):
        """
        Leer raster existente por prefijo y nombre de índice.
        Retorna: numpy array y perfil del raster
        """
        path = self.input_dir / f"{prefix}_{name}.tif"
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            profile = src.profile
        return data, profile

    def save_raster(self, array, profile, prefix, name):
        """
        Guardar raster con el nombre de índice especificado.
        """
        output_path = self.output_dir / f"{prefix}_{name}.tif"
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype(rasterio.float32), 1)
        print(f"✅ Raster guardado: {output_path}")
