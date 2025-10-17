from pathlib import Path
import rasterio
from abc import ABC, abstractmethod
import numpy as np

class BiofisicaBase(ABC):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_band(self, prefix: str, band: str):
        """Leer una banda específica según el prefijo de la imagen"""
        search_name = f"{prefix}_{band}.tif"
        path = self.input_dir / search_name
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la banda: {path}")
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            profile = src.profile
        return data, profile

    def save_raster(self, data: np.ndarray, profile: dict, prefix: str, index_name: str):
        """Guarda raster usando el prefijo de la imagen y el índice"""
        out_name = f"{prefix}_{index_name}.tif"
        out_path = self.output_dir / out_name
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
        print(f"✅ Guardado: {out_path}")

    @abstractmethod
    def calculate(self):
        pass
