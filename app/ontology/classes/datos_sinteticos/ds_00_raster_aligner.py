import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os


class RasterAlignerToDEM3m:
    """
    Alinea cualquier raster a la malla del DEM 3m (raster maestro).
    """

    def __init__(self, dem_path):
        self.dem_path = dem_path

        # Leer metadata del raster maestro
        with rasterio.open(dem_path) as src:
            self.master_meta = src.meta.copy()
            self.master_transform = src.transform
            self.master_crs = src.crs
            self.master_shape = (src.height, src.width)

    def align(self, input_path, output_path):
        """
        Alinea un raster al DEM usando resampling nearest.
        """

        with rasterio.open(input_path) as src:
            src_data = src.read(1)
            src_transform = src.transform
            src_crs = src.crs

        # Matriz destino con dimensión EXACTA del DEM
        dst_data = np.empty(self.master_shape, dtype=src_data.dtype)

        # Reproject correctamente
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=self.master_transform,
            dst_crs=self.master_crs,
            resampling=Resampling.nearest
        )

        # Guardar alineado
        meta = self.master_meta.copy()
        meta.update(dtype=dst_data.dtype, count=1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(dst_data, 1)

        return output_path
