import math
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject   # IMPORT CORRECTO


class RasterResampler3mWGS84:
    """
    Remuestrea un raster EPSG:4326 a resolución equivalente a 3 metros,
    convirtiendo metros a grados según la latitud fija de Bogotá.
    """

    PIXEL_SIZE_DEG = 3 / 110574  # Bogotá

    def __init__(self, raster_path, method="bilinear"):
        self.raster_path = raster_path
        self.method = method
        self.data = None
        self.meta = None

    def load(self):
        with rasterio.open(self.raster_path) as src:
            self.meta = src.meta.copy()

    def resample(self):
        if self.meta is None:
            self.load()

        with rasterio.open(self.raster_path) as src:
            left, bottom, right, top = src.bounds

            pixel_size = RasterResampler3mWGS84.PIXEL_SIZE_DEG

            new_width = int((right - left) / pixel_size)
            new_height = int((top - bottom) / pixel_size)

            if new_width <= 0 or new_height <= 0:
                raise ValueError("Dimensiones inválidas calculadas.")

            new_transform = Affine(
                pixel_size, 0, left,
                0, -pixel_size, top
            )

            dst = np.empty((new_height, new_width), dtype="float32")

            resample_method = Resampling.bilinear if self.method == "bilinear" else Resampling.nearest

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=resample_method
            )

        self.data = dst
        self.meta.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "dtype": "float32"
        })

        return self.data

    def save(self, output_path):
        if self.data is None:
            raise ValueError("Debe ejecutar resample() antes.")

        with rasterio.open(output_path, "w", **self.meta) as dst:
            dst.write(self.data, 1)

        return output_path
