import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


class DistanceToWater:
    """
    Genera un raster de distancia euclidiana a cuerpos de agua (en metros),
    a partir de un raster binario 0/1 (1 = agua).
    """

    def __init__(self, water_raster_1m_path, output_resolution=1):
        """
        water_raster_1m_path: raster binario 1 m (1 = agua)
        output_resolution: resolución del raster de salida (1 o 3 metros)
        """
        self.water_raster_1m_path = water_raster_1m_path
        self.output_resolution = output_resolution
        self.meta = None
        self.distances = None

    def load_raster(self):
        """Carga el raster binario de agua (1 = agua, 0 = no agua)."""

        with rasterio.open(self.water_raster_1m_path) as src:
            water = src.read(1).astype(np.float32)
            self.meta = src.meta.copy()

        # distance_transform_edt calcula distancia a píxeles == 0,
        # así que invertimos la máscara: queremos distancia hacia el agua
        mask = (water == 0).astype(np.uint8)

        return mask

    def compute_distance(self):
        """Calcula la distancia euclidiana a cuerpos de agua (en metros)."""

        mask = self.load_raster()

        # Obtener la resolución espacial (X, Y) en metros reales
        res_x = abs(self.meta["transform"][0])
        res_y = abs(self.meta["transform"][4])

        # Distancia euclidiana
        dist_px = distance_transform_edt(mask, sampling=(res_y, res_x))

        self.distances = dist_px.astype(np.float32)

        # Si el usuario quiere resolución 3 m → submuestrear
        if self.output_resolution == 3:
            self.distances = self.distances[::3, ::3]

        return self.distances

    def save_raster(self, output_path):
        """Guarda el raster de distancia en metros."""

        if self.distances is None:
            self.compute_distance()

        meta_out = self.meta.copy()

        # Ajustar metadatos en caso de salida 3 metros
        if self.output_resolution == 3:
            meta_out.update({
                "height": self.distances.shape[0],
                "width": self.distances.shape[1],
                "transform": rasterio.Affine(
                    self.meta["transform"].a * 3, 0, self.meta["transform"].c,
                    0, self.meta["transform"].e * 3, self.meta["transform"].f
                )
            })

        meta_out.update({"dtype": "float32"})

        with rasterio.open(output_path, "w", **meta_out) as dst:
            dst.write(self.distances.astype(np.float32), 1)

        return output_path
