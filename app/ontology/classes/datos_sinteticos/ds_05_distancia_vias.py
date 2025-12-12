import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


class DistanceToRoads:
    """
    Genera un raster de distancia euclidiana a vías principales (en metros)
    a partir de un raster binario 0/1 (1 = vía).
    """

    def __init__(self, roads_raster_1m_path, output_resolution=1):
        """
        roads_raster_1m_path: raster binario 1 m (1 = vía)
        output_resolution: resolución del raster de salida (1 m o 3 m)
        """
        self.roads_raster_1m_path = roads_raster_1m_path
        self.output_resolution = output_resolution  # 1 = full, 3 = coarse
        self.meta = None
        self.distances = None

    def load_raster(self):
        """Carga el raster binario de vías (1 = vía, 0 = no vía)."""
        with rasterio.open(self.roads_raster_1m_path) as src:
            roads = src.read(1).astype(np.float32)
            self.meta = src.meta.copy()

        # roads = 1 -> queremos distancia a esos píxeles → invertimos para EDT
        mask = (roads == 0).astype(np.uint8)

        return mask

    def compute_distance(self):
        """Calcula la distancia euclidiana a la vía más cercana (en metros)."""

        mask = self.load_raster()

        # Transformación del DEM: resolución X y Y
        res_x = abs(self.meta["transform"][0])
        res_y = abs(self.meta["transform"][4])

        # Distancia euclidiana en píxeles reales
        dist_px = distance_transform_edt(mask, sampling=(res_y, res_x))

        # Guardar el resultado
        self.distances = dist_px.astype(np.float32)

        # Si la salida es 3 m → remuestrear manualmente
        if self.output_resolution == 3:
            self.distances = self.distances[::3, ::3]

        return self.distances

    def save_raster(self, output_path):
        """Guarda el raster de distancias (en metros)."""

        if self.distances is None:
            self.compute_distance()

        meta_out = self.meta.copy()

        # Cambiar resolución si es 3 m
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
