import rasterio
import numpy as np


class SlopeGenerator:
    """
    Genera un mapa de pendientes (slope) a partir de un DEM.
    La pendiente se calcula SIEMPRE en grados (°).
    """

    def __init__(self, dem_path):
        """
        dem_path: ruta al DEM de entrada
        """
        self.dem_path = dem_path
        self.dem = None
        self.meta = None
        self.slope = None
        self.res_x = None
        self.res_y = None

    def load_dem(self):
        """Carga el DEM y su metadata."""
        with rasterio.open(self.dem_path) as src:
            self.dem = src.read(1).astype(float)
            self.meta = src.meta.copy()
            self.res_x = src.transform[0]
            self.res_y = -src.transform[4]  # valor absoluto de la resolución Y

    def compute_slope(self):
        """Calcula la pendiente en grados (°)."""
        if self.dem is None:
            self.load_dem()

        # Calcular gradientes en Y y X (dZ/dy, dZ/dx)
        dz_dy, dz_dx = np.gradient(self.dem, self.res_y, self.res_x)

        # Magnitud del gradiente
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

        # Convertir a grados
        self.slope = np.degrees(slope_rad)

        return self.slope

    def save_slope(self, output_path):
        """Guarda el raster de pendiente en grados."""
        if self.slope is None:
            self.compute_slope()

        self.meta.update({
            "dtype": "float32",
            "count": 1
        })

        with rasterio.open(output_path, "w", **self.meta) as dst:
            dst.write(self.slope.astype(np.float32), 1)

        return output_path
