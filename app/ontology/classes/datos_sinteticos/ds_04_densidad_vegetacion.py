import numpy as np
import rasterio

class VegetationDensity3m:
    """
    Genera un raster del porcentaje de vegetación (cobertura verde)
    a resolución 3 metros, a partir de un raster binario a 1 metro (0/1).
    """

    def __init__(self, raster_vegetacion_1m, factor=3):
        """
        raster_vegetacion_1m: ruta al raster binario de vegetación (1 m, 0/1)
        factor: tamaño del bloque de agregación (3 -> 3x3 píxeles)
        """
        self.raster_vegetacion_1m = raster_vegetacion_1m
        self.factor = factor
        self.meta = None
        self.veg_1m = None
        self.result = None

    def load_raster(self):
        """Carga el raster binario de vegetación a 1 metro."""

        with rasterio.open(self.raster_vegetacion_1m) as src:
            self.veg_1m = src.read(1).astype(np.float32)
            self.meta = src.meta.copy()

    def compute_density(self):
        """Convierte bloques 3x3 en porcentaje de vegetación."""

        if self.veg_1m is None:
            self.load_raster()

        arr = self.veg_1m
        f = self.factor

        # Ajustar tamaño para que sea divisible por 3
        rows = arr.shape[0] - (arr.shape[0] % f)
        cols = arr.shape[1] - (arr.shape[1] % f)
        arr = arr[:rows, :cols]

        # Reorganizar en bloques de 3x3
        arr_blocks = arr.reshape(rows // f, f, cols // f, f)

        # Promedio dentro de cada bloque -> porcentaje (0..1)
        density = arr_blocks.mean(axis=(1, 3))

        self.result = density
        return density

    def save_raster(self, output_path):
        """Guarda el raster 3 m con porcentaje de vegetación."""
        if self.result is None:
            self.compute_density()

        out_meta = self.meta.copy()
        out_meta.update({
            "height": self.result.shape[0],
            "width": self.result.shape[1],
            "transform": rasterio.Affine(
                self.meta["transform"].a * self.factor, 0, self.meta["transform"].c,
                0, self.meta["transform"].e * self.factor, self.meta["transform"].f
            ),
            "dtype": "float32"
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(self.result.astype(np.float32), 1)

        return output_path
