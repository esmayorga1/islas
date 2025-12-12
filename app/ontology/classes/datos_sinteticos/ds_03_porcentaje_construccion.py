import numpy as np
import rasterio


class UrbanDensity3m:
    """
    Genera un raster de porcentaje urbano (construcciones + vías)
    a resolución 3 metros, a partir de dos ráster binarios a 1 metro (0/1).
    """

    def __init__(self, raster_construcciones_1m, raster_vias_1m, factor=3):
        """
        raster_construcciones_1m: ruta al raster binario de construcciones (1 m, 0/1)
        raster_vias_1m: ruta al raster binario de vías (1 m, 0/1)
        factor: tamaño del bloque de agregación (3 -> 3x3 píxeles)
        """
        self.raster_construcciones_1m = raster_construcciones_1m
        self.raster_vias_1m = raster_vias_1m
        self.factor = factor
        self.meta = None
        self.urbano_1m = None
        self.result = None

    def load_rasters(self):
        """Carga ambos ráster binarios (construcciones y vías)."""

        # Construcciones
        with rasterio.open(self.raster_construcciones_1m) as src_c:
            construcciones = src_c.read(1).astype(np.float32)
            self.meta = src_c.meta.copy()

        # Vías (se asume misma resolución, tamaño y transform)
        with rasterio.open(self.raster_vias_1m) as src_v:
            vias = src_v.read(1).astype(np.float32)

        # Combinar: 1 si hay construcción O avenida
        self.urbano_1m = np.where((construcciones == 1) | (vias == 1), 1.0, 0.0)

    def compute_density(self):
        """Convierte bloques 3x3 en porcentaje urbano."""

        if self.urbano_1m is None:
            self.load_rasters()

        arr = self.urbano_1m
        f = self.factor

        # Ajustar tamaño para que sea divisible por 3
        rows = arr.shape[0] - (arr.shape[0] % f)
        cols = arr.shape[1] - (arr.shape[1] % f)
        arr = arr[:rows, :cols]

        # Reorganizar en bloques de 3x3
        arr_blocks = arr.reshape(rows // f, f, cols // f, f)

        # Promedio dentro de cada bloque (0..1)
        density = arr_blocks.mean(axis=(1, 3))

        self.result = density
        return density

    def save_raster(self, output_path):
        """Guarda el raster 3 m con porcentaje urbano."""
        if self.result is None:
            self.compute_density()

        # Actualizar metadatos para la nueva resolución
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
