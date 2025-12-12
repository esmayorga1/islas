import numpy as np
import rasterio


class CoverageClassifier3m:
    """
    Clasifica coberturas (agua, vegetación, urbano, otros)
    a resolución 3 metros, combinando capas binarias a 1 metro.
    
    Códigos de salida:
        1 = Agua
        2 = Vegetación
        3 = Urbano (construcciones + vías)
        0 = Otro / sin datos
    """

    def __init__(self, raster_agua_1m, raster_vegetacion_1m, raster_urbano_1m, factor=3):
        """
        raster_agua_1m: raster 0/1 a 1 metro (agua)
        raster_vegetacion_1m: raster 0/1 a 1 metro (vegetación)
        raster_urbano_1m: raster 0/1 a 1 metro (urbano = vías + construcciones)
        factor: escala de remuestreo (3 -> 3x3 píxeles)
        """
        self.raster_agua_1m = raster_agua_1m
        self.raster_vegetacion_1m = raster_vegetacion_1m
        self.raster_urbano_1m = raster_urbano_1m
        self.factor = factor

        self.meta = None
        self.cover_1m = None
        self.cover_3m = None

    def load_rasters(self):
        """Carga las capas binarias a 1 metro."""

        with rasterio.open(self.raster_agua_1m) as src_a:
            agua = src_a.read(1).astype(np.uint8)
            self.meta = src_a.meta.copy()

        with rasterio.open(self.raster_vegetacion_1m) as src_v:
            vegetacion = src_v.read(1).astype(np.uint8)

        with rasterio.open(self.raster_urbano_1m) as src_u:
            urbano = src_u.read(1).astype(np.uint8)

        # Clasificación por prioridad:
        # Agua > Vegetación > Urbano > Otro
        cover = np.zeros_like(agua)

        cover[agua == 1] = 1           # Agua
        cover[(agua == 0) & (vegetacion == 1)] = 2  # Vegetación
        cover[(agua == 0) & (vegetacion == 0) & (urbano == 1)] = 3  # Urbano

        self.cover_1m = cover

    def aggregate_to_3m(self):
        """Agrupa bloques 3x3 aplicando mayoría."""

        if self.cover_1m is None:
            self.load_rasters()

        arr = self.cover_1m
        f = self.factor

        # Ajustar a múltiplos de 3
        rows = arr.shape[0] - (arr.shape[0] % f)
        cols = arr.shape[1] - (arr.shape[1] % f)
        arr = arr[:rows, :cols]

        # Remuestreo en bloques
        arr_blocks = arr.reshape(rows // f, f, cols // f, f)
        cover_3m = np.zeros((rows // f, cols // f), dtype=np.uint8)

        # Mayoría por bloque (algoritmo exacto)
        for i in range(cover_3m.shape[0]):
            for j in range(cover_3m.shape[1]):
                bloque = arr_blocks[i, :, j, :].flatten()
                valores, cuentas = np.unique(bloque, return_counts=True)
                cover_3m[i, j] = valores[np.argmax(cuentas)]

        self.cover_3m = cover_3m
        return cover_3m

    def save_raster(self, output_path):
        """Guarda el raster final de clasificación a 3 metros."""

        if self.cover_3m is None:
            self.aggregate_to_3m()

        out_meta = self.meta.copy()
        out_meta.update({
            "height": self.cover_3m.shape[0],
            "width": self.cover_3m.shape[1],
            "transform": rasterio.Affine(
                self.meta["transform"].a * self.factor, 0, self.meta["transform"].c,
                0, self.meta["transform"].e * self.factor, self.meta["transform"].f
            ),
            "dtype": "uint8",
            "count": 1
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(self.cover_3m, 1)

        return output_path
