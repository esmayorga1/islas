from .climatica import Climatica
import numpy as np
import rasterio
import geopandas as gpd
from noise import pnoise2              # pip install noise
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from pathlib import Path


class TemperaturaAireSintetica(Climatica):
    """Genera versiones sintéticas de temperatura del aire a partir de un raster
        de baja resolución. Remuestrea la imagen a resolución fina,
    aplica un campo térmico suave tipo Perlin para crear variación espacial
    realista, elimina valores NoData y garantiza continuidad dentro del AOI."""

    def __init__(self, input_dir, output_dir, aoi=None,
                 pixel_size=0.000027,   # ≈ 3 m en grados
                 intensidad=0.4,        # controla cuánto “se deforma” respecto al original
                 escala=0.001):         # frecuencia del Perlin (suavidad)
        super().__init__(input_dir, output_dir)
        self.pixel_size = pixel_size
        self.intensidad = intensidad
        self.escala = escala

        self.aoi = None
        if aoi is not None:
            if isinstance(aoi, Path):
                aoi = str(aoi)
            gdf = gpd.read_file(aoi)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)
            self.aoi = gdf

    def _perlin(self, h, w):
        """
        Genera un campo Perlin suave en [-1, 1] aproximadamente.
        """
        campo = np.zeros((h, w), dtype=np.float32) # Matriz vacia
        for i in range(h):
            for j in range(w):
                campo[i, j] = pnoise2(i * self.escala, j * self.escala, octaves=4) #Rellena cada pixel con ruido Perlin
        # normalizar a media 0, std 1
        campo = campo - np.nanmean(campo)
        std = np.nanstd(campo)
        if std == 0:
            std = 1.0
        campo = campo / std
        return campo  # media 0, std 1

    def calculate(self):
        files = sorted(self.input_dir.glob("TempAire_*.tif"))
        if not files:
            print("❌ No rasters encontrados")
            return

        shapes = [geom for geom in self.aoi.geometry] if self.aoi is not None else None

        for f in files:
            print(f"Procesando {f.name}...")

            # 1) Leer raster original
            with rasterio.open(f) as src:
                base = src.read(1).astype(np.float32)
                nodata = src.nodata
                transform = src.transform
                crs = src.crs
                H = src.height
                W = src.width

            # 2) Calcular límites y nueva grilla (3 m aprox.)
            xmin = transform[2]
            ymax = transform[5]
            xmax = xmin + W * transform[0]
            ymin = ymax + H * transform[4]

            new_width = int((xmax - xmin) / self.pixel_size)
            new_height = int((ymax - ymin) / self.pixel_size)

            new_transform = from_bounds(xmin, ymin, xmax, ymax, new_width, new_height)

            # 3) Remuestrear a resolución fina
            up = np.empty((new_height, new_width), dtype=np.float32) # Array vacio
            
            reproject(
                source=base,
                destination=up,
                src_transform=transform,
                src_crs=crs,
                dst_transform=new_transform,
                dst_crs=crs,
                resampling=Resampling.cubic
            ) # Del pixel original lo distrbuye en los nievos pixeles en caso de 3 x 3 

            # 4) Estadísticas del raster remuestreado
            valid = up[~np.isnan(up)]
            if valid.size == 0:
                print("⚠ Raster sin datos válidos, se genera completamente sintético.")
                mean_val = 18.0
                std_val = 2.0
                up[:] = mean_val
            else:
                mean_val = float(np.nanmean(valid))
                std_val = float(np.nanstd(valid))
                if std_val == 0:
                    std_val = max(0.5, abs(mean_val) * 0.05)

            # 5) Campo Perlin suave (media 0, std 1) → escala climática
            perlin = self._perlin(new_height, new_width)

            # Campo en unidades de temperatura (°C)
            # amplitud inicial ~ std_val
            campo = perlin * std_val

            # 6) Combinar original + campo sintético
            #    intensidad controla cuánto nos alejamos del raster original.
            sint = up.copy()
            # donde up tenga NaN, poner la media antes de combinar
            sint[np.isnan(sint)] = mean_val

            # deformación
            sint = sint + campo * self.intensidad

            # 7) REESCALAR DESVIACIÓN ESTÁNDAR → AUMENTAR RANGO
            #    Queremos que la std final sea mayor que la original.
            sint_valid = sint[~np.isnan(sint)]
            if sint_valid.size > 0:
                std_sint = float(np.nanstd(sint_valid))
                # factor deseado: entre 1x y 4x la std original (según intensidad)
                factor_std = 1.0 + 3.0 * self.intensidad
                desired_std = std_val * factor_std
                if std_sint > 0:
                    scale = desired_std / std_sint
                    sint = mean_val + (sint - mean_val) * scale

            # 8) Reparar cualquier NaN residual con vecino más cercano
            nan_mask = np.isnan(sint)
            if nan_mask.any():
                dist, inds = distance_transform_edt(
                    nan_mask, return_indices=True
                )
                sint[nan_mask] = sint[tuple(inds[:, nan_mask])]
            sint[np.isnan(sint)] = mean_val

            # 9) Aplicar AOI sin introducir NoData
            if shapes is not None:
                from rasterio.features import geometry_mask
                mask_aoi = geometry_mask(
                    shapes,
                    transform=new_transform,
                    invert=True,
                    out_shape=(new_height, new_width)
                )
                sint[~mask_aoi] = mean_val

            # 10) Guardar raster sintético
            new_profile = {
                "driver": "GTiff",
                "height": new_height,
                "width": new_width,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": new_transform,
                "nodata": None
            }

            output_name = f"{f.stem}_SinteticoPerlin3m"
            self.save_raster(sint.astype(np.float32), new_profile, "MODIS", output_name)

            print(f"✔ Guardado {output_name}.tif  (media≈{mean_val:.2f}, std_orig≈{std_val:.2f})")
