import os
import rasterio
import numpy as np
from rasterio.enums import Resampling


class SyntheticRelativeHumidity3m:
    """
    Genera Humedad Relativa Sintética (HR, %) a 3 m
    mediante una sintetización formal basada en variables definidas.

    Variables utilizadas:
    - Humedad relativa base mensual (CSV)
    - NDVI mensual (10 m → 3 m)
    - NDBI mensual (10 m → 3 m)
    - Temperatura del aire sintética mensual (3 m)
    - Distancia a cuerpos de agua (3 m)
    - Porcentaje de construcción (1 m → 3 m)
    - DEM relativo (3 m)
    - Pendiente (3 m)
    """

    DEG_TO_M = 111_000.0  # Conversión válida para Bogotá

    def __init__(
        self,
        hr_monthly_dict,
        ndvi_dir,
        ndbi_dir,
        tair_dir,
        dist_water_path,
        urban_path,
        dem_path,
        slope_path,
        output_dir,
    ):

        self.hr_monthly = hr_monthly_dict
        self.ndvi_dir = ndvi_dir
        self.ndbi_dir = ndbi_dir
        self.tair_dir = tair_dir

        self.dist_water = self._load_raster(dist_water_path)
        self.urban = self._load_raster(urban_path)
        self.dem = self._load_raster(dem_path)
        self.slope = self._load_raster(slope_path)

        self.meta_base = self._read_meta(dem_path)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Utilidades
    # -----------------------------------------------------
    def _load_raster(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def _read_meta(self, path):
        with rasterio.open(path) as src:
            return src.meta.copy()

    def _align_to_3m(self, raster_path):
        """Reescala raster a grilla 3 m (DEM)."""
        with rasterio.open(raster_path) as src:
            data = src.read(
                1,
                out_shape=(
                    self.meta_base["height"],
                    self.meta_base["width"],
                ),
                resampling=Resampling.nearest,
            )
        return data.astype(np.float32)

    # -----------------------------------------------------
    # Modelo sintético
    # -----------------------------------------------------
    def _compute_hr(self, month, ndvi, ndbi, tair):

        HR_base = self.hr_monthly[f"2021-{month:02d}"]

        # Normalizaciones
        dem_norm = (self.dem - np.nanmin(self.dem)) / (
            np.nanmax(self.dem) - np.nanmin(self.dem) + 1e-6
        )
        slope_norm = np.clip(self.slope / 30.0, 0, 1)

        # Promedio térmico mensual
        tair_mean = np.nanmean(tair)

        # Distancia a agua en metros
        dist_water_m = self.dist_water * self.DEG_TO_M
        water_effect = 10.0 * np.exp(-dist_water_m / 500.0)

        # Efectos
        temp_effect = -1.5 * (tair - tair_mean)
        ndvi_effect = 8.0 * ndvi
        ndbi_effect = -6.0 * ndbi
        urban_effect = -6.0 * self.urban
        dem_effect = -3.0 * dem_norm
        slope_effect = 2.0 * slope_norm

        HR = (
            HR_base
            + temp_effect
            + ndvi_effect
            + ndbi_effect
            + water_effect
            + urban_effect
            + dem_effect
            + slope_effect
        )

        return np.clip(HR, 30, 100)

    # -----------------------------------------------------
    # Procesar un mes
    # -----------------------------------------------------
    def process_month(self, month):

        ndvi_path = os.path.join(
            self.ndvi_dir, f"S2_2021_{month}_PatioBonito_NDVI.tif"
        )
        ndbi_path = os.path.join(
            self.ndbi_dir, f"S2_2021_{month}_PatioBonito_NDBI.tif"
        )
        tair_path = os.path.join(
            self.tair_dir, f"TAIRE_SINTETICA_3M_2021_{month:02d}.tif"
        )

        ndvi = self._align_to_3m(ndvi_path)
        ndbi = self._align_to_3m(ndbi_path)
        tair = self._load_raster(tair_path)

        hr = self._compute_hr(month, ndvi, ndbi, tair)

        out_path = os.path.join(
            self.output_dir,
            f"HR_SINTETICA_3M_2021_{month:02d}.tif"
        )

        meta = self.meta_base.copy()
        meta.update(dtype="float32", count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(hr.astype(np.float32), 1)

        return out_path

    # -----------------------------------------------------
    # Procesar todo el año
    # -----------------------------------------------------
    def process_all(self):
        outputs = []
        for month in range(1, 13):
            out = self.process_month(month)
            outputs.append(out)
            print(f"✓ HR sintética mes {month:02d} generada → {out}")
        return outputs
