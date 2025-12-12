import os
import rasterio
import numpy as np
from rasterio.enums import Resampling


class SyntheticAirTemperature3m:
    """
    Genera Temperatura del Aire Sintética (3 m) combinando:

    - LST mensual (índice 0–1, 10 m → 3 m)
    - NDVI mensual (0–1, 10 m → 3 m)
    - Tipo de cobertura (3 m)
    - Porcentaje de construcción (3 m, 0–1)
    - Porcentaje de vegetación (3 m, 0–1)
    - Distancia a vías (3 m, EN GRADOS) → convertido a METROS (Bogotá)
    - Distancia a cuerpos de agua (3 m, EN GRADOS) → convertido a METROS
    - DEM relativo (3 m, 0–15 m aprox)
    - Pendiente (3 m, grados)
    - Temperatura base mensual (CSV, °C)
    """

    # Conversión fija aproximada válida para Bogotá (lat ~4.6°)
    DEG_TO_M = 111_000.0

    def __init__(
        self,
        lst_dir,
        ndvi_dir,
        coverage_path,
        urban_path,
        vegetation_path,
        dist_vias_path,
        dist_agua_path,
        dem_path,
        slope_path,
        temp_monthly_dict,
        output_dir,
    ):

        self.lst_dir = lst_dir
        self.ndvi_dir = ndvi_dir

        self.coverage_path = coverage_path
        self.urban_path = urban_path
        self.vegetation_path = vegetation_path
        self.dist_vias_path = dist_vias_path
        self.dist_agua_path = dist_agua_path
        self.dem_path = dem_path
        self.slope_path = slope_path

        self.temp_monthly = temp_monthly_dict
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Cargar rasters 3 m (todos ya alineados al DEM)
        self.coverage = self._load_raster(coverage_path)      # clases 1,2,3,...
        self.urban = self._load_raster(urban_path)            # 0–1
        self.vegetation = self._load_raster(vegetation_path)  # 0–1
        self.dist_vias = self._load_raster(dist_vias_path)    # en grados
        self.dist_agua = self._load_raster(dist_agua_path)    # en grados
        self.dem = self._load_raster(dem_path)                # 0–15 m (relativo)
        self.slope = self._load_raster(slope_path)            # grados

        # Metadata base = DEM
        self.meta_base = self._read_meta(dem_path)

    # ----------------------------------------------------
    # Utilidades de carga
    # ----------------------------------------------------
    def _load_raster(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def _read_meta(self, path):
        with rasterio.open(path) as src:
            return src.meta.copy()

    # ----------------------------------------------------
    # Reescalar LST y NDVI (10 m → 3 m)
    # ----------------------------------------------------
    def _align_to_3m(self, raster_path):
        """
        Reescala un raster (ej. LST/NDVI 10 m) a la grilla del DEM 3 m,
        usando nearest neighbour.
        """
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

    # ----------------------------------------------------
    # Ecuación formal ajustada para índices 0–1
    # ----------------------------------------------------
    def _compute_synthetic_temperature(self, month, lst_idx, ndvi_idx):
        """
        Calcula la temperatura del aire sintética (°C) para un mes dado.

        - lst_idx y ndvi_idx están normalizados 0–1.
        - Se usan como moduladores de calentamiento/enfriamiento alrededor de Tbase.
        """

        # 1) Temperatura base mensual (CSV, constante por UPL)
        Tbase = self.temp_monthly[f"2021-{month:02d}"]  # °C

        # 2) Convertir distancias de grados → metros (Bogotá)
        dist_vias_m = self.dist_vias * self.DEG_TO_M
        dist_agua_m = self.dist_agua * self.DEG_TO_M

        # 3) Normalizar DEM relativo a 0–1 (0–15 m aprox)
        dem_min = np.nanmin(self.dem)
        dem_max = np.nanmax(self.dem)
        dem_norm = (self.dem - dem_min) / (dem_max - dem_min + 1e-6)

        # 4) Efecto relativo de LST (0–1) y NDVI (0–1)
        #    LST alto → más calor (máx ±5 °C alrededor de Tbase)
        #    NDVI alto → más enfriamiento (máx ±4 °C)
        lst_effect = (lst_idx - 0.5) * 10.0      # -5 a +5 °C
        ndvi_effect = (0.5 - ndvi_idx) * 8.0     # verde alto: -4°C, suelo desnudo: +4°C

        # 5) Factores por cobertura
        # 1 = urbano, 2 = vegetación, 3 = agua, otros = 0
        coverage_raw = np.where(
            self.coverage == 1, 1.0,      # urbano
            np.where(
                self.coverage == 2, -1.0, # vegetación
                np.where(self.coverage == 3, -1.5, 0.0)  # agua / otros
            )
        )
        coverage_effect = coverage_raw * 1.0  # máx ±1.5 °C aprox

        # 6) Densidades: urbano / vegetación (0–1)
        urban_effect = 3.0 * self.urban          # máx +3 °C
        vegetation_effect = -2.0 * self.vegetation  # máx -2 °C

        # 7) Distancias en metros (ya razonables: ~0–2000 m)
        vias_effect = 0.0015 * dist_vias_m       # máx ~+3 °C
        agua_effect = -0.0008 * dist_agua_m      # máx ~-1.5 °C

        # 8) DEM relativo: máximo 1 °C de diferencia por relieve local
        dem_effect = -1.0 * dem_norm             # máx -1 °C en zonas más altas relativas

        # 9) Pendiente: suavizado
        #    Asumimos pendiente 0–30° típico, máx +1.5 °C en las más soleadas
        slope_effect = 0.05 * self.slope         # 0–4.5 si 0–90°, en la práctica menos

        # 10) Suma total
        Tsint = (
            Tbase
            + lst_effect
            + ndvi_effect
            + urban_effect
            + vegetation_effect
            + vias_effect
            + agua_effect
            + dem_effect
            + slope_effect
            + coverage_effect
        )

        # 11) Seguridad: limitar a un rango razonable para Bogotá
        Tsint = np.clip(Tsint, 5, 35)

        return Tsint

    # ----------------------------------------------------
    # Procesar un mes
    # ----------------------------------------------------
    def process_month(self, month):
        """
        Genera un raster sintético 3 m de temperatura del aire (°C) para un mes.
        """

        lst_path = os.path.join(self.lst_dir,  f"S2_2021_{month}_PatioBonito_LST.tif")
        ndvi_path = os.path.join(self.ndvi_dir, f"S2_2021_{month}_PatioBonito_NDVI.tif")

        # LST y NDVI en 0–1, reescalados a grilla 3 m
        lst_idx = self._align_to_3m(lst_path)
        ndvi_idx = self._align_to_3m(ndvi_path)

        result = self._compute_synthetic_temperature(month, lst_idx, ndvi_idx)

        out_path = os.path.join(
            self.output_dir,
            f"TAIRE_SINTETICA_3M_2021_{month:02d}.tif"
        )

        meta = self.meta_base.copy()
        meta.update(dtype="float32", count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(result.astype(np.float32), 1)

        return out_path

    # ----------------------------------------------------
    # Procesar los 12 meses
    # ----------------------------------------------------
    def process_all(self):
        outputs = []
        for month in range(1, 13):
            out = self.process_month(month)
            outputs.append(out)
            print(f"✓ Mes {month:02d} generado → {out}")
        return outputs
