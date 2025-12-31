import os
from pathlib import Path
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

    DEG_TO_M = 111_000.0  # aproximación válida para Bogotá (EPSG:4326)

    def __init__(
        self,
        year: int,
        hr_monthly_dict,
        ndvi_dir,
        ndbi_dir,
        tair_dir,
        dist_water_path,
        urban_path,
        dem_path,
        slope_path,
        output_dir,
        ndvi_tag="NDVI",
        ndbi_tag="NDBI",
        tair_tag="TAIRE_SINTETICA_3M",
    ):
        self.year = int(year)

        self.hr_monthly = hr_monthly_dict
        self.ndvi_dir = str(ndvi_dir)
        self.ndbi_dir = str(ndbi_dir)
        self.tair_dir = str(tair_dir)

        self.ndvi_tag = str(ndvi_tag)
        self.ndbi_tag = str(ndbi_tag)
        self.tair_tag = str(tair_tag)

        self.dist_water = self._load_raster(str(dist_water_path))
        self.urban = self._load_raster(str(urban_path))
        self.dem = self._load_raster(str(dem_path))
        self.slope = self._load_raster(str(slope_path))

        self.meta_base = self._read_meta(str(dem_path))

        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Utilidades
    # -----------------------------------------------------
    def _load_raster(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster requerido no encontrado: {path}")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def _read_meta(self, path):
        with rasterio.open(path) as src:
            return src.meta.copy()

    def _align_to_3m(self, raster_path):
        """Reescala raster a grilla 3 m (DEM)."""
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"No existe raster mensual: {raster_path}")
        with rasterio.open(raster_path) as src:
            data = src.read(
                1,
                out_shape=(self.meta_base["height"], self.meta_base["width"]),
                resampling=Resampling.nearest,
            )
        return data.astype(np.float32)

    # -----------------------------------------------------
    # Buscar archivo mensual por patrón (robusto)
    # -----------------------------------------------------
    def _find_month_file(self, directory: str, tag: str, month: int) -> str:
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {d}")

        y = str(self.year)
        m1 = str(int(month))
        m2 = f"{int(month):02d}"

        patterns = [
            f"*{y}*_{m1}_*{tag}*.tif",
            f"*{y}*_{m2}_*{tag}*.tif",
            f"*_{y}_*_{m1}_*{tag}*.tif",
            f"*_{y}_*_{m2}_*{tag}*.tif",
            f"*{tag}*{y}*{m1}*.tif",
            f"*{tag}*{y}*{m2}*.tif",
        ]

        candidates = []
        for p in patterns:
            candidates.extend(d.glob(p))

        tag_upper = tag.upper()
        filtered = [c for c in candidates if tag_upper in c.name.upper()]

        seen = set()
        uniq = []
        for c in filtered:
            if str(c) not in seen:
                uniq.append(c)
                seen.add(str(c))

        if not uniq:
            sample = list(d.glob("*.tif"))[:10]
            raise FileNotFoundError(
                f"No encontré archivo {tag} para {self.year}-{month:02d} en: {d}\n"
                f"Patrones probados: {patterns}\n"
                f"Ejemplos de .tif (primeros 10): {[s.name for s in sample]}"
            )

        return str(uniq[0])

    # -----------------------------------------------------
    # Obtener HR_base robusto desde el diccionario
    # -----------------------------------------------------
    def _get_hr_base(self, month: int) -> float:
        y = self.year
        mm2 = f"{month:02d}"
        mm1 = str(int(month))

        keys_to_try = [
            f"{y}-{mm2}",
            f"{y}-{mm1}",
            f"{y}{mm2}",
            f"{y}{mm1}",
            (y, int(month)),
            (str(y), mm2),
            (str(y), int(month)),
        ]

        for k in keys_to_try:
            if k in self.hr_monthly:
                return float(self.hr_monthly[k])

        suffixes = {f"{y}{mm2}", f"{y}{mm1}"}
        for k in self.hr_monthly.keys():
            ks = str(k)
            if ks in suffixes or ks.endswith(f"{y}{mm2}") or ks.endswith(f"{y}{mm1}"):
                try:
                    return float(self.hr_monthly[k])
                except Exception:
                    pass

        raise KeyError(
            f"No encontré HR_base para {y}-{mm2} en hr_monthly_dict.\n"
            f"Claves de ejemplo (primeras 20): {list(self.hr_monthly.keys())[:20]}"
        )

    # -----------------------------------------------------
    # Modelo sintético
    # -----------------------------------------------------
    def _compute_hr(self, month, ndvi, ndbi, tair):
        HR_base = self._get_hr_base(month)

        dem_norm = (self.dem - np.nanmin(self.dem)) / (
            np.nanmax(self.dem) - np.nanmin(self.dem) + 1e-6
        )
        slope_norm = np.clip(self.slope / 30.0, 0, 1)

        tair_mean = np.nanmean(tair)

        dist_water_m = self.dist_water * self.DEG_TO_M
        water_effect = 10.0 * np.exp(-dist_water_m / 500.0)

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
    def process_month(self, month: int):
        ndvi_path = self._find_month_file(self.ndvi_dir, self.ndvi_tag, month)
        ndbi_path = self._find_month_file(self.ndbi_dir, self.ndbi_tag, month)

        # TAIR puede venir con nombre exacto o con patrón
        # 1) intentar con el nombre típico del pipeline
        tair_guess_1 = os.path.join(self.tair_dir, f"{self.tair_tag}_{self.year}_{month:02d}.tif")
        tair_guess_2 = os.path.join(self.tair_dir, f"{self.tair_tag}_{self.year}_{int(month)}.tif")

        if os.path.exists(tair_guess_1):
            tair_path = tair_guess_1
        elif os.path.exists(tair_guess_2):
            tair_path = tair_guess_2
        else:
            tair_path = self._find_month_file(self.tair_dir, self.tair_tag, month)

        ndvi = self._align_to_3m(ndvi_path)
        ndbi = self._align_to_3m(ndbi_path)
        tair = self._load_raster(tair_path)

        hr = self._compute_hr(month, ndvi, ndbi, tair)

        out_path = os.path.join(self.output_dir, f"HR_SINTETICA_3M_{self.year}_{month:02d}.tif")

        meta = self.meta_base.copy()
        meta.update(dtype="float32", count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(hr.astype(np.float32), 1)

        return out_path

    # -----------------------------------------------------
    # Procesar todo el año
    # -----------------------------------------------------
    def process_all(self, months=range(1, 13)):
        outputs = []
        for month in months:
            out = self.process_month(int(month))
            outputs.append(out)
            print(f"✓ HR sintética {self.year}-{int(month):02d} → {out}")
        return outputs
