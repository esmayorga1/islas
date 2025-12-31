import os
from pathlib import Path
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

    DEG_TO_M = 111_000.0  # aproximación válida para Bogotá (EPSG:4326)

    def __init__(
        self,
        year: int,
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
        lst_tag="LST",
        ndvi_tag="NDVI",
    ):
        self.year = int(year)

        self.lst_dir = str(lst_dir)
        self.ndvi_dir = str(ndvi_dir)

        self.coverage_path = str(coverage_path)
        self.urban_path = str(urban_path)
        self.vegetation_path = str(vegetation_path)
        self.dist_vias_path = str(dist_vias_path)
        self.dist_agua_path = str(dist_agua_path)
        self.dem_path = str(dem_path)
        self.slope_path = str(slope_path)

        self.temp_monthly = temp_monthly_dict
        self.output_dir = str(output_dir)

        self.lst_tag = str(lst_tag)
        self.ndvi_tag = str(ndvi_tag)

        os.makedirs(self.output_dir, exist_ok=True)

        # Cargar rasters 3 m (todos ya alineados al DEM)
        self.coverage = self._load_raster(self.coverage_path)      # clases 1,2,3,...
        self.urban = self._load_raster(self.urban_path)            # 0–1
        self.vegetation = self._load_raster(self.vegetation_path)  # 0–1
        self.dist_vias = self._load_raster(self.dist_vias_path)    # en grados
        self.dist_agua = self._load_raster(self.dist_agua_path)    # en grados
        self.dem = self._load_raster(self.dem_path)                # 0–15 m (relativo)
        self.slope = self._load_raster(self.slope_path)            # grados

        # Metadata base = DEM
        self.meta_base = self._read_meta(self.dem_path)

    # ----------------------------------------------------
    # Utilidades de carga
    # ----------------------------------------------------
    def _load_raster(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster requerido no encontrado: {path}")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    def _read_meta(self, path):
        with rasterio.open(path) as src:
            return src.meta.copy()

    # ----------------------------------------------------
    # Reescalar LST y NDVI (10 m → 3 m)
    # ----------------------------------------------------
    def _align_to_3m(self, raster_path):
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"No existe raster mensual: {raster_path}")
        with rasterio.open(raster_path) as src:
            data = src.read(
                1,
                out_shape=(self.meta_base["height"], self.meta_base["width"]),
                resampling=Resampling.nearest,
            )
            return data.astype(np.float32)

    # ----------------------------------------------------
    # Buscar archivo mensual por patrón (robusto)
    # ----------------------------------------------------
    def _find_month_file(self, directory: str, tag: str, month: int) -> str:
        """
        Busca un .tif de un mes y año en un directorio, tolerante a:
        - mes con 1 o 01
        - prefijos distintos (S2_, L8_, etc.)
        - tokens separados por _ o -
        """
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {d}")

        m1 = str(int(month))
        m2 = f"{int(month):02d}"
        y = str(self.year)

        # Patrones comunes
        patterns = [
            f"*{y}*_{m1}_*{tag}*.tif",
            f"*{y}*_{m2}_*{tag}*.tif",
            f"*{y}*{m1}*{tag}*.tif",
            f"*{y}*{m2}*{tag}*.tif",
            f"*_{y}_*_{m1}_*{tag}*.tif",
            f"*_{y}_*_{m2}_*{tag}*.tif",
        ]

        candidates = []
        for p in patterns:
            candidates.extend(d.glob(p))

        # Filtrar por tag real en nombre (por si glob trae cosas raras)
        tag_upper = tag.upper()
        filtered = [c for c in candidates if tag_upper in c.name.upper()]

        # Quitar duplicados conservando orden
        seen = set()
        uniq = []
        for c in filtered:
            if str(c) not in seen:
                uniq.append(c)
                seen.add(str(c))

        if not uniq:
            # ayudar con debug: listar algunos .tif
            sample = list(d.glob("*.tif"))[:10]
            sample_names = [s.name for s in sample]
            raise FileNotFoundError(
                f"No encontré archivo {tag} para {self.year}-{month:02d} en: {d}\n"
                f"Patrones probados: {patterns}\n"
                f"Ejemplos de .tif en carpeta (primeros 10): {sample_names}"
            )

        # Si hay varios, elegimos el primero; si quieres, aquí puedes priorizar por nombre exacto
        return str(uniq[0])

    # ----------------------------------------------------
    # Obtener Tbase robusto desde el diccionario
    # ----------------------------------------------------
    def _get_tbase(self, month: int) -> float:
        """
        Soporta diccionarios con claves como:
        - 'YYYY-MM'        (ej. '2022-01')
        - 'YYYY-M'         (ej. '2022-1')
        - 'YYYYMM'         (ej. '202201')
        - (YYYY, MM)       (tupla)
        - 'yyyymm' si viene así
        """
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
            if k in self.temp_monthly:
                return float(self.temp_monthly[k])

        # Si tu dict viene como {'yyyymm': valor} pero k es int:
        # intentamos un fallback buscando por sufijo
        suffixes = {f"{y}{mm2}", f"{y}{mm1}"}
        for k in self.temp_monthly.keys():
            ks = str(k)
            if ks in suffixes or ks.endswith(f"{y}{mm2}") or ks.endswith(f"{y}{mm1}"):
                try:
                    return float(self.temp_monthly[k])
                except Exception:
                    pass

        raise KeyError(
            f"No encontré Tbase para {y}-{mm2} en temp_monthly_dict.\n"
            f"Claves de ejemplo (primeras 20): {list(self.temp_monthly.keys())[:20]}"
        )

    # ----------------------------------------------------
    # Ecuación formal ajustada para índices 0–1
    # ----------------------------------------------------
    def _compute_synthetic_temperature(self, month, lst_idx, ndvi_idx):
        # 1) Temperatura base mensual (CSV)
        Tbase = self._get_tbase(month)

        # 2) Convertir distancias de grados → metros
        dist_vias_m = self.dist_vias * self.DEG_TO_M
        dist_agua_m = self.dist_agua * self.DEG_TO_M

        # 3) Normalizar DEM relativo a 0–1
        dem_min = np.nanmin(self.dem)
        dem_max = np.nanmax(self.dem)
        dem_norm = (self.dem - dem_min) / (dem_max - dem_min + 1e-6)

        # 4) Efecto relativo de LST (0–1) y NDVI (0–1)
        lst_effect = (lst_idx - 0.5) * 10.0      # -5 a +5 °C
        ndvi_effect = (0.5 - ndvi_idx) * 8.0     # -4 a +4 °C

        # 5) Factores por cobertura
        coverage_raw = np.where(
            self.coverage == 1, 1.0,      # urbano
            np.where(
                self.coverage == 2, -1.0, # vegetación
                np.where(self.coverage == 3, -1.5, 0.0)  # agua / otros
            )
        )
        coverage_effect = coverage_raw * 1.0

        # 6) Densidades 0–1
        urban_effect = 3.0 * self.urban
        vegetation_effect = -2.0 * self.vegetation

        # 7) Distancias en metros
        vias_effect = 0.0015 * dist_vias_m
        agua_effect = -0.0008 * dist_agua_m

        # 8) DEM
        dem_effect = -1.0 * dem_norm

        # 9) Pendiente
        slope_effect = 0.05 * self.slope

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

        return np.clip(Tsint, 5, 35)

    # ----------------------------------------------------
    # Procesar un mes
    # ----------------------------------------------------
    def process_month(self, month: int):
        lst_path = self._find_month_file(self.lst_dir, self.lst_tag, month)
        ndvi_path = self._find_month_file(self.ndvi_dir, self.ndvi_tag, month)

        lst_idx = self._align_to_3m(lst_path)
        ndvi_idx = self._align_to_3m(ndvi_path)

        result = self._compute_synthetic_temperature(month, lst_idx, ndvi_idx)

        out_path = os.path.join(self.output_dir, f"TAIRE_SINTETICA_3M_{self.year}_{month:02d}.tif")

        meta = self.meta_base.copy()
        meta.update(dtype="float32", count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(result.astype(np.float32), 1)

        return out_path

    # ----------------------------------------------------
    # Procesar los 12 meses
    # ----------------------------------------------------
    def process_all(self, months=range(1, 13)):
        outputs = []
        for month in months:
            out = self.process_month(int(month))
            outputs.append(out)
            print(f"✓ {self.year}-{int(month):02d} generado → {out}")
        return outputs
