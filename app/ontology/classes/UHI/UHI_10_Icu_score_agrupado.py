from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.features import shapes

from scipy.ndimage import distance_transform_edt, gaussian_filter

from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.ops import unary_union


class ICUIsobandPolygonGenerator:
    """
    Genera polígonos (isobandas) desde ICU_SCORE(_AGRUPADO) y DISUELVE por clase.

    ✅ NUEVO (automático):
    - Detecta year/month desde el nombre del tif: ICU_SCORE_YYYY_MM_...
    - Si out_geojson es carpeta: crea automáticamente:
      ICU_SCORE_YYYY_MM_..._ISOBANDS_DISSOLVE.geojson
    - Escribe year/month en properties automáticamente.

    Además:
    - run_folder(): procesa todos los meses de una carpeta
    - run_year_unified(): genera 1 GeoJSON anual unificado (dissolve por banda en todos los meses)
    """

    def __init__(
        self,
        nodata: int = -1,
        min_class: int = 3,
        pixel_size_m: float = 3.0,
        decay_m: float = 60.0,
        smooth_sigma_m: float = 18.0,
        band_edges: Optional[List[float]] = None,
        drop_class0: bool = True,
        min_polygon_area_m2: float = 300.0,
        compress: str = "lzw",
        dissolve: bool = True,
    ):
        self.nodata = int(nodata)
        self.min_class = int(min_class)
        self.pixel_size_m = float(pixel_size_m)

        self.decay_m = float(decay_m)
        self.smooth_sigma_m = float(smooth_sigma_m)

        self.band_edges = band_edges or [0.25, 0.40, 0.55, 0.70, 0.85]
        self.drop_class0 = bool(drop_class0)
        self.min_polygon_area_m2 = float(min_polygon_area_m2)
        self.compress = str(compress)

        self.dissolve = bool(dissolve)

    # -------------------------
    # Helpers: parse name + output name
    # -------------------------
    def _parse_year_month_from_stem(self, stem: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Espera algo tipo:
        ICU_SCORE_2021_01_10x10_AGRUPADO
        """
        parts = stem.split("_")
        # ICU SCORE YYYY MM ...
        if len(parts) >= 4 and parts[0] == "ICU" and parts[1] == "SCORE":
            y = parts[2]
            m = parts[3]
            if y.isdigit() and len(y) == 4 and m.isdigit() and len(m) in (1, 2):
                return y, m.zfill(2)
        return None, None

    def _auto_out_geojson_path(self, in_tif: Path, out_geojson: str) -> str:
        """
        Si out_geojson es carpeta -> crea nombre automático:
        {in_stem}_ISOBANDS_DISSOLVE.geojson (si dissolve=True)
        {in_stem}_ISOBANDS.geojson (si dissolve=False)
        """
        outp = Path(out_geojson)
        suffix = "_ISOBANDS_DISSOLVE" if self.dissolve else "_ISOBANDS"

        if outp.suffix.lower() == ".geojson":
            return str(outp)

        # si no tiene extensión -> lo tratamos como carpeta
        outp.mkdir(parents=True, exist_ok=True)
        return str(outp / f"{in_tif.stem}{suffix}.geojson")

    # -------------------------
    # Heat surface
    # -------------------------
    def _build_heat_surface(self, icu: np.ndarray) -> np.ndarray:
        valid = (icu != self.nodata)
        core = valid & (icu >= self.min_class)

        if np.count_nonzero(core) == 0:
            return np.zeros_like(icu, dtype=np.float32)

        dist_pix = distance_transform_edt(~core)
        dist_m = dist_pix.astype(np.float32) * self.pixel_size_m

        heat = np.exp(-dist_m / max(self.decay_m, 1e-6)).astype(np.float32)
        heat[~valid] = 0.0

        sigma_pix = self.smooth_sigma_m / max(self.pixel_size_m, 1e-6)
        if sigma_pix > 0:
            heat = gaussian_filter(heat, sigma=sigma_pix).astype(np.float32)

        return np.clip(heat, 0.0, 1.0)

    # -------------------------
    # Classify heat into bands
    # -------------------------
    def _classify(self, heat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        edges = np.array(self.band_edges, dtype=np.float32)
        cls = np.digitize(heat, edges, right=False).astype(np.uint8)
        cls[~valid_mask] = np.uint8(255)
        return cls

    def _band_range(self, cls_val: int) -> Tuple[float, float]:
        edges = self.band_edges
        if cls_val <= 0:
            return 0.0, float(edges[0])
        if cls_val >= len(edges):
            return float(edges[-1]), 1.0
        return float(edges[cls_val - 1]), float(edges[cls_val])

    # -------------------------
    # Vectorize polygons per class
    # -------------------------
    def _polygons_by_class(
        self,
        cls: np.ndarray,
        transform: Affine,
        valid_mask: np.ndarray,
    ) -> Dict[int, List[Dict[str, Any]]]:
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        valid = valid_mask & (cls != 255)

        for geom, value in shapes(cls, mask=valid, transform=transform):
            v = int(value)
            if v == 255:
                continue
            if self.drop_class0 and v == 0:
                continue
            by_class.setdefault(v, []).append(geom)

        return by_class

    # -------------------------
    # Dissolve per class (union)
    # -------------------------
    def _dissolve_by_class(
        self,
        by_class: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for cid, geom_list in by_class.items():
            if not geom_list:
                continue
            geoms = [shp_shape(g) for g in geom_list]
            merged = unary_union(geoms)
            out[cid] = shp_mapping(merged)
        return out

    # -------------------------
    # Public API (mensual automático)
    # -------------------------
    def run(
        self,
        in_icu_tif: str,
        out_geojson: str,
        out_heat_tif: Optional[str] = None,
        out_class_tif: Optional[str] = None,
    ) -> str:
        in_path = Path(in_icu_tif)
        out_geojson = self._auto_out_geojson_path(in_path, out_geojson)

        year, month = self._parse_year_month_from_stem(in_path.stem)

        with rasterio.open(str(in_path)) as src:
            icu = src.read(1).astype(np.int16)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs if src.crs is not None else CRS.from_epsg(4326)

        valid = (icu != self.nodata)
        heat = self._build_heat_surface(icu)
        cls = self._classify(heat, valid_mask=valid)

        # opcionales tif
        if out_heat_tif:
            out_heat_tif = str(out_heat_tif)
            hp = profile.copy()
            hp.update(dtype=rasterio.float32, count=1, nodata=0.0, compress=self.compress)
            Path(out_heat_tif).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_heat_tif, "w", **hp) as dst:
                dst.write(heat.astype(np.float32), 1)

        if out_class_tif:
            out_class_tif = str(out_class_tif)
            cp = profile.copy()
            cp.update(dtype=rasterio.uint8, count=1, nodata=255, compress=self.compress)
            Path(out_class_tif).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_class_tif, "w", **cp) as dst:
                dst.write(cls.astype(np.uint8), 1)

        by_class = self._polygons_by_class(cls=cls, transform=transform, valid_mask=valid)

        if self.dissolve:
            dissolved = self._dissolve_by_class(by_class)
            features = []
            for cid, geom in sorted(dissolved.items(), key=lambda x: x[0]):
                lo, hi = self._band_range(cid)
                props = {
                    "class_id": int(cid),
                    "strength": int(cid),
                    "range_min": lo,
                    "range_max": hi,
                    "label": f"{lo:.2f}-{hi:.2f}",
                }
                if year is not None:
                    props["year"] = year
                if month is not None:
                    props["month"] = month
                props["source"] = in_path.stem  # ICU_SCORE_YYYY_MM_...

                features.append({"type": "Feature", "properties": props, "geometry": geom})
        else:
            features = []
            for cid, geom_list in sorted(by_class.items(), key=lambda x: x[0]):
                lo, hi = self._band_range(cid)
                for geom in geom_list:
                    props = {
                        "class_id": int(cid),
                        "strength": int(cid),
                        "range_min": lo,
                        "range_max": hi,
                        "label": f"{lo:.2f}-{hi:.2f}",
                    }
                    if year is not None:
                        props["year"] = year
                    if month is not None:
                        props["month"] = month
                    props["source"] = in_path.stem
                    features.append({"type": "Feature", "properties": props, "geometry": geom})

        fc = {
            "type": "FeatureCollection",
            # ✅ aquí queda exactamente como tú quieres: ICU_SCORE_2021_01_..._ISOBANDS_DISSOLVE
            "name": Path(out_geojson).stem,
            "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": features,
        }

        Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        return out_geojson

    # -------------------------
    # Procesar carpeta completa (mensual automático)
    # -------------------------
    def run_folder(
        self,
        in_dir: str,
        out_dir: str,
        pattern: str = "ICU_SCORE_*_*_10x10_AGRUPADO.tif",
    ) -> List[str]:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No hay TIF con patrón {pattern} en:\n{in_dir}")

        outs = []
        for f in files:
            # out_dir como carpeta → nombre automático
            outs.append(self.run(str(f), str(out_dir)))
        return outs

    # -------------------------
    # Unificado anual: 1 GeoJSON con todos los meses
    # -------------------------
    def run_year_unified(
        self,
        in_dir: str,
        year: str,
        out_geojson_year: str,
        pattern: str = "ICU_SCORE_{year}_??_10x10_AGRUPADO.tif",
    ) -> str:
        """
        Produce 1 GeoJSON anual (dissolve por class_id con TODAS las geometrías de todos los meses).
        """
        in_dir = Path(in_dir)
        files = sorted(in_dir.glob(pattern.format(year=year)))
        if not files:
            raise FileNotFoundError(f"No se encontraron TIF del año {year} en:\n{in_dir}")

        annual_by_class: Dict[int, List[Dict[str, Any]]] = {}
        crs_final = CRS.from_epsg(4326)

        for f in files:
            with rasterio.open(str(f)) as src:
                icu = src.read(1).astype(np.int16)
                transform = src.transform
                crs_final = src.crs if src.crs is not None else CRS.from_epsg(4326)

            valid = (icu != self.nodata)
            heat = self._build_heat_surface(icu)
            cls = self._classify(heat, valid_mask=valid)

            by_class = self._polygons_by_class(cls=cls, transform=transform, valid_mask=valid)
            for cid, geom_list in by_class.items():
                annual_by_class.setdefault(cid, []).extend(geom_list)

        dissolved_year = self._dissolve_by_class(annual_by_class)

        features = []
        for cid, geom in sorted(dissolved_year.items(), key=lambda x: x[0]):
            lo, hi = self._band_range(cid)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class_id": int(cid),
                        "strength": int(cid),
                        "range_min": lo,
                        "range_max": hi,
                        "label": f"{lo:.2f}-{hi:.2f}",
                        "year": str(year),
                        "month": "ALL",
                    },
                    "geometry": geom,
                }
            )

        out_geojson_year = str(out_geojson_year)
        fc = {
            "type": "FeatureCollection",
            "name": Path(out_geojson_year).stem,
            "crs": {"type": "name", "properties": {"name": str(crs_final)}},
            "features": features,
        }

        Path(out_geojson_year).parent.mkdir(parents=True, exist_ok=True)
        with open(out_geojson_year, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        return out_geojson_year
