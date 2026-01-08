from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.features import shapes, rasterize

from scipy.ndimage import distance_transform_edt, gaussian_filter

import geopandas as gpd
from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.ops import unary_union


class ICUIsobandPolygonGenerator:
    """
    Genera isobandas (polígonos) desde ICU_SCORE(_AGRUPADO).

    ✅ Cumple:
    - Usa UPL (SHP/GeoJSON) como máscara: clasifica SOLO dentro de la UPL (sin huecos).
    - Genera subclases (band_edges) tipo curvas/isobandas.
    - Disuelve por class_id (1 polígono por banda) si dissolve=True.
    - Clip final a la UPL (no se sale) con clip_polygons_to_upl=True.
    - Detecta year/month desde el nombre: ICU_SCORE_YYYY_MM_...
    - Si out_geojson es carpeta: crea nombre automático por mes.
    - run_folder(): procesa todos los meses.
    - run_year_unified(): 1 GeoJSON anual disuelto por banda (month="ALL").
    - ✅ NUEVO: run_year_by_month(): 1 GeoJSON anual PERO con features MES A MES
      (disuelve por (month, class_id)) y mantiene columnas como antes:
      class_id, strength, range_min, range_max, label, year, month, source
    """

    def __init__(
        self,
        nodata: int = -1,
        min_class: int = 3,
        pixel_size_m: float = 3.0,
        decay_m: float = 60.0,
        smooth_sigma_m: float = 18.0,
        band_edges: Optional[List[float]] = None,
        upl_path: Optional[str] = None,           # ✅ SHP/GeoJSON UPL
        clip_polygons_to_upl: bool = True,        # ✅ clip final para que nada salga
        drop_class0: bool = False,                # False para conservar todas las bandas
        compress: str = "lzw",
        dissolve: bool = True,
    ):
        self.nodata = int(nodata)
        self.min_class = int(min_class)
        self.pixel_size_m = float(pixel_size_m)

        self.decay_m = float(decay_m)
        self.smooth_sigma_m = float(smooth_sigma_m)

        # Bandas (subclases): 0..len(edges)
        self.band_edges = band_edges or [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]

        self.upl_path = str(upl_path) if upl_path else None
        self.clip_polygons_to_upl = bool(clip_polygons_to_upl)

        self.drop_class0 = bool(drop_class0)
        self.compress = str(compress)
        self.dissolve = bool(dissolve)

    # -------------------------
    # Helpers: parse name + output name
    # -------------------------
    def _parse_year_month_from_stem(self, stem: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Espera: ICU_SCORE_2021_01_10x10_AGRUPADO
        """
        parts = stem.split("_")
        if len(parts) >= 4 and parts[0] == "ICU" and parts[1] == "SCORE":
            y = parts[2]
            m = parts[3]
            if y.isdigit() and len(y) == 4 and m.isdigit() and len(m) in (1, 2):
                return y, m.zfill(2)
        return None, None

    def _auto_out_geojson_path(self, in_tif: Path, out_geojson: str) -> str:
        outp = Path(out_geojson)
        suffix = "_ISOBANDS_DISSOLVE" if self.dissolve else "_ISOBANDS"

        if outp.suffix.lower() == ".geojson":
            return str(outp)

        outp.mkdir(parents=True, exist_ok=True)
        return str(outp / f"{in_tif.stem}{suffix}.geojson")

    # -------------------------
    # UPL geometry + raster mask
    # -------------------------
    def _load_upl_geom(self, raster_crs: CRS):
        """
        Carga UPL (SHP/GeoJSON) y la reproyecta al CRS del raster si aplica.
        Retorna shapely geometry (unary_union).
        """
        if not self.upl_path:
            return None

        gdf = gpd.read_file(self.upl_path)
        if gdf.empty:
            raise ValueError(f"UPL vacía: {self.upl_path}")

        if gdf.crs is not None and raster_crs is not None:
            if str(gdf.crs) != str(raster_crs):
                gdf = gdf.to_crs(raster_crs)

        return unary_union(gdf.geometry)

    def _upl_mask(self, upl_geom, transform: Affine, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Rasteriza la geometría UPL a máscara booleana (True dentro de UPL).
        """
        if upl_geom is None:
            return None

        mask = rasterize(
            [(upl_geom, 1)],
            out_shape=shape_hw,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        return mask

    # -------------------------
    # Heat surface
    # -------------------------
    def _build_heat_surface(self, icu: np.ndarray, inside_mask: np.ndarray) -> np.ndarray:
        """
        heat en 0..1.
        core = ICU >= min_class dentro de inside_mask.
        """
        core = inside_mask & (icu >= self.min_class)

        if np.count_nonzero(core) == 0:
            return np.zeros_like(icu, dtype=np.float32)

        dist_pix = distance_transform_edt(~core)
        dist_m = dist_pix.astype(np.float32) * self.pixel_size_m

        heat = np.exp(-dist_m / max(self.decay_m, 1e-6)).astype(np.float32)
        heat[~inside_mask] = 0.0

        sigma_pix = self.smooth_sigma_m / max(self.pixel_size_m, 1e-6)
        if sigma_pix > 0:
            heat = gaussian_filter(heat, sigma=sigma_pix).astype(np.float32)

        return np.clip(heat, 0.0, 1.0)

    # -------------------------
    # Classify heat into bands
    # -------------------------
    def _classify(self, heat: np.ndarray, inside_mask: np.ndarray) -> np.ndarray:
        """
        Clasifica TODA la UPL (inside_mask) sin huecos.
        Fuera de UPL => 255 (nodata clase).
        """
        edges = np.array(self.band_edges, dtype=np.float32)
        cls = np.full(heat.shape, 255, dtype=np.uint8)

        cls_inside = np.digitize(heat[inside_mask], edges, right=False).astype(np.uint8)
        cls[inside_mask] = cls_inside
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
    def _polygons_by_class(self, cls: np.ndarray, transform: Affine) -> Dict[int, List[Dict[str, Any]]]:
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        valid = (cls != 255)

        for geom, value in shapes(cls, mask=valid, transform=transform):
            cid = int(value)
            if cid == 255:
                continue
            if self.drop_class0 and cid == 0:
                continue
            by_class.setdefault(cid, []).append(geom)

        return by_class

    # -------------------------
    # Dissolve per class (union)
    # -------------------------
    def _dissolve_by_class(self, by_class: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for cid, geom_list in by_class.items():
            if not geom_list:
                continue
            geoms = [shp_shape(g) for g in geom_list]
            merged = unary_union(geoms)
            out[cid] = shp_mapping(merged)
        return out

    # -------------------------
    # Clip geometry to UPL (final safety)
    # -------------------------
    def _clip_geom_to_upl(self, geom_geojson: Dict[str, Any], upl_geom) -> Optional[Dict[str, Any]]:
        if not self.clip_polygons_to_upl or upl_geom is None:
            return geom_geojson

        g = shp_shape(geom_geojson)
        g2 = g.intersection(upl_geom)
        if g2.is_empty:
            return None
        return shp_mapping(g2)

    # -------------------------
    # Public API (mensual)
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

        # ✅ UPL geometry + mask
        upl_geom = self._load_upl_geom(crs)
        upl_mask = self._upl_mask(upl_geom, transform=transform, shape_hw=icu.shape)

        # dentro = UPL (si existe), si no existe -> valid del raster
        inside = upl_mask if upl_mask is not None else (icu != self.nodata)

        # heat + clases para toda la UPL
        heat = self._build_heat_surface(icu=icu, inside_mask=inside)
        cls = self._classify(heat=heat, inside_mask=inside)

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

        # vectorizar y disolver
        by_class = self._polygons_by_class(cls=cls, transform=transform)

        features: List[Dict[str, Any]] = []

        if self.dissolve:
            dissolved = self._dissolve_by_class(by_class)
            for cid, geom in sorted(dissolved.items(), key=lambda x: x[0]):
                geom2 = self._clip_geom_to_upl(geom, upl_geom)
                if geom2 is None:
                    continue

                lo, hi = self._band_range(cid)
                props = {
                    "class_id": int(cid),
                    "strength": int(cid),
                    "range_min": lo,
                    "range_max": hi,
                    "label": f"{lo:.2f}-{hi:.2f}",
                    "source": in_path.stem,
                }
                if year is not None:
                    props["year"] = year
                if month is not None:
                    props["month"] = month

                features.append({"type": "Feature", "properties": props, "geometry": geom2})
        else:
            for cid, geom_list in sorted(by_class.items(), key=lambda x: x[0]):
                lo, hi = self._band_range(cid)
                for geom in geom_list:
                    geom2 = self._clip_geom_to_upl(geom, upl_geom)
                    if geom2 is None:
                        continue

                    props = {
                        "class_id": int(cid),
                        "strength": int(cid),
                        "range_min": lo,
                        "range_max": hi,
                        "label": f"{lo:.2f}-{hi:.2f}",
                        "source": in_path.stem,
                    }
                    if year is not None:
                        props["year"] = year
                    if month is not None:
                        props["month"] = month

                    features.append({"type": "Feature", "properties": props, "geometry": geom2})

        fc = {
            "type": "FeatureCollection",
            "name": Path(out_geojson).stem,
            "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": features,
        }

        Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        return out_geojson

    # -------------------------
    # Procesar carpeta completa (mensual)
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
            outs.append(self.run(str(f), str(out_dir)))
        return outs

    # -------------------------
    # ✅ NUEVO: ANUAL "MES A MES" (un solo GeoJSON con features por mes)
    # -------------------------
    def run_year_by_month(
        self,
        in_dir: str,
        year: str,
        out_geojson_year: str,
        pattern: str = "ICU_SCORE_{year}_??_10x10_AGRUPADO.tif",
        dissolve_per_month: bool = True,
    ) -> str:
        """
        1 archivo ANUAL pero con geometrías "mes a mes".
        - Mantiene columnas como antes:
          class_id, strength, range_min, range_max, label, year, month, source
        - dissolve_per_month=True:
          1 polígono por banda POR MES (disuelve dentro de cada mes).
        """
        in_dir = Path(in_dir)
        files = sorted(in_dir.glob(pattern.format(year=year)))
        if not files:
            raise FileNotFoundError(f"No se encontraron TIF del año {year} en:\n{in_dir}")

        crs_final = CRS.from_epsg(4326)
        upl_geom_final = None
        features: List[Dict[str, Any]] = []

        for f in files:
            y, m = self._parse_year_month_from_stem(f.stem)

            with rasterio.open(str(f)) as src:
                icu = src.read(1).astype(np.int16)
                transform = src.transform
                crs_final = src.crs if src.crs is not None else CRS.from_epsg(4326)

            if upl_geom_final is None:
                upl_geom_final = self._load_upl_geom(crs_final)

            upl_mask = self._upl_mask(upl_geom_final, transform=transform, shape_hw=icu.shape)
            inside = upl_mask if upl_mask is not None else (icu != self.nodata)

            heat = self._build_heat_surface(icu=icu, inside_mask=inside)
            cls = self._classify(heat=heat, inside_mask=inside)

            by_class = self._polygons_by_class(cls=cls, transform=transform)

            if dissolve_per_month:
                dissolved = self._dissolve_by_class(by_class)
                for cid, geom in sorted(dissolved.items(), key=lambda x: x[0]):
                    if self.drop_class0 and cid == 0:
                        continue

                    geom2 = self._clip_geom_to_upl(geom, upl_geom_final)
                    if geom2 is None:
                        continue

                    lo, hi = self._band_range(cid)
                    props = {
                        "class_id": int(cid),
                        "strength": int(cid),
                        "range_min": lo,
                        "range_max": hi,
                        "label": f"{lo:.2f}-{hi:.2f}",
                        "year": str(y or year),
                        "month": str(m or "??"),
                        "source": f.stem,
                    }
                    features.append({"type": "Feature", "properties": props, "geometry": geom2})
            else:
                for cid, geom_list in sorted(by_class.items(), key=lambda x: x[0]):
                    if self.drop_class0 and cid == 0:
                        continue

                    lo, hi = self._band_range(cid)
                    for geom in geom_list:
                        geom2 = self._clip_geom_to_upl(geom, upl_geom_final)
                        if geom2 is None:
                            continue

                        props = {
                            "class_id": int(cid),
                            "strength": int(cid),
                            "range_min": lo,
                            "range_max": hi,
                            "label": f"{lo:.2f}-{hi:.2f}",
                            "year": str(y or year),
                            "month": str(m or "??"),
                            "source": f.stem,
                        }
                        features.append({"type": "Feature", "properties": props, "geometry": geom2})

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

    # -------------------------
    # Unificado anual: 1 GeoJSON con todos los meses (disuelto por banda)
    # -------------------------
    def run_year_unified(
        self,
        in_dir: str,
        year: str,
        out_geojson_year: str,
        pattern: str = "ICU_SCORE_{year}_??_10x10_AGRUPADO.tif",
    ) -> str:
        """
        Produce 1 GeoJSON anual disuelto por class_id con TODAS las geometrías de todos los meses.
        Mantiene columnas como antes (incluye source y month="ALL").
        """
        in_dir = Path(in_dir)
        files = sorted(in_dir.glob(pattern.format(year=year)))
        if not files:
            raise FileNotFoundError(f"No se encontraron TIF del año {year} en:\n{in_dir}")

        annual_by_class: Dict[int, List[Dict[str, Any]]] = {}
        crs_final = CRS.from_epsg(4326)
        upl_geom_final = None

        for f in files:
            with rasterio.open(str(f)) as src:
                icu = src.read(1).astype(np.int16)
                transform = src.transform
                crs_final = src.crs if src.crs is not None else CRS.from_epsg(4326)

            if upl_geom_final is None:
                upl_geom_final = self._load_upl_geom(crs_final)

            upl_mask = self._upl_mask(upl_geom_final, transform=transform, shape_hw=icu.shape)
            inside = upl_mask if upl_mask is not None else (icu != self.nodata)

            heat = self._build_heat_surface(icu=icu, inside_mask=inside)
            cls = self._classify(heat=heat, inside_mask=inside)

            by_class = self._polygons_by_class(cls=cls, transform=transform)
            for cid, geom_list in by_class.items():
                if self.drop_class0 and cid == 0:
                    continue
                annual_by_class.setdefault(cid, []).extend(geom_list)

        dissolved_year = self._dissolve_by_class(annual_by_class)

        features: List[Dict[str, Any]] = []
        for cid, geom in sorted(dissolved_year.items(), key=lambda x: x[0]):
            geom2 = self._clip_geom_to_upl(geom, upl_geom_final)
            if geom2 is None:
                continue

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
                        "source": f"ICU_SCORE_{year}_ALL_UNIFIED",
                    },
                    "geometry": geom2,
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
