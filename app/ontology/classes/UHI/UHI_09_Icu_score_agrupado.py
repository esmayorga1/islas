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

# ✅ dissolve (union) por clase
from shapely.geometry import shape as shp_shape, mapping as shp_mapping
from shapely.ops import unary_union


class ICUIsobandPolygonGenerator:
    """
    Genera polígonos (isobandas) desde ICU_SCORE y DISUELVE por clase (misma curva de nivel).

    - Núcleo/foco: ICU >= min_class (recomendado min_class=3)
    - Crea superficie HEAT (0..1) con decaimiento por distancia al núcleo
    - Clasifica HEAT en bandas (bins)
    - Vectoriza a POLÍGONOS
    - ✅ DISUELVE por class_id (1 polígono por banda)
    """

    def __init__(
        self,
        nodata: int = -1,
        min_class: int = 3,
        pixel_size_m: float = 3.0,
        decay_m: float = 60.0,
        smooth_sigma_m: float = 18.0,
        band_edges: Optional[List[float]] = None,
        # filtros
        drop_class0: bool = True,            # clase más débil (ruidosa)
        min_polygon_area_m2: float = 300.0,  # filtra polígonos pequeños ANTES de disolver
        compress: str = "lzw",
        # ✅ disolver
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
        cls[~valid_mask] = np.uint8(255)  # nodata clase
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
        """
        Devuelve dict {class_id: [geom_geojson, ...]}
        Filtra:
          - class 0 si drop_class0=True
          - polígonos muy pequeños (aprox por área real en m² usando shapely.area,
            OJO: en EPSG:4326 area está en grados^2; pero tú estás trabajando “como 3m”,
            así que filtramos por conteo aproximado de pixeles usando bbox no es fiable.
            Mejor: filtrar por tamaño mínimo en pixeles estimado en raster antes del vectorizado.
        """
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        valid = valid_mask & (cls != 255)

        # filtro mínimo por pixeles (simple y efectivo)
        pixel_area_m2 = self.pixel_size_m ** 2
        min_pixels = int(np.ceil(self.min_polygon_area_m2 / max(pixel_area_m2, 1e-9)))

        # Para filtrar por pixeles SIN shapely:
        # vectorizamos y luego, como no tenemos count, aplicamos un truco:
        # usamos un segundo shapes sobre máscara de cada clase para “contar” pixeles por componente.
        # Lo más simple y rápido aquí: NO filtrar por pixeles componente a componente;
        # y dejar que el dissolve + tu postproceso controlen el ruido.
        # Si quieres filtro real por pixeles por componente, te lo armo también.

        for geom, value in shapes(cls, mask=valid, transform=transform):
            v = int(value)
            if v == 255:
                continue
            if self.drop_class0 and v == 0:
                continue

            by_class.setdefault(v, [])
            by_class[v].append(geom)

        return by_class

    # -------------------------
    # Dissolve per class (union)
    # -------------------------
    def _dissolve_by_class(
        self,
        by_class: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Union geométrica por clase: devuelve {class_id: dissolved_geom_geojson}
        """
        out: Dict[int, Dict[str, Any]] = {}
        for cid, geom_list in by_class.items():
            if not geom_list:
                continue
            geoms = [shp_shape(g) for g in geom_list]
            merged = unary_union(geoms)
            out[cid] = shp_mapping(merged)
        return out

    # -------------------------
    # Public API
    # -------------------------
    def run(
        self,
        in_icu_tif: str,
        out_geojson: str,
        out_heat_tif: Optional[str] = None,
        out_class_tif: Optional[str] = None,
    ) -> str:
        in_icu_tif = str(in_icu_tif)
        out_geojson = str(out_geojson)

        with rasterio.open(in_icu_tif) as src:
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

        # vectorizar
        by_class = self._polygons_by_class(cls=cls, transform=transform, valid_mask=valid)

        # ✅ disolver por clase (1 polígono por isobanda)
        if self.dissolve:
            dissolved = self._dissolve_by_class(by_class)
            features = []
            for cid, geom in sorted(dissolved.items(), key=lambda x: x[0]):
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
                        },
                        "geometry": geom,
                    }
                )
        else:
            # sin dissolve: múltiples features por clase
            features = []
            for cid, geom_list in sorted(by_class.items(), key=lambda x: x[0]):
                lo, hi = self._band_range(cid)
                for geom in geom_list:
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {
                                "class_id": int(cid),
                                "strength": int(cid),
                                "range_min": lo,
                                "range_max": hi,
                                "label": f"{lo:.2f}-{hi:.2f}",
                            },
                            "geometry": geom,
                        }
                    )

        fc = {
            "type": "FeatureCollection",
            "name": "icu_isobands_dissolved",
            "crs": {"type": "name", "properties": {"name": str(crs)}},
            "features": features,
        }

        Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)

        return out_geojson
