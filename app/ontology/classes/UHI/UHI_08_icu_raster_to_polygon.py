from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union


ICUMode = Literal["eq", "gte"]  # eq: == threshold, gte: >= threshold


class ICURasterToPolygonDissolve:
    """
    Convierte un raster ICU_SCORE a polígonos y los UNE (dissolve) para obtener
    una geometría continua de la "isla de calor".

    Recomendado:
      - threshold=3 y mode="eq"  -> ICU alta
      - threshold=2 y mode="gte" -> ICU media + alta

    Salida:
      - GeoJSON o Shapefile con 1 fila (geom disuelta) o multiparte.
    """

    def __init__(
        self,
        threshold: int = 2,
        mode: ICUMode = "eq",
        nodata: int = -1,
        min_area_m2: float = 500.0,
        simplify_tolerance_m: Optional[float] = None,
        connectivity: Literal[4, 8] = 8,
    ):
        self.threshold = int(threshold)
        self.mode = mode
        self.nodata = int(nodata)
        self.min_area_m2 = float(min_area_m2)
        self.simplify_tolerance_m = simplify_tolerance_m
        self.connectivity = connectivity

    def _make_binary_mask(self, arr: np.ndarray) -> np.ndarray:
        valid = arr != self.nodata
        if self.mode == "eq":
            return valid & (arr == self.threshold)
        return valid & (arr >= self.threshold)

    def run(
        self,
        icu_tif: str,
        out_vector_path: str,
        month_key: str,
        year: str,
        layer_name: str = "ICU",
    ) -> str:
        icu_tif = str(icu_tif)
        out_vector_path = str(out_vector_path)
        Path(out_vector_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(icu_tif) as src:
            arr = src.read(1)
            transform = src.transform
            crs = src.crs

        mask = self._make_binary_mask(arr)

        if mask.sum() == 0:
            raise ValueError(
                f"No hay píxeles que cumplan el criterio ICU (mode={self.mode}, threshold={self.threshold}).\n"
                f"Raster: {icu_tif}"
            )

        # rasterio.features.shapes produce polígonos por regiones contiguas del "mask"
        # usamos arr_bin = 1 donde mask True
        arr_bin = np.where(mask, 1, 0).astype(np.uint8)

        # connectivity: 4 u 8
        # Nota: shapes() usa conectividad 4 por defecto en algunas versiones;
        # el parámetro 'connectivity' existe en rasterio >= 1.3.
        # Si tu rasterio no lo soporta, ignora y quedará 4.
        try:
            gen = shapes(arr_bin, mask=mask, transform=transform, connectivity=self.connectivity)
        except TypeError:
            gen = shapes(arr_bin, mask=mask, transform=transform)

        geoms = []
        for geom, value in gen:
            if int(value) != 1:
                continue
            geoms.append(shape(geom))

        if not geoms:
            raise ValueError("No se pudieron extraer polígonos desde el raster (shapes devolvió vacío).")

        gdf = gpd.GeoDataFrame({"value": [1] * len(geoms)}, geometry=geoms, crs=crs)

        # ⚠️ Área: si el raster está en EPSG:4326, el área sale en grados^2.
        # Para filtrar por área real, re-proyectamos temporalmente a una CRS métrica.
        # Como estás en Bogotá, una opción robusta es EPSG:3116 (MAGNA-SIRGAS / Colombia Bogotá).
        # No cambia tu raster; solo es para calcular/filtrar.
        gdf_m = gdf.to_crs("EPSG:3116")
        gdf_m["area_m2"] = gdf_m.area

        # Filtrar ruido por área mínima
        gdf_m = gdf_m[gdf_m["area_m2"] >= self.min_area_m2].copy()
        if gdf_m.empty:
            raise ValueError(
                f"Tras filtrar por min_area_m2={self.min_area_m2}, no quedó ningún polígono.\n"
                f"Baja min_area_m2 o revisa el raster."
            )

        # Dissolve / unir todo en 1 geometría
        union_geom = unary_union(list(gdf_m.geometry))

        # Simplificar opcional (en metros)
        if self.simplify_tolerance_m is not None and self.simplify_tolerance_m > 0:
            union_geom = union_geom.simplify(self.simplify_tolerance_m, preserve_topology=True)

        out = gpd.GeoDataFrame(
            {
                "year": [str(year)],
                "month": [month_key],
                "mode": [self.mode],
                "threshold": [self.threshold],
                "min_area_m2": [self.min_area_m2],
            },
            geometry=[union_geom],
            crs="EPSG:3116",
        ).to_crs(crs)  # volver al CRS original del raster (probablemente EPSG:4326)

        # Guardar según extensión
        ext = Path(out_vector_path).suffix.lower()
        if ext in (".geojson", ".json"):
            out.to_file(out_vector_path, driver="GeoJSON")
        elif ext == ".shp":
            out.to_file(out_vector_path)
        else:
            # default GeoJSON si no reconoce
            out_vector_path = str(Path(out_vector_path).with_suffix(".geojson"))
            out.to_file(out_vector_path, driver="GeoJSON")

        return out_vector_path
