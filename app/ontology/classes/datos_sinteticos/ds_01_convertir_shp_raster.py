import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import math


class ShapefileToRaster:

    def __init__(self, shp_path, pixel_size=0.0001, raster_value=1, shp_referencia_path=None):
        """
        shp_path: SHP que será rasterizado
        shp_referencia_path: SHP que define la extensión y tamaño del raster (UPL)
        """
        self.shp_path = shp_path
        self.shp_referencia_path = shp_referencia_path
        self.pixel_size = pixel_size
        self.raster_value = raster_value
        self.gdf = None
        self.gdf_ref = None
        self.transform = None
        self.width = None
        self.height = None
        self.bounds = None

    def load_and_validate(self):
        """Carga el SHP a rasterizar y valida que sea EPSG:4326"""
        self.gdf = gpd.read_file(self.shp_path)

        if self.gdf.crs is None:
            raise ValueError("El shapefile no tiene sistema de referencia (CRS).")

        if str(self.gdf.crs.to_epsg()) != "4326":
            raise ValueError(f"El shapefile debe estar en EPSG:4326 y está en {self.gdf.crs}")

        # Si hay un SHP de referencia, cargarlo también
        if self.shp_referencia_path:
            self.gdf_ref = gpd.read_file(self.shp_referencia_path)

            if str(self.gdf_ref.crs.to_epsg()) != "4326":
                raise ValueError(
                    f"El SHP de referencia debe estar en EPSG:4326 y está en {self.gdf_ref.crs}"
                )

    def compute_raster_grid(self):
        """Calcula el tamaño del raster usando el SHP de referencia (UPL) si existe"""

        # Si hay SHP de referencia → usar ese bounding box
        if self.gdf_ref is not None:
            minx, miny, maxx, maxy = self.gdf_ref.total_bounds
        else:
            minx, miny, maxx, maxy = self.gdf.total_bounds  # usa el SHP original

        self.bounds = (minx, miny, maxx, maxy)

        width_deg = maxx - minx
        height_deg = maxy - miny

        # Asegurar cubrir el área completa
        self.width = math.ceil(width_deg / self.pixel_size)
        self.height = math.ceil(height_deg / self.pixel_size)

        # Transformación espacial
        self.transform = from_origin(minx, maxy, self.pixel_size, self.pixel_size)

        return self.width, self.height

    def rasterize_shp(self, output_path):
        """Convierte el SHP a raster respetando dimensiones del SHP referencia"""

        if self.gdf is None:
            self.load_and_validate()

        self.compute_raster_grid()

        meta = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 1,
            "dtype": rasterio.uint8,
            "crs": "EPSG:4326",
            "transform": self.transform
        }

        shapes = [(geom, self.raster_value) for geom in self.gdf.geometry]

        with rasterio.open(output_path, "w", **meta) as dst:
            raster = rasterize(
                shapes=shapes,
                out_shape=(self.height, self.width),
                transform=self.transform,
                fill=0,
                dtype=rasterio.uint8
            )
            dst.write(raster, 1)

        return output_path
