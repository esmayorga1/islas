from sqlalchemy import create_engine, text
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os
from app.core.config import settings

class SocioeconomicaBase:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.engine = self._connect_db()

    def _connect_db(self):
        db_url = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        return create_engine(db_url)

    def read_table(self, table_name, geom_col='geom', value_col=None, query=None):
        """
        Leer la tabla de PostGIS como GeoDataFrame.
        """
        if query is None:
            query = f"SELECT {geom_col}, {value_col} FROM {table_name};"
        gdf = gpd.read_postgis(query, con=self.engine, geom_col=geom_col)
        return gdf

    def rasterize(self, gdf, value_col, pixel_size=10, crs=None, output_name="raster.tif"):
        """
        Rasterizar la columna `value_col` del GeoDataFrame.
        """
        if crs is None:
            crs = gdf.crs

        bounds = gdf.total_bounds  # xmin, ymin, xmax, ymax
        width = int((bounds[2] - bounds[0]) / pixel_size)
        height = int((bounds[3] - bounds[1]) / pixel_size)
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], pixel_size, pixel_size)

        raster_data = rasterize(
            ((geom, value) for geom, value in zip(gdf.geometry, gdf[value_col])),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='float32'
        )

        output_path = os.path.join(self.output_dir, output_name)
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(raster_data, 1)

        return output_path
