from .urbana import Urbana
import numpy as np
import geopandas as gpd
from sqlalchemy import create_engine
import rasterio

class DensidadEdificatoria(Urbana):
    """
    Calcula la densidad edificatoria a partir de:
    - Raster de área construida (IBI)
    - Shapefile de edificaciones almacenado en PostGIS
    """

    def __init__(self, input_dir, output_dir, table_name="construciones", geom_col="geom", crs="EPSG:4326"):
        super().__init__(input_dir=input_dir, output_dir=output_dir)
        self.table_name = table_name
        self.geom_col = geom_col
        self.crs = crs
        self.engine = self._connect_db()

    def _connect_db(self):
        from app.core.config import settings
        db_url = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        return create_engine(db_url)

    def read_construcciones(self):
        """Leer las construcciones desde PostGIS como GeoDataFrame"""
        query = f"SELECT {self.geom_col} FROM {self.table_name};"
        gdf = gpd.read_postgis(query, self.engine, geom_col=self.geom_col)
        # Calcular área en m² usando CRS proyectado (EPSG:3857)
        gdf['area_m2'] = gdf.geometry.to_crs(epsg=3857).area
        return gdf

    def calculate(self, output_prefix="DensidadEdificatoria"):
        """
        Calcula la densidad edificatoria promedio y guarda un raster con la densidad
        manteniendo el prefijo de cada imagen Sentinel-2.
        """
        # Leer construcciones
        construcciones = self.read_construcciones()
        num_edificios = len(construcciones)
        if num_edificios == 0:
            print("⚠️ No se encontraron edificaciones en la base de datos")
            return

        # Detectar rasters de área construida en la carpeta
        raster_files = list(self.input_dir.glob("*_IndiceAreaConstruida.tif"))
        if not raster_files:
            raise FileNotFoundError("No se encontraron rasters *_IndiceAreaConstruida.tif en la carpeta de entrada")

        # Iterar sobre cada raster
        for raster_path in raster_files:
            with rasterio.open(raster_path) as src:
                ibi = src.read(1)
                profile = src.profile
                transform = src.transform
                pixel_area = abs(transform.a * transform.e)  # área de cada pixel en m²

            # Área total construida
            area_total_construida = np.sum(ibi) * pixel_area

            # Densidad edificatoria promedio
            densidad = area_total_construida / num_edificios

            # Crear raster con densidad constante
            densidad_raster = np.full(ibi.shape, densidad, dtype=np.float32)

            # Guardar raster manteniendo prefijo original
            output_name = raster_path.stem.replace("IndiceAreaConstruida", output_prefix) + ".tif"
            output_path = self.output_dir / output_name
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(densidad_raster, 1)

            print(f"✅ Raster de densidad edificatoria guardado en: {output_path}")
