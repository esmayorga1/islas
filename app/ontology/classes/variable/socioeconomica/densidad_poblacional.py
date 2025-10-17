import os
import sys
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
from sqlalchemy import create_engine
from app.core.config import settings

# === 🔧 CONFIGURAR PROJ CORRECTAMENTE ===
possible_proj_paths = [
    os.path.join(os.path.dirname(sys.executable), "Library", "share", "proj"),  # conda
    os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "rasterio", "proj_data"),  # venv
    os.path.join(os.path.dirname(rasterio.__file__), "proj_data")  # fallback
]

proj_path = next((p for p in possible_proj_paths if os.path.exists(os.path.join(p, "proj.db"))), None)

if proj_path:
    os.environ["PROJ_LIB"] = proj_path
    print(f"✅ PROJ_LIB configurado correctamente: {proj_path}")
else:
    raise RuntimeError("❌ No se encontró proj.db válida. Rasterio fallará al reproyectar.")


class DensidadPoblacionalDB:
    """
    Variable socioeconómica: densidad poblacional a partir de tabla en PostGIS.
    """

    def __init__(self, output_dir, pixel_size_m=10):
        """
        pixel_size_m: tamaño del píxel en metros (por defecto 10 m, se convierte a grados automáticamente)
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        db_url = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        self.engine = create_engine(db_url)

        # Convertir metros a grados (aprox)
        self.pixel_size = pixel_size_m / 111320  # 1° ≈ 111.32 km

    def load_data(self, table_name="censo_2018", geom_col="geom", value_col="personas"):
        """
        Carga geometrías y valores desde la BD.
        """
        query = f"SELECT {geom_col}, {value_col} FROM {table_name}"
        gdf = gpd.read_postgis(query, self.engine, geom_col=geom_col)
        print(f"📌 Datos cargados: {len(gdf)} geometrías")
        return gdf, value_col

    def rasterize(self, raster_name="censo_2018_raster.tif",
                  table_name="censo_2018", geom_col="geom", value_col="personas"):
        """
        Genera un raster a partir de los datos de la BD.
        """

        gdf, val_col = self.load_data(table_name, geom_col, value_col)

        if gdf.empty:
            raise ValueError("⚠️ No se encontraron geometrías para rasterizar.")

        crs = "EPSG:4326"

        minx, miny, maxx, maxy = gdf.total_bounds

        width = max(int((maxx - minx) / self.pixel_size), 1)
        height = max(int((maxy - miny) / self.pixel_size), 1)

        transform = from_origin(minx, maxy, self.pixel_size, self.pixel_size)

        raster_path = os.path.join(self.output_dir, raster_name)
        print(f"📁 Raster de salida: {raster_path}")
        print(f"📏 Tamaño del raster: {width}x{height} píxeles (~10 m/pixel)")

        burned = rasterize(
            [(geom, value) for geom, value in zip(gdf.geometry, gdf[val_col])],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='float32'
        )

        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=burned.shape[0],
            width=burned.shape[1],
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(burned, 1)

        print("✅ Rasterización completada.")
        return raster_path
