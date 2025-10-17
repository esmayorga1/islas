import geopandas as gpd
from sqlalchemy import create_engine
from app.config.settings import settings  # ✅ Importa las credenciales desde el backend


def subir_islas_calor(ruta_shp, tabla_destino="islas_de_calor"):
    """
    Sube un shapefile de islas de calor a la base de datos PostGIS usando las credenciales
    definidas en el archivo .env y gestionadas por settings.py.
    """

    # ✅ Usa la URL generada automáticamente por settings.database_url
    engine = create_engine(settings.database_url)

    print(f"📂 Cargando shapefile: {ruta_shp}")
    gdf = gpd.read_file(ruta_shp)

    # Verificar CRS y estandarizar
    if gdf.crs is None:
        print("⚠️ El shapefile no tiene CRS definido. Se asigna EPSG:4326 por defecto.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"🔄 Reproyectando de {gdf.crs.to_string()} a EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Subir al esquema especificado en settings
    schema = settings.DB_SCHEMA

    print(f"🚀 Subiendo a la base de datos: {settings.DB_NAME} ({schema}.{tabla_destino}) ...")
    gdf.to_postgis(
        name=tabla_destino,
        con=engine,
        schema=schema,
        if_exists="replace",  # o "append" si no quieres borrar la tabla anterior
        index=False
    )

    print(f"✅ Capa '{tabla_destino}' subida correctamente a {settings.DB_NAME}.{schema}")
