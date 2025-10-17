import geopandas as gpd

class SuavizadorIslasCalor:
    """
    Clase para suavizar los bordes de las islas de calor vectorizadas,
    eliminando la apariencia pixelada sin alterar significativamente el área.
    """

    def __init__(self, ruta_shp_entrada: str, ruta_shp_salida: str, buffer_metros: float = 30, tolerancia: float = 10):
        """
        Parámetros:
        - ruta_shp_entrada: Ruta del shapefile de entrada (islas de calor originales)
        - ruta_shp_salida: Ruta donde se guardará el shapefile suavizado
        - buffer_metros: Distancia del buffer positivo/negativo (en metros)
        - tolerancia: Tolerancia de simplificación (en metros)
        """
        self.ruta_shp_entrada = ruta_shp_entrada
        self.ruta_shp_salida = ruta_shp_salida
        self.buffer_metros = buffer_metros
        self.tolerancia = tolerancia

    # ------------------------------------------------------------
    def suavizar(self):
        """Aplica suavizado a los polígonos del shapefile."""
        print(f"🌀 Suavizando bordes del shapefile: {self.ruta_shp_entrada}")

        # Leer shapefile
        gdf = gpd.read_file(self.ruta_shp_entrada)

        # Reproyectar a sistema en metros (para aplicar buffer correctamente)
        gdf = gdf.to_crs("EPSG:3116")

        # Aplicar buffer positivo y negativo para suavizar bordes
        gdf["geometry"] = gdf.geometry.buffer(self.buffer_metros)
        gdf["geometry"] = gdf.geometry.buffer(-self.buffer_metros)

        # Aplicar simplificación adicional
        gdf["geometry"] = gdf.geometry.simplify(self.tolerancia, preserve_topology=True)

        # Volver a CRS original (EPSG:4326)
        gdf = gdf.to_crs("EPSG:4326")

        # Guardar shapefile suavizado
        gdf.to_file(self.ruta_shp_salida)

        print(f"✅ Shapefile suavizado guardado en: {self.ruta_shp_salida}")
        print(f"📉 Parámetros usados → buffer: {self.buffer_metros} m | tolerancia: {self.tolerancia} m")