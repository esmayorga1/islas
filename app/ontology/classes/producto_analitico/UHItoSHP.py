import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

class UHItoSHP:
    """
    Genera un shapefile con un solo polígono por mes (fusionado),
    a partir de los rasters mensuales del índice UHI.
    """

    def __init__(self, ruta_uhi: str, anio: int, salida_shp: str, umbral: float = 0.6):
        self.ruta_uhi = ruta_uhi
        self.anio = anio
        self.salida_shp = salida_shp
        self.umbral = umbral

    # ------------------------------------------------------------
    def generar(self):
        """Genera el shapefile consolidado de islas de calor fusionadas por mes."""
        print(f"🔥 Generando shapefile fusionado de islas de calor para {self.anio}...")

        registros_mes = []

        for mes in range(1, 13):
            archivo = os.path.join(self.ruta_uhi, f"UHI_{self.anio}_{mes:02d}.tif")
            if not os.path.exists(archivo):
                print(f"⚠️ No se encontró {archivo}, se omite.")
                continue

            print(f"➡️ Procesando {os.path.basename(archivo)}...")

            with rasterio.open(archivo) as src:
                data = src.read(1).astype(np.float32)
                transform = src.transform

                # Reemplazar NoData y valores no válidos
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Crear máscara
                mask = (data > self.umbral)
                total_pix = np.sum(mask)
                print(f"   🔹 Píxeles sobre el umbral ({self.umbral}): {total_pix}")

                if total_pix == 0:
                    print(f"   ⚠️ Ningún píxel supera el umbral en {os.path.basename(archivo)}.")
                    continue

                # Vectorizar zonas calientes
                shapes_gen = shapes(data, mask=mask, transform=transform)

                geometries = []
                for geom, value in shapes_gen:
                    geom_shape = shape(geom)
                    if geom_shape.is_valid and geom_shape.area > 0:
                        geometries.append(geom_shape)

                # Unir todos los polígonos del mes en uno solo
                if geometries:
                    gdf_mes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geometries), crs="EPSG:4326")
                    gdf_union = gdf_mes.dissolve()  # fusiona todos los polígonos
                    merged_geom = gdf_union.geometry.iloc[0]

                    registros_mes.append({
                        "geometry": merged_geom,
                        "mes": mes,
                        "anio": self.anio
                    })

                    print(f"   ✅ Polígonos fusionados para mes {mes}: {len(geometries)} → 1")

        # ------------------------------------------------------------
        if not registros_mes:
            print("⚠️ No se detectaron zonas de calor en ningún mes.")
            return

        # Crear shapefile final con un polígono por mes
        gdf_final = gpd.GeoDataFrame(registros_mes, crs="EPSG:4326")
        gdf_final.to_file(self.salida_shp)

        print(f"\n✅ Shapefile fusionado generado: {self.salida_shp}")
        print(f"📊 Total de polígonos (uno por mes): {len(gdf_final)}")
