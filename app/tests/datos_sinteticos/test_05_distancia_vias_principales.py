import os
from app.ontology.classes.datos_sinteticos.ds_05_distancia_vias import DistanceToRoads
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

raster_vias_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "vias_principales_1m.tif")
salida_distancias = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "distancia_vias_3m.tif")


dist = DistanceToRoads(
    roads_raster_1m_path=raster_vias_1m,
    output_resolution=3  # salida a 3 metros
)

dist.compute_distance()
dist.save_raster(salida_distancias)

print("Mapa de distancias generado:", salida_distancias)
