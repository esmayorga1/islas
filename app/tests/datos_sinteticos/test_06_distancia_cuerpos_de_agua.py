import os
from app.ontology.classes.datos_sinteticos.ds_06_distancia_cuerpos_agua import DistanceToWater
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

raster_agua_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "cuerpos_de_agua.tif")
salida_dist_agua_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "distancia_agua_3m.tif")

dist_agua = DistanceToWater(
    water_raster_1m_path=raster_agua_1m,
    output_resolution=3  # salida en 3 metros
)

dist_agua.compute_distance()
dist_agua.save_raster(salida_dist_agua_3m)

print("Mapa de distancia a agua generado:", salida_dist_agua_3m)

import os
from app.ontology.classes.datos_sinteticos.ds_06_distancia_cuerpos_agua import DistanceToWater
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

raster_agua_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "cuerpos_de_agua.tif")
salida_dist_agua_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "distancia_agua_3m.tif")

dist_agua = DistanceToWater(
    water_raster_1m_path=raster_agua_1m,
    output_resolution=3  # salida en 3 metros
)

dist_agua.compute_distance()
dist_agua.save_raster(salida_dist_agua_3m)

print("Mapa de distancia a agua generado:", salida_dist_agua_3m)

