import os
from app.ontology.classes.datos_sinteticos.ds_02_mapa_pendientes import SlopeGenerator
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

print("Generando mapa de pendientes a partir del DEM")

dem_path = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "dem_3m.tif")
output_slope = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "slope_3m.tif")


slope_gen = SlopeGenerator(dem_path)
slope_gen.compute_slope()
slope_gen.save_slope(output_slope)

print("Mapa de pendientes generado en:", output_slope)
