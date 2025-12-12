import os
from app.ontology.classes.datos_sinteticos.ds_03_porcentaje_construccion import UrbanDensity3m
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

construcciones_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "construcciones.tif")
vias_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "vias.tif")
salida_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "porcentaje_urbano_3m.tif")


urban = UrbanDensity3m(construcciones_1m, vias_1m)
urban.compute_density()
urban.save_raster(salida_3m)

print("Porcentaje urbano generado en:", salida_3m)
