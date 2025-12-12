import os
from app.ontology.classes.datos_sinteticos.ds_04_densidad_vegetacion import VegetationDensity3m
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

vegetacion_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "vegetacion.tif")
salida_veg_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "porcentaje_vegetacion_3m.tif")


veg = VegetationDensity3m(vegetacion_1m)
veg.compute_density()
veg.save_raster(salida_veg_3m)

print("Porcentaje de vegetación generado en:", salida_veg_3m)