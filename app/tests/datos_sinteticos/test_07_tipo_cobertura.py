import os
from app.ontology.classes.datos_sinteticos.ds_07_tipo_de_cobertura import CoverageClassifier3m
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

agua_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "cuerpos_de_agua.tif")
vegetacion_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "vegetacion.tif")
urbano_1m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "construcciones.tif")
salida_cover_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "tipo_cobertura_3m.tif")

classifier = CoverageClassifier3m(
    raster_agua_1m=agua_1m,
    raster_vegetacion_1m=vegetacion_1m,
    raster_urbano_1m=urbano_1m,
    factor=3
)

classifier.aggregate_to_3m()
classifier.save_raster(salida_cover_3m)

print("Mapa de cobertura generado:", salida_cover_3m)
