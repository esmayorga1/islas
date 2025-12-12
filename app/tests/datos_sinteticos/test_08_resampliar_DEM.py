import os
from app.ontology.classes.datos_sinteticos.ds_08_resampliar_DEM import RasterResampler3mWGS84
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

# Rutas del DEM original (EPSG:4326) y salida remuestreada a ~3 metros equivalentes (en grados)
dem_4_5m = os.path.join(ruta_base, "01_temperatura_del_aire", "01_insumos", "03_DEM", "dem_zona.tif")
dem_3m = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "dem_3m.tif")

# Crear instancia del remuestreador
resampler = RasterResampler3mWGS84(
    raster_path=dem_4_5m,
    method="bilinear"  
)

# Ejecutar remuestreo
resampler.resample()

# Guardar raster final
resampler.save(dem_3m)

print("DEM remuestreado directamente en EPSG:4326 a tamaño equivalente a 3 m:", dem_3m)
