import os
from app.ontology.classes.datos_sinteticos.ds_01_convertir_shp_raster import ShapefileToRaster

# Ruta base donde se encuentran los SHP y donde se guardarán los rasters
ruta_inicial = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"
ruta_base = os.path.join(ruta_inicial, "01_temperatura_del_aire", "01_insumos", "01_shp")
ruta_base_salida_base = os.path.join(ruta_inicial, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster")

# SHP de entrada
ruta_shp_cuerpos_de_agua = os.path.join(ruta_base, "Cuerpos_De_Agua.shp")
ruta_shp_upl = os.path.join(ruta_base, "Upl_Modificada.shp")
shp_vias = os.path.join(ruta_base, "vias.shp")
shp_construcciones = os.path.join(ruta_base, "construcciones2.shp")
shp_vegetacion = os.path.join(ruta_base, "Vegetacion2.shp")
shp_vias_principales = os.path.join(ruta_base, "vias_Principales.shp")

# Rutas de salida (rasters generados)
ruta_salida_raster_cuerpos_de_agua = os.path.join(ruta_base_salida_base, "cuerpos_de_agua.tif")
raster_vias = os.path.join(ruta_base_salida_base, "vias.tif")
raster_construcciones = os.path.join(ruta_base_salida_base, "construcciones.tif")
raster_vegetacion = os.path.join(ruta_base_salida_base, "vegetacion.tif")
raster_vias_principales = os.path.join(ruta_base_salida_base, "vias_principales_1m.tif")

# Convirtiendo de SHP a Raster para cuerpos de agua
print("Convirtiendo de SHP a Raster")

converter = ShapefileToRaster(
    shp_path=ruta_shp_cuerpos_de_agua, 
    shp_referencia_path=ruta_shp_upl,   # Dimensiones EXACTAS según UPL
    pixel_size=0.000009,                # ~1 metro
    raster_value=1
)
output = converter.rasterize_shp(ruta_salida_raster_cuerpos_de_agua)
print("Raster generado en:", output)

# Convirtiendo SHP de vías a raster
converter = ShapefileToRaster(
    shp_path=shp_vias,
    shp_referencia_path=ruta_shp_upl,   # Dimensiones EXACTAS según UPL
    pixel_size=0.000009,                # ~1 metro
    raster_value=1
)
output = converter.rasterize_shp(raster_vias)
print("Raster generado en:", output)

# Convirtiendo SHP de construcciones a raster
converter = ShapefileToRaster(
    shp_path=shp_construcciones,
    shp_referencia_path=ruta_shp_upl,   # Dimensiones EXACTAS según UPL
    pixel_size=0.000009,                # ~1 metro
    raster_value=1
)
output = converter.rasterize_shp(raster_construcciones)
print("Raster generado en:", output)

# Convirtiendo SHP de vegetación a raster
converter = ShapefileToRaster(
    shp_path=shp_vegetacion,
    shp_referencia_path=ruta_shp_upl,   # Dimensiones EXACTAS según UPL
    pixel_size=0.000009,                # ~1 metro
    raster_value=1
)
output = converter.rasterize_shp(raster_vegetacion)
print("Raster generado en:", output)

# Convirtiendo SHP de vías principales a raster
converter = ShapefileToRaster(
    shp_path=shp_vias_principales, 
    # Sin SHP de referencia → dimensiones según el SHP original
    pixel_size=0.00001,                # ~1 metro
    raster_value=1
)
output = converter.rasterize_shp(raster_vias_principales)
print("Raster generado en:", output)
print("Conversión de SHP a Raster completada.")