from app.ontology.classes.variable.socioeconomica.densidad_poblacional import DensidadPoblacionalDB


# Carpeta de salida
output_dir = r"F:\LAURA COTE\CAPAS RASTER\INDICES"

# Crear instancia, pixel de 15 metros
dp = DensidadPoblacionalDB(output_dir, pixel_size_m=15)

# Rasterizar usando la tabla censo_2018
raster_path = dp.rasterize(
    raster_name="censo_2018_raster.tif",
    table_name="censo_2018",
    geom_col="geom",
    value_col="personas"
)

print(f"✅ Raster generado en: {raster_path}")
