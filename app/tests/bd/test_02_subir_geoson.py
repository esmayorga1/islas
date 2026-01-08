from app.ontology.classes.producto_analitico.subir_geojson import SubirGeoJSONIslasCalor

uploader = SubirGeoJSONIslasCalor(
    tabla_destino="islas_de_calor",
    if_exists="append",   # siempre append; el reemplazo lo hace el DELETE
)

n = uploader.subir_geojson(
    ruta_geojson=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2024\06_UHI\06_salidas_SOM\2024\ICU_ISOBANDS_POLY.geojson",
    replace_year=True,     
    year="2024",           
)
print("Listo:", n)
