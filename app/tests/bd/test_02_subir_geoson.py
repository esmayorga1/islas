from app.ontology.classes.producto_analitico.subir_geojson import SubirGeoJSONIslasCalor

uploader = SubirGeoJSONIslasCalor(
    tabla_destino="islas_de_calor",
    if_exists="append",   # siempre append; el reemplazo lo hace el DELETE
)

n = uploader.subir_geojson(
    ruta_geojson=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM\2022\ICU_ISOBANDS_POLY\ICU_SCORE_2022_MERGE_ALL_MONTHS_ISOBANDS_DISSOLVE.geojson",
    replace_year=True,     # ✅ BORRA TODO 2022 y luego INSERTA
    year="2022",           # ✅ explícito para que no falle
)
print("Listo:", n)
