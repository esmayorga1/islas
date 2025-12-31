from app.ontology.classes.UHI.UHI_00_normalizar_raster import RasterNormalizer01

norm = RasterNormalizer01(
    input_dir=r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\01_temperatura_del_aire\02_resultados\01_resultado_raster\alineados\normalizar",
    output_dir=r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\01_temperatura_del_aire\02_resultados\01_resultado_raster\alineados\normalizar\normalizar",
    overwrite=False
)

salidas = norm.process_folder()
print("Normalizados:", len(salidas))