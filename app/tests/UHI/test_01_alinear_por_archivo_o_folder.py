from app.ontology.classes.UHI.UHI_01_Alinear_por_archivo_o_folder import GridAlignerFile

aligner = GridAlignerFile(
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    overwrite=False,
    suffix="_ALINEADO",
)

# ----------------- Construcciones -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\construcciones.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\construcciones_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

# ----------------- Cuerpos de agua -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\cuerpos_de_agua.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\cuerpos_de_agua_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

# ----------------- Vías -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\vias.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\01_resultado_raster\vias_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)
