from app.ontology.classes.UHI.UHI_01_Alinear import FolderGridAligner


aligner = FolderGridAligner(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\05_normalizado_y_aliado",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\2022\01_temperatura_del_aire\01_insumos\02_resultados\05_normalizado_y_aliado",
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    overwrite=False,
    suffix="_3m",  # si quieres mantener el sufijo que ya manejas
)

outputs = aligner.process_folder(kind_mode="auto", copy_if_already_aligned=True)
print("Generados:", len(outputs))
