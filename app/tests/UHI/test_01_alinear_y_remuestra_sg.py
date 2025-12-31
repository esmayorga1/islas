from app.ontology.classes.UHI.UHI_01_Alinear_Y_Remuestrar import FolderTo3mResampler4326

res = FolderTo3mResampler4326(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\Calculos",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022\Calculos2",
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",  
    mode="direct",
    overwrite=False
)

salidas = res.process_folder(
    apply_clip=True,
    clip_range=(0.0, 1.0),
    kind_mode="auto"
)

print("Generados:", len(salidas))
