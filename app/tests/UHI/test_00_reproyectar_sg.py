from app.ontology.classes.UHI.UHI_00_definir_proyectar import (
    Reproject9377_To4326_AlignAndClip
)

proc = Reproject9377_To4326_AlignAndClip(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\04_sg",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\04_sg",
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    overwrite=True,
    kind_mode="auto"
)

salidas = proc.process_folder()
print("Generados:", len(salidas))
