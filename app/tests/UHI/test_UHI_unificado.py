from app.ontology.classes.UHI.UHI import UHIFullYearPipeline, UHIPipelineConfig


if __name__ == "__main__":
    cfg = UHIPipelineConfig(
        year="2021",
        root=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021",
        ref_align=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
        upl_shp=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\otros_Insumos\01_temperatura_del_aire\01_shp\Upl_Modificada.shp",
        overwrite=False,
        som_rows=10,
        som_cols=10,
    )

    pipeline = UHIFullYearPipeline(cfg)
    pipeline.run_all()
