from app.ontology.classes.observacion.observacion_final import ObservacionPipeline, ObservacionConfig
from app.ontology.classes.producto_analitico.producto_uhi import ProductoUHIPipeline, ProductoUHIConfig

def main():
    base = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022"
    year = 2022
    overwrite = False

   
    ObservacionPipeline(
        ObservacionConfig(base=base, year=year, overwrite=overwrite)
    ).run()

    # PRODUCTO ANALÍTICO (4–5)
    out = ProductoUHIPipeline(
        ProductoUHIConfig(
            base=base,
            year=year,
            overwrite=overwrite,
            ref_align=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
            upl_shp=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\otros_Insumos\01_temperatura_del_aire\01_shp\Upl_Modificada.shp",
            tabla_destino="islas_de_calor",
            if_exists="append",
            replace_year=True,
        )
    ).run()

    print("✅ Listo:", out["rows"])
    print("📄", out["geojson_path"])

if __name__ == "__main__":
    main()
