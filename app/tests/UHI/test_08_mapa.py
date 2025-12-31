from pathlib import Path
from app.ontology.classes.UHI.UHI_08_icu_raster_to_polygon import ICURasterToPolygonDissolve

def main():
    year = "2021"
    month_key = "2021_02"

    root = Path(
        r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2"
        rf"\{year}\{year}_salida\SALIDAS_SOM\{year}"
    )

    icu_tif = root / "ICU_SCORE" / f"ICU_SCORE_{month_key}_10x10.tif"
    out_vec = root / "ICU_POLYGONS" / f"ICU_POLY_{month_key}_high.geojson"

    tool = ICURasterToPolygonDissolve(
        threshold=3,
        mode="eq",               # ==3
        min_area_m2=100.0,       # ajusta según tu escala
        simplify_tolerance_m=2.0 # opcional
    )

    saved = tool.run(
        icu_tif=str(icu_tif),
        out_vector_path=str(out_vec),
        month_key=month_key,
        year=year
    )

    print("✅ Polígono ICU guardado:", saved)

if __name__ == "__main__":
    main()
