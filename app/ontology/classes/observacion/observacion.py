from pathlib import Path

from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar
from app.ontology.classes.variable.biofisica.ndvi import NDVI
from app.ontology.classes.variable.biofisica.ndbi import NDBI
from app.ontology.classes.variable.biofisica.albedo import Albedo
from app.ontology.classes.variable.biofisica.lst import LST
from app.ontology.classes.variable.biofisica.mnwi import MNDWI
from app.ontology.classes.variable.climatica.temperatura_del_aire_sintetica import TAIRPipeline
from app.ontology.classes.variable.climatica.humedad_relativa_sintetica import HRPipeline
from app.ontology.classes.UHI.UHI import UHIFullYearPipeline, UHIPipelineConfig

from app.ontology.classes.producto_analitico.subir_geojson import SubirGeoJSONIslasCalor


def has(dir_: Path, pattern: str) -> bool:
    return dir_.exists() and any(dir_.glob(pattern))


def pick_geojson_by_month(icu_poly_dir: Path, year: int) -> Path:
    """
    Prioriza el nombre nuevo:
      ICU_ISOBANDS_<YEAR>_ANUAL_BY_MONTH.geojson

    Fallbacks (por compatibilidad):
      ICU_SCORE_<YEAR>_ISOBANDS_ANUAL_BY_MONTH.geojson
      *<YEAR>*ANUAL*BY_MONTH*.geojson  (el primero que encuentre)
    """
    candidates = [
        icu_poly_dir / f"ICU_ISOBANDS_{year}_ANUAL_BY_MONTH.geojson",
        icu_poly_dir / f"ICU_SCORE_{year}_ISOBANDS_ANUAL_BY_MONTH.geojson",
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = sorted(icu_poly_dir.glob(f"*{year}*ANUAL*BY_MONTH*.geojson"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        "No se encontró el GeoJSON anual by month en:\n"
        f"{icu_poly_dir}\n\n"
        "Esperados (en orden):\n"
        f" - ICU_ISOBANDS_{year}_ANUAL_BY_MONTH.geojson\n"
        f" - ICU_SCORE_{year}_ISOBANDS_ANUAL_BY_MONTH.geojson\n"
        f" - *{year}*ANUAL*BY_MONTH*.geojson"
    )


def main(
    base: str,
    year: int = 2021,
    overwrite: bool = False,
    dst_crs: str = "EPSG:4326",
    px: float = 0.000009,
    ref_align: str = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    upl_shp: str = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\otros_Insumos\01_temperatura_del_aire\01_shp\Upl_Modificada.shp",
    tabla_destino: str = "islas_de_calor",
    if_exists: str = "append",
    replace_year: bool = True,
):
    base = Path(base)

    # -------------------------
    # Carpetas base de trabajo
    # -------------------------
    repro_dir, idx_dir = base / "01_reproyectadas", base / "02_Calculos"
    tair_out, hr_out = base / "04_Temperatura_Aire", base / "05_humedad_relativa"
    repro_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Reproyección
    # -------------------------
    if overwrite or not has(repro_dir, "*.tif"):
        Reproyectar(str(base), str(repro_dir), dst_crs=dst_crs).procesar_carpeta()

    # -------------------------
    # 2) Índices biofísicos
    # -------------------------
    for cls, pat in [
        (NDVI, "*_NDVI.tif"),
        (NDBI, "*_NDBI.tif"),
        (Albedo, "*_ALBEDO.tif"),
        (LST, "*_LST.tif"),
        (MNDWI, "*_MNDWI.tif"),
    ]:
        if overwrite or not has(idx_dir, pat):
            cls(input_dir=str(repro_dir), output_dir=str(idx_dir)).calculate()

    # -------------------------
    # 3) Variables climáticas sintéticas
    # -------------------------
    TAIRPipeline(input_root=base, output_root=tair_out, year=year, overwrite=overwrite, px=px).run()
    HRPipeline(input_root=base, output_root=hr_out, year=year, overwrite=overwrite).run()

    # -------------------------
    # 4) Pipeline UHI completo
    # -------------------------
    UHIFullYearPipeline(
        UHIPipelineConfig(
            year=str(year),
            root=str(base),
            ref_align=ref_align,
            upl_shp=upl_shp,
            overwrite=overwrite,
            som_rows=10,
            som_cols=10,
        )
    ).run_all()

    # -------------------------
    # 5) SUBIR GEOJSON a PostGIS (DINÁMICO)
    # -------------------------
    icu_poly_dir = base / "06_UHI" / "06_salidas_SOM" / str(year) / "ICU_ISOBANDS_POLY"
    geojson_path = pick_geojson_by_month(icu_poly_dir, year)

    uploader = SubirGeoJSONIslasCalor(tabla_destino=tabla_destino, if_exists=if_exists)
    n = uploader.subir_geojson(
        ruta_geojson=str(geojson_path),
        replace_year=replace_year,
        year=str(year),
    )

    print("✅ Listo:", n)
    print("📄", geojson_path)


if __name__ == "__main__":
    main(
        base=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2023",
        year=2023,
        overwrite=False,
    )
