from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from app.ontology.classes.UHI.UHI import UHIFullYearPipeline, UHIPipelineConfig
from app.ontology.classes.producto_analitico.subir_geojson import SubirGeoJSONIslasCalor


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


@dataclass
class ProductoUHIConfig:
    base: str
    year: int = 2021
    overwrite: bool = False
    ref_align: str = ""
    upl_shp: str = ""
    tabla_destino: str = "islas_de_calor"
    if_exists: str = "append"
    replace_year: bool = True
    som_rows: int = 10
    som_cols: int = 10


class ProductoUHIPipeline:
    """
    PRODUCTO_ANALITICO:
    - Ejecuta UHI (SOM, ICU, isobandas, etc.)
    - Obtiene el GeoJSON final anual_by_month
    - Sube el GeoJSON a PostGIS
    """

    def __init__(self, cfg: ProductoUHIConfig):
        self.cfg = cfg
        self.base = Path(cfg.base)

    def run(self) -> dict:
        # 4) Pipeline UHI completo
        UHIFullYearPipeline(
            UHIPipelineConfig(
                year=str(self.cfg.year),
                root=str(self.base),
                ref_align=self.cfg.ref_align,
                upl_shp=self.cfg.upl_shp,
                overwrite=self.cfg.overwrite,
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
            )
        ).run_all()

        # 5) SUBIR GEOJSON a PostGIS
        icu_poly_dir = self.base / "06_UHI" / "06_salidas_SOM" / str(self.cfg.year) / "ICU_ISOBANDS_POLY"
        geojson_path = pick_geojson_by_month(icu_poly_dir, self.cfg.year)

        uploader = SubirGeoJSONIslasCalor(
            tabla_destino=self.cfg.tabla_destino,
            if_exists=self.cfg.if_exists
        )
        n = uploader.subir_geojson(
            ruta_geojson=str(geojson_path),
            replace_year=self.cfg.replace_year,
            year=str(self.cfg.year),
        )

        return {"icu_poly_dir": icu_poly_dir, "geojson_path": geojson_path, "rows": n}
