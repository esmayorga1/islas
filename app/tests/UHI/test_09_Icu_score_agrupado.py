from pathlib import Path
from app.ontology.classes.UHI.UHI_09_Icu_score_agrupado import ICUIsobandPolygonGenerator

base = Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM\2022")
in_dir = base / "ICU_SCORE_AGRUPADO"
out_dir = base / "ICU_ISOBANDS_POLY"
out_dir.mkdir(parents=True, exist_ok=True)

gen = ICUIsobandPolygonGenerator(
    nodata=-1,
    min_class=3,           # ✅ foco fuerte
    decay_m=60.0,
    smooth_sigma_m=18.0,   # ✅ agrupa más
    band_edges=[0.25, 0.40, 0.55, 0.70, 0.85],
    dissolve=True          # ✅ 1 polígono por banda
)

for f in sorted(in_dir.glob("ICU_SCORE_2022_??_10x10_AGRUPADO.tif")):
    out_geojson = out_dir / f"{f.stem}_ISOBANDS_DISSOLVE.geojson"
    gen.run(str(f), str(out_geojson))
    print("✅", out_geojson.name)
