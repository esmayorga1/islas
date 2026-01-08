from pathlib import Path
import json

from app.ontology.classes.UHI.UHI_10_Icu_score_agrupado import ICUIsobandPolygonGenerator

base = Path(
    r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\06_UHI\06_salidas_SOM\2021"
)

YEAR = "2021"

in_dir = base / "ICU_SCORE_AGRUPADO"
out_dir = base / "ICU_ISOBANDS_POLY"
out_dir.mkdir(parents=True, exist_ok=True)

gen = ICUIsobandPolygonGenerator(
    nodata=-1,
    min_class=3,
    decay_m=60.0,
    smooth_sigma_m=18.0,
    band_edges=[0.25, 0.40, 0.55, 0.70, 0.85],
    dissolve=True  # <- esto es para el mensual (1 polígono por banda en cada mes)
)

# =========================
# 1) Generar GeoJSON mensuales (con year/month automático)
# =========================
monthly_geojsons = []

for tif in sorted(in_dir.glob(f"ICU_SCORE_{YEAR}_??_10x10_AGRUPADO.tif")):
    out_geojson_path = gen.run(in_icu_tif=str(tif), out_geojson=str(out_dir))
    monthly_geojsons.append(Path(out_geojson_path))
    print("✅", Path(out_geojson_path).name)

# =========================
# 2) MERGE (sin disolver): unir TODOS los GeoJSON mensuales en 1 solo
#    Mantiene atributos originales de cada feature (month/year/etc.)
# =========================
merged_features = []
crs = None

for gj in monthly_geojsons:
    with open(gj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if crs is None:
        crs = data.get("crs")

    feats = data.get("features", [])
    merged_features.extend(feats)

out_merged = out_dir / f"ICU_SCORE_{YEAR}_MERGE_ALL_MONTHS_ISOBANDS_DISSOLVE.geojson"

fc = {
    "type": "FeatureCollection",
    "name": out_merged.stem,
    "crs": crs,
    "properties": {
        "year": YEAR,
        "merge_type": "concat_features_keep_attributes",
        "merged_from": [p.name for p in monthly_geojsons],
    },
    "features": merged_features,
}

with open(out_merged, "w", encoding="utf-8") as f:
    json.dump(fc, f, ensure_ascii=False)

print("\n🎉 MERGE FINAL CREADO (SIN DISOLVER):")
print("✅", out_merged)
print("📌 Features total:", len(merged_features))
