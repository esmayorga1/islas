from pathlib import Path
from app.ontology.classes.UHI.UHI_11_Icu_score_agrupado_v2 import ICUIsobandPolygonGenerator

# =====================================================
# CONFIGURACIÓN MÍNIMA
# =====================================================
BASE = Path(
    r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\06_UHI\06_salidas_SOM\2021"
)

YEAR = "2021"

UPL_SHP = Path(
    r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\otros_Insumos\01_temperatura_del_aire\01_shp\Upl_Modificada.shp"
)

# =====================================================
# VALIDACIONES
# =====================================================
if not BASE.exists():
    raise FileNotFoundError(f"BASE no existe:\n{BASE}")

if not UPL_SHP.exists():
    raise FileNotFoundError(f"UPL_SHP no existe:\n{UPL_SHP}")

# =====================================================
# RUTAS
# =====================================================
IN_DIR = BASE / "ICU_SCORE_AGRUPADO"
OUT_DIR = BASE / "ICU_ISOBANDS_POLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not IN_DIR.exists():
    raise FileNotFoundError(f"IN_DIR no existe:\n{IN_DIR}")

# =====================================================
# GENERADOR
# =====================================================
gen = ICUIsobandPolygonGenerator(
    nodata=-1,
    min_class=3,
    pixel_size_m=3.0,
    decay_m=60.0,
    smooth_sigma_m=18.0,

    # 🔹 Isobandas / gradiente
    band_edges=[0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85],

    # 🔹 UPL (máscara + clip final)
    upl_path=str(UPL_SHP),
    clip_polygons_to_upl=True,

    # 🔹 comportamiento
    dissolve=True,      # mensual: 1 polígono por banda
    drop_class0=False,  # conserva banda 0 (cobertura total)
)

# =====================================================
# 1) GEOJSON MENSUALES (uno por mes)
# =====================================================
pattern = f"ICU_SCORE_{YEAR}_??_10x10_AGRUPADO.tif"
print("▶ Buscando TIF mensuales con patrón:", pattern)

monthly = gen.run_folder(
    in_dir=str(IN_DIR),
    out_dir=str(OUT_DIR),
    pattern=pattern,
)

print(f"\n✅ GeoJSON mensuales generados: {len(monthly)}")
for f in monthly:
    print("  -", Path(f).name)

# =====================================================
# 2) GEOJSON ANUAL **MES A MES** (SOLUCIÓN REAL)
# =====================================================
out_year_by_month = OUT_DIR / f"ICU_SCORE_{YEAR}_ISOBANDS_ANUAL_BY_MONTH.geojson"

gen.run_year_by_month(
    in_dir=str(IN_DIR),
    year=YEAR,
    out_geojson_year=str(out_year_by_month),
    dissolve_per_month=True,  # ✅ 1 polígono por banda POR MES
)

print("\n✅ Anual MES A MES generado:")
print("  -", out_year_by_month.name)

# =====================================================
# (OPCIONAL) 3) ANUAL TOTAL DISUELTO (month = 'ALL')
# =====================================================
# Si lo necesitas para mapas de resumen
out_year_unified = OUT_DIR / f"ICU_SCORE_{YEAR}_ISOBANDS_ANUAL_UNIFIED.geojson"

gen.run_year_unified(
    in_dir=str(IN_DIR),
    year=YEAR,
    out_geojson_year=str(out_year_unified),
)

print("\n✅ Anual UNIFICADO generado:")
print("  -", out_year_unified.name)

print("\n🎉 PROCESO COMPLETO – ISOBANDAS CORRECTAS")
