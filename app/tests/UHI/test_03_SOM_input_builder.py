import os
import rasterio as rio
from app.ontology.classes.UHI.UHI_03_SOM_input_builder import SOMInputBuilder

# --------------------------------------------------
# ✅ GRILLA MAESTRA (REFERENCIA ÚNICA)
# --------------------------------------------------
MASTER_GRID = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif"

input_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2022"
out_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI"
os.makedirs(out_dir, exist_ok=True)

builder = SOMInputBuilder(
    input_dir=input_dir,

   
    month_regex=r"(20\d{2})[_-](0?[1-9]|1[0-2])(?=[_-]|$)",

    monthly_patterns={
        "NDVI": r"NDVI",
        "NDBI": r"NDBI",
        "ALBEDO": r"ALBEDO",
        "LST": r"LST",
        "MNDWI": r"MNDWI|MNWI",
        "TAIR": r"TAIR|TEMP|AIRE",
        "HR": r"HR|HUM|RH",
    },

    static_patterns={
        "CONSTRUCCIONES_BIN": r"construcciones_ALINEADO",
        "CUERPOS_AGUA_BIN": r"cuerpos_de_agua_ALINEADO",
        "VEGETACION_BIN": r"vegetacion_ALINEADO",
        "VIAS_BIN": r"vias_ALINEADO",
        "PORC_URBANO": r"porcentaje_urbano_3m_ALINEADO",
        "DIST_AGUA_NORM": r"^distancia_agua_3m_ALINEADO_NORMALIZADO",
        "DIST_VIAS_NORM": r"distancia_vias_3m_ALINEADO_NORMALIZADO",
    },

    mask_mode="all_valid",
    sample_max_pixels=200_000,
    allow_missing_monthly_vars=False,
    require_static=True,
)

# ✅ Diagnóstico (mensuales + estáticas)
diag = builder.diagnose("2022")
print("DIAG:", diag)


# =====================================================================
# ✅ VALIDACIÓN CONTRA GRILLA MAESTRA (MASTER_GRID)
#     - NO usa NDVI como ref
#     - compara TODO contra MASTER_GRID
# =====================================================================

def validate_against_master(builder: SOMInputBuilder, year: str = "2022", master_path: str = MASTER_GRID):
    monthly = builder._collect_monthly()
    static_map = builder._collect_static()

    with rio.open(master_path) as m:
        m_w, m_h = m.width, m.height
        m_tr = m.transform
        m_crs = str(m.crs)

    print("\n" + "=" * 80)
    print("🔎 VALIDACIÓN DE GRILLAS (mensuales + estáticas) CONTRA GRILLA MAESTRA")
    print("=" * 80)
    print("MASTER:", master_path)
    print(f"MASTER GRID: {m_w}x{m_h} | {m_crs}")

    any_problem = False

    for month_key, var_map in sorted(monthly.items()):
        if not month_key.startswith(f"{year}_"):
            continue

        # unir mensuales + estáticas como hace build()
        merged = dict(var_map)
        if static_map:
            merged.update(static_map)

        # orden igual al builder: mensuales (en orden) + estáticas (en orden)
        ordered_vars = []
        for v in builder.monthly_patterns.keys():
            if v in merged:
                ordered_vars.append(v)
        for v in builder.static_patterns.keys():
            if v in merged:
                ordered_vars.append(v)

        if not ordered_vars:
            print(f"\n📅 Mes: {month_key}  ⚠️ (sin variables)")
            continue

        print(f"\n📅 Mes: {month_key}")
        print(f"  🔹 Ref MASTER → {m_w}x{m_h} | {m_crs}")

        # comparar
        for v in ordered_vars:
            p = merged[v]
            with rio.open(p) as ds:
                problems = []

                if (ds.width, ds.height) != (m_w, m_h):
                    problems.append(f"SIZE {ds.width}x{ds.height}")
                if ds.transform != m_tr:
                    problems.append("TRANSFORM")
                if str(ds.crs) != m_crs:
                    problems.append(f"CRS {ds.crs}")

            if problems:
                any_problem = True
                print(f"  - {v:<18} ❌ DIFERENTE → {', '.join(problems)}")
                print(f"      ↳ {p}")
            else:
                print(f"  - {v:<18} ✅ OK")

    print("\n" + "=" * 80)
    if any_problem:
        print("❌ Se detectaron capas con grilla distinta a MASTER. Alínea esas rutas y vuelve a ejecutar.")
    else:
        print("✅ Todo está alineado a MASTER. Puedes ejecutar build() sin errores de tamaño/transform/crs.")
    print("=" * 80 + "\n")


# Ejecuta el validador ANTES de build()
validate_against_master(builder, year="2022", master_path=MASTER_GRID)


# =====================================================================
# ⚠️ Si todo está OK, ahora sí construyes los NPZ
# =====================================================================

packs = builder.build()
packs_2022 = {k: v for k, v in packs.items() if k.startswith("2022_")}

for month_key, pack in sorted(packs_2022.items()):
    out_path = os.path.join(out_dir, f"SOM_INPUT_{month_key}.npz")
    builder.save_npz(out_path, pack["X"], pack["coords"], pack["meta"])
    print(f"✅ {month_key} → X shape: {pack['X'].shape}")

print("🎉 NPZ generados:", len(packs_2022))
