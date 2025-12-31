import os
import numpy as np
from pathlib import Path

from app.ontology.classes.UHI.UHI_00_normalizar_raster import RasterNormalizer01
from app.ontology.classes.UHI.UHI_00_normalizar_por_archivo import RasterNormalizer01File
from app.ontology.classes.UHI.UHI_01_Alinear import FolderGridAligner
from app.ontology.classes.UHI.UHI_01_Alinear_por_archivo_o_folder import GridAlignerFile
from app.ontology.classes.UHI.UHI_01_Alinear_Y_Remuestrar import FolderTo3mResampler4326

from app.ontology.classes.UHI.UHI_03_SOM_input_builder import SOMInputBuilder
from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner
from app.ontology.classes.UHI.UHI_05_cluster_geotiff_writer import ClusterGeoTIFFWriter
from app.ontology.classes.UHI.UHI_06_cluster_stats import ClusterStatsCalculator
from app.ontology.classes.UHI.UHI_07_icu_cluster_labeler import ICUClusterLabeler
from app.ontology.classes.UHI.UHI_08_ICUPostProcessor import ICUPostProcessor
from app.ontology.classes.UHI.UHI_09_Icu_score_agrupado import ICUIsobandPolygonGenerator


# =========================
# HELPERS
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def has_any(dir_path: Path, pattern: str) -> bool:
    return dir_path.exists() and any(dir_path.glob(pattern))

def run_if_needed(name: str, done: bool, overwrite: bool, fn):
    if done and not overwrite:
        print(f"⏭️  {name} ya existe → omitido")
        return None
    print(f"\n▶ {name}")
    return fn()


# =========================
# CONFIG
# =========================
YEAR = "2022"
OVERWRITE = False  # True = fuerza recalcular todo

ROOT = Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1")
REF_ALIGN = Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif")

UHI = ROOT / "06_UHI"

NORM_DIR     = UHI / "01_datos_normalizados"
ALIGNED_DIR  = UHI / "02_datos_normalizados_y_aliados"
RESAMP_DIR   = UHI / "03_alinear_resampliar"
NPZ_DIR      = UHI / "04_SOM_input_builder"
SOM_YEAR_DIR = UHI / "06_salidas_SOM" / YEAR

CLUSTERS_DIR = SOM_YEAR_DIR / "CLUSTERS_TIF"
STATS_DIR    = SOM_YEAR_DIR / "STATS"
ICU_DIR      = SOM_YEAR_DIR / "ICU_SCORE"
ICU_AGR_DIR  = SOM_YEAR_DIR / "ICU_SCORE_AGRUPADO"
ICU_POLY_DIR = SOM_YEAR_DIR / "ICU_ISOBANDS_POLY"

for d in [NORM_DIR, ALIGNED_DIR, RESAMP_DIR, NPZ_DIR, SOM_YEAR_DIR, CLUSTERS_DIR, STATS_DIR, ICU_DIR, ICU_AGR_DIR, ICU_POLY_DIR]:
    ensure_dir(d)

# Fuentes sintéticas
TAIR_SYN_DIR = ROOT / "04_Temperatura_Aire" / YEAR / "02_resultados" / "03_resultado_temperatura_sintetica"
HR_SYN_DIR   = ROOT / "05_humedad_relativa" / YEAR / "00_resultados_humedad_relativa_sintetica"

# Fuentes estáticas (pipeline TAIR)
TAIR_RAS_DIR = ROOT / "04_Temperatura_Aire" / YEAR / "02_resultados" / "01_resultado_raster"
TAIR_ALI_DIR = TAIR_RAS_DIR / "alineados"


# =========================
# 1) Normalizar TAIR + HR
# =========================
def step_norm(src_dir: Path):
    norm = RasterNormalizer01(input_dir=str(src_dir), output_dir=str(NORM_DIR), overwrite=OVERWRITE)
    outs = norm.process_folder()
    print(f"Normalizados {src_dir.name}:", len(outs))
    return outs

run_if_needed(
    "1a) Normalizar TAIR",
    done=has_any(NORM_DIR, f"*TAIR*{YEAR}*_norm_3m*.tif") or has_any(NORM_DIR, f"*TAIRE*{YEAR}*_norm_3m*.tif"),
    overwrite=OVERWRITE,
    fn=lambda: step_norm(TAIR_SYN_DIR)
)

run_if_needed(
    "1b) Normalizar HR",
    done=has_any(NORM_DIR, f"*HR*{YEAR}*_norm_3m*.tif"),
    overwrite=OVERWRITE,
    fn=lambda: step_norm(HR_SYN_DIR)
)


# =========================
# 2) Alinear normalizados (TAIR/HR)
# =========================
def step_align_norm():
    aligner = FolderGridAligner(
        input_dir=str(NORM_DIR),
        output_dir=str(ALIGNED_DIR),
        align_to_path=str(REF_ALIGN),
        overwrite=OVERWRITE,
        suffix="_3m",
    )
    outs = aligner.process_folder(kind_mode="auto", copy_if_already_aligned=True)
    print("Alineados:", len(outs))
    return outs

run_if_needed(
    "2) Alinear normalizados (TAIR/HR)",
    done=has_any(ALIGNED_DIR, f"*{YEAR}*_3m.tif"),
    overwrite=OVERWRITE,
    fn=step_align_norm
)


# =========================
# 3) Normalizar distancias (agua/vías) por archivo
# =========================
dist_jobs = [
    ("distancia_agua_3m_ALINEADO.tif", ALIGNED_DIR / "distancia_agua_3m_ALINEADO_NORMALIZADO.tif"),
    ("distancia_vias_3m_ALINEADO.tif", ALIGNED_DIR / "distancia_vias_3m_ALINEADO_NORMALIZADO.tif"),
]

def step_norm_dist():
    outs = []
    for src_name, dst in dist_jobs:
        src = TAIR_ALI_DIR / src_name
        norm_file = RasterNormalizer01File(input_path=str(src), output_path=str(dst), overwrite=OVERWRITE)
        print(norm_file.process_file())
        outs.append(dst)
    return outs

run_if_needed(
    "3) Normalizar distancias (agua/vías)",
    done=all(dst.exists() for _, dst in dist_jobs),
    overwrite=OVERWRITE,
    fn=step_norm_dist
)


# =========================
# 4) Alinear estáticos categóricos (construcciones/agua/vías/vegetación)
# =========================
static_jobs = [
    ("construcciones.tif", "construcciones_ALINEADO.tif"),
    ("cuerpos_de_agua.tif", "cuerpos_de_agua_ALINEADO.tif"),
    ("vias.tif", "vias_ALINEADO.tif"),
    ("vegetacion.tif", "vegetacion_ALINEADO.tif"),
]

def step_align_static():
    aligner = GridAlignerFile(align_to_path=str(REF_ALIGN), overwrite=OVERWRITE, suffix="_ALINEADO")
    outs = []
    for src_name, out_name in static_jobs:
        out = aligner.process_file(
            input_path=str(TAIR_RAS_DIR / src_name),
            output_path=str(ALIGNED_DIR / out_name),
            kind_mode="categorical",
            copy_if_already_aligned=True,
        )
        print("Archivo alineado:", out)
        outs.append(ALIGNED_DIR / out_name)
    return outs

run_if_needed(
    "4) Alinear estáticos (categorical)",
    done=all((ALIGNED_DIR / out_name).exists() for _, out_name in static_jobs),
    overwrite=OVERWRITE,
    fn=step_align_static
)


# =========================
# 5) Resample+align índices 3m (02_Calculos → 03_alinear_resampliar)
# =========================
def step_resample_indices():
    res = FolderTo3mResampler4326(
        input_dir=str(ROOT / "02_Calculos"),
        output_dir=str(RESAMP_DIR),
        align_to_path=str(REF_ALIGN),
        mode="direct",
        overwrite=OVERWRITE
    )
    salidas = res.process_folder(apply_clip=True, clip_range=(0.0, 1.0), kind_mode="auto")
    print("Generados:", len(salidas))
    return salidas

run_if_needed(
    "5) Resample+align índices 3m",
    done=has_any(RESAMP_DIR, f"*_{YEAR}_*_NDVI_3m*.tif"),
    overwrite=OVERWRITE,
    fn=step_resample_indices
)


# =========================
# 6) SOM INPUTS (NPZ)
# =========================
def step_build_npz():
    builder = SOMInputBuilder(
        input_dir=str(ROOT),
        month_regex=r"(20\d{2})[_-](0?[1-9]|1[0-2])(?=[_-]|$)",
        monthly_patterns={
            "NDVI":   r"_NDVI_3m",
            "NDBI":   r"_NDBI_3m",
            "ALBEDO": r"_Albedo_3m",
            "LST":    r"_LST_3m",
            "MNDWI":  r"_MNDWI_3m|_MNWI_3m",
            "TAIR":   r"TAIRE_SINTETICA_3M_.*_norm_3m",
            "HR":     r"HR_SINTETICA_3M_.*_norm_3m",
        },
        static_patterns={
            "CONSTRUCCIONES_BIN": r"construcciones_ALINEADO",
            "CUERPOS_AGUA_BIN":   r"cuerpos_de_agua_ALINEADO",
            "VEGETACION_BIN":     r"vegetacion_ALINEADO",
            "VIAS_BIN":           r"vias_ALINEADO",
            "PORC_URBANO":        r"porcentaje_urbano_3m_ALINEADO",
            "DIST_AGUA_NORM":     r"^distancia_agua_3m_ALINEADO_NORMALIZADO",
            "DIST_VIAS_NORM":     r"distancia_vias_3m_ALINEADO_NORMALIZADO",
        },
        mask_mode="all_valid",
        sample_max_pixels=200_000,
        allow_missing_monthly_vars=False,
        require_static=True,
    )

    print("DIAG:", builder.diagnose(YEAR))

    packs = builder.build()
    packs_year = {k: v for k, v in packs.items() if k.startswith(f"{YEAR}_")}

    ok = 0
    for month_key, pack in sorted(packs_year.items()):
        out_path = NPZ_DIR / f"SOM_INPUT_{month_key}.npz"
        if out_path.exists() and not OVERWRITE:
            continue
        builder.save_npz(str(out_path), pack["X"], pack["coords"], pack["meta"])
        print(f"✅ {month_key} → X shape: {pack['X'].shape}")
        ok += 1

    print("🎉 NPZ generados/actualizados:", ok)
    return True

run_if_needed(
    "6) Generar NPZ (SOM inputs)",
    done=has_any(NPZ_DIR, f"SOM_INPUT_{YEAR}_??.npz"),
    overwrite=OVERWRITE,
    fn=step_build_npz
)


# =========================
# 7) SOM anual + export modelo
# =========================
MODEL_OUT = SOM_YEAR_DIR / f"SOM_MODEL_ANNUAL_{YEAR}_10x10.npz"

def step_som_annual():
    runner = SOMYearRunner(
        npz_dir=str(NPZ_DIR),
        som_rows=10,
        som_cols=10,
        standardize=True,
        sample_per_month=200000,
    )
    packs = runner.load_year(YEAR)
    print("Meses cargados:", [p.month_key for p in packs])

    runner.fit_annual(packs)
    results = runner.predict_year(packs)

    model = runner.export_model()
    np.savez_compressed(MODEL_OUT, **model)
    print("✅ Modelo anual guardado:", MODEL_OUT)

    for month_key, pack in results.items():
        print("✅", month_key, "N:", pack["cluster_ids"].shape[0], "clusters únicos:", len(np.unique(pack["cluster_ids"])))

    return True

run_if_needed(
    "7) SOM anual + export modelo",
    done=MODEL_OUT.exists(),
    overwrite=OVERWRITE,
    fn=step_som_annual
)


# =========================
# 8) Rasterizar clusters FULL a GeoTIFF  ✅ (TU PARTE QUE FALTABA)
# =========================
def step_clusters_full_tif():
    runner = SOMYearRunner(
        npz_dir=str(NPZ_DIR),
        som_rows=10,
        som_cols=10,
        standardize=False,
        sample_per_month=200_000,
    )

    packs = runner.load_year(YEAR)
    runner.fit_annual(packs)
    results = runner.predict_year(packs)
    som_model = runner.export_model()

    writer = ClusterGeoTIFFWriter(
        nodata=-1,
        tile_size=256,
        mask_mode="all_valid",
    )

    ok = 0
    for month_key, pack in results.items():
        out_tif = CLUSTERS_DIR / f"SOM_CLUSTER_{month_key}_{runner.som_rows}x{runner.som_cols}.tif"
        if out_tif.exists() and not OVERWRITE:
            continue

        writer.save(
            out_path=str(out_tif),
            meta=pack["meta"],
            mode="full",
            som_model=som_model,
        )
        print("✅", out_tif.name, "| FULL raster clusters")
        ok += 1

    print(f"🎉 Listo: {ok} GeoTIFF de clusters FULL en {CLUSTERS_DIR}")
    return True

run_if_needed(
    "8) Rasterizar clusters FULL (CLUSTERS_TIF)",
    done=has_any(CLUSTERS_DIR, f"SOM_CLUSTER_{YEAR}_??_10x10.tif"),
    overwrite=OVERWRITE,
    fn=step_clusters_full_tif
)


# =========================
# 9) Cluster stats (CSV)
# =========================
def step_cluster_stats():
    runner = SOMYearRunner(
        npz_dir=str(NPZ_DIR),
        som_rows=10,
        som_cols=10,
        standardize=False,
        sample_per_month=200000,
    )
    packs = runner.load_year(YEAR)
    runner.fit_annual(packs)
    results = runner.predict_year(packs)

    calc = ClusterStatsCalculator(min_pixels_per_cluster=50)

    for month_key, pack in results.items():
        out_csv = STATS_DIR / f"CLUSTER_STATS_{month_key}_10x10.csv"
        if out_csv.exists() and not OVERWRITE:
            continue

        rows = calc.compute(
            X=pack["X"],
            cluster_ids=pack["cluster_ids"],
            variables=pack["meta"]["variables"],
            extra={"year": YEAR, "month": month_key},
        )
        calc.save_csv(rows, str(out_csv))
        print("✅", out_csv.name, "| clusters en CSV:", len(rows))

    return True

run_if_needed(
    "9) Cluster stats CSV",
    done=has_any(STATS_DIR, f"CLUSTER_STATS_{YEAR}_??_10x10.csv"),
    overwrite=OVERWRITE,
    fn=step_cluster_stats
)


# =========================
# 10) ICU_SCORE (tif)
# =========================
def step_icu_score():
    labeler = ICUClusterLabeler(
        p_hot=92, p_cool=8,
        score_p1=80, score_p2=90, score_p3=97,
        min_pixels_per_cluster=400,
        weights={
            "TAIR": 3.0, "LST": 0.5, "HR": 2.0,
            "NDVI": 1.0, "MNDWI": 0.8, "CUERPOS_AGUA_BIN": 0.7, "VEGETACION_BIN": 0.7,
            "NDBI": 1.0, "PORC_URBANO": 1.0, "CONSTRUCCIONES_BIN": 0.9, "VIAS_BIN": 0.7,
            "DIST_AGUA_NORM": 0.9, "DIST_VIAS_NORM": 0.4,
            "ALBEDO": 0.2,
        },
    )

    cluster_files = sorted(CLUSTERS_DIR.glob(f"SOM_CLUSTER_{YEAR}_??_10x10.tif"))
    if not cluster_files:
        raise FileNotFoundError(f"No se encontraron clusters en:\n{CLUSTERS_DIR}")

    ok = 0
    skipped = 0

    for cpath in cluster_files:
        parts = cpath.stem.split("_")
        month_key = f"{parts[2]}_{parts[3]}"

        stats_csv = STATS_DIR / f"CLUSTER_STATS_{month_key}_10x10.csv"
        out_tif = ICU_DIR / f"ICU_SCORE_{month_key}_10x10.tif"

        if not stats_csv.exists():
            print(f"⚠️  SKIP {month_key} → falta CSV {stats_csv.name}")
            skipped += 1
            continue

        if out_tif.exists() and not OVERWRITE:
            continue

        labeler.run(cluster_tif=str(cpath), stats_csv=str(stats_csv), out_tif=str(out_tif))
        print(f"✅ {month_key} → {out_tif.name}")
        ok += 1

    print("\n📊 RESUMEN ICU_SCORE")
    print("✅ Generados:", ok)
    print("⚠️  Omitidos:", skipped)
    print("📁 Salida:", ICU_DIR)
    return True

run_if_needed(
    "10) ICU_SCORE (tif)",
    done=has_any(ICU_DIR, f"ICU_SCORE_{YEAR}_??_10x10.tif"),
    overwrite=OVERWRITE,
    fn=step_icu_score
)


# =========================
# 11) Postproceso ICU (AGRUPADO)
# =========================
def step_post():
    post = ICUPostProcessor(
        nodata=-1,
        min_class=3,
        kernel_size=9,
        close_iters=2,
        min_area_m2=9000.0,
        connectivity=8,
        keep_low_class=False,
        fill_nodata_to_zero=True
    )

    icu_files = sorted(ICU_DIR.glob(f"ICU_SCORE_{YEAR}_??_10x10.tif"))
    if not icu_files:
        raise FileNotFoundError(f"No se encontraron ICU_SCORE en: {ICU_DIR}")

    for in_tif in icu_files:
        out_tif = ICU_AGR_DIR / f"{in_tif.stem}_AGRUPADO.tif"
        if out_tif.exists() and not OVERWRITE:
            continue
        post.run(str(in_tif), str(out_tif))
        print(f"✅ {out_tif.name}")

    print("\n🎉 Postproceso terminado")
    print("📁 Resultados en:", ICU_AGR_DIR)
    return True

run_if_needed(
    "11) ICU postproceso (AGRUPADO)",
    done=has_any(ICU_AGR_DIR, f"ICU_SCORE_{YEAR}_??_10x10_AGRUPADO.tif"),
    overwrite=OVERWRITE,
    fn=step_post
)


# =========================
# 12) Isobandas (geojson)
# =========================
def step_isobands():
    gen = ICUIsobandPolygonGenerator(
        nodata=-1,
        min_class=3,
        decay_m=60.0,
        smooth_sigma_m=18.0,
        band_edges=[0.25, 0.40, 0.55, 0.70, 0.85],
        dissolve=True
    )

    icu_agr_files = sorted(ICU_AGR_DIR.glob(f"ICU_SCORE_{YEAR}_??_10x10_AGRUPADO.tif"))
    if not icu_agr_files:
        raise FileNotFoundError(f"No se encontraron ICU_SCORE_AGRUPADO en: {ICU_AGR_DIR}")

    for f in icu_agr_files:
        out_geojson = ICU_POLY_DIR / f"{f.stem}_ISOBANDS_DISSOLVE.geojson"
        if out_geojson.exists() and not OVERWRITE:
            continue
        gen.run(str(f), str(out_geojson))
        print("✅", out_geojson.name)

    return True

run_if_needed(
    "12) Isobandas (geojson)",
    done=has_any(ICU_POLY_DIR, f"ICU_SCORE_{YEAR}_??_10x10_AGRUPADO_ISOBANDS_DISSOLVE.geojson"),
    overwrite=OVERWRITE,
    fn=step_isobands
)

print("\n🎉 PIPELINE COMPLETO OK.")
