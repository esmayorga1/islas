from app.ontology.classes.UHI.UHI_00_normalizar_raster import RasterNormalizer01
from app.ontology.classes.UHI.UHI_01_Alinear import FolderGridAligner
from app.ontology.classes.UHI.UHI_00_normalizar_por_archivo import RasterNormalizer01File


# Normalizando temperatura del aire sintetica
norm = RasterNormalizer01(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\03_resultado_temperatura_sintetica",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\01_datos_normalizados",
    overwrite=False
)
salidas = norm.process_folder()
print("Normalizados TAIR:", len(salidas))


# Normalizando humedad relativa sintetica
norm = RasterNormalizer01(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\05_humedad_relativa\2022\00_resultados_humedad_relativa_sintetica",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\01_datos_normalizados",
    overwrite=False
)
salidas = norm.process_folder()
print("Normalizados HR:", len(salidas))


# Alineando temperatura del aire sintetica y humedad relativa sintetica
aligner = FolderGridAligner(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\01_datos_normalizados",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados",
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    overwrite=False,
    suffix="_3m",
)
outputs = aligner.process_folder(kind_mode="auto", copy_if_already_aligned=True)
print("Alineados:", len(outputs))


# Normalizar por archivo (distancia agua)  
norm_file = RasterNormalizer01File(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\alineados\distancia_agua_3m_ALINEADO.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\distancia_agua_3m_ALINEADO_NORMALIZADO.tif",
    overwrite=False
)
print(norm_file.process_file())



# Normalizar por archivo (distancia vías) 
norm_file = RasterNormalizer01File(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\alineados\distancia_vias_3m_ALINEADO.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\distancia_vias_3m_ALINEADO_NORMALIZADO.tif",
    overwrite=False
)
print(norm_file.process_file())


from app.ontology.classes.UHI.UHI_01_Alinear_por_archivo_o_folder import GridAlignerFile

aligner = GridAlignerFile(
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
    overwrite=False,
    suffix="_ALINEADO",
)

# ----------------- Construcciones -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\construcciones.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\construcciones_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

# ----------------- Cuerpos de agua -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\cuerpos_de_agua.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\cuerpos_de_agua_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

# ----------------- Vías -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\vias.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\vias_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

# ----------------- Vegetación| -----------------
out = aligner.process_file(
    input_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\04_Temperatura_Aire\2022\02_resultados\01_resultado_raster\vegetacion.tif",
    output_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\02_datos_normalizados_y_aliados\vegetacion_ALINEADO.tif",
    kind_mode="categorical",
    copy_if_already_aligned=True,
)
print("Archivo alineado:", out)

from app.ontology.classes.UHI.UHI_01_Alinear_Y_Remuestrar import FolderTo3mResampler4326

res = FolderTo3mResampler4326(
    input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\02_Calculos",
    output_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\03_alinear_resampliar",
    align_to_path=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",  
    mode="direct",
    overwrite=False
)

salidas = res.process_folder(
    apply_clip=True,
    clip_range=(0.0, 1.0),
    kind_mode="auto"
)

print("Generados:", len(salidas))


#03-----------------------------
import os
from app.ontology.classes.UHI.UHI_03_SOM_input_builder import SOMInputBuilder

input_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1"
out_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\04_SOM_input_builder"
os.makedirs(out_dir, exist_ok=True)

builder = SOMInputBuilder(
    input_dir=input_dir,

    
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

    # ✅ si quieres que falle si falta alguna estática
    require_static=True,
)

# ✅ Diagnóstico (mensuales + estáticas)
diag = builder.diagnose("2022")
print("DIAG:", diag)

packs = builder.build()
packs_2022 = {k: v for k, v in packs.items() if k.startswith("2022_")}

for month_key, pack in sorted(packs_2022.items()):
    out_path = os.path.join(out_dir, f"SOM_INPUT_{month_key}.npz")
    builder.save_npz(out_path, pack["X"], pack["coords"], pack["meta"])
    print(f"✅ {month_key} → X shape: {pack['X'].shape}")

print("🎉 NPZ generados:", len(packs_2022))



#4  -------------------

import os
import numpy as np
from pathlib import Path

from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner


def main():
    npz_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\04_SOM_input_builder"
    out_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\05_SOM_year_annual"
    os.makedirs(out_dir, exist_ok=True)

    year = "2022"

    runner = SOMYearRunner(
        npz_dir=npz_dir,
        som_rows=10,
        som_cols=10,
        standardize=True,        # ✅ tus variables 0–1
        sample_per_month=200000,  # opcional
    )

    packs = runner.load_year(year)
    print("Meses cargados:", [p.month_key for p in packs])

    runner.fit_annual(packs)

    results = runner.predict_year(packs)

    model = runner.export_model()
    out_model = Path(out_dir) / f"SOM_MODEL_ANNUAL_{year}_{runner.som_rows}x{runner.som_cols}.npz"
    np.savez_compressed(out_model, **model)
    print("✅ Modelo anual guardado:", out_model)

    for month_key, pack in results.items():
        print("✅", month_key, "N:", pack["cluster_ids"].shape[0], "clusters únicos:", len(np.unique(pack["cluster_ids"])))

    print("🎉 OK. Siguiente paso: rasterizar clusters a GeoTIFF.")


if __name__ == "__main__":
    main()





#05-------------------------------

import numpy as np
from pathlib import Path

from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner
from app.ontology.classes.UHI.UHI_05_cluster_geotiff_writer import ClusterGeoTIFFWriter


def main():
    # Carpeta donde están los NPZ del año
    npz_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\04_SOM_input_builder"

    year = "2022"

    out_root = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM"
    out_dir = Path(out_root) / year / "CLUSTERS_TIF"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Entrenas con muestra (rápido), pero luego exportas modelo para predicción completa
    runner = SOMYearRunner(
        npz_dir=npz_dir,
        som_rows=10,
        som_cols=10,
        standardize=False,
        sample_per_month=200_000,  # solo para entrenar (opcional)
    )

    packs = runner.load_year(year)
    runner.fit_annual(packs)
    results = runner.predict_year(packs)

    # ✅ modelo exportado (pesos SOM + mu/sd si standardize=True)
    som_model = runner.export_model()

    # ✅ Writer FULL: predice por tiles y escribe raster COMPLETO
    writer = ClusterGeoTIFFWriter(
        nodata=-1,
        tile_size=256,          # 256 o 512 según RAM
        mask_mode="all_valid",  # consistente con tu pipeline
    )

    ok = 0
    for month_key, pack in results.items():
        meta = pack["meta"]

        out_tif = out_dir / f"SOM_CLUSTER_{month_key}_{runner.som_rows}x{runner.som_cols}.tif"

        # ✅ CLAVE: mode="full" (no usa coords)
        writer.save(
            out_path=str(out_tif),
            meta=meta,
            mode="full",
            som_model=som_model,
        )

        print("✅", out_tif.name, "| FULL raster clusters")
        ok += 1

    print(f"🎉 Listo: {ok} GeoTIFF de clusters FULL por mes para el año {year}")
    print("📁 Salida:", out_dir)


if __name__ == "__main__":
    main()


#06-------------------------------
import os
from pathlib import Path

import numpy as np

from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner
from app.ontology.classes.UHI.UHI_06_cluster_stats import ClusterStatsCalculator


def main():
    # Donde están los NPZ del año (por ahora 2021)
    npz_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\04_SOM_input_builder"

    year = "2022"

    out_root = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM\2022"
    out_dir = Path(out_root) / "STATS"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = SOMYearRunner(
        npz_dir=npz_dir,
        som_rows=10,
        som_cols=10,
        standardize=False,
        sample_per_month=200000,
    )

    packs = runner.load_year(year)
    runner.fit_annual(packs)
    results = runner.predict_year(packs)

    calc = ClusterStatsCalculator(min_pixels_per_cluster=50)

    for month_key, pack in results.items():
        X = pack["X"]
        cluster_ids = pack["cluster_ids"]
        variables = pack["meta"]["variables"]

        rows = calc.compute(
            X=X,
            cluster_ids=cluster_ids,
            variables=variables,
            extra={"year": year, "month": month_key},
        )

        out_csv = out_dir / f"CLUSTER_STATS_{month_key}_10x10.csv"
        calc.save_csv(rows, str(out_csv))

        print("✅", out_csv.name, "| clusters en CSV:", len(rows))

    print("🎉 Listo: CSV de métricas por cluster (por mes) para el año", year)


if __name__ == "__main__":
    main()
#07-------------------------------
from pathlib import Path
from app.ontology.classes.UHI.UHI_07_icu_cluster_labeler import ICUClusterLabeler


def main():
    year = "2022"

    # root = Path(
        # r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2"
        # rf"\{year}\{year}_salida\SALIDAS_SOM\{year}"
    # )

    root = Path(
     r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM\2022"
 )

    clusters_dir = root / "CLUSTERS_TIF"
    stats_dir = root / "STATS"
    out_dir = root / "ICU_SCORE"
    out_dir.mkdir(parents=True, exist_ok=True)

    labeler = ICUClusterLabeler(
        # ✅ más estricto en lo que se considera "alto/bajo"
        p_hot=92,
        p_cool=8,

        # ✅ MUCHÍSIMO más estricto para que ICU 2-3 sean solo hotspots
        score_p1=80,
        score_p2=90,
        score_p3=97,

        # ✅ menos ruido: ignora clusters pequeños
        min_pixels_per_cluster=400,

        weights={
            "TAIR": 3.0,
            "LST": 0.5,
            "HR": 2.0,

            "NDVI": 1.0,
            "MNDWI": 0.8,
            "CUERPOS_AGUA_BIN": 0.7,
            "VEGETACION_BIN": 0.7,

            "NDBI": 1.0,
            "PORC_URBANO": 1.0,
            "CONSTRUCCIONES_BIN": 0.9,
            "VIAS_BIN": 0.7,

            "DIST_AGUA_NORM": 0.9,
            "DIST_VIAS_NORM": 0.4,

            "ALBEDO": 0.2,
        },
    )

    cluster_files = sorted(clusters_dir.glob(f"SOM_CLUSTER_{year}_??_10x10.tif"))
    if not cluster_files:
        raise FileNotFoundError(f"No se encontraron clusters en:\n{clusters_dir}")

    ok = 0
    skipped = 0

    for cpath in cluster_files:
        parts = cpath.stem.split("_")
        month_key = f"{parts[2]}_{parts[3]}"

        stats_csv = stats_dir / f"CLUSTER_STATS_{month_key}_10x10.csv"
        out_tif = out_dir / f"ICU_SCORE_{month_key}_10x10.tif"

        if not stats_csv.exists():
            print(f"⚠️  SKIP {month_key} → falta CSV {stats_csv.name}")
            skipped += 1
            continue

        labeler.run(
            cluster_tif=str(cpath),
            stats_csv=str(stats_csv),
            out_tif=str(out_tif),
        )

        print(f"✅ {month_key} → {out_tif.name}")
        ok += 1

    print("\n📊 RESUMEN")
    print("✅ ICU_SCORE generados:", ok)
    print("⚠️  Meses omitidos:", skipped)
    print("📁 Salida:", out_dir)


if __name__ == "__main__":
    main()
#08-------------------------------
from pathlib import Path
from app.ontology.classes.UHI.UHI_08_ICUPostProcessor import ICUPostProcessor


def main():
    post = ICUPostProcessor(
        nodata=-1,
        min_class=3,              # ICU >= 2 forman islas
        kernel_size=9,            # ⬅️ más grande = más agrupación
        close_iters=2,            # ⬅️ une parches cercanos
        min_area_m2=9000.0,       # ⬅️ elimina islas pequeñas (≈ 1000 píxeles a 3m)
        connectivity=8,
        keep_low_class=False,
        fill_nodata_to_zero=True  # ⬅️ reduce valores nulos visibles
    )

    # Directorio base
    base_dir = Path(
        r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1\06_UHI\06_salidas_SOM\2022"
       
    )

    in_dir = base_dir / "ICU_SCORE"
    out_dir = base_dir / "ICU_SCORE_AGRUPADO"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ICU_SCORE mensuales
    icu_files = sorted(in_dir.glob("ICU_SCORE_2022_??_10x10.tif"))

    if not icu_files:
        raise FileNotFoundError(f"No se encontraron ICU_SCORE en: {in_dir}")

    for in_tif in icu_files:
        out_tif = out_dir / f"{in_tif.stem}_AGRUPADO.tif"
        post.run(str(in_tif), str(out_tif))
        print(f"✅ {out_tif.name}")

    print("\n🎉 Postproceso terminado")
    print("📁 Resultados en:", out_dir)


if __name__ == "__main__":
    main()
#09-------------------------------
from pathlib import Path
from backend.app.ontology.classes.UHI.UHI_09_Icu_score_agrupado import ICUIsobandPolygonGenerator

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
