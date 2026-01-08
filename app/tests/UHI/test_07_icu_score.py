from pathlib import Path
from app.ontology.classes.UHI.UHI_07_icu_cluster_labeler import ICUClusterLabeler


def main():
    year = "2021"

    # root = Path(
        # r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2"
        # rf"\{year}\{year}_salida\SALIDAS_SOM\{year}"
    # )

    root = Path(
     r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\06_UHI\06_salidas_SOM\2021"
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
            "TAIR": 10.0,
            "LST": 1.4,
            "HR": 2.2,

            "NDVI": 1.1,
            "MNDWI": 0.9,
            "CUERPOS_AGUA_BIN": 0.8,
            "VEGETACION_BIN": 0.8,

            "NDBI": 1.1,
            "PORC_URBANO": 3.0,
            "CONSTRUCCIONES_BIN": 3.0,
            "VIAS_BIN": 0.7,

            "DIST_AGUA_NORM": 0.8,
            "DIST_VIAS_NORM": 3.0,

            "ALBEDO": 0.4,
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
