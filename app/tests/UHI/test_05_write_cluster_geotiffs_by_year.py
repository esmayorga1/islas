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
