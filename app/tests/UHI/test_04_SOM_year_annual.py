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
