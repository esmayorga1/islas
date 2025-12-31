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
