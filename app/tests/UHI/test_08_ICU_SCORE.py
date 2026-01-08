from pathlib import Path
from app.ontology.classes.UHI.UHI_08_ICUPostProcessor import ICUPostProcessor


def main():
    post = ICUPostProcessor(
        nodata=-1,
        min_class=2,              # ICU >= 2 forman islas
        kernel_size=9,            # ⬅️ más grande = más agrupación
        close_iters=2,            # ⬅️ une parches cercanos
        min_area_m2=9000.0,       # ⬅️ elimina islas pequeñas (≈ 1000 píxeles a 3m)
        connectivity=8,
        keep_low_class=False,
        fill_nodata_to_zero=True  # ⬅️ reduce valores nulos visibles
    )

    # Directorio base
    base_dir = Path(
        r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\06_UHI\06_salidas_SOM\2021"
       
    )

    in_dir = base_dir / "ICU_SCORE"
    out_dir = base_dir / "ICU_SCORE_AGRUPADO"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ICU_SCORE mensuales
    icu_files = sorted(in_dir.glob("ICU_SCORE_2021_??_10x10.tif"))

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
