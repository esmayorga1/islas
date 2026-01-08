import os
import sys
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

# === 🔧 CONFIGURAR PROJ CORRECTAMENTE ===
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

possible_proj_paths = [
    os.path.join(os.path.dirname(sys.executable), "Library", "share", "proj"),
    os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "rasterio", "proj_data"),
    os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
]

proj_path = next((p for p in possible_proj_paths if os.path.exists(os.path.join(p, "proj.db"))), None)
if proj_path:
    os.environ["PROJ_LIB"] = proj_path

# === 📁 CONFIGURACIÓN ===
input_dir = Path(r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021")
output_dir = input_dir / "reproyectadas"
os.makedirs(output_dir, exist_ok=True)

dst_crs = "EPSG:4326"

# Bandas que tienes en el orden del TIFF unificado
band_names = ["B2","B3","B4","B8","B11","B12"]

# Listar imágenes
images = [f for f in input_dir.glob("*.tif")]

for img_path in images:
    output_path = output_dir / img_path.name

    with rasterio.open(img_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)

        # Guardar archivo reproyectado unificado
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

    # Guardar bandas individuales con nombres originales
    with rasterio.open(output_path) as src:
        for i, bname in enumerate(band_names, start=1):
            band_profile = src.profile.copy()
            band_profile.update(count=1)
            band_out = output_dir / f"{img_path.stem}_{bname}.tif"
            with rasterio.open(band_out, "w", **band_profile) as dst_band:
                dst_band.write(src.read(i), 1)
    print(f"✅ Reproyectado y separado: {img_path.name}")

print("\n🎯 Reproyección y separación de bandas completadas.")
