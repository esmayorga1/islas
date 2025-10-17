import os
import sys
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

class Reproyectar:
    """
    Reproyecta imágenes raster a EPSG:4326 y separa las bandas individuales.
    Puede aplicarse a carpetas completas de imágenes Sentinel o MODIS.
    """

    def __init__(self, input_dir, output_dir=None, dst_crs="EPSG:4326"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "reproyectadas"
        self.dst_crs = dst_crs
        os.makedirs(self.output_dir, exist_ok=True)
        self._configurar_proj()

        # Orden típico de bandas Sentinel-2
        self.band_names = ["B2", "B3", "B4", "B8", "B11", "B12"]

    def _configurar_proj(self):
        """Configura correctamente la ruta de PROJ para evitar errores."""
        os.environ.pop("PROJ_LIB", None)
        os.environ.pop("PROJ_DATA", None)

        possible_proj_paths = [
            os.path.join(os.path.dirname(sys.executable), "Library", "share", "proj"),  # conda
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "rasterio", "proj_data"),
            os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
        ]

        proj_path = next((p for p in possible_proj_paths if os.path.exists(os.path.join(p, "proj.db"))), None)
        if proj_path:
            os.environ["PROJ_LIB"] = proj_path

    def reproyectar_imagen(self, img_path):
        """Reproyecta una imagen raster y genera sus bandas individuales."""
        output_path = self.output_dir / img_path.name

        with rasterio.open(img_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.dst_crs, src.width, src.height, *src.bounds
            )

            profile = src.profile.copy()
            profile.update(crs=self.dst_crs, transform=transform, width=width, height=height)

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.dst_crs,
                        resampling=Resampling.bilinear
                    )

        # Guardar bandas individuales
        with rasterio.open(output_path) as src:
            for i, bname in enumerate(self.band_names[:src.count], start=1):
                band_profile = src.profile.copy()
                band_profile.update(count=1)
                band_out = self.output_dir / f"{img_path.stem}_{bname}.tif"
                with rasterio.open(band_out, "w", **band_profile) as dst_band:
                    dst_band.write(src.read(i), 1)
        print(f"✅ Reproyectado y separado: {img_path.name}")

    def procesar_carpeta(self):
        """Reproyecta todas las imágenes TIFF de la carpeta."""
        images = [f for f in self.input_dir.glob("*.tif")]
        if not images:
            print("⚠️ No se encontraron imágenes TIFF en la carpeta.")
            return

        print(f"📂 Procesando {len(images)} imágenes...")
        for img_path in images:
            self.reproyectar_imagen(img_path)

        print("\n🎯 Reproyección y separación de bandas completadas.")

# === Ejemplo de uso ===












# import rasterio
# from rasterio.warp import calculate_default_transform, reproject, Resampling
# from pathlib import Path

# def reproyectar_imagen(input_path: str, output_path: str, dst_crs: str = "EPSG:4326"):
#     """Reproyecta una imagen raster al sistema de coordenadas especificado (por defecto EPSG:4326)."""
#     input_path = Path(input_path)
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     with rasterio.open(input_path) as src:
#         print(f"🗺️ Reproyectando: {input_path.name} desde {src.crs} → {dst_crs}")
#         transform, width, height = calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds
#         )

#         kwargs = src.meta.copy()
#         kwargs.update({
#             "crs": dst_crs,
#             "transform": transform,
#             "width": width,
#             "height": height
#         })

#         with rasterio.open(output_path, "w", **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest
#                 )

#     print(f"✅ Imagen reproyectada correctamente: {output_path.name}")