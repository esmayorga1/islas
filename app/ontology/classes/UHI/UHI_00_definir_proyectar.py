import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Literal
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

KindMode = Literal["auto", "continuous", "categorical"]


class Reproject9377_To4326_AlignAndClip:
    """
    1) Asigna CRS EPSG:9377 si el raster no lo tiene o es incorrecto
    2) Reproyecta a EPSG:4326
    3) Alinea EXACTAMENTE a una plantilla (.tif)
    4) Recorta al extent de la plantilla
    5) Rellena zonas sin datos con 0 (cero)
    6) Importante: el raster de salida NO marca 0 como NoData (para que 0 sea dato real)
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        align_to_path: str,
        overwrite: bool = False,
        suffix: str = "_4326_ALINEADO.tif",
        kind_mode: KindMode = "auto",
        tmp_dir: Optional[str] = None,
        output_nodata: Optional[float] = None,  # ✅ por defecto NO nodata en salida
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.align_to_path = Path(align_to_path)

        self.overwrite = overwrite
        self.suffix = suffix
        self.kind_mode = kind_mode
        self.output_nodata = output_nodata

        self.tmp_dir = Path(tmp_dir) if tmp_dir else (self.output_dir / "__tmp__")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        if not self.align_to_path.exists():
            raise FileNotFoundError(f"No existe plantilla: {self.align_to_path}")

        with rasterio.open(self.align_to_path) as tmpl:
            if tmpl.crs is None:
                raise ValueError("La plantilla debe tener CRS (ideal EPSG:4326).")
            self.t_crs = tmpl.crs
            self.t_transform = tmpl.transform
            self.t_width = tmpl.width
            self.t_height = tmpl.height

    def _decide_kind(self, src_path: Path) -> str:
        if self.kind_mode != "auto":
            return self.kind_mode

        try:
            with rasterio.open(src_path) as src:
                arr = src.read(1, masked=True).compressed()
                if arr.size == 0:
                    return "continuous"

                if np.issubdtype(arr.dtype, np.integer):
                    if np.unique(arr[:100000]).size <= 32:
                        return "categorical"

                u = np.unique(arr[:100000])
                if u.size <= 4 and np.all(np.isin(u, [0, 1, 0.0, 1.0])):
                    return "categorical"
        except:
            pass

        return "continuous"

    def _force_crs_9377(self, src_path: Path) -> Path:
        epsg_9377 = CRS.from_epsg(9377)

        with rasterio.open(src_path) as src:
            if src.crs == epsg_9377:
                return src_path

            tmp = self.tmp_dir / f"{src_path.stem}__crs9377.tif"
            if tmp.exists() and not self.overwrite:
                return tmp

            meta = src.meta.copy()
            meta.update(crs=epsg_9377)

            data = src.read()
            with rasterio.open(tmp, "w", **meta) as dst:
                dst.write(data)

            return tmp

    def _reproject_align_clip(self, src_path: Path, out_path: Path, resampling, dtype):
        with rasterio.open(src_path) as src:
            src_data = src.read(1)
            src_nodata = src.nodata  # puede ser None

            # ✅ Inicializamos todo en 0 -> así las zonas fuera / sin datos quedan en 0.
            dst = np.zeros((self.t_height, self.t_width), dtype=np.dtype(dtype))

            # ✅ No marcamos dst_nodata=0 (porque entonces 0 sería NoData)
            reproject(
                source=src_data,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=self.t_transform,
                dst_crs=self.t_crs,
                dst_nodata=self.output_nodata,  # None por defecto
                resampling=resampling,
            )

            # ✅ Si quedaron NaN por alguna interpolación, los volvemos 0.
            if np.issubdtype(dst.dtype, np.floating):
                dst = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)

            profile = {
                "driver": "GTiff",
                "height": self.t_height,
                "width": self.t_width,
                "count": 1,
                "dtype": dtype,
                "crs": self.t_crs,
                "transform": self.t_transform,
                "nodata": self.output_nodata,  # ✅ None => el archivo NO marca nodata
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst_ds:
                dst_ds.write(dst, 1)

    def process_folder(self) -> list[str]:
        outputs = []
        tifs = list(self.input_dir.rglob("*.tif"))
        if not tifs:
            raise FileNotFoundError("No se encontraron TIFF en la carpeta de entrada")

        for src in tifs:
            rel = src.relative_to(self.input_dir)
            out_path = self.output_dir / rel.parent / f"{src.stem}{self.suffix}"

            if out_path.exists() and not self.overwrite:
                outputs.append(str(out_path))
                continue

            kind = self._decide_kind(src)
            resampling = Resampling.nearest if kind == "categorical" else Resampling.bilinear
            dtype = "uint8" if kind == "categorical" else "float32"

            src_9377 = self._force_crs_9377(src)
            self._reproject_align_clip(src_9377, out_path, resampling, dtype)

            outputs.append(str(out_path))

        return outputs

    def cleanup_tmp(self):
        for p in self.tmp_dir.glob("*.tif"):
            try:
                p.unlink()
            except:
                pass
