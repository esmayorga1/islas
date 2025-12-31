from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Literal

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


ResamplingMode = Literal["nearest", "bilinear", "cubic"]
KindMode = Literal["auto", "continuous", "categorical"]


class FolderGridAligner:
    """
    Alinea TODOS los .tif (recursivo) de input_dir a la grilla EXACTA de align_to_path:
      - mismo CRS
      - mismo transform
      - mismo width/height

    ✅ SOLO alineación (warp a grilla), NO crea grilla 3m, NO estima resolución, NO multistage.
    ✅ Mantiene estructura de carpetas y agrega sufijo (por defecto _ALIGNED).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        align_to_path: str,  # raster referencia (la "grilla buena")
        overwrite: bool = False,
        suffix: str = "_ALIGNED",
        continuous_resampling: ResamplingMode = "bilinear",
        categorical_resampling: ResamplingMode = "nearest",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.align_to_path = Path(align_to_path)
        if not self.align_to_path.exists():
            raise FileNotFoundError(f"No existe align_to_path: {self.align_to_path}")

        self.overwrite = overwrite
        self.suffix = suffix

        self.continuous_resampling = self._to_resampling(continuous_resampling)
        self.categorical_resampling = self._to_resampling(categorical_resampling)

        self.template = self._read_template(self.align_to_path)

    # ---------------- Template ----------------

    @staticmethod
    def _read_template(path: Path) -> dict:
        with rasterio.open(path) as ref:
            prof = ref.profile.copy()
            prof.update(count=1)  # escribiremos 1 banda por archivo (asumiendo 1 banda)
            # Asegurar nodata consistente
            if prof.get("nodata") is None:
                prof["nodata"] = -9999.0
            return prof

    @staticmethod
    def _to_resampling(name: ResamplingMode) -> Resampling:
        return {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }[name]

    # ---------------- Helpers ----------------

    def _build_output_path(self, src_path: Path) -> Path:
        rel = src_path.relative_to(self.input_dir)
        out_dir = self.output_dir / rel.parent
        out_name = f"{src_path.stem}{self.suffix}.tif"
        return out_dir / out_name

    @staticmethod
    def _infer_kind_from_name(name_upper: str) -> str:
        # Ajusta la lista a tus nombres reales si quieres
        categorical_hints = ["SEG", "SEGMENT", "CLASE", "CLASIF", "COBERT", "URBAN", "BUILT", "MASK"]
        if any(h in name_upper for h in categorical_hints):
            return "categorical"
        return "continuous"

    def _is_already_aligned(self, src_profile: dict) -> bool:
        t = self.template
        return (
            str(src_profile.get("crs")) == str(t.get("crs"))
            and src_profile.get("transform") == t.get("transform")
            and src_profile.get("width") == t.get("width")
            and src_profile.get("height") == t.get("height")
        )

    def _warp_to_template(
        self,
        src_path: Path,
        resampling: Resampling,
        dst_dtype: str,
    ) -> np.ndarray:
        t = self.template

        with rasterio.open(src_path) as src:
            src_arr = src.read(1)
            src_prof = src.profile

            dst = np.full(
                (t["height"], t["width"]),
                t.get("nodata", -9999.0),
                dtype=np.dtype(dst_dtype),
            )

            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src_prof["transform"],
                src_crs=src_prof.get("crs", t.get("crs")),
                dst_transform=t["transform"],
                dst_crs=t.get("crs", src_prof.get("crs")),
                resampling=resampling,
                src_nodata=src_prof.get("nodata"),
                dst_nodata=t.get("nodata"),
            )

        return dst

    def _write(self, arr: np.ndarray, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prof = self.template.copy()
        prof.update(dtype=str(arr.dtype), count=1)

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr, 1)

    # ---------------- Public API ----------------

    def process_folder(
        self,
        kind_mode: KindMode = "auto",
        copy_if_already_aligned: bool = True,
    ) -> List[str]:
        """
        - kind_mode:
            - auto: heurística por nombre
            - continuous: fuerza bilinear/cubic
            - categorical: fuerza nearest
        - copy_if_already_aligned:
            - True: si ya está alineado, lo copia tal cual (más rápido y evita re-muestreo extra)
            - False: igual lo reprojecta/warpea a la plantilla
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"No existe input_dir: {self.input_dir}")

        src_files = sorted([p for p in self.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff")])
        if not src_files:
            raise FileNotFoundError(f"No se encontraron .tif/.tiff dentro de: {self.input_dir}")

        outputs: List[str] = []

        for src in src_files:
            dst = self._build_output_path(src)
            if dst.exists() and not self.overwrite:
                continue

            with rasterio.open(src) as ds:
                src_prof = ds.profile.copy()

            # decidir tipo
            if kind_mode == "auto":
                kind = self._infer_kind_from_name(src.stem.upper())
            else:
                kind = kind_mode

            if kind == "categorical":
                resampling = self.categorical_resampling
                dst_dtype = "uint8"
            else:
                resampling = self.continuous_resampling
                dst_dtype = "float32"

            # si ya está alineado, copiar o re-escribir
            if self._is_already_aligned(src_prof) and copy_if_already_aligned:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # copia binaria (rápido)
                dst.write_bytes(src.read_bytes())
                outputs.append(str(dst))
                continue

            arr = self._warp_to_template(src, resampling=resampling, dst_dtype=dst_dtype)

            if kind == "categorical":
                arr = arr.astype(np.uint8, copy=False)
            else:
                arr = arr.astype(np.float32, copy=False)

            self._write(arr, dst)
            outputs.append(str(dst))

        return outputs
