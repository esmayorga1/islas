from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


ResamplingMode = Literal["nearest", "bilinear", "cubic"]
KindMode = Literal["auto", "continuous", "categorical"]


class GridAlignerFile:
    """
    Alinea rasters a la grilla EXACTA de un raster plantilla (align_to_path):
      - mismo CRS
      - mismo transform
      - mismo width/height

    Soporta:
      - process_file(): alinear un solo archivo
      - process_folder(): alinear toda una carpeta (recursivo)

    ✅ SOLO alineación (warp a grilla)
    ✅ Mantiene estructura en modo carpeta y agrega sufijo.
    ✅ Corrige nodata según dtype (evita: dst_nodata must be in valid range for destination dtype)
    """

    def __init__(
        self,
        align_to_path: str,
        overwrite: bool = False,
        suffix: str = "_ALIGNED",
        continuous_resampling: ResamplingMode = "bilinear",
        categorical_resampling: ResamplingMode = "nearest",
        nodata_default: float = -9999.0,
    ):
        self.align_to_path = Path(align_to_path)
        if not self.align_to_path.exists():
            raise FileNotFoundError(f"No existe align_to_path: {self.align_to_path}")

        self.overwrite = overwrite
        self.suffix = suffix
        self.nodata_default = float(nodata_default)

        self.continuous_resampling = self._to_resampling(continuous_resampling)
        self.categorical_resampling = self._to_resampling(categorical_resampling)

        self.template = self._read_template(self.align_to_path)

    # ---------------- Template ----------------

    def _read_template(self, path: Path) -> dict:
        with rasterio.open(path) as ref:
            prof = ref.profile.copy()
            prof.update(count=1)
            if prof.get("nodata") is None:
                prof["nodata"] = self.nodata_default
            return prof

    @staticmethod
    def _to_resampling(name: ResamplingMode) -> Resampling:
        return {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }[name]

    # ---------------- nodata helper (FIX) ----------------

    def _nodata_for_dtype(self, dtype: str) -> float:
        """
        Devuelve un nodata válido para el dtype de salida.
        - Para uint8 (categóricas/binarias): 255 (típico y seguro si tus clases no usan 255)
        - Para float32: usa nodata del template o nodata_default
        """
        d = str(dtype).lower()
        if d in ("uint8", "ubyte"):
            return 255.0
        if d in ("uint16", "int16", "int32", "uint32"):
            return 0.0
        # floats
        return float(self.template.get("nodata", self.nodata_default))

    # ---------------- Helpers ----------------

    @staticmethod
    def _infer_kind_from_name(name_upper: str) -> str:
        categorical_hints = [
            "SEG", "SEGMENT", "CLASE", "CLASIF", "COBERT", "MASK",
            "BIN", "CLASS", "LABEL"
        ]
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

    def _pick_resampling_and_dtype(self, src_name: str, kind_mode: KindMode):
        if kind_mode == "auto":
            kind = self._infer_kind_from_name(src_name.upper())
        else:
            kind = kind_mode

        if kind == "categorical":
            return kind, self.categorical_resampling, "uint8"
        return kind, self.continuous_resampling, "float32"

    def _warp_to_template(
        self,
        src_path: Path,
        resampling: Resampling,
        dst_dtype: str,
    ) -> np.ndarray:
        t = self.template
        dst_nodata = self._nodata_for_dtype(dst_dtype)

        with rasterio.open(src_path) as src:
            src_arr = src.read(1)
            src_prof = src.profile.copy()

            dst = np.full(
                (t["height"], t["width"]),
                dst_nodata,
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
                dst_nodata=dst_nodata,
            )

        return dst

    def _write(self, arr: np.ndarray, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prof = self.template.copy()
        prof.update(dtype=str(arr.dtype), count=1)
        prof["nodata"] = self._nodata_for_dtype(str(arr.dtype))

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr, 1)

    def _default_output_file(self, input_path: Path, output_dir: Optional[Path] = None) -> Path:
        base = output_dir if output_dir is not None else input_path.parent
        return base / f"{input_path.stem}{self.suffix}{input_path.suffix}"

    def _build_output_path_folder(self, src_path: Path, input_dir: Path, output_dir: Path) -> Path:
        rel = src_path.relative_to(input_dir)
        out_dir = output_dir / rel.parent
        out_name = f"{src_path.stem}{self.suffix}.tif"
        return out_dir / out_name

    # ---------------- Public API (FILE) ----------------

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        kind_mode: KindMode = "auto",
        copy_if_already_aligned: bool = True,
    ) -> str:
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"No existe input_path: {src}")

        dst = Path(output_path) if output_path else self._default_output_file(src)

        if dst.exists() and not self.overwrite:
            return str(dst)

        with rasterio.open(src) as ds:
            src_prof = ds.profile.copy()

        kind, resampling, dst_dtype = self._pick_resampling_and_dtype(src.stem, kind_mode)

        if self._is_already_aligned(src_prof) and copy_if_already_aligned:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            return str(dst)

        arr = self._warp_to_template(src, resampling=resampling, dst_dtype=dst_dtype)

        if kind == "categorical":
            arr = arr.astype(np.uint8, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)

        self._write(arr, dst)
        return str(dst)

    # ---------------- Public API (FOLDER) ----------------

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        kind_mode: KindMode = "auto",
        copy_if_already_aligned: bool = True,
    ) -> List[str]:
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            raise FileNotFoundError(f"No existe input_dir: {in_dir}")

        src_files = sorted(
            [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
        )
        if not src_files:
            raise FileNotFoundError(f"No se encontraron .tif/.tiff dentro de: {in_dir}")

        outputs: List[str] = []

        for src in src_files:
            dst = self._build_output_path_folder(src, in_dir, out_dir)

            if dst.exists() and not self.overwrite:
                continue

            out = self.process_file(
                input_path=str(src),
                output_path=str(dst),
                kind_mode=kind_mode,
                copy_if_already_aligned=copy_if_already_aligned,
            )
            outputs.append(out)

        return outputs
