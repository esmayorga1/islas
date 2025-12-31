from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio


class RasterNormalizer01File:
    """
    Normaliza un raster continuo a rango [0,1] (POR ARCHIVO)
    y (opcional) también puede normalizar una carpeta completa.

    - No remuestrea
    - No reproyecta
    - No alinea
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        overwrite: bool = False,
        nodata_default: float = -9999.0,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.overwrite = overwrite
        self.nodata_default = nodata_default

    # ----------------- helpers -----------------

    @staticmethod
    def _is_categorical(arr: np.ndarray) -> bool:
        vals = np.unique(arr)
        return len(vals) <= 10 and np.all(np.mod(vals, 1) == 0)

    @staticmethod
    def _normalize_01(arr: np.ndarray, nodata: float) -> np.ndarray:
        mask = arr != nodata
        valid = arr[mask]

        if valid.size == 0:
            return arr.astype(np.float32)

        vmin = float(valid.min())
        vmax = float(valid.max())

        out = np.full(arr.shape, nodata, dtype=np.float32)

        if vmax == vmin:
            out[mask] = 0.0
            return out

        out[mask] = (valid - vmin) / (vmax - vmin)
        return out

    def _build_output_path_for_folder(self, src: Path, input_dir: Path, output_dir: Path) -> Path:
        rel = src.relative_to(input_dir)
        return output_dir / rel.parent / f"{src.stem}_norm.tif"

    def _resolve_output_file(self, src: Path, out: Path) -> Path:
        # Si output_path es carpeta → autogenera nombre *_norm.tif
        if out.exists() and out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            return out / f"{src.stem}_norm.tif"

        if out.suffix.lower() not in [".tif", ".tiff"]:
            # Si no parece archivo tif, lo tratamos como carpeta aunque no exista
            out.mkdir(parents=True, exist_ok=True)
            return out / f"{src.stem}_norm.tif"

        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    # ----------------- API -----------------

    def process_file(self) -> str:
        """
        Normaliza UN tif: input_path (archivo) -> output_path (archivo o carpeta).
        """
        if not self.input_path.exists() or self.input_path.is_dir():
            raise FileNotFoundError(f"El input_path no es un archivo .tif válido: {self.input_path}")

        dst = self._resolve_output_file(self.input_path, self.output_path)

        if dst.exists() and not self.overwrite:
            return str(dst)

        with rasterio.open(self.input_path) as ds:
            arr = ds.read(1)
            profile = ds.profile.copy()
            nodata = ds.nodata if ds.nodata is not None else self.nodata_default

        valid = arr[arr != nodata]

        # Si parece categórico, lo copia (no normaliza)
        if valid.size > 0 and self._is_categorical(valid):
            out_arr = arr.astype(np.uint8)
            profile.update(dtype="uint8", nodata=nodata)
        else:
            out_arr = self._normalize_01(arr, nodata)
            profile.update(dtype="float32", nodata=nodata)

        with rasterio.open(dst, "w", **profile) as out:
            out.write(out_arr, 1)

        return str(dst)

    def process_folder(self) -> List[str]:
        """
        Compatibilidad: si input_path es carpeta, normaliza todos los tif dentro
        y los escribe manteniendo estructura en output_path.
        """
        if not self.input_path.exists() or not self.input_path.is_dir():
            raise FileNotFoundError(f"El input_path no es una carpeta válida: {self.input_path}")

        input_dir = self.input_path
        output_dir = self.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        rasters = list(input_dir.rglob("*.tif"))
        if not rasters:
            raise FileNotFoundError("No se encontraron archivos .tif")

        outputs = []

        for src in rasters:
            dst = self._build_output_path_for_folder(src, input_dir, output_dir)

            if dst.exists() and not self.overwrite:
                continue

            with rasterio.open(src) as ds:
                arr = ds.read(1)
                profile = ds.profile.copy()
                nodata = ds.nodata if ds.nodata is not None else self.nodata_default

            valid = arr[arr != nodata]

            if valid.size > 0 and self._is_categorical(valid):
                out_arr = arr.astype(np.uint8)
                profile.update(dtype="uint8", nodata=nodata)
            else:
                out_arr = self._normalize_01(arr, nodata)
                profile.update(dtype="float32", nodata=nodata)

            dst.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst, "w", **profile) as out:
                out.write(out_arr, 1)

            outputs.append(str(dst))

        return outputs
