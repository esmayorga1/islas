from pathlib import Path
from typing import List

import numpy as np
import rasterio


class RasterNormalizer01:
    """
    Normaliza TODOS los rasters continuos a rango [0,1].

    - Mantiene estructura de carpetas
    - Agrega sufijo _norm
    - No remuestrea
    - No reproyecta
    - No alinea
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        overwrite: bool = False,
        nodata_default: float = -9999.0,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.overwrite = overwrite
        self.nodata_default = nodata_default

    # ----------------- helpers -----------------

    @staticmethod
    def _is_categorical(arr: np.ndarray) -> bool:
        """
        Heurística simple:
        - pocos valores únicos
        - valores enteros
        """
        vals = np.unique(arr)
        return len(vals) <= 10 and np.all(np.mod(vals, 1) == 0)

    @staticmethod
    def _normalize_01(arr: np.ndarray, nodata: float) -> np.ndarray:
        mask = arr != nodata
        valid = arr[mask]

        if valid.size == 0:
            return arr.astype(np.float32)

        vmin = valid.min()
        vmax = valid.max()

        if vmax == vmin:
            out = np.full(arr.shape, nodata, dtype=np.float32)
            out[mask] = 0.0
            return out

        out = np.full(arr.shape, nodata, dtype=np.float32)
        out[mask] = (valid - vmin) / (vmax - vmin)
        return out

    def _build_output_path(self, src: Path) -> Path:
        rel = src.relative_to(self.input_dir)
        return self.output_dir / rel.parent / f"{src.stem}_norm.tif"

    # ----------------- API -----------------

    def process_folder(self) -> List[str]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"No existe: {self.input_dir}")

        rasters = list(self.input_dir.rglob("*.tif"))
        if not rasters:
            raise FileNotFoundError("No se encontraron archivos .tif")

        outputs = []

        for src in rasters:
            dst = self._build_output_path(src)

            if dst.exists() and not self.overwrite:
                continue

            with rasterio.open(src) as ds:
                arr = ds.read(1)
                profile = ds.profile.copy()
                nodata = ds.nodata if ds.nodata is not None else self.nodata_default

            # excluir nodata
            valid = arr[arr != nodata]

            # capas categóricas se copian
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
