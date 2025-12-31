from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio


class ICUPostProcessor:
    """
    Postproceso para agrupar ICU_SCORE y obtener islas coherentes.

    Mejoras vs versión anterior:
      - Reconstrucción por componentes: cada isla final toma un SOLO valor (máximo ICU original dentro del componente)
        => resultado MUCHO más agrupado y consistente.
      - Opción de convertir NODATA a 0 (solo para salida/visualización, no "inventa" ICU).

    Pasos:
      1) mask = valid & (ICU >= min_class)
      2) closing: dilate -> erode (close_iters veces)
      3) remove_small_patches por min_area_m2
      4) reconstrucción:
           - fuera de máscara: 0 (o conserva 1 si keep_low_class=True)
           - dentro de cada componente: asigna el MAX ICU original del componente
    """

    def __init__(
        self,
        nodata: int = -1,
        min_class: int = 2,
        kernel_size: int = 9,         # ⬅️ sube a 9/11 para agrupar más
        close_iters: int = 2,         # ⬅️ sube a 2/3 para agrupar más
        min_area_m2: float = 1500.0,  # ⬅️ sube para eliminar islas pequeñas
        connectivity: int = 8,
        compress: str = "lzw",
        keep_low_class: bool = False,

        # ✅ opcional: si quieres que no queden nodata en la salida
        fill_nodata_to_zero: bool = True,
    ):
        self.nodata = int(nodata)
        self.min_class = int(min_class)
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar (3,5,7,9,11...).")

        self.close_iters = int(close_iters)
        self.min_area_m2 = float(min_area_m2)
        self.connectivity = int(connectivity)
        self.compress = str(compress)
        self.keep_low_class = bool(keep_low_class)
        self.fill_nodata_to_zero = bool(fill_nodata_to_zero)

    # -------------------------
    # Binary morphology (numpy)
    # -------------------------
    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        r = k // 2
        pad = np.pad(mask, r, mode="edge")
        out = np.zeros_like(mask, dtype=bool)

        h, w = mask.shape
        for i in range(h):
            for j in range(w):
                out[i, j] = np.any(pad[i:i + k, j:j + k])
        return out

    def _erode(self, mask: np.ndarray) -> np.ndarray:
        k = self.kernel_size
        r = k // 2
        pad = np.pad(mask, r, mode="edge")
        out = np.zeros_like(mask, dtype=bool)

        h, w = mask.shape
        for i in range(h):
            for j in range(w):
                out[i, j] = np.all(pad[i:i + k, j:j + k])
        return out

    def _closing(self, mask: np.ndarray) -> np.ndarray:
        out = mask
        for _ in range(max(1, self.close_iters)):
            out = self._dilate(out)
            out = self._erode(out)
        return out

    # -------------------------
    # Connected components helpers
    # -------------------------
    def _neighbors(self, r: int, c: int):
        if self.connectivity == 4:
            return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
            (r, c - 1),                 (r, c + 1),
            (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
        ]

    def _connected_components(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        comps: List[List[Tuple[int, int]]] = []

        for i in range(h):
            for j in range(w):
                if visited[i, j] or not mask[i, j]:
                    continue

                stack = [(i, j)]
                visited[i, j] = True
                comp = [(i, j)]

                while stack:
                    r, c = stack.pop()
                    for nr, nc in self._neighbors(r, c):
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if visited[nr, nc]:
                            continue
                        if mask[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            comp.append((nr, nc))

                comps.append(comp)

        return comps

    def _remove_small_patches(self, mask: np.ndarray, pixel_area_m2: float) -> np.ndarray:
        min_pixels = int(np.ceil(self.min_area_m2 / pixel_area_m2))
        out = mask.copy()

        comps = self._connected_components(out)
        for comp in comps:
            if len(comp) < min_pixels:
                for (r, c) in comp:
                    out[r, c] = False

        return out

    # -------------------------
    # Reconstruction (component-wise)
    # -------------------------
    def _reconstruct_scores_by_components(self, arr: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
        out = arr.copy()

        valid = (arr != self.nodata)

        # fuera de isla
        if self.keep_low_class:
            out[~final_mask & valid & (out != 1)] = 0
        else:
            out[~final_mask & valid] = 0

        # dentro de islas: asignar por componente el MAX ICU original (más agrupado)
        comps = self._connected_components(final_mask)
        for comp in comps:
            rr = np.array([p[0] for p in comp], dtype=np.int64)
            cc = np.array([p[1] for p in comp], dtype=np.int64)

            vals = arr[rr, cc]
            vals = vals[(vals != self.nodata)]
            if vals.size == 0:
                # componente raro sin valores válidos
                out[rr, cc] = self.min_class
                continue

            max_val = int(np.max(vals))
            if max_val < self.min_class:
                max_val = self.min_class

            out[rr, cc] = max_val

        # opcional: no dejar nodata en salida
        if self.fill_nodata_to_zero:
            out[~valid] = 0

        return out.astype(np.int16)

    # -------------------------
    # Public API
    # -------------------------
    def run(self, in_tif: str, out_tif: str) -> str:
        with rasterio.open(in_tif) as src:
            arr = src.read(1).astype(np.int16)
            profile = src.profile.copy()

        # Pixel ~3m => 9 m² (tu pipeline 3m)
        pixel_area_m2 = 9.0

        valid = (arr != self.nodata)

        # 1) máscara binaria de isla
        mask = valid & (arr >= self.min_class)

        # 2) closing para unir y rellenar
        mask2 = self._closing(mask)

        # 3) eliminar parches pequeños
        mask3 = self._remove_small_patches(mask2, pixel_area_m2=pixel_area_m2)

        # 4) reconstruir ICU_SCORE (más agrupado)
        out_arr = self._reconstruct_scores_by_components(arr, mask3)

        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=(0 if self.fill_nodata_to_zero else self.nodata),
            compress=self.compress,
        )

        Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(out_arr, 1)

        return out_tif
