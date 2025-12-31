from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

import numpy as np
import rasterio
from rasterio.windows import Window


WriteMode = Literal["sample", "full"]
MaskMode = Literal["all_valid", "any_valid"]


class ClusterGeoTIFFWriter:
    """
    ✅ Dos modos:

    1) mode="sample" (como lo tenías):
       Reconstruye raster (H,W) desde coords (N,2) + cluster_ids (N,) y guarda GeoTIFF.

       ⚠️ Esto deja MUCHOS nodata si X fue muestreado (sample_max_pixels).

    2) mode="full" (NUEVO, recomendado):
       Usa meta['sources'] (rutas a rasters alineados por variable) y el modelo SOM exportado
       para predecir clusters para TODO el raster por ventanas (tiles).
       => Sale CLUSTERS_TIF completo (sin huecos salvo nodata real).
    """

    def __init__(
        self,
        nodata: int = -1,
        compress: str = "lzw",
        # FULL mode
        tile_size: int = 256,
        mask_mode: MaskMode = "all_valid",
        nodata_default: float = -9999.0,
    ):
        self.nodata = int(nodata)
        self.compress = str(compress)

        self.tile_size = int(tile_size)
        self.mask_mode = mask_mode
        self.nodata_default = float(nodata_default)

    # ------------------------------------------------------------------
    # MODE 1: SAMPLE (igual que antes)
    # ------------------------------------------------------------------

    def build_raster(self, coords: np.ndarray, cluster_ids: np.ndarray, shape_hw) -> np.ndarray:
        h, w = shape_hw
        out = np.full((h, w), self.nodata, dtype=np.int32)

        rr = coords[:, 0].astype(np.int64)
        cc = coords[:, 1].astype(np.int64)
        out[rr, cc] = cluster_ids.astype(np.int32)

        return out

    # ------------------------------------------------------------------
    # MODE 2: FULL (NUEVO)
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return (X - mu) / (sd + 1e-9)

    def _make_valid_mask(self, arrays: List[np.ndarray], nodata: float) -> np.ndarray:
        valids = [(a != nodata) for a in arrays]
        if self.mask_mode == "all_valid":
            return np.logical_and.reduce(valids)
        return np.logical_or.reduce(valids)

    def _predict_clusters_block(
        self,
        X: np.ndarray,
        w_flat: np.ndarray,
        som_cols: int,
    ) -> np.ndarray:
        """
        X: (N,C)
        w_flat: (K,C)
        retorna cluster_id (N,)
        """
        # distancia euclidiana a cada neurona: argmin ||x-w||
        # (N,1,C) - (1,K,C) -> (N,K,C) -> sum -> (N,K)
        d2 = np.sum((X[:, None, :] - w_flat[None, :, :]) ** 2, axis=2)
        winners = np.argmin(d2, axis=1).astype(np.int32)
        return winners  # ya es r*som_cols + c por cómo aplanamos

    def _full_write(
        self,
        out_path: str,
        meta: Dict[str, Any],
        som_model: Dict[str, Any],
    ) -> str:
        """
        som_model es lo que retorna SOMYearRunner.export_model().
        meta['sources'] debe tener rutas a los GeoTIFF por variable.
        """
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        variables: List[str] = meta["variables"]
        sources: Dict[str, str] = meta["sources"]

        # leer perfil desde cualquier raster fuente
        any_source = next(iter(sources.values()))
        with rasterio.open(any_source) as ref:
            profile = ref.profile.copy()
            H, W = ref.height, ref.width
            nodata = ref.nodata if ref.nodata is not None else self.nodata_default

        # cargar modelo SOM
        weights = som_model["weights"]  # (rows, cols, C)
        som_rows = int(som_model["som_rows"])
        som_cols = int(som_model["som_cols"])
        standardize = bool(som_model.get("standardize", False))
        mu = som_model.get("mu", None)
        sd = som_model.get("sd", None)

        w_flat = np.array(weights, dtype=np.float32).reshape(som_rows * som_cols, -1)

        if standardize:
            if mu is None or sd is None:
                raise ValueError("standardize=True pero som_model no trae mu/sd.")
            mu = np.array(mu, dtype=np.float32)
            sd = np.array(sd, dtype=np.float32)
            w_flat = self._standardize(w_flat, mu, sd)

        # preparar salida (escritura por bloques)
        profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=self.nodata,
            compress=self.compress,
        )

        # abrir todas las fuentes una vez
        srcs = {v: rasterio.open(sources[v]) for v in variables}
        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                for row0 in range(0, H, self.tile_size):
                    for col0 in range(0, W, self.tile_size):
                        h = min(self.tile_size, H - row0)
                        w = min(self.tile_size, W - col0)
                        win = Window(col0, row0, w, h)

                        # leer bloque por variable
                        arrays = []
                        for v in variables:
                            a = srcs[v].read(1, window=win).astype(np.float32)
                            arrays.append(a)

                        # máscara válida SOLO en este bloque
                        valid = self._make_valid_mask(arrays, nodata)

                        # salida bloque: nodata por defecto
                        out_block = np.full((h, w), self.nodata, dtype=np.int32)

                        if np.any(valid):
                            # construir X solo para píxeles válidos
                            X = np.stack([a[valid] for a in arrays], axis=1).astype(np.float32)

                            if standardize:
                                X = self._standardize(X, mu, sd)

                            winners = self._predict_clusters_block(X, w_flat=w_flat, som_cols=som_cols)
                            out_block[valid] = winners

                        dst.write(out_block, 1, window=win)

        finally:
            for s in srcs.values():
                s.close()

        return out_path

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def save(
        self,
        out_path: str,
        meta: Dict[str, Any],
        mode: WriteMode = "sample",
        # sample mode
        cluster_raster: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
        shape_hw: Optional[tuple] = None,
        # full mode
        som_model: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        mode="sample":
          - provee (coords, cluster_ids, shape_hw) o cluster_raster ya armado.

        mode="full":
          - provee som_model (export_model) y meta con sources+variables.
        """
        out_path = str(out_path)

        if mode == "full":
            if som_model is None:
                raise ValueError("mode='full' requiere som_model=export_model().")
            return self._full_write(out_path, meta=meta, som_model=som_model)

        # mode == "sample"
        if cluster_raster is None:
            if coords is None or cluster_ids is None or shape_hw is None:
                raise ValueError("mode='sample' requiere cluster_raster o (coords, cluster_ids, shape_hw).")
            cluster_raster = self.build_raster(coords, cluster_ids, shape_hw)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        any_source = next(iter(meta["sources"].values()))
        with rasterio.open(any_source) as ref:
            profile = ref.profile.copy()

        profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=self.nodata,
            compress=self.compress,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(cluster_raster.astype(np.int32), 1)

        return out_path
