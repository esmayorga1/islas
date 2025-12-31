import os
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin


ResampleMode = Literal["direct", "multistage"]


class FolderTo3mResampler4326:
    """
    Convierte TODOS los .tif dentro de input_dir (recursivo) a una grilla ~3 m (EPSG:4326),
    guardando en output_dir con la MISMA estructura y agregando _3m al nombre.

    - Si align_to_path se entrega: todo se alinea EXACTO a ese raster (recomendado).
    - Si no se entrega: crea plantilla 3 m con bounds del primer .tif encontrado.
    - Clip (0-1) se decide cuando se llama process_folder(apply_clip=True).
    """

    DEG_TO_M = 111_000.0  # aproximación válida para Bogotá

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        align_to_path: Optional[str] = None,   # <<< raster al que se alinea TODO (opcional pero recomendado)
        target_m: float = 3.0,
        mode: ResampleMode = "direct",
        multistage_m: Optional[List[float]] = None,  # p.ej [6.0, 3.0]
        continuous_method: Literal["bilinear", "cubic"] = "bilinear",
        categorical_method: Literal["nearest"] = "nearest",
        overwrite: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.align_to_path = align_to_path
        self.target_m = float(target_m)
        self.target_deg = self.target_m / self.DEG_TO_M

        self.mode = mode
        self.multistage_m = multistage_m or [6.0, 3.0]
        self.overwrite = overwrite

        self.continuous_resampling = Resampling.bilinear if continuous_method == "bilinear" else Resampling.cubic
        self.categorical_resampling = Resampling.nearest  # por ahora solo nearest

        self.template_profile: Optional[dict] = None

    # ----------------- Template / Grid -----------------

    def _read_template_profile(self, template_path: str) -> dict:
        with rasterio.open(template_path) as src:
            prof = src.profile.copy()
            prof.update(count=1)  # nos sirve 1 banda como plantilla
            return prof

    def _make_template_from_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        crs="EPSG:4326",
        nodata: float = -9999.0,
        dtype="float32",
        res_deg: Optional[float] = None,
    ) -> dict:
        minx, miny, maxx, maxy = bounds
        res = res_deg if res_deg is not None else self.target_deg

        width = int(np.ceil((maxx - minx) / res))
        height = int(np.ceil((maxy - miny) / res))
        transform = from_origin(minx, maxy, res, res)

        return {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

    def _ensure_template(self, fallback_src_path: str) -> dict:
        """
        Prioridad:
        1) align_to_path (si existe)
        2) template generado por bounds del primer raster encontrado
        """
        if self.template_profile is not None:
            return self.template_profile

        if self.align_to_path:
            self.template_profile = self._read_template_profile(self.align_to_path)
            return self.template_profile

        # si no hay align_to_path: creamos plantilla desde el primer raster (fallback)
        with rasterio.open(fallback_src_path) as src:
            b = src.bounds
            self.template_profile = self._make_template_from_bounds(
                (b.left, b.bottom, b.right, b.top),
                crs=src.crs or "EPSG:4326",
                nodata=src.nodata if src.nodata is not None else -9999.0,
                dtype="float32",
                res_deg=self.target_deg,
            )
        return self.template_profile

    # ----------------- Core warp -----------------

    def _reproject_to_profile(
        self,
        src_path: str,
        dst_profile: dict,
        resampling: Resampling,
        dst_dtype: str,
    ) -> np.ndarray:
        with rasterio.open(src_path) as src:
            src_arr = src.read(1)
            src_prof = src.profile

            dst = np.full(
                (dst_profile["height"], dst_profile["width"]),
                dst_profile.get("nodata", -9999.0),
                dtype=np.dtype(dst_dtype),
            )

            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src_prof["transform"],
                src_crs=src_prof.get("crs", dst_profile.get("crs")),
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile.get("crs", src_prof.get("crs")),
                resampling=resampling,
                src_nodata=src_prof.get("nodata"),
                dst_nodata=dst_profile.get("nodata"),
            )
            return dst

    def _write_raster(self, array: np.ndarray, profile: dict, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prof = profile.copy()
        prof.update(count=1, dtype=str(array.dtype))

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(array, 1)

    # ----------------- Helpers -----------------

    @staticmethod
    def _clip_range(arr: np.ndarray, nodata: float, clip_range: Tuple[float, float]) -> np.ndarray:
        cmin, cmax = clip_range
        mask = (arr != nodata)
        arr = arr.astype(np.float32, copy=False)
        arr[mask] = np.clip(arr[mask], cmin, cmax)
        return arr

    @staticmethod
    def _infer_kind_from_name(name_upper: str) -> str:
        """
        Heurística simple: ajusta si quieres.
        - categórico: segmentaciones, suelo/clase/cobertura, binario.
        """
        categorical_hints = ["SEG", "SEGMENT", "SUELO", "SOIL", "CLASE", "CLASIF", "COBERT", "URBAN", "BUILT"]
        if any(h in name_upper for h in categorical_hints):
            return "categorical"
        return "continuous"

    def _build_output_path(self, src_path: Path) -> Path:
        """
        Mantiene estructura:
        output_dir / (ruta_relativa_desde_input_dir) / nombre_3m.tif
        """
        rel = src_path.relative_to(self.input_dir)
        out_rel_dir = rel.parent

        stem = src_path.stem
        out_name = f"{stem}_3m.tif"
        return self.output_dir / out_rel_dir / out_name

    # ----------------- Public API -----------------

    def process_folder(
        self,
        apply_clip: bool = True,
        clip_range: Tuple[float, float] = (0.0, 1.0),
        kind_mode: Literal["auto", "continuous", "categorical"] = "auto",
    ) -> List[str]:
        """
        Procesa todos los .tif del input_dir (recursivo).
        - apply_clip: si True, recorta continuos al rango (ej 0-1) (NO aplica a categóricos)
        - kind_mode:
            - "auto": infiere por nombre
            - "continuous": fuerza todo como continuo
            - "categorical": fuerza todo como categórico (NO recomendado si hay índices)
        Devuelve lista de rutas generadas.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"No existe input_dir: {self.input_dir}")

        src_files = sorted(self.input_dir.rglob("*.tif"))
        if not src_files:
            raise FileNotFoundError(f"No se encontraron .tif dentro de: {self.input_dir}")

        # inicializa template con el primer raster (o align_to)
        template = self._ensure_template(str(src_files[0]))
        nodata = template.get("nodata", -9999.0)

        outputs = []

        for src in src_files:
            dst = self._build_output_path(src)

            if dst.exists() and not self.overwrite:
                continue

            # determinar tipo
            if kind_mode == "auto":
                kind = self._infer_kind_from_name(src.stem.upper())
            else:
                kind = kind_mode

            # elegir resampling y dtype
            if kind == "categorical":
                resampling = self.categorical_resampling
                dst_dtype = "uint8"
            else:
                resampling = self.continuous_resampling
                dst_dtype = "float32"

            # direct (por ahora; multistage lo puedo reactivar si lo necesitas)
            if self.mode == "direct":
                arr = self._reproject_to_profile(str(src), template, resampling=resampling, dst_dtype=dst_dtype)
            else:
                # MULTISTAGE: aplica etapas SOLO a continuos; categóricos mejor directo
                if kind == "categorical":
                    arr = self._reproject_to_profile(str(src), template, resampling=resampling, dst_dtype=dst_dtype)
                else:
                    # bounds del template para coherencia
                    tr = template["transform"]
                    minx = tr.c
                    maxy = tr.f
                    maxx = minx + template["width"] * tr.a
                    miny = maxy + template["height"] * tr.e
                    bounds = (minx, miny, maxx, maxy)

                    current_path = str(src)
                    tmp_paths: List[Path] = []

                    for i, m in enumerate(self.multistage_m):
                        if m <= self.target_m:
                            continue
                        deg = m / self.DEG_TO_M

                        stage_profile = self._make_template_from_bounds(
                            bounds=bounds,
                            crs=template.get("crs", "EPSG:4326"),
                            nodata=nodata,
                            dtype="float32",
                            res_deg=deg,
                        )

                        stage_arr = self._reproject_to_profile(
                            current_path,
                            stage_profile,
                            resampling=self.continuous_resampling,
                            dst_dtype="float32",
                        )

                        tmp_path = self.output_dir / f"__tmp_stage_{i}__{src.stem}.tif"
                        self._write_raster(stage_arr, stage_profile, tmp_path)
                        tmp_paths.append(tmp_path)
                        current_path = str(tmp_path)

                    arr = self._reproject_to_profile(current_path, template, resampling=self.continuous_resampling, dst_dtype="float32")

                    # limpiar temporales
                    for tp in tmp_paths:
                        try:
                            tp.unlink()
                        except:
                            pass

            # post
            if kind == "categorical":
                arr = arr.astype(np.uint8)
            else:
                arr = arr.astype(np.float32)
                if apply_clip:
                    arr = self._clip_range(arr, nodata=nodata, clip_range=clip_range)

            self._write_raster(arr, template, dst)
            outputs.append(str(dst))

        return outputs
