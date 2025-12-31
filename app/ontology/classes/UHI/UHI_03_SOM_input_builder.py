from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, List

import numpy as np
import rasterio

MaskMode = Literal["all_valid", "any_valid"]


class SOMInputBuilder:
    """
    Construye matrices de entrada para SOM por mes (YYYY_MM) a partir de rasters ya:
      - remuestreados a 3 m
      - alineados (misma grilla)
      - EPSG:4326

    Soporta 2 tipos de variables:
      1) Mensuales (con mes):
         S2_2021_2_PatioBonito_LST_3m.tif   -> 2021_02
         S2_2021_02_PatioBonito_NDVI_3m.tif -> 2021_02

      2) Estáticas (una sola vez, sin mes; pueden o no tener año):
         construcciones_3m.tif
         vias_3m.tif
         distancia_agua_3m_ALINEADO_norm.tif
         ...
    """

    def __init__(
        self,
        input_dir: str,
        monthly_patterns: Dict[str, str],
        static_patterns: Optional[Dict[str, str]] = None,  # ✅ NUEVO (y único extra)
        # ✅ mes como token: 2021_2_ o 2021_02_ o 2021_02(fin). NO toma 2021_3m
        month_regex: str = r"(20\d{2})[_-](0?[1-9]|1[0-2])(?=[_-]|$)",
        nodata_default: float = -9999.0,
        mask_mode: MaskMode = "all_valid",
        sample_max_pixels: Optional[int] = None,
        random_seed: int = 42,
        allow_missing_monthly_vars: bool = False,
        require_static: bool = False,  # ✅ si True, exige TODAS las static_patterns
        debug_examples: int = 12,
    ):
        self.input_dir = Path(input_dir)
        self.monthly_patterns = monthly_patterns
        self.static_patterns = static_patterns or {}

        self.month_regex_str = month_regex
        self.month_re = re.compile(month_regex)

        self.nodata_default = float(nodata_default)
        self.mask_mode = mask_mode

        self.sample_max_pixels = sample_max_pixels
        self.rng = np.random.default_rng(random_seed)

        self.allow_missing_monthly_vars = allow_missing_monthly_vars
        self.require_static = require_static
        self.debug_examples = int(debug_examples)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_month_key(self, stem: str) -> Optional[str]:
        m = self.month_re.search(stem)
        if not m:
            return None
        year = m.group(1)
        month = str(int(m.group(2))).zfill(2)
        return f"{year}_{month}"

    def _match_var(self, stem: str, patterns: Dict[str, str]) -> Optional[str]:
        for var, pat in patterns.items():
            if re.search(pat, stem, flags=re.IGNORECASE):
                return var
        return None

    # ------------------------------------------------------------------
    # Collect files
    # ------------------------------------------------------------------

    def _all_tifs(self) -> List[Path]:
        """
        ✅ Corrección Windows:
        - encuentra .tif/.tiff aunque sean .TIF/.TIFF
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"No existe input_dir: {self.input_dir}")

        files = sorted(self.input_dir.rglob("*"))
        tifs = [p for p in files if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]

        if not tifs:
            raise FileNotFoundError(f"No se encontraron .tif/.tiff en {self.input_dir}")
        return tifs

    def _collect_monthly(self) -> Dict[str, Dict[str, Path]]:
        tifs = self._all_tifs()
        groups: Dict[str, Dict[str, Path]] = {}

        for p in tifs:
            stem = p.stem
            month_key = self._extract_month_key(stem)
            if month_key is None:
                continue

            var = self._match_var(stem, self.monthly_patterns)
            if var is None:
                continue

            groups.setdefault(month_key, {})
            if var not in groups[month_key]:
                groups[month_key][var] = p

        return groups

    def _collect_static(self) -> Dict[str, Path]:
        """
        Capas estáticas (sin mes) que se aplican a todos los meses.
        - Se excluyen archivos mensuales (si tienen month_key).
        - Si hay múltiples candidatos para una var, toma el primero (ordenado).
        """
        if not self.static_patterns:
            return {}

        tifs = self._all_tifs()
        static_map: Dict[str, Path] = {}

        for p in tifs:
            stem = p.stem

            # ✅ estáticas: NO deben tener mes
            if self._extract_month_key(stem) is not None:
                continue

            var = self._match_var(stem, self.static_patterns)
            if var is None:
                continue

            if var not in static_map:
                static_map[var] = p

        return static_map

    # ------------------------------------------------------------------
    # Raster helpers
    # ------------------------------------------------------------------

    def _read_float(self, path: Path) -> Tuple[np.ndarray, dict, float]:
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            prof = ds.profile.copy()
            nodata = float(ds.nodata) if ds.nodata is not None else self.nodata_default
        return arr, prof, nodata

    def _validate_alignment(self, ref_prof: dict, prof: dict, key: str, var: str):
        if (prof["width"], prof["height"]) != (ref_prof["width"], ref_prof["height"]):
            raise ValueError(f"[{key}] {var}: tamaño no coincide.")
        if prof["transform"] != ref_prof["transform"]:
            raise ValueError(f"[{key}] {var}: transform distinto (NO alineado).")
        if str(prof.get("crs")) != str(ref_prof.get("crs")):
            raise ValueError(f"[{key}] {var}: CRS distinto.")

    def _make_mask(self, arrays: List[np.ndarray], nodata: float) -> np.ndarray:
        valids = [(a != nodata) for a in arrays]
        if self.mask_mode == "all_valid":
            return np.logical_and.reduce(valids)
        return np.logical_or.reduce(valids)

    def _sample_idx(self, mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask.ravel())
        if self.sample_max_pixels is None or idx.size <= self.sample_max_pixels:
            return idx
        return np.sort(self.rng.choice(idx, size=self.sample_max_pixels, replace=False))

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def _fmt_examples(self, paths: List[Path], limit: Optional[int] = None) -> str:
        limit = self.debug_examples if limit is None else int(limit)
        show = paths[:limit]
        lines = "\n".join([f"  - {p.name}" for p in show])
        if len(paths) > limit:
            lines += f"\n  ... (+{len(paths) - limit} más)"
        return lines if lines else "  (sin ejemplos)"

    def _monthly_candidates_for_key(self, month_key: str) -> List[Path]:
        candidates = []
        for p in self._all_tifs():
            mk = self._extract_month_key(p.stem)
            if mk == month_key:
                candidates.append(p)
        return candidates

    def diagnose(self, year: str = "2021") -> dict:
        monthly = self._collect_monthly()
        static_map = self._collect_static()

        months = sorted([k for k in monthly.keys() if k.startswith(f"{year}_")])
        month_vars = {m: sorted(list(monthly[m].keys())) for m in months}

        return {
            "input_dir": str(self.input_dir),
            "tif_total": len(self._all_tifs()),
            "month_regex": self.month_regex_str,
            "months_detected": months,
            "month_to_vars": month_vars,
            "static_vars_detected": sorted(list(static_map.keys())),
            "static_sources": {k: str(v) for k, v in static_map.items()},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Dict[str, object]]:
        monthly = self._collect_monthly()
        static_map = self._collect_static()

        if not monthly:
            all_tifs = self._all_tifs()
            raise ValueError(
                "No se detectaron capas mensuales.\n"
                f"input_dir: {self.input_dir}\n"
                f"month_regex='{self.month_regex_str}'\n"
                f"monthly_patterns={self.monthly_patterns}\n\n"
                f"Ejemplos vistos:\n{self._fmt_examples(all_tifs)}\n"
            )

        # ✅ validar estáticas si el usuario lo exige
        if self.static_patterns and self.require_static:
            missing_s = [v for v in self.static_patterns.keys() if v not in static_map]
            if missing_s:
                raise ValueError(
                    f"Faltan capas estáticas requeridas: {missing_s}\n"
                    f"Encontradas: {sorted(list(static_map.keys()))}\n"
                    f"static_patterns: {self.static_patterns}"
                )

        results: Dict[str, Dict[str, object]] = {}
        expected_monthly = list(self.monthly_patterns.keys())
        expected_static = list(self.static_patterns.keys())

        for month_key, var_map in monthly.items():
            # -------------------------
            # Validación de mensuales
            # -------------------------
            if not self.allow_missing_monthly_vars:
                missing_m = [v for v in expected_monthly if v not in var_map]
                if missing_m:
                    candidates = self._monthly_candidates_for_key(month_key)
                    raise ValueError(
                        f"[{month_key}] faltan variables mensuales: {missing_m}.\n"
                        f"Encontradas: {sorted(var_map.keys())}\n\n"
                        f"Ejemplos para {month_key}:\n{self._fmt_examples(candidates)}\n"
                        f"Patrones mensuales: {self.monthly_patterns}"
                    )

            merged: Dict[str, Path] = dict(var_map)

            # -------------------------
            # ✅ Añadir estáticas (si existen)
            # -------------------------
            if self.static_patterns and static_map:
                merged.update(static_map)

            # Orden columnas: mensuales en orden + estáticas en orden
            ordered_vars: List[str] = []
            for v in expected_monthly:
                if v in merged:
                    ordered_vars.append(v)
            for v in expected_static:
                if v in merged:
                    ordered_vars.append(v)

            if not ordered_vars:
                raise ValueError(f"[{month_key}] no hay variables para construir X.")

            # Leer ref
            ref_arr, ref_prof, nodata = self._read_float(merged[ordered_vars[0]])
            arrays = [ref_arr]

            # Leer y validar alineación
            for v in ordered_vars[1:]:
                arr, prof, _ = self._read_float(merged[v])
                self._validate_alignment(ref_prof, prof, month_key, v)
                arrays.append(arr)

            mask = self._make_mask(arrays, nodata)
            idx = self._sample_idx(mask)

            width = ref_prof["width"]
            rows = (idx // width).astype(np.int32)
            cols = (idx % width).astype(np.int32)
            coords = np.stack([rows, cols], axis=1)

            X = np.stack([a.ravel()[idx] for a in arrays], axis=1).astype(np.float32)

            meta = {
                "month": month_key,
                "variables": ordered_vars,
                "shape": (ref_prof["height"], ref_prof["width"]),
                "crs": str(ref_prof.get("crs")),
                "transform": ref_prof["transform"],
                "nodata": nodata,
                "sources": {v: str(merged[v]) for v in ordered_vars},
                "static_vars": sorted(list(static_map.keys())),
            }

            results[month_key] = {"X": X, "coords": coords, "meta": meta}

        return results

    @staticmethod
    def save_npz(out_path: str, X: np.ndarray, coords: np.ndarray, meta: dict):
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, X=X, coords=coords, meta=str(meta))
