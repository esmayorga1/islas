from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import ast

# pip install minisom
from minisom import MiniSom

# ✅ para reconstruir Affine(...) dentro del meta
from affine import Affine


@dataclass
class MonthPack:
    month_key: str
    X: np.ndarray
    coords: np.ndarray
    meta: Dict[str, Any]


class SOMYearRunner:
    """
    Runner GENERAL por año para tus NPZ:
      SOM_INPUT_YYYY_MM.npz

    Modo recomendado:
      - SOM ANUAL: entrena 1 SOM con X concatenado de todos los meses del año
        y luego predice clusters para cada mes (comparables entre meses).
    """

    def __init__(
        self,
        npz_dir: str,
        som_rows: int = 10,
        som_cols: int = 10,
        sigma: float = 1.2,
        learning_rate: float = 0.5,
        random_seed: int = 42,
        standardize: bool = False,
        sample_per_month: Optional[int] = None,
    ):
        self.npz_dir = Path(npz_dir)

        self.som_rows = int(som_rows)
        self.som_cols = int(som_cols)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.random_seed = int(random_seed)

        self.standardize = bool(standardize)
        self.sample_per_month = sample_per_month

        self.som: Optional[MiniSom] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.variables: Optional[List[str]] = None

        self.rng = np.random.default_rng(self.random_seed)

    # -------------------------
    # NPZ IO (SIN eval inseguro)
    # -------------------------
    @staticmethod
    def _safe_parse_meta(meta_str: str) -> Dict[str, Any]:
        """
        Parsea meta guardado como str(meta) donde puede venir:
          'transform': Affine(a, b, c, d, e, f)

        Usamos ast.literal_eval (seguro) con un preprocesamiento mínimo:
        - reemplazamos "Affine(" por "('AFFINE', " para poder parsear
        - y luego reconstruimos Affine con los 6 valores.
        """
        s = meta_str

        # Convertir Affine(...) a una tupla parseable
        # Ej: Affine(1,0,0,0,-1,0) -> ('AFFINE', 1,0,0,0,-1,0)
        s = s.replace("Affine(", "('AFFINE', ")

        meta = ast.literal_eval(s)

        # reconstruir transform si viene como ('AFFINE', a,b,c,d,e,f)
        if isinstance(meta, dict) and "transform" in meta:
            tr = meta["transform"]
            if isinstance(tr, tuple) and len(tr) == 7 and tr[0] == "AFFINE":
                _, a, b, c, d, e, f = tr
                meta["transform"] = Affine(a, b, c, d, e, f)

        return meta

    @classmethod
    def load_npz(cls, npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        d = np.load(str(npz_path), allow_pickle=True)
        X = d["X"].astype(np.float32)
        coords = d["coords"].astype(np.int32)

        meta_raw = d["meta"]
        meta_str = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw

        if isinstance(meta_str, str):
            meta = cls._safe_parse_meta(meta_str)
        else:
            meta = meta_str

        return X, coords, meta

    def list_year_npz(self, year: str) -> List[Path]:
        year = str(year)
        return sorted(self.npz_dir.glob(f"SOM_INPUT_{year}_??.npz"))

    def load_year(self, year: str) -> List[MonthPack]:
        files = self.list_year_npz(year)
        if not files:
            raise FileNotFoundError(
                f"No encontré NPZ para el año {year} en {self.npz_dir}\n"
                f"Esperado: SOM_INPUT_{year}_01.npz ... SOM_INPUT_{year}_12.npz"
            )

        packs: List[MonthPack] = []
        for f in files:
            X, coords, meta = self.load_npz(f)
            month_key = meta.get("month", f.stem.replace("SOM_INPUT_", ""))
            packs.append(MonthPack(month_key=month_key, X=X, coords=coords, meta=meta))

        # validar variables consistentes
        vars0 = packs[0].meta.get("variables")
        for p in packs[1:]:
            if p.meta.get("variables") != vars0:
                raise ValueError(
                    f"Variables no consistentes en el año {year}.\n"
                    f"{packs[0].month_key}: {vars0}\n"
                    f"{p.month_key}: {p.meta.get('variables')}"
                )

        self.variables = list(vars0) if vars0 is not None else None
        return packs

    # -------------------------
    # Prep / Standardize
    # -------------------------
    def _maybe_sample(self, X: np.ndarray) -> np.ndarray:
        if self.sample_per_month is None or X.shape[0] <= self.sample_per_month:
            return X
        idx = self.rng.choice(X.shape[0], size=self.sample_per_month, replace=False)
        return X[idx, :]

    def _prep_fit_X(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        self.mu = X.mean(axis=0, keepdims=True)
        self.sd = X.std(axis=0, keepdims=True) + 1e-9
        return (X - self.mu) / self.sd

    def _prep_pred_X(self, X: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return X
        if self.mu is None or self.sd is None:
            raise RuntimeError("standardize=True pero no hay mu/sd (falta entrenar).")
        return (X - self.mu) / self.sd

    # -------------------------
    # Train annual SOM
    # -------------------------
    def fit_annual(self, packs: List[MonthPack], iters: Optional[int] = None) -> "SOMYearRunner":
        X_list = [self._maybe_sample(p.X) for p in packs]
        X_all = np.vstack(X_list).astype(np.float32)

        X_fit = self._prep_fit_X(X_all)

        self.som = MiniSom(
            self.som_rows,
            self.som_cols,
            input_len=X_fit.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function="gaussian",
            random_seed=self.random_seed,
        )
        self.som.random_weights_init(X_fit)

        if iters is None:
            iters = min(80000, max(10000, 5 * X_fit.shape[0]))

        self.som.train_random(X_fit, iters)
        return self

    # -------------------------
    # Predict
    # -------------------------
    def predict_cluster_ids(self, X: np.ndarray) -> np.ndarray:
        if self.som is None:
            raise RuntimeError("Primero entrena el SOM (fit_annual).")

        Xt = self._prep_pred_X(X)
        out = np.zeros((Xt.shape[0],), dtype=np.int32)
        for i, x in enumerate(Xt):
            r, c = self.som.winner(x)
            out[i] = int(r * self.som_cols + c)
        return out

    def predict_year(self, packs: List[MonthPack]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for p in packs:
            cluster_ids = self.predict_cluster_ids(p.X)
            results[p.month_key] = {
                "cluster_ids": cluster_ids,
                "coords": p.coords,
                "meta": p.meta,
                "X": p.X,
            }
        return results

    # -------------------------
    # Export model
    # -------------------------
    def export_model(self) -> Dict[str, Any]:
        if self.som is None:
            raise RuntimeError("No hay SOM entrenado.")
        return {
            "weights": self.som.get_weights(),
            "som_rows": self.som_rows,
            "som_cols": self.som_cols,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_seed,
            "standardize": self.standardize,
            "mu": self.mu,
            "sd": self.sd,
            "variables": np.array(self.variables, dtype=object) if self.variables else None,
        }
