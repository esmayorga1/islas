from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import csv


class ClusterStatsCalculator:
    """
    Calcula métricas por cluster a partir de:
      - X: (N,C) valores por pixel muestreado
      - cluster_ids: (N,) cluster asignado por pixel
      - variables: lista de nombres (C)

    Devuelve una lista de dicts lista para CSV.
    """

    def __init__(self, min_pixels_per_cluster: int = 50):
        self.min_pixels_per_cluster = int(min_pixels_per_cluster)

    def compute(
        self,
        X: np.ndarray,
        cluster_ids: np.ndarray,
        variables: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (N,C).")
        if cluster_ids.ndim != 1:
            raise ValueError("cluster_ids debe ser 1D (N,).")
        if X.shape[0] != cluster_ids.shape[0]:
            raise ValueError("X y cluster_ids no tienen el mismo N.")
        if len(variables) != X.shape[1]:
            raise ValueError("variables no coincide con C columnas de X.")

        extra = extra or {}

        rows: List[Dict[str, Any]] = []
        unique_clusters = np.unique(cluster_ids)

        for cid in unique_clusters:
            idx = np.where(cluster_ids == cid)[0]
            n = int(idx.size)
            if n < self.min_pixels_per_cluster:
                continue

            Xi = X[idx, :]
            means = Xi.mean(axis=0)
            stds = Xi.std(axis=0)

            row: Dict[str, Any] = {
                **extra,
                "cluster_id": int(cid),
                "n_pixels": n,
            }

            for j, v in enumerate(variables):
                row[f"mean_{v}"] = float(means[j])
                row[f"std_{v}"] = float(stds[j])

            rows.append(row)

        rows.sort(key=lambda r: r["cluster_id"])
        return rows

    @staticmethod
    def save_csv(rows: List[Dict[str, Any]], out_csv: str) -> str:
        out_csv = str(out_csv)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            raise ValueError("No hay filas para guardar (rows vacío).")

        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        return out_csv
