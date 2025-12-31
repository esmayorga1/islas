from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import numpy as np
import rasterio


class ICUClusterLabeler:
    """
    Etiqueta clusters SOM como Islas de Calor Urbano (ICU/UHI) usando TODAS las variables
    (mensuales + estáticas) y genera un raster ICU_SCORE:

      0 = no ICU
      1 = ICU baja
      2 = ICU media
      3 = ICU alta

    ✅ Idea central:
    - Se calcula un "HOT_SCORE" ponderado por cluster (a partir de estadísticas mean_* por cluster).
    - Se le da MÁS PESO a la Temperatura del Aire (TAIR).
    - LST se usa como variable indirecta (peso menor que TAIR).
    - Variables de urbanización aumentan la probabilidad de ICU.
    - Variables de agua/vegetación/humedad tienden a disminuir ICU (se evalúan como "bajas").
    - Distancias normalizadas: mayor distancia al agua suele asociarse a mayor ICU.

    Requisitos de entrada:
      - cluster_tif: raster de cluster_id (int), nodata=-1
      - stats_csv: CSV con columnas:
          cluster_id, n_pixels,
          mean_TAIR, mean_LST, mean_HR,
          mean_NDVI, mean_NDBI, mean_MNDWI, mean_ALBEDO,
          mean_CONSTRUCCIONES_BIN, mean_VIAS_BIN, mean_PORC_URBANO,
          mean_CUERPOS_AGUA_BIN, mean_VEGETACION_BIN,
          mean_DIST_AGUA_NORM, mean_DIST_VIAS_NORM

    Nota: si tus columnas se llaman distinto, ajusta `aliases` en __init__.
    """

    # ----------------------------
    # Constructor
    # ----------------------------
    def __init__(
        self,
        nodata_cluster: int = -1,
        nodata_out: int = -1,
        min_pixels_per_cluster: int = 800,
        compress: str = "lzw",
        # Percentiles para convertir variables en "condiciones" hot/cool:
        p_hot: float = 90.0,   # "alto" (caliente/urbano)
        p_cool: float = 10.0,  # "bajo" (fresco/húmedo/vegetado/acuoso)
        # Percentiles para mapear HOT_SCORE -> ICU 0..3:
        score_p1: float = 60.0,  # >= p1 -> ICU 1
        score_p2: float = 75.0,  # >= p2 -> ICU 2
        score_p3: float = 90.0,  # >= p3 -> ICU 3
        # Pesos (TAIR > LST)
        weights: Optional[Dict[str, float]] = None,
        # Aliases por si el CSV usa otros nombres de columna
        aliases: Optional[Dict[str, List[str]]] = None,
    ):
        self.nodata_cluster = int(nodata_cluster)
        self.nodata_out = int(nodata_out)
        self.min_pixels_per_cluster = int(min_pixels_per_cluster)
        self.compress = str(compress)

        self.p_hot = float(p_hot)
        self.p_cool = float(p_cool)

        self.score_p1 = float(score_p1)
        self.score_p2 = float(score_p2)
        self.score_p3 = float(score_p3)

        # ✅ Pesos por defecto (TAIR con más fuerza; LST indirecta)
        self.weights = weights or {
            # Núcleo térmico
            "TAIR": 2.5,   # 🔥 MÁS peso
            "LST": 1.2,    # indirecta
            "HR": 1.0,     # baja HR => más ICU

            # Vegetación / agua (bajas => más ICU)
            "NDVI": 1.0,
            "MNDWI": 0.8,
            "CUERPOS_AGUA_BIN": 0.7,
            "VEGETACION_BIN": 0.7,

            # Urbanización / superficies
            "NDBI": 1.0,
            "PORC_URBANO": 1.0,
            "CONSTRUCCIONES_BIN": 0.9,
            "VIAS_BIN": 0.7,

            # Distancias normalizadas
            "DIST_AGUA_NORM": 0.9,  # más lejos del agua => más ICU
            "DIST_VIAS_NORM": 0.4,  # opcional (proxy urbano, depende de tu normalización)

            # Radiación / materialidad (asumimos: albedo bajo tiende a más calentamiento)
            "ALBEDO": 0.5,
        }

        # ✅ Aliases: intenta leer cualquiera de estas columnas para cada variable
        self.aliases = aliases or {
            "TAIR": ["mean_TAIR", "mean_TEMP_AIRE", "mean_TA", "mean_AIR_TEMP", "mean_TAIR_3m"],
            "LST": ["mean_LST", "mean_LST_3m"],
            "HR": ["mean_HR", "mean_RH", "mean_HUM", "mean_HUMEDAD_RELATIVA"],

            "NDVI": ["mean_NDVI"],
            "NDBI": ["mean_NDBI"],
            "MNDWI": ["mean_MNDWI", "mean_MNWI"],
            "ALBEDO": ["mean_ALBEDO"],

            "CONSTRUCCIONES_BIN": ["mean_CONSTRUCCIONES_BIN", "mean_CONSTRUCCIONES", "mean_BUILT_BIN"],
            "VIAS_BIN": ["mean_VIAS_BIN", "mean_VIAS", "mean_ROADS_BIN"],
            "PORC_URBANO": ["mean_PORC_URBANO", "mean_PORC_URBANO_ALINEADO", "mean_URB_PCT"],

            "CUERPOS_AGUA_BIN": ["mean_CUERPOS_AGUA_BIN", "mean_CUERPOS_AGUA", "mean_WATER_BIN"],
            "VEGETACION_BIN": ["mean_VEGETACION_BIN", "mean_VEGETACION", "mean_VEG_BIN"],

            "DIST_AGUA_NORM": ["mean_DIST_AGUA_NORM", "mean_DISTANCIA_AGUA_NORM", "mean_DIST_WATER_NORM"],
            "DIST_VIAS_NORM": ["mean_DIST_VIAS_NORM", "mean_DISTANCIA_VIAS_NORM", "mean_DIST_ROADS_NORM"],
        }

        # Definición de "dirección" para ICU:
        #   - high: valores altos aumentan ICU
        #   - low : valores bajos aumentan ICU
        self.direction = {
            "TAIR": "high",
            "LST": "high",
            "NDBI": "high",
            "PORC_URBANO": "high",
            "CONSTRUCCIONES_BIN": "high",
            "VIAS_BIN": "high",
            "DIST_AGUA_NORM": "high",
            "DIST_VIAS_NORM": "high",

            "NDVI": "low",
            "MNDWI": "low",
            "HR": "low",
            "CUERPOS_AGUA_BIN": "low",
            "VEGETACION_BIN": "low",
            "ALBEDO": "low",
        }

    # ------------------------------------------------------------
    # CSV reading
    # ------------------------------------------------------------
    def _get_float(self, row: dict, candidates: List[str]) -> float:
        for c in candidates:
            if c in row and row[c] not in (None, "", "nan", "NaN"):
                try:
                    return float(row[c])
                except ValueError:
                    continue
        raise KeyError(f"No se encontró ninguna columna válida entre: {candidates}")

    def read_stats(self, stats_csv: str) -> List[Dict[str, float]]:
        """
        Lee CSV y devuelve filas con todas las variables como claves normalizadas:
          cluster_id, n_pixels, mean_<VAR>
        """
        rows: List[Dict[str, float]] = []

        with open(stats_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])

            base_required = {"cluster_id", "n_pixels"}
            missing_base = base_required - fieldnames
            if missing_base:
                raise ValueError(
                    f"El CSV no tiene columnas base requeridas: {sorted(list(missing_base))}\n"
                    f"Columnas encontradas: {sorted(list(fieldnames))}"
                )

            # Verificar que cada VAR tenga al menos 1 alias presente
            missing_vars = []
            for var, cols in self.aliases.items():
                if not any(c in fieldnames for c in cols):
                    missing_vars.append((var, cols))

            if missing_vars:
                msg = "\n".join([f" - {v}: {cols}" for v, cols in missing_vars])
                raise ValueError(
                    "El CSV no contiene TODAS las variables requeridas (ningún alias encontrado para algunas):\n"
                    f"{msg}\n\n"
                    f"Columnas encontradas: {sorted(list(fieldnames))}"
                )

            for row in reader:
                n = int(float(row["n_pixels"]))
                if n < self.min_pixels_per_cluster:
                    continue

                r: Dict[str, float] = {
                    "cluster_id": int(float(row["cluster_id"])),
                    "n_pixels": n,
                }

                # Cargar todas las variables requeridas como mean_<VAR>
                for var, cols in self.aliases.items():
                    r[f"mean_{var}"] = float(self._get_float(row, cols))

                rows.append(r)

        if not rows:
            raise ValueError(
                "No quedaron filas válidas después del filtro min_pixels_per_cluster.\n"
                "Baja min_pixels_per_cluster o revisa el CSV."
            )

        return rows

    # ------------------------------------------------------------
    # Threshold helpers (percentiles)
    # ------------------------------------------------------------
    def _percentile_thresholds(self, stats_rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Devuelve thresholds por variable:
          thr_hot[var]  = percentile(p_hot)
          thr_cool[var] = percentile(p_cool)
        """
        thr: Dict[str, Tuple[float, float]] = {}

        for var in self.weights.keys():
            vals = np.array([r[f"mean_{var}"] for r in stats_rows], dtype=np.float32)
            # si hubiera nans, filtrarlos
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                # si esto pasa, no debería, porque exigimos columnas, pero por seguridad:
                thr[var] = (np.nan, np.nan)
                continue

            thr_hot = float(np.percentile(vals, self.p_hot))
            thr_cool = float(np.percentile(vals, self.p_cool))
            thr[var] = (thr_hot, thr_cool)

        return thr

    # ------------------------------------------------------------
    # Scoring rules (ALL VARIABLES)
    # ------------------------------------------------------------
    def build_score_map(self, stats_rows: List[Dict[str, float]]) -> Dict[int, int]:
        """
        Retorna dict: {cluster_id: ICU_SCORE} usando score ponderado con TODAS las variables.
        """
        thresholds = self._percentile_thresholds(stats_rows)

        # 1) calcular hot_score ponderado por cluster
        hot_scores: List[float] = []
        cluster_ids: List[int] = []

        for r in stats_rows:
            cid = int(r["cluster_id"])
            score = 0.0

            for var, w in self.weights.items():
                mean_val = float(r[f"mean_{var}"])
                thr_hot, thr_cool = thresholds[var]
                if np.isnan(mean_val) or np.isnan(thr_hot) or np.isnan(thr_cool):
                    continue

                direction = self.direction.get(var, "high")

                if direction == "high":
                    cond = mean_val >= thr_hot
                else:  # "low"
                    cond = mean_val <= thr_cool

                if cond:
                    score += float(w)

            hot_scores.append(score)
            cluster_ids.append(cid)

        hot_scores_arr = np.array(hot_scores, dtype=np.float32)

        # 2) convertir HOT_SCORE -> ICU 0..3 por percentiles del score
        t1 = float(np.percentile(hot_scores_arr, self.score_p1))
        t2 = float(np.percentile(hot_scores_arr, self.score_p2))
        t3 = float(np.percentile(hot_scores_arr, self.score_p3))

        score_map: Dict[int, int] = {}
        for cid, s in zip(cluster_ids, hot_scores_arr):
            if s >= t3:
                icu = 3
            elif s >= t2:
                icu = 2
            elif s >= t1:
                icu = 1
            else:
                icu = 0
            score_map[int(cid)] = int(icu)

        return score_map

    # ------------------------------------------------------------
    # Raster transform
    # ------------------------------------------------------------
    def rasterize_icu(self, cluster_arr: np.ndarray, score_map: Dict[int, int]) -> np.ndarray:
        """
        Convierte raster de clusters -> raster ICU_SCORE.
        """
        out = np.full(cluster_arr.shape, self.nodata_out, dtype=np.int16)

        mask = (cluster_arr != self.nodata_cluster)
        clusters = cluster_arr[mask].astype(np.int32)

        icu_vals = np.zeros_like(clusters, dtype=np.int16)
        for cid in np.unique(clusters):
            icu_vals[clusters == cid] = np.int16(score_map.get(int(cid), 0))

        out[mask] = icu_vals
        return out

    # ------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------
    def run(self, cluster_tif: str, stats_csv: str, out_tif: str) -> str:
        """
        Ejecuta:
          stats_csv -> score_map
          cluster_tif -> icu_score raster -> out_tif
        """
        stats_rows = self.read_stats(stats_csv)
        score_map = self.build_score_map(stats_rows)

        with rasterio.open(cluster_tif) as src:
            cluster_arr = src.read(1)
            profile = src.profile.copy()

        icu_arr = self.rasterize_icu(cluster_arr, score_map)

        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=self.nodata_out,
            compress=self.compress,
        )

        Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(icu_arr, 1)

        return out_tif
