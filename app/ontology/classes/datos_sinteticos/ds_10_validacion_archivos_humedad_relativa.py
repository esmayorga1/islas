# ======================================================
# app/ontology/classes/datos_sinteticos/ds_15_validacion_insumos_humedad_relativa.py
# ======================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict


@dataclass
class HRInputs:
    # ===== INSUMOS DE ENTRADA =====
    hr_csv: Path                # HUMEDAD_R_YYYY.csv
    ndvi_dir: Path              # carpeta NDVI del año
    ndbi_dir: Path              # carpeta NDBI del año
    tair_dir: Path              # carpeta temperatura aire sintética
    dist_agua: Path             # distancia_agua_3m_ALINEADO.tif
    urban: Path                 # porcentaje_urbano_3m_ALINEADO.tif
    dem_3m: Path                # dem_3m.tif
    slope_3m: Path              # slope_3m.tif


class HRInputFinder:
    """
    Valida ÚNICAMENTE los INSUMOS de entrada para SyntheticRelativeHumidity3m.

    - Parte de una sola ruta base.
    - No exige estructura fija.
    - Busca archivos de forma recursiva.
    - NO incluye resultados finales.
    """

    def __init__(
        self,
        base_root: Path,
        year: int,
        recursive: bool = True,
    ):
        self.base_root = Path(base_root)
        self.year = int(year)
        self.recursive = bool(recursive)

        if not self.base_root.exists():
            raise FileNotFoundError(f"base_root no existe: {self.base_root}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _glob(self, pattern: str) -> List[Path]:
        return list(self.base_root.rglob(pattern)) if self.recursive else list(self.base_root.glob(pattern))

    def _require_file(self, label: str, patterns: Sequence[str]) -> Path:
        hits: List[Path] = []
        for pat in patterns:
            hits.extend(self._glob(pat))

        hits = [h for h in hits if h.is_file()]

        if not hits:
            raise FileNotFoundError(
                f"Falta insumo: {label}\n"
                f"Patrones: {list(patterns)}\n"
                f"base_root: {self.base_root}"
            )

        hits.sort(key=lambda p: (len(p.parts), len(p.name)))
        return hits[0]

    def _find_dir_by_tag_and_year(self, tag: str) -> Path:
        tag_u = tag.upper()
        y = str(self.year)

        hits = self._glob(f"*{tag_u}*.tif") + self._glob(f"*{tag_u.lower()}*.tif")
        hits = [h for h in hits if tag_u in h.name.upper() and y in h.name]

        if not hits:
            raise FileNotFoundError(
                f"No encontré insumo {tag} para el año {self.year}"
            )

        count: Dict[Path, int] = {}
        for h in hits:
            count[h.parent] = count.get(h.parent, 0) + 1

        return sorted(count.items(), key=lambda kv: (-kv[1], len(kv[0].parts)))[0][0]

    # --------------------------------------------------
    # API
    # --------------------------------------------------
    def find_all(self) -> HRInputs:
        hr_csv = self._require_file(
            f"CSV Humedad Relativa {self.year}",
            patterns=[f"*HUMEDAD*{self.year}*.csv", f"*humedad*{self.year}*.csv"],
        )

        ndvi_dir = self._find_dir_by_tag_and_year("NDVI")
        ndbi_dir = self._find_dir_by_tag_and_year("NDBI")

        tair_dir = self._find_dir_by_tag_and_year("TAIRE_SINTETICA")

        dist_agua = self._require_file(
            "Distancia a cuerpos de agua 3m",
            patterns=["*distancia*agua*3m*ALINEADO*.tif"],
        )

        urban = self._require_file(
            "Porcentaje urbano 3m",
            patterns=["*porcentaje*urbano*3m*ALINEADO*.tif"],
        )

        dem_3m = self._require_file(
            "DEM 3m",
            patterns=["*dem_3m*.tif"],
        )

        slope_3m = self._require_file(
            "Pendiente 3m",
            patterns=["*slope_3m*.tif", "*pend*3m*.tif"],
        )

        return HRInputs(
            hr_csv=hr_csv,
            ndvi_dir=ndvi_dir,
            ndbi_dir=ndbi_dir,
            tair_dir=tair_dir,
            dist_agua=dist_agua,
            urban=urban,
            dem_3m=dem_3m,
            slope_3m=slope_3m,
        )
