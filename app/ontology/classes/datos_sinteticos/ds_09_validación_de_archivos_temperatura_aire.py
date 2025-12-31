
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict


@dataclass
class TAIRInputs:
    # ===== INSUMOS DE ENTRADA =====
    dem_original: Path           # dem_zona.tif

    shp_agua: Path               # Cuerpos_De_Agua.shp
    shp_vias: Path               # vias.shp
    shp_construcciones: Path     # construcciones*.shp
    shp_vegetacion: Path         # Vegetacion*.shp
    shp_upl: Path                # Upl_Modificada.shp

    tair_csv: Path               # TAIRE_YYYY.csv  ← INSUMO CLAVE
    lst_dir: Path                # carpeta LST del año
    ndvi_dir: Path               # carpeta NDVI del año


class TAIRInputFinder:
    """
    Valida ÚNICAMENTE los INSUMOS de entrada para SyntheticAirTemperature3m.

    INSUMOS:
    - DEM original (dem_zona.tif)
    - SHP: agua, vías, construcciones, vegetación, UPL
    - CSV de temperatura del aire (TAIRE_YYYY.csv)
    - Carpeta LST del año (TIFFs con 'LST' + año)
    - Carpeta NDVI del año (TIFFs con 'NDVI' + año)

    NO valida resultados derivados:
    - dem_3m, slope, rasters alineados, distancias, porcentajes, etc.
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

    def _find_by_exact_name(self, filename: str) -> List[Path]:
        return [p for p in self._glob(f"*{filename}") if p.is_file()]

    def _find_by_patterns(self, patterns: Sequence[str]) -> List[Path]:
        hits: List[Path] = []
        for pat in patterns:
            hits.extend(self._glob(pat))
        return [h for h in hits if h.is_file()]

    def _require_file(self, label: str, exact_names: Sequence[str], patterns: Sequence[str]) -> Path:
        candidates: List[Path] = []
        for n in exact_names:
            candidates.extend(self._find_by_exact_name(n))
        if not candidates:
            candidates.extend(self._find_by_patterns(patterns))

        uniq: List[Path] = []
        seen = set()
        for c in candidates:
            s = str(c).lower()
            if s not in seen:
                uniq.append(c)
                seen.add(s)

        if not uniq:
            raise FileNotFoundError(
                f"Falta insumo: {label}\n"
                f"Nombres: {list(exact_names)}\n"
                f"Patrones: {list(patterns)}\n"
                f"base_root: {self.base_root}"
            )

        uniq.sort(key=lambda p: (len(p.parts), len(p.name)))
        return uniq[0]

    def _find_dir_by_tag_and_year(self, tag: str) -> Path:
        tag_u = tag.upper()
        y = str(self.year)

        patterns = [
            f"*{tag_u}*.tif", f"*{tag_u.lower()}*.tif",
            f"*{tag_u}*.tiff", f"*{tag_u.lower()}*.tiff",
        ]

        hits = self._find_by_patterns(patterns)
        hits = [h for h in hits if tag_u in h.name.upper() and y in h.name]

        if not hits:
            raise FileNotFoundError(
                f"No encontré insumo {tag} para el año {self.year} en {self.base_root}"
            )

        count: Dict[Path, int] = {}
        for h in hits:
            count[h.parent] = count.get(h.parent, 0) + 1

        return sorted(count.items(), key=lambda kv: (-kv[1], len(kv[0].parts)))[0][0]

    # --------------------------------------------------
    # API
    # --------------------------------------------------
    def find_all(self) -> TAIRInputs:
        dem_original = self._require_file(
            "DEM original (dem_zona.tif)",
            exact_names=["dem_zona.tif"],
            patterns=["*dem_zona*.tif", "*dem*zona*.tif", "*DEM*zona*.tif"],
        )

        shp_agua = self._require_file(
            "SHP Cuerpos de Agua",
            exact_names=["Cuerpos_De_Agua.shp"],
            patterns=["*cuerpos*agua*.shp", "*agua*.shp"],
        )

        shp_vias = self._require_file(
            "SHP Vías",
            exact_names=["vias.shp"],
            patterns=["*vias*.shp"],
        )

        shp_construcciones = self._require_file(
            "SHP Construcciones",
            exact_names=["construcciones.shp", "construcciones2.shp"],
            patterns=["*construcciones*.shp"],
        )

        shp_vegetacion = self._require_file(
            "SHP Vegetación",
            exact_names=["Vegetacion.shp", "Vegetacion2.shp"],
            patterns=["*vegetacion*.shp"],
        )

        shp_upl = self._require_file(
            "SHP UPL",
            exact_names=["Upl_Modificada.shp"],
            patterns=["*upl*.shp"],
        )

        tair_csv = self._require_file(
            f"CSV Temperatura del Aire {self.year}",
            exact_names=[f"TAIRE_{self.year}.csv"],
            patterns=[f"*TAIRE*{self.year}*.csv", f"*tair*{self.year}*.csv"],
        )

        lst_dir = self._find_dir_by_tag_and_year("LST")
        ndvi_dir = self._find_dir_by_tag_and_year("NDVI")

        return TAIRInputs(
            dem_original=dem_original,
            shp_agua=shp_agua,
            shp_vias=shp_vias,
            shp_construcciones=shp_construcciones,
            shp_vegetacion=shp_vegetacion,
            shp_upl=shp_upl,
            tair_csv=tair_csv,
            lst_dir=lst_dir,
            ndvi_dir=ndvi_dir,
        )