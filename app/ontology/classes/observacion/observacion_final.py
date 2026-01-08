from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar
from app.ontology.classes.variable.biofisica.ndvi import NDVI
from app.ontology.classes.variable.biofisica.ndbi import NDBI
from app.ontology.classes.variable.biofisica.albedo import Albedo
from app.ontology.classes.variable.biofisica.lst import LST
from app.ontology.classes.variable.biofisica.mnwi import MNDWI
from app.ontology.classes.variable.climatica.temperatura_del_aire_sintetica import TAIRPipeline
from app.ontology.classes.variable.climatica.humedad_relativa_sintetica import HRPipeline


def has(dir_: Path, pattern: str) -> bool:
    return dir_.exists() and any(dir_.glob(pattern))


@dataclass
class ObservacionConfig:
    base: str
    year: int = 2021
    overwrite: bool = False
    dst_crs: str = "EPSG:4326"
    px: float = 0.000009


class ObservacionPipeline:
    """
    OBSERVACION:
    - Reproyección
    - Índices biofísicos (NDVI, NDBI, ALBEDO, LST, MNDWI)
    - Variables climáticas sintéticas (TAIR, HR)
    """

    def __init__(self, cfg: ObservacionConfig):
        self.cfg = cfg
        self.base = Path(cfg.base)

        self.repro_dir = self.base / "01_reproyectadas"
        self.idx_dir = self.base / "02_Calculos"
        self.tair_out = self.base / "04_Temperatura_Aire"
        self.hr_out = self.base / "05_humedad_relativa"

        self.repro_dir.mkdir(parents=True, exist_ok=True)
        self.idx_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        # 1) Reproyección
        if self.cfg.overwrite or not has(self.repro_dir, "*.tif"):
            Reproyectar(str(self.base), str(self.repro_dir), dst_crs=self.cfg.dst_crs).procesar_carpeta()

        # 2) Índices biofísicos
        for cls, pat in [
            (NDVI, "*_NDVI.tif"),
            (NDBI, "*_NDBI.tif"),
            (Albedo, "*_ALBEDO.tif"),
            (LST, "*_LST.tif"),
            (MNDWI, "*_MNDWI.tif"),
        ]:
            if self.cfg.overwrite or not has(self.idx_dir, pat):
                cls(input_dir=str(self.repro_dir), output_dir=str(self.idx_dir)).calculate()

        # 3) Variables climáticas sintéticas
        TAIRPipeline(
            input_root=self.base,
            output_root=self.tair_out,
            year=self.cfg.year,
            overwrite=self.cfg.overwrite,
            px=self.cfg.px
        ).run()

        HRPipeline(
            input_root=self.base,
            output_root=self.hr_out,
            year=self.cfg.year,
            overwrite=self.cfg.overwrite
        ).run()

        return {
            "repro_dir": self.repro_dir,
            "idx_dir": self.idx_dir,
            "tair_out": self.tair_out,
            "hr_out": self.hr_out,
        }
