import os
from pathlib import Path

from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar
from app.ontology.classes.variable.biofisica.ndvi import NDVI
from app.ontology.classes.variable.biofisica.ndbi import NDBI
from app.ontology.classes.variable.biofisica.albedo import Albedo
from app.ontology.classes.variable.biofisica.lst import LST
from app.ontology.classes.variable.biofisica.mnwi import MNDWI
from app.ontology.classes.variable.climatica.temperatura_del_aire_sintetica import TAIRPipeline
from app.ontology.classes.variable.climatica.humedad_relativa_sintetica import HRPipeline



# ======================================================
# UTILIDADES MÍNIMAS
# ======================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def dir_has_tifs(path: Path) -> bool:
    """True si el directorio existe y ya tiene productos .tif."""
    return path.exists() and any(path.glob("*.tif"))


# ======================================================
# CONFIGURACIÓN
# ======================================================
YEAR = 2021
OVERWRITE = False

BASE_INPUT = Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021")

REPROJECTED_DIR = BASE_INPUT / "01_reproyectadas"
INDICES_DIR = BASE_INPUT / "02_Calculos"


# ======================================================
# 1) REPROYECCIÓN
# ======================================================
print("\n[1] Reproyección Sentinel-2")

if dir_has_tifs(REPROJECTED_DIR) and not OVERWRITE:
    print("⏭️  Reproyección ya ejecutada (hay .tif en reproyectadas) → omitido")
else:
    ensure_dir(REPROJECTED_DIR)
    reproyectar = Reproyectar(
        input_dir=str(BASE_INPUT),
        output_dir=str(REPROJECTED_DIR),
        dst_crs="EPSG:4326",
    )
    reproyectar.procesar_carpeta()
    print("✓ Reproyección finalizada")


# ======================================================
# 2) ÍNDICES BIOFÍSICOS
# ======================================================
print("\n[2] Cálculo de índices biofísicos")
ensure_dir(INDICES_DIR)

indices = [
    ("NDVI", NDVI, "*_NDVI.tif"),
    ("NDBI", NDBI, "*_NDBI.tif"),
    ("ALBEDO", Albedo, "*_ALBEDO.tif"),
    ("LST", LST, "*_LST.tif"),
    ("MNDWI", MNDWI, "*_MNDWI.tif"),
]

for name, IndexClass, pattern in indices:

    # Omitir solo si YA EXISTEN los productos esperados
    if INDICES_DIR.exists() and any(INDICES_DIR.glob(pattern)) and not OVERWRITE:
        print(f"⏭️  Índice {name} ya generado → omitido")
        continue

    print(f"→ Calculando {name}")

    index = IndexClass(
        input_dir=str(REPROJECTED_DIR),
        output_dir=str(INDICES_DIR),
    )
    index.calculate()

    print(f"✓ {name} generado")

print("\n✅ ÍNDICES BIOFÍSICOS FINALIZADOS")


# ======================================================
# PIPELINE TAIR
# ======================================================

pipeline = TAIRPipeline(
    input_root=Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021"),
    output_root=Path(os.path.join(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021", "04_Temperatura_Aire")), 
    year=2021,
    overwrite=False,
    px=0.000009
)
pipeline.run()


# # ======================================================
# # PIPELINE HR
# # ======================================================


pipeline = HRPipeline(
    input_root=Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021"),
    output_root=Path(os.path.join(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021", "05_humedad_relativa")),
    year=2021,
    overwrite=False,
)
pipeline.run()

# # ======================================================
#Genereando las capas UHI
# # ======================================================

from app.ontology.classes.UHI.UHI import UHIFullYearPipeline, UHIPipelineConfig


if __name__ == "__main__":
    cfg = UHIPipelineConfig(
        year="2021",
        root=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021",
        ref_align=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif",
        upl_shp=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\prueba_3_2021\otros_Insumos\01_temperatura_del_aire\01_shp\Upl_Modificada.shp",
        overwrite=False,
        som_rows=10,
        som_cols=10,
    )

    pipeline = UHIFullYearPipeline(cfg)
    pipeline.run_all()

