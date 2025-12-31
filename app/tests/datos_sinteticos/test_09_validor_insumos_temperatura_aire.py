# ======================================================
# CÓMO LLAMARLO (VALIDACIÓN DE INSUMOS DE ENTRADA)
# ======================================================

from pathlib import Path
from app.ontology.classes.datos_sinteticos.ds_09_validación_de_archivos_temperatura_aire import TAIRInputFinder

BASE_ROOT = Path(r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\2022\01_temperatura_del_aire\01_insumos\01_shp")
YEAR = 2022

finder = TAIRInputFinder(base_root=BASE_ROOT, year=YEAR)
inputs = finder.find_all()

print("DEM original:", inputs.dem_original)
print("SHP Agua:", inputs.shp_agua)
print("SHP Vías:", inputs.shp_vias)
print("SHP Construcciones:", inputs.shp_construcciones)
print("SHP Vegetación:", inputs.shp_vegetacion)
print("SHP UPL:", inputs.shp_upl)
print("CSV TAIR:", inputs.tair_csv)
print("LST dir:", inputs.lst_dir)
print("NDVI dir:", inputs.ndvi_dir)
