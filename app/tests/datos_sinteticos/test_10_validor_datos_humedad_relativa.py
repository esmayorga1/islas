from pathlib import Path
from app.ontology.classes.datos_sinteticos.ds_10_validacion_archivos_humedad_relativa import HRInputFinder

BASE_ROOT = Path(r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\2022\02_humedad_relativa")
YEAR = 2022

finder = HRInputFinder(base_root=BASE_ROOT, year=YEAR)
inputs = finder.find_all()

print("CSV HR:", inputs.hr_csv)
print("NDVI dir:", inputs.ndvi_dir)
print("NDBI dir:", inputs.ndbi_dir)
print("TAIR dir:", inputs.tair_dir)
print("Dist agua:", inputs.dist_agua)
print("Urbano:", inputs.urban)
print("DEM 3m:", inputs.dem_3m)
print("Slope 3m:", inputs.slope_3m)
