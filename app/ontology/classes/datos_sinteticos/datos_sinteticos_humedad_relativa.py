from pathlib import Path

from app.ontology.classes.datos_sinteticos.ds_13_humeda_relativa import RelativeHumidityMonthly
from app.ontology.classes.datos_sinteticos.ds_14_calcular_humedad_sintetica import SyntheticRelativeHumidity3m
from app.ontology.classes.datos_sinteticos.ds_10_validacion_archivos_humedad_relativa import HRInputFinder


# ======================================================
# UTILIDADES MÍNIMAS (IGUALES AL PIPELINE TAIR)
# ======================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def should_skip(path: Path, overwrite: bool, label: str = "") -> bool:
    if path.exists() and not overwrite:
        print(f"⏭️  {label or path.name} existe → omitido")
        return True
    return False


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
YEAR = 2022
OVERWRITE = False  

BASE_ROOT = Path(r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\2022\02_humedad_relativa")


# ======================================================
# 0) VALIDAR INSUMOS (FUENTE DE VERDAD)
# ======================================================
print("\n[0] Validando insumos de Humedad Relativa")

finder = HRInputFinder(base_root=BASE_ROOT, year=YEAR)
ins = finder.find_all()


# ======================================================
# 1) DEFINIR DIRECTORIOS DE RESULTADOS
#    (RELATIVOS A DONDE ESTÁN LOS INSUMOS)
# ======================================================
HR_BASE = ins.hr_csv.parent.parent.parent  # …/02_humedad_relativa
RESULTS_BASE = HR_BASE / "00_resultados_humedad_relativa_sintetica"
ensure_dir(RESULTS_BASE)

CSV_OUT_DIR = HR_BASE / "03_Insumos_datos_sinteticos" / "01_archivo_cvs_promedio_humedad_relativa"
ensure_dir(CSV_OUT_DIR)

hr_out_csv = CSV_OUT_DIR / f"humedad_relativa_mensual_{YEAR}.csv"


# ======================================================
# 2) HR MENSUAL (CSV → dict)
# ======================================================
print("\n[1] HR mensual (CSV → dict)")

hr_reader = RelativeHumidityMonthly(str(ins.hr_csv))
hr_reader.load()
hr_reader.compute_monthly()
hr_dict = hr_reader.to_dict("yyyymm")

if not should_skip(hr_out_csv, OVERWRITE, "CSV HR mensual"):
    hr_reader.save_csv(str(hr_out_csv))
    print("✓ CSV mensual HR generado")


# ======================================================
# 3) HR SINTÉTICA (3 m)
# ======================================================
print("\n[2] HR sintética (3 m)")

synthetic_hr = SyntheticRelativeHumidity3m(
    year=YEAR,
    hr_monthly_dict=hr_dict,
    ndvi_dir=str(ins.ndvi_dir),
    ndbi_dir=str(ins.ndbi_dir),
    tair_dir=str(ins.tair_dir),
    dist_water_path=str(ins.dist_agua),
    urban_path=str(ins.urban),
    dem_path=str(ins.dem_3m),
    slope_path=str(ins.slope_3m),
    output_dir=str(RESULTS_BASE),
)

synthetic_hr.process_all()

print("\n✅ PROCESO DE HUMEDAD RELATIVA FINALIZADO")
