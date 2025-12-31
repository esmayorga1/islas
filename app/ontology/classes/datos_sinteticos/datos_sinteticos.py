from pathlib import Path

from app.ontology.classes.datos_sinteticos.ds_01_convertir_shp_raster import ShapefileToRaster
from app.ontology.classes.datos_sinteticos.ds_02_leer_archivo_csv_temperatura_aire import AirTemperatureMonthly
from app.ontology.classes.datos_sinteticos.ds_03_porcentaje_construccion import UrbanDensity3m
from app.ontology.classes.datos_sinteticos.ds_04_densidad_vegetacion import VegetationDensity3m
from app.ontology.classes.datos_sinteticos.ds_05_distancia_vias import DistanceToRoads
from app.ontology.classes.datos_sinteticos.ds_06_distancia_cuerpos_agua import DistanceToWater
from app.ontology.classes.datos_sinteticos.ds_07_tipo_de_cobertura import CoverageClassifier3m
from app.ontology.classes.datos_sinteticos.ds_08_resampliar_DEM import RasterResampler3mWGS84
from app.ontology.classes.datos_sinteticos.ds_02_mapa_pendientes import SlopeGenerator
from app.ontology.classes.datos_sinteticos.ds_00_raster_aligner import RasterAlignerToDEM3m
from app.ontology.classes.datos_sinteticos.ds_11_temperatura_aire_sintetica import SyntheticAirTemperature3m
from app.ontology.classes.datos_sinteticos.ds_13_humeda_relativa import RelativeHumidityMonthly
from app.ontology.classes.datos_sinteticos.ds_14_calcular_humedad_sintetica import SyntheticRelativeHumidity3m

# ======================================================
# 0) RUTAS BASE
# ======================================================
BASE = Path(r"D:\002trabajos\21_islas_de_calor\datos_sinteticos\2022")
TAIR = BASE / "01_temperatura_del_aire"
SHP = TAIR / "01_insumos" / "01_shp"
RAS = TAIR / "02_resultados" / "01_resultado_raster"
ALI = RAS / "alineados"
OUT_TAIR = TAIR / "02_resultados" / "03_resultado_temperatura_sintetica"

UPL = SHP / "Upl_Modificada.shp"
PX_UPL, PX_FREE = 0.000009, 0.00001

RAS.mkdir(parents=True, exist_ok=True)
ALI.mkdir(parents=True, exist_ok=True)
OUT_TAIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 0) DEM → DEM 3 m
# ======================================================
print("\n[0] DEM 3 m")
dem_original = TAIR / "01_insumos" / "03_DEM" / "dem_zona.tif"
dem_3m = RAS / "dem_3m.tif"

if not dem_original.exists():
    raise FileNotFoundError(f"DEM original no encontrado: {dem_original}")

if not dem_3m.exists():
    resampler = RasterResampler3mWGS84(str(dem_original), method="bilinear")
    resampler.resample()
    resampler.save(str(dem_3m))
    print("✓ DEM 3 m generado:", dem_3m)
else:
    print("✓ DEM 3 m ya existe")

# ======================================================
# 1) SHP → Raster
# ======================================================
print("\n[1] SHP → Raster")
jobs = [
    ("Cuerpos_De_Agua.shp",  "cuerpos_de_agua.tif",     True,  PX_UPL),
    ("vias.shp",             "vias.tif",                True,  PX_UPL),
    ("construcciones2.shp",  "construcciones.tif",      True,  PX_UPL),
    ("Vegetacion2.shp",      "vegetacion.tif",          True,  PX_UPL),
    ("vias_Principales.shp", "vias_principales_1m.tif", False, PX_FREE),
]

for shp_name, tif_name, use_ref, px in jobs:
    conv = ShapefileToRaster(
        shp_path=str(SHP / shp_name),
        shp_referencia_path=str(UPL) if use_ref else None,
        pixel_size=px,
        raster_value=1,
    )
    conv.rasterize_shp(str(RAS / tif_name))
    print("✓", tif_name)

# ======================================================
# 2) CSV Temperatura del aire
# ======================================================
print("\n[2] Temperatura del aire mensual")
tair_csv = TAIR / "01_insumos" / "02_archivo_cvc_temperatura" / "TAIRE_2022.csv"
tair_out_csv = TAIR / "02_resultados" / "02_resultado_promedio_temperatura_aire" / "temperatura_mensual_2022.csv"
tair_out_csv.parent.mkdir(parents=True, exist_ok=True)

tair_reader = AirTemperatureMonthly(str(tair_csv))
tair_reader.load()
tair_reader.compute_monthly()
temp_dict = tair_reader.to_dict("yyyymm")
tair_reader.save_csv(str(tair_out_csv))
print("✓ CSV mensual generado")

# ======================================================
# 3) % Urbano 3 m
# ======================================================
print("\n[3] Porcentaje urbano")
urban = UrbanDensity3m(str(RAS / "construcciones.tif"), str(RAS / "vias.tif"))
urban.compute_density()
urban.save_raster(str(RAS / "porcentaje_urbano_3m.tif"))

# ======================================================
# 4) % Vegetación 3 m
# ======================================================
print("\n[4] Porcentaje vegetación")
veg = VegetationDensity3m(str(RAS / "vegetacion.tif"))
veg.compute_density()
veg.save_raster(str(RAS / "porcentaje_vegetacion_3m.tif"))

# ======================================================
# 5) Distancia a vías (3 m)
# ======================================================
print("\n[5] Distancia a vías")
dist_vias = DistanceToRoads(str(RAS / "vias_principales_1m.tif"), output_resolution=3)
dist_vias.compute_distance()
dist_vias.save_raster(str(RAS / "distancia_vias_3m.tif"))

# ======================================================
# 5B) Distancia a cuerpos de agua (3 m)
# ======================================================
print("\n[5B] Distancia a cuerpos de agua")
dist_agua = DistanceToWater(str(RAS / "cuerpos_de_agua.tif"), output_resolution=3)
dist_agua.compute_distance()
dist_agua.save_raster(str(RAS / "distancia_agua_3m.tif"))

# ======================================================
# 6) Tipo de cobertura 3 m
# ======================================================
print("\n[6] Tipo de cobertura")
cov = CoverageClassifier3m(
    raster_agua_1m=str(RAS / "cuerpos_de_agua.tif"),
    raster_vegetacion_1m=str(RAS / "vegetacion.tif"),
    raster_urbano_1m=str(RAS / "construcciones.tif"),
    factor=3,
)
cov.aggregate_to_3m()
cov.save_raster(str(RAS / "tipo_cobertura_3m.tif"))

# ======================================================
# 7) Pendiente (Slope)
# ======================================================
print("\n[7] Pendiente")
slope = RAS / "slope_3m.tif"
sg = SlopeGenerator(str(dem_3m))
sg.compute_slope()
sg.save_slope(str(slope))

# ======================================================
# 8) Alinear rasters al DEM
# ======================================================
print("\n[8] Alineando rasters")
aligner = RasterAlignerToDEM3m(str(dem_3m))

to_align = {
    "tipo_cobertura_3m.tif":        "tipo_cobertura_3m_ALINEADO.tif",
    "porcentaje_urbano_3m.tif":     "porcentaje_urbano_3m_ALINEADO.tif",
    "porcentaje_vegetacion_3m.tif": "porcentaje_vegetacion_3m_ALINEADO.tif",
    "distancia_vias_3m.tif":        "distancia_vias_3m_ALINEADO.tif",
    "distancia_agua_3m.tif":        "distancia_agua_3m_ALINEADO.tif",
}

for inp, out in to_align.items():
    aligner.align(str(RAS / inp), str(ALI / out))
    print("✓", out)

# ======================================================
# 9) Temperatura del aire sintética
# ======================================================

print("\n[9] Temperatura del aire sintética")

synth_t = SyntheticAirTemperature3m(
    year=2022,
    lst_dir=str(TAIR/"01_insumos"/"04_LST"),
    ndvi_dir=str(TAIR/"01_insumos"/"05_NDVI"),
    coverage_path=str(ALI/"tipo_cobertura_3m_ALINEADO.tif"),
    urban_path=str(ALI/"porcentaje_urbano_3m_ALINEADO.tif"),
    vegetation_path=str(ALI/"porcentaje_vegetacion_3m_ALINEADO.tif"),
    dist_vias_path=str(ALI/"distancia_vias_3m_ALINEADO.tif"),
    dist_agua_path=str(ALI/"distancia_agua_3m_ALINEADO.tif"),
    dem_path=str(dem_3m),
    slope_path=str(slope),
    temp_monthly_dict=temp_dict,
    output_dir=str(OUT_TAIR),
)
synth_t.process_all()

# ======================================================
# 10) Humedad relativa sintética
# ======================================================
print("\n[10] Humedad relativa sintética")
HR = BASE / "02_humedad_relativa"

hr_csv = HR / "01_archivo_cvs_promedio_humedad_relativa" / "HUMEDAD_R_2022.csv"
hr_out_csv = HR / "03_Insumos_datos_sinteticos" / "01_archivo_cvs_promedio_humedad_relativa" / "humedad_relativa_mensual_2022.csv"
hr_out_csv.parent.mkdir(parents=True, exist_ok=True)

hr_reader = RelativeHumidityMonthly(str(hr_csv))
hr_reader.load()
hr_reader.compute_monthly()
hr_dict = hr_reader.to_dict("yyyymm")
hr_reader.save_csv(str(hr_out_csv))

synthetic_hr = SyntheticRelativeHumidity3m(
    year=2022,
    hr_monthly_dict=hr_dict,
    ndvi_dir=str(HR/"03_NDVI"),
    ndbi_dir=str(HR/"02_NDBI"),
    tair_dir=str(HR/"05_temperatura_del_aire"),
    dist_water_path=str(HR/"04_distancia_cuerpo_de_agua"/"distancia_agua_3m_ALINEADO.tif"),
    urban_path=str(HR/"06_porcentaje_de_construccion"/"porcentaje_urbano_3m_ALINEADO.tif"),
    dem_path=str(HR/"07_DEM"/"dem_3m.tif"),
    slope_path=str(HR/"08_pendiente"/"slope_3m.tif"),
    output_dir=str(HR/"00_resultados_humedad_relativa_sintetica"),
)
synthetic_hr.process_all()


print("\n PROCESO COMPLETO FINALIZADO SIN ERRORES")
