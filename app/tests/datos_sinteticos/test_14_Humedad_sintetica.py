import os
from app.ontology.classes.datos_sinteticos.ds_14_calcular_humedad_sintetica import SyntheticRelativeHumidity3m
from app.ontology.classes.datos_sinteticos.ds_13_humeda_relativa import RelativeHumidityMonthly
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"
# -----------------------------------------------------
# 1. Humedad relativa mensual (ancla temporal)
# -----------------------------------------------------
reader = RelativeHumidityMonthly(
    csv_path=os.path.join(ruta_base, "02_humedad_relativa", "01_archivo_cvs_promedio_humedad_relativa", "HUMEDAD_R_2021.csv")
)

reader.load()
reader.compute_monthly()
hr_monthly_dict = reader.to_dict("yyyymm")

# -----------------------------------------------------
# 2. Rutas base
# -----------------------------------------------------
carpeta_base = os.path.join(ruta_base, "02_humedad_relativa")
# -----------------------------------------------------
# 3. Humedad relativa sintética (3 m)
#    Sintetización formal basada en variables definidas
# -----------------------------------------------------
synthetic_hr = SyntheticRelativeHumidity3m(
    hr_monthly_dict=hr_monthly_dict,
    ndvi_dir=os.path.join(carpeta_base, "03_NDVI"),
    ndbi_dir=os.path.join(carpeta_base, "02_NDBI"),
    tair_dir=os.path.join(carpeta_base, "05_temperatura_del_aire"),
    dist_water_path=os.path.join(carpeta_base, "04_distancia_cuerpo_de_agua", "distancia_agua_3m_ALINEADO.tif"),
    urban_path=os.path.join(carpeta_base, "06_porcentaje_de_construccion", "porcentaje_urbano_3m_ALINEADO.tif"),
    dem_path=os.path.join(carpeta_base, "07_DEM", "dem_3m.tif"),
    slope_path=os.path.join(carpeta_base, "08_pendiente", "slope_3m.tif"),
    output_dir=os.path.join(carpeta_base, "00_resultados_humedad_relativa_sintetica")
)

synthetic_hr.process_all()
