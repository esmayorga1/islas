import os
from app.ontology.classes.datos_sinteticos.ds_11_temperatura_aire_sintetica import SyntheticAirTemperature3m
from app.ontology.classes.datos_sinteticos.ds_02_leer_archivo_csv_temperatura_aire import AirTemperatureMonthly
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

# Leer temperatura base mensual
reader = AirTemperatureMonthly(os.path.join(ruta_base, "01_temperatura_del_aire", "01_insumos", "02_archivo_cvc_temperatura", "TAIRE_2021.csv"))
reader.load()
reader.compute_monthly()
temp_dict = reader.to_dict("yyyymm")

synth = SyntheticAirTemperature3m(
    lst_dir=os.path.join(ruta_base, "01_temperatura_del_aire", "01_insumos", "04_LST"),
    ndvi_dir=os.path.join(ruta_base, "01_temperatura_del_aire", "01_insumos", "05_NDVI"),
    coverage_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "alineados", "tipo_cobertura_3m_ALINEADO.tif"),     
    urban_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "alineados", "porcentaje_urbano_3m_ALINEADO.tif"),     
    vegetation_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "alineados", "porcentaje_vegetacion_3m_ALINEADO.tif"),
    dist_vias_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "alineados", "distancia_vias_3m_ALINEADO.tif"), 
    dist_agua_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "alineados", "distancia_agua_3m_ALINEADO.tif"), 
    dem_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "dem_3m.tif"), 
    slope_path=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "slope_3m.tif"), 
    temp_monthly_dict=temp_dict,
    output_dir=os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "03_resultado_temperatura_sintetica")
)

synth.process_all()

