import os
from app.ontology.classes.datos_sinteticos.ds_02_leer_archivo_csv_temperatura_aire import AirTemperatureMonthly
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

reader = AirTemperatureMonthly(
    csv_path=os.path.join(ruta_base, "01_temperatura_del_aire", "01_insumos", "02_archivo_cvc_temperatura", "TAIRE_2021.csv")
)

df = reader.load()
print(df.head())

monthly = reader.compute_monthly()
print(monthly)

diccionario = reader.to_dict(key_format="yyyymm")
print(diccionario)

reader.save_csv(os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "02_resultado_promedio_temperatura_aire", "temperatura_mensual_2021.csv"))
