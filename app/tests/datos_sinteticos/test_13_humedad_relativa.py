import os
from app.ontology.classes.datos_sinteticos.ds_13_humeda_relativa import RelativeHumidityMonthly
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

reader = RelativeHumidityMonthly(
    csv_path=os.path.join(ruta_base, "02_humedad_relativa", "01_archivo_cvs_promedio_humedad_relativa", "HUMEDAD_R_2021.csv")  
  
)

# 1. Cargar CSV
df = reader.load()
print(df.head())

# 2. Calcular promedio mensual
monthly = reader.compute_monthly()
print(monthly)

# 3. Convertir a diccionario (clave YYYY-MM)
diccionario = reader.to_dict(key_format="yyyymm")
print(diccionario)

# 4. Guardar CSV mensual
reader.save_csv(
    os.path.join(ruta_base, "02_humedad_relativa", "03_Insumos_datos_sinteticos", "01_archivo_cvs_promedio_humedad_relativa", "humedad_relativa_mensual_2021.csv")  
    
)