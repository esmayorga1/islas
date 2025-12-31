from app.ontology.classes.producto_analitico.riesgo_termico import RiesgoTermico
import os

ruta_indices = r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Calculos"
ruta_temp = r"F:\Pruebas\2021\reproyectadas"
ruta_cobertura = ruta_indices
salida_shp = os.path.join(ruta_indices, "Islas_Calor")



#riesgo = RiesgoTermico(ruta_temp, ruta_cobertura, salida_shp, umbral_fijo=10, year=2021, umbral_temp=12.5, clase_urbana_fija=3, min_size_pixels=10 )

riesgo = RiesgoTermico(
    ruta_temp=r"F:\Pruebas\2021\reproyectadas",
    ruta_cobertura=r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Calculos",
    salida_dir=r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Islas_Calor",
    year=2021,
    umbral_temp=14,         # 🔥 nuevo umbral fijo bajo
    clase_urbana_fija=3,      # clase urbana
    min_size_pixels=5         # elimina manchas pequeñas
)

riesgo.generar()