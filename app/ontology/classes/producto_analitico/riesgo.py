import os
from app.ontology.classes.producto_analitico.VariableValidator import VariableValidator
from app.ontology.classes.producto_analitico.UHIGenerator import UHIGenerator
from app.ontology.classes.producto_analitico.UHItoSHP import UHItoSHP
from app.ontology.classes.producto_analitico.SuavizadorIslasCalor import SuavizadorIslasCalor
from app.ontology.classes.producto_analitico.subir_shp import subir_islas_calor


ruta_calculos=r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Calculos"
validador = VariableValidator(ruta_calculos, ruta_temp=r"F:\Pruebas\2021\reproyectadas", anio=2021)
validador.validar()

uhi = UHIGenerator(ruta_calculos, anio=2021, salida=ruta_calculos)
uhi.generar()
ruta_calculos_shp = os.path.join(ruta_calculos, "islas_calor.shp")

uhi_shp = UHItoSHP(
    ruta_uhi=ruta_calculos,         # Carpeta con los UHI mensuales
    anio=2021,
    salida_shp=ruta_calculos_shp,
    umbral=0.75                        # Valor UHI a partir del cual se considera zona caliente
)
uhi_shp.generar()

#=============================================

ruta_calculos_shp_suavizado = os.path.join(ruta_calculos, "islas_calor_suavisado.shp")
suavizador = SuavizadorIslasCalor(
    ruta_shp_entrada=ruta_calculos_shp ,
    ruta_shp_salida=ruta_calculos_shp_suavizado,
    buffer_metros=30,     # valor pequeño: 20–50 m suaviza bordes sin distorsionar
    tolerancia=10         # simplificación ligera
)
suavizador.suavizar()


from app.core.db.connection import SessionLocal
from app.modules.processing.islas.service import upsert_islas_calor
import geopandas as gpd

ruta = ruta_calculos_shp_suavizado
gdf = gpd.read_file(ruta)

db = SessionLocal()

for _, row in gdf.iterrows():
    anio = int(row["anio"])
    mes = int(row["mes"])
    geom_wkt = row.geometry.wkt
    propiedades = {k: v for k, v in row.items() if k not in ["geometry", "anio", "mes"]}

    upsert_islas_calor(db, anio, mes, geom_wkt, propiedades)