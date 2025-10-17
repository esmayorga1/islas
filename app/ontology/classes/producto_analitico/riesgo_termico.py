import os
import geopandas as gpd
from app.ontology.classes.producto_analitico.VariableValidator import VariableValidator
from app.ontology.classes.producto_analitico.UHIGenerator import UHIGenerator
from app.ontology.classes.producto_analitico.UHItoSHP import UHItoSHP
from app.ontology.classes.producto_analitico.SuavizadorIslasCalor import SuavizadorIslasCalor
from app.modules.processing.islas.service import upsert_islas_calor
from app.core.db.connection import SessionLocal


class IslasDeCalorPipeline:
    """
    🔥 Clase que ejecuta el flujo completo para generar, suavizar y subir
    las islas de calor urbanas (UHI) a la base de datos.

    Etapas:
    1️⃣ Validar variables (NDVI, NDBI, LST, etc.)
    2️⃣ Generar mapas mensuales de UHI
    3️⃣ Convertir a shapefile
    4️⃣ Suavizar polígonos
    5️⃣ Subir/actualizar en PostGIS
    """

    def __init__(self, ruta_calculos: str, ruta_temp: str, anio: int):
        self.ruta_calculos = ruta_calculos
        self.ruta_temp = ruta_temp
        self.anio = anio
        self.db = SessionLocal()

        # Rutas intermedias
        self.shp_islas = os.path.join(self.ruta_calculos, "islas_calor.shp")
        self.shp_islas_suav = os.path.join(self.ruta_calculos, "islas_calor_suavizado.shp")

    # ====================================================
    # 1️⃣ Validar variables de entrada
    # ====================================================
    def validar_variables(self):
        print(f"🧩 Validando variables para el año {self.anio}...")
        validador = VariableValidator(self.ruta_calculos, ruta_temp=self.ruta_temp, anio=self.anio)
        validador.validar()
        print("✅ Variables validadas correctamente.")

    # ====================================================
    # 2️⃣ Generar mapas de UHI
    # ====================================================
    def generar_uhi(self):
        print("🔥 Generando mapas UHI...")
        uhi = UHIGenerator(self.ruta_calculos, anio=self.anio, salida=self.ruta_calculos)
        uhi.generar()
        print("✅ Mapas UHI generados correctamente.")

    # ====================================================
    # 3️⃣ Convertir UHI a shapefile
    # ====================================================
    def generar_shp(self, umbral: float = 0.75):
        print(f"📦 Generando shapefile de islas de calor (umbral={umbral})...")
        uhi_shp = UHItoSHP(
            ruta_uhi=self.ruta_calculos,
            anio=self.anio,
            salida_shp=self.shp_islas,
            umbral=umbral
        )
        uhi_shp.generar()
        print(f"✅ Shapefile generado en: {self.shp_islas}")

    # ====================================================
    # 4️⃣ Suavizar polígonos de islas
    # ====================================================
    def suavizar_islas(self, buffer_m: int = 30, tolerancia: int = 10):
        print("🌀 Suavizando polígonos...")
        suavizador = SuavizadorIslasCalor(
            ruta_shp_entrada=self.shp_islas,
            ruta_shp_salida=self.shp_islas_suav,
            buffer_metros=buffer_m,
            tolerancia=tolerancia
        )
        suavizador.suavizar()
        print(f"✅ Shapefile suavizado: {self.shp_islas_suav}")

    # ====================================================
    # 5️⃣ Subir shapefile a PostGIS
    # ====================================================
    def subir_a_postgis(self):
        print("⬆️ Subiendo o actualizando islas de calor en la base de datos...")

        gdf = gpd.read_file(self.shp_islas_suav)
        total = len(gdf)
        print(f"📊 Registros a procesar: {total}")

        for _, row in gdf.iterrows():
            anio = int(row["anio"])
            mes = int(row["mes"])
            geom_wkt = row.geometry.wkt
            propiedades = {k: v for k, v in row.items() if k not in ["geometry", "anio", "mes"]}

            upsert_islas_calor(self.db, anio, mes, geom_wkt, propiedades)

        print(f"✅ {total} registros subidos o actualizados correctamente.")

    # ====================================================
    # 🚀 Ejecutar todo el flujo
    # ====================================================
    def ejecutar_pipeline(self):
        print(f"\n🚀 Iniciando pipeline completo de islas de calor ({self.anio})...\n")

        self.validar_variables()
        self.generar_uhi()
        self.generar_shp()
        self.suavizar_islas()
        self.subir_a_postgis()

        print(f"\n🎯 Pipeline completo para el año {self.anio}\n")


# ====================================================
# 💡 USO
# ====================================================




