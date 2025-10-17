# backend/app/tests/test_fuente_sentinel2.py

from pathlib import Path
from app.ontology.classes.fuente_de_datos.sentinel2 import Sentinel2Fuente


def test_lectura_sentinel2():
    """
    Prueba la lectura de las imágenes Sentinel-2 unificadas.
    """
    ruta_carpeta = Path(r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021")  # 🔁 Ajusta esta ruta
    fuente = Sentinel2Fuente(ruta_carpeta, 2021, "PatioBonito")

    print(f"\n📂 Verificando imágenes en: {fuente.ruta}")
    imagenes = fuente.listar_imagenes()
    print(f"🗂️ Se encontraron {len(imagenes)} imágenes:")

    for img in imagenes:
        print(f"   - {img.name}")

    # Probar lectura del primer archivo
    if imagenes:
        with fuente.leer_imagen(1) as src:
            print("\n✅ Imagen abierta correctamente:")
            print(f"   Bandas: {src.count}")
            print(f"   Dimensiones: {src.width}x{src.height}")
            print(f"   CRS: {src.crs}")
            print(f"   Resolución: {src.res}")
    else:
        print("\n⚠️ No se encontraron imágenes para leer.")

if __name__ == "__main__":
    test_lectura_sentinel2()