from app.ontology.classes.variable.climatica.temperatura_del_aire import TemperaturaAire

# === Rutas ===
input_dir = r"F:\LAURA COTE\CAPAS RASTER\TEMPERATURA AIRE\2021"     # carpeta donde tienes TempAire_YYYY_MM.tif
output_dir = r"F:\LAURA COTE\CAPAS RASTER\TEMPERATURA AIRE\2021RESULTADOS"

# === Ejecutar cálculo ===
if __name__ == "__main__":
    print("🧪 Iniciando prueba del cálculo de temperatura del aire...")
    temp_aire = TemperaturaAire(input_dir, output_dir)
    resultado = temp_aire.calculate()

    if resultado is not None:
        print(f"✅ Prueba completada. Temperatura media anual calculada con éxito.")
    else:
        print("⚠️ No se generó resultado. Verifica los archivos de entrada.")
