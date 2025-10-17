import os

class VariableValidator:
    """
    Clase para validar la existencia de todas las variables necesarias
    para el cálculo de islas de calor por mes (biofísicas, temperatura, cobertura, densidad).
    """

    def __init__(self, ruta_indices: str, ruta_temp: str, anio: int):
        self.ruta_indices = ruta_indices
        self.ruta_temp = ruta_temp
        self.anio = anio

        # Variables biofísicas mensuales
        self.biofisicas = ["Albedo", "LST", "MNDWI", "NDBI", "NDVI"]

        # Variable socioeconómica (única, fija)
        self.densidad_poblacional = os.path.join(ruta_indices, "censo_2018_raster.tif")

        # Resultados
        self.faltantes = {}

    # -----------------------------------------------------------
    def validar(self):
        """Ejecuta la validación completa."""
        print(f"🔍 Iniciando validación de variables para el año {self.anio}\n")
        self.faltantes.clear()

        self._validar_mensuales()
        self._validar_anuales()
        self._resumen()

        return self.faltantes

    # -----------------------------------------------------------
    def _validar_mensuales(self):
        """Valida las variables mensuales (biofísicas, temperatura, cobertura del suelo)."""
        for mes in range(1, 13):
            archivos_faltantes = []
            mes_str = str(mes)
            mes_temp = str(mes).zfill(2)

            # Validar variables biofísicas
            for var in self.biofisicas:
                nombre = f"S2_{self.anio}_{mes_str}_PatioBonito_{var}.tif"
                ruta_archivo = os.path.join(self.ruta_indices, nombre)
                if not os.path.exists(ruta_archivo):
                    archivos_faltantes.append(nombre)

            # Validar temperatura del aire
            nombre_temp = f"TempAire_{self.anio}_{mes_temp}.tif"
            ruta_temp_mes = os.path.join(self.ruta_temp, nombre_temp)
            if not os.path.exists(ruta_temp_mes):
                archivos_faltantes.append(nombre_temp)

            # Validar cobertura del suelo
            nombre_cobertura = f"S2_{self.anio}_{mes_str}_PatioBonito_ClasificacionSuperficie.tif"
            ruta_cobertura = os.path.join(self.ruta_indices, nombre_cobertura)
            if not os.path.exists(ruta_cobertura):
                archivos_faltantes.append(nombre_cobertura)

            # Resultado del mes
            if archivos_faltantes:
                self.faltantes[mes] = archivos_faltantes
                print(f"⚠️  Mes {mes}: faltan {len(archivos_faltantes)} archivos")
                for a in archivos_faltantes:
                    print(f"   └── {a}")
            else:
                print(f"✅ Mes {mes}: todas las variables encontradas")

    # -----------------------------------------------------------
    def _validar_anuales(self):
        """Valida variables fijas (densidad poblacional)."""
        print("\n🗺️ Validando variables fijas...\n")
        faltantes_fijas = []

        if not os.path.exists(self.densidad_poblacional):
            faltantes_fijas.append(os.path.basename(self.densidad_poblacional))

        if faltantes_fijas:
            self.faltantes["fijas"] = faltantes_fijas
            print(f"⚠️ Faltan {len(faltantes_fijas)} variables fijas:")
            for a in faltantes_fijas:
                print(f"   └── {a}")
        else:
            print("✅ Todas las variables fijas encontradas")

    # -----------------------------------------------------------
    def _resumen(self):
        """Muestra un resumen general."""
        print("\n📊 Resumen general")
        if not self.faltantes:
            print("✔️ Todas las variables (mensuales y fijas) están disponibles.")
        else:
            print(f"❌ Se detectaron faltantes en {len(self.faltantes)} conjuntos de datos.")

    # -----------------------------------------------------------
    def guardar_reporte(self, salida="validacion_variables.txt"):
        """Guarda un reporte de validación en un archivo de texto."""
        with open(salida, "w", encoding="utf-8") as f:
            f.write(f"Reporte de validación de variables ({self.anio})\n")
            f.write("=" * 40 + "\n\n")
            if not self.faltantes:
                f.write("Todas las variables están disponibles.\n")
            else:
                for k, v in self.faltantes.items():
                    if k == "fijas":
                        f.write("\n[Variables fijas faltantes]\n")
                    else:
                        f.write(f"\n[Mes {k}]\n")
                    for item in v:
                        f.write(f"- {item}\n")

        print(f"\n📝 Reporte guardado en: {os.path.abspath(salida)}")
