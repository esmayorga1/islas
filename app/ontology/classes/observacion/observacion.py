from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar
from app.ontology.classes.variable.biofisica.ndvi import NDVI
from app.ontology.classes.variable.biofisica.ndbi import NDBI
from app.ontology.classes.variable.biofisica.albedo import Albedo
from app.ontology.classes.variable.biofisica.lst import LST
from app.ontology.classes.variable.biofisica.mnwi import MNDWI
from app.ontology.classes.variable.urbana.cobertura_suelo import ClasificacionSuperficie
from app.ontology.classes.variable.urbana.indice_area_construida import IndiceAreaConstruida
from app.ontology.classes.variable.socioeconomica.densidad_poblacional import DensidadPoblacionalDB
from app.ontology.classes.variable.climatica.temperatura_del_aire import TemperaturaAire
from app.ontology.classes.producto_analitico.riesgo_termico import IslasDeCalorPipeline
import os

# 1. Proyectar
input_dir=r"F:\Pruebas\IMAGENES SENTINEL 2\2021"
proyectar = Reproyectar(input_dir)
proyectar.procesar_carpeta()

input_dir_t=r"F:\Pruebas\2021"
proyectar = Reproyectar(input_dir_t)
proyectar.procesar_carpeta()


# 2. Construir los indices
input_dir_pro = os.path.join(input_dir, "reproyectadas")
output_dir_indices = os.path.join(input_dir, "Calculos")
NDVI(input_dir_pro, output_dir_indices).calculate()
NDBI(input_dir_pro, output_dir_indices).calculate()
Albedo(input_dir_pro, output_dir_indices).calculate()
LST(input_dir_pro, output_dir_indices).calculate()
MNDWI(input_dir_pro, output_dir_indices).calculate()

# 3. Variable urbana
clasificador = ClasificacionSuperficie(output_dir_indices, output_dir_indices)
clasificador.calculate()
indice = IndiceAreaConstruida(input_dir, input_dir)
indice.calculate()

# 4. Variable Socieconomica
dens = DensidadPoblacionalDB(output_dir_indices)
dens.rasterize(table_name="censo_2018", geom_col="geom", value_col="personas")

# Variable climatologica
input_dir_t_pro = os.path.join(input_dir_t, "reproyectadas")
temp_aire = TemperaturaAire(input_dir_t_pro, output_dir_indices)
resultado = temp_aire.calculate()


ruta_calculos = r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Calculos"
ruta_temp = r"F:\Pruebas\2021\reproyectadas"

pipeline = IslasDeCalorPipeline(ruta_calculos, ruta_temp, anio=2021)
pipeline.ejecutar_pipeline()
