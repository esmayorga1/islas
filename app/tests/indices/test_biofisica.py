from app.ontology.classes.variable.biofisica.ndvi import NDVI
from app.ontology.classes.variable.biofisica.ndbi import NDBI
from app.ontology.classes.variable.biofisica.albedo import Albedo
from app.ontology.classes.variable.biofisica.lst import LST
from app.ontology.classes.variable.biofisica.mnwi import MNDWI


input_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\reproyectadas"
output_dir = r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\calculos"

NDVI(input_dir, output_dir).calculate()
NDBI(input_dir, output_dir).calculate()
Albedo(input_dir, output_dir).calculate()
LST(input_dir, output_dir).calculate()
MNDWI(input_dir, output_dir).calculate()