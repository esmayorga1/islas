from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar

# 1. Proyectar
input_dir=r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021"
proyectar = Reproyectar(input_dir)
proyectar.procesar_carpeta()


