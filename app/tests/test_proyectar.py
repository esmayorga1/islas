from app.ontology.classes.fuente_de_datos.reproyectar import Reproyectar

# 1. Proyectar
input_dir=r"F:\Pruebas\IMAGENES SENTINEL 2\2021"
proyectar = Reproyectar(input_dir)
proyectar.procesar_carpeta()


