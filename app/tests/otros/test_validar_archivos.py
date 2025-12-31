from app.ontology.classes.producto_analitico.VariableValidator import VariableValidator
validador = VariableValidator(ruta_indices=r"F:\Pruebas\IMAGENES SENTINEL 2\2021\Calculos", ruta_temp=r"F:\Pruebas\2021\reproyectadas", anio=2021)
validador.validar()
#validador.guardar_reporte()