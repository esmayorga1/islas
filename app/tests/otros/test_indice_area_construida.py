from app.ontology.classes.variable.urbana.indice_area_construida import IndiceAreaConstruida

input_dir = r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021\indices"
output_dir = r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021\indices"

indice = IndiceAreaConstruida(input_dir, output_dir)
indice.calculate()
