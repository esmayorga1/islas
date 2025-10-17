from app.ontology.classes.variable.urbana.densidad_edificatoria import DensidadEdificatoria
from pathlib import Path

input_dir = Path(r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021\indices")
output_dir = Path(r"F:\LAURA COTE\CAPAS RASTER\IMAGENES SENTINEL 2\2021\indices")

densidad = DensidadEdificatoria(input_dir=input_dir, output_dir=output_dir)
densidad.calculate()
