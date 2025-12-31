from backend.app.ontology.classes.variable.climatica.temperatura_del_aire_sintetica import TemperaturaAireSintetica
from pathlib import Path

proc = TemperaturaAireSintetica(
    input_dir=Path(r"D:\002trabajos\21_islas_de_calor\insumos\TEMPERATURA AIRE\2021\Reproyectadas"),
    output_dir=Path(r"D:\002trabajos\21_islas_de_calor\insumos\TEMPERATURA AIRE\2021\Datos_Sinteticos"),
    aoi=Path(r"D:\002trabajos\21_islas_de_calor\insumos\SHP\UPL\upl.shp"),
    pixel_size=0.000027,
    intensidad=0.4,
    escala=0.001
)

proc.calculate()
