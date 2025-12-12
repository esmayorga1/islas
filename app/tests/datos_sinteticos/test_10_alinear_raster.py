import os
from app.ontology.classes.datos_sinteticos.ds_00_raster_aligner import RasterAlignerToDEM3m
ruta_base = r"D:\002trabajos\21_islas_de_calor\datos_sinteticos"

dem = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster", "dem_3m.tif")

aligner = RasterAlignerToDEM3m(dem)

inputs_outputs = {
    "tipo_cobertura_3m.tif": "tipo_cobertura_3m_ALINEADO.tif",
    "porcentaje_urbano_3m.tif": "porcentaje_urbano_3m_ALINEADO.tif",
    "porcentaje_vegetacion_3m.tif": "porcentaje_vegetacion_3m_ALINEADO.tif",
    "distancia_vias_3m.tif": "distancia_vias_3m_ALINEADO.tif",
    "distancia_agua_3m.tif": "distancia_agua_3m_ALINEADO.tif",
}

base = dem = os.path.join(ruta_base, "01_temperatura_del_aire", "02_resultados", "01_resultado_raster")


for inp, out in inputs_outputs.items():
    inp_path = base + "\\" + inp
    out_path = base + "\\alineados\\" + out
    aligner.align(inp_path, out_path)
    print("✓ Alineado:", out_path)
