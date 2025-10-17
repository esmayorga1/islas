from .urbana import Urbana
import numpy as np

class IndiceAreaConstruida(Urbana):
    """
    Construye el índice de área construida (IBI) a partir de NDVI, NDBI y MNDWI.
    Optimizado para proyectos mayormente urbanos:
    - Casi todos los píxeles urbanos se consideran.
    - Valores positivos normalizados proporcionalmente al área del pixel.
    """

    def calculate(self):
        prefixes = set("_".join(f.stem.split("_")[:-1]) for f in self.input_dir.glob("*_NDBI.tif"))

        for prefix in prefixes:
            # Leer índices ya calculados
            ndbi, profile = self.read_raster(prefix, "NDBI")
            ndvi, _ = self.read_raster(prefix, "NDVI")
            mndwi, _ = self.read_raster(prefix, "MNDWI")

            # Calcular IBI
            ibi = (ndbi - (ndvi + mndwi)) / (ndbi + (ndvi + mndwi) + 1e-10)

            # Mantener solo valores positivos y normalizar
            ibi_positive = np.clip(ibi, 0, None)
            if ibi_positive.max() > 0:
                ibi_norm = ibi_positive / ibi_positive.max()  # escala 0–1
            else:
                ibi_norm = ibi_positive  # todos ceros, muy raro en urbano

            # Calcular área por pixel (m²)
            transform = profile['transform']
            pixel_area = abs(transform.a * transform.e)

            # Área construida proporcional por pixel
            ibi_area = ibi_norm * pixel_area

            # Guardar raster final
            self.save_raster(ibi_area, profile, prefix, "IndiceAreaConstruida")
            print(f"✅ Raster de área construida guardado: {prefix}_IndiceAreaConstruida.tif")
