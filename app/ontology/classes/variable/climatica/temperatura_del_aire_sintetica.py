from pathlib import Path
from app.ontology.classes.datos_sinteticos.ds_01_convertir_shp_raster import ShapefileToRaster
from app.ontology.classes.datos_sinteticos.ds_02_leer_archivo_csv_temperatura_aire import AirTemperatureMonthly
from app.ontology.classes.datos_sinteticos.ds_03_porcentaje_construccion import UrbanDensity3m
from app.ontology.classes.datos_sinteticos.ds_04_densidad_vegetacion import VegetationDensity3m
from app.ontology.classes.datos_sinteticos.ds_05_distancia_vias import DistanceToRoads
from app.ontology.classes.datos_sinteticos.ds_06_distancia_cuerpos_agua import DistanceToWater
from app.ontology.classes.datos_sinteticos.ds_07_tipo_de_cobertura import CoverageClassifier3m
from app.ontology.classes.datos_sinteticos.ds_08_resampliar_DEM import RasterResampler3mWGS84
from app.ontology.classes.datos_sinteticos.ds_02_mapa_pendientes import SlopeGenerator
from app.ontology.classes.datos_sinteticos.ds_00_raster_aligner import RasterAlignerToDEM3m
from app.ontology.classes.datos_sinteticos.ds_11_temperatura_aire_sintetica import SyntheticAirTemperature3m
from app.ontology.classes.datos_sinteticos.ds_09_validación_de_archivos_temperatura_aire import TAIRInputFinder


class TAIRPipeline:
    """
    Pipeline TAIR con rutas controladas:
    - input_root: donde están los insumos (SHP, DEM, CSV, ndvi/lst dirs, etc.)
    - output_root: donde se crean TODAS las carpetas de salida (02_resultados, 03_resultado_..., etc.)
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        year: int,
        overwrite: bool = False,
        px: float = 0.000009,
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.year = int(year)
        self.overwrite = bool(overwrite)
        self.px = float(px)

    @staticmethod
    def _ensure_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)

    def _should_skip(self, path: Path, label: str = "") -> bool:
        if path.exists() and not self.overwrite:
            print(f"⏭️  {label or path.name} existe → omitido")
            return True
        return False

    def run(self):
        print("\n[0] Validando insumos de entrada")
        finder = TAIRInputFinder(base_root=self.input_root, year=self.year)
        ins = finder.find_all()

        # ==============================
        # SALIDAS (AHORA SALEN DE output_root)
        # ==============================
        OUT_BASE = self.output_root / str(self.year)
        RAS = OUT_BASE / "02_resultados" / "01_resultado_raster"
        ALI = RAS / "alineados"
        OUT_TAIR = OUT_BASE / "02_resultados" / "03_resultado_temperatura_sintetica"
        OUT_CSV = OUT_BASE / "02_resultados" / "02_resultado_promedio_temperatura_aire"

        self._ensure_dir(RAS)
        self._ensure_dir(ALI)
        self._ensure_dir(OUT_TAIR)
        self._ensure_dir(OUT_CSV)

        # 1) DEM → DEM 3m
        print("\n[1] DEM 3 m")
        dem_3m = RAS / "dem_3m.tif"
        if not self._should_skip(dem_3m, "DEM 3 m"):
            resampler = RasterResampler3mWGS84(str(ins.dem_original), method="bilinear")
            resampler.resample()
            resampler.save(str(dem_3m))
            print("✓ DEM 3 m generado")

        # 2) SHP → Raster
        print("\n[2] SHP → Raster")
        jobs = [
            (ins.shp_agua, "cuerpos_de_agua.tif"),
            (ins.shp_vias, "vias.tif"),
            (ins.shp_construcciones, "construcciones.tif"),
            (ins.shp_vegetacion, "vegetacion.tif"),
        ]
        for shp, name in jobs:
            out = RAS / name
            if self._should_skip(out, name):
                continue
            conv = ShapefileToRaster(
                shp_path=str(shp),
                shp_referencia_path=str(ins.shp_upl),
                pixel_size=self.px,
                raster_value=1,
            )
            conv.rasterize_shp(str(out))
            print("✓", name)

        # 3) CSV Temperatura del Aire → mensual
        print("\n[3] Temperatura del aire mensual")
        tair_out_csv = OUT_CSV / f"temperatura_mensual_{self.year}.csv"

        reader = AirTemperatureMonthly(str(ins.tair_csv))
        reader.load()
        reader.compute_monthly()
        temp_dict = reader.to_dict("yyyymm")

        if not self._should_skip(tair_out_csv, "CSV temperatura mensual"):
            reader.save_csv(str(tair_out_csv))
            print("✓ CSV mensual generado")

        # 4) % Urbano
        print("\n[4] Porcentaje urbano")
        urban_out = RAS / "porcentaje_urbano_3m.tif"
        if not self._should_skip(urban_out, urban_out.name):
            urban = UrbanDensity3m(str(RAS / "construcciones.tif"), str(RAS / "vias.tif"))
            urban.compute_density()
            urban.save_raster(str(urban_out))
            print("✓", urban_out.name)

        # 5) % Vegetación
        print("\n[5] Porcentaje vegetación")
        veg_out = RAS / "porcentaje_vegetacion_3m.tif"
        if not self._should_skip(veg_out, veg_out.name):
            veg = VegetationDensity3m(str(RAS / "vegetacion.tif"))
            veg.compute_density()
            veg.save_raster(str(veg_out))
            print("✓", veg_out.name)

        # 6) Distancia a vías
        print("\n[6] Distancia a vías")
        vias_out = RAS / "distancia_vias_3m.tif"
        if not self._should_skip(vias_out, vias_out.name):
            dist_vias = DistanceToRoads(str(RAS / "vias.tif"), output_resolution=3)
            dist_vias.compute_distance()
            dist_vias.save_raster(str(vias_out))
            print("✓", vias_out.name)

        # 7) Distancia a cuerpos de agua
        print("\n[7] Distancia a cuerpos de agua")
        agua_out = RAS / "distancia_agua_3m.tif"
        if not self._should_skip(agua_out, agua_out.name):
            dist_agua = DistanceToWater(str(RAS / "cuerpos_de_agua.tif"), output_resolution=3)
            dist_agua.compute_distance()
            dist_agua.save_raster(str(agua_out))
            print("✓", agua_out.name)

        # 8) Tipo de cobertura
        print("\n[8] Tipo de cobertura")
        cov_out = RAS / "tipo_cobertura_3m.tif"
        if not self._should_skip(cov_out, cov_out.name):
            cov = CoverageClassifier3m(
                raster_agua_1m=str(RAS / "cuerpos_de_agua.tif"),
                raster_vegetacion_1m=str(RAS / "vegetacion.tif"),
                raster_urbano_1m=str(RAS / "construcciones.tif"),
                factor=3,
            )
            cov.aggregate_to_3m()
            cov.save_raster(str(cov_out))
            print("✓", cov_out.name)

        # 9) Pendiente
        print("\n[9] Pendiente")
        slope = RAS / "slope_3m.tif"
        if not self._should_skip(slope, slope.name):
            sg = SlopeGenerator(str(dem_3m))
            sg.compute_slope()
            sg.save_slope(str(slope))
            print("✓", slope.name)

        # 10) Alinear rasters
        print("\n[10] Alineando rasters")
        aligner = RasterAlignerToDEM3m(str(dem_3m))
        align_jobs = [
            ("tipo_cobertura_3m.tif", "tipo_cobertura_3m_ALINEADO.tif"),
            ("porcentaje_urbano_3m.tif", "porcentaje_urbano_3m_ALINEADO.tif"),
            ("porcentaje_vegetacion_3m.tif", "porcentaje_vegetacion_3m_ALINEADO.tif"),
            ("distancia_vias_3m.tif", "distancia_vias_3m_ALINEADO.tif"),
            ("distancia_agua_3m.tif", "distancia_agua_3m_ALINEADO.tif"),
        ]
        for src, dst in align_jobs:
            out = ALI / dst
            if self._should_skip(out, dst):
                continue
            aligner.align(str(RAS / src), str(out))
            print("✓", dst)

        # 11) Temperatura del aire sintética
        print("\n[11] Temperatura del aire sintética")
        synth = SyntheticAirTemperature3m(
            year=self.year,
            lst_dir=str(ins.lst_dir),
            ndvi_dir=str(ins.ndvi_dir),
            coverage_path=str(ALI / "tipo_cobertura_3m_ALINEADO.tif"),
            urban_path=str(ALI / "porcentaje_urbano_3m_ALINEADO.tif"),
            vegetation_path=str(ALI / "porcentaje_vegetacion_3m_ALINEADO.tif"),
            dist_vias_path=str(ALI / "distancia_vias_3m_ALINEADO.tif"),
            dist_agua_path=str(ALI / "distancia_agua_3m_ALINEADO.tif"),
            dem_path=str(dem_3m),
            slope_path=str(slope),
            temp_monthly_dict=temp_dict,
            output_dir=str(OUT_TAIR),
        )
        synth.process_all()

        print("\n✅ PROCESO COMPLETO FINALIZADO")
        print(f"📌 Salidas en: {OUT_BASE}")
