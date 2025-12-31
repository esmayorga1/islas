from pathlib import Path
from app.ontology.classes.datos_sinteticos.ds_13_humeda_relativa import RelativeHumidityMonthly
from app.ontology.classes.datos_sinteticos.ds_14_calcular_humedad_sintetica import SyntheticRelativeHumidity3m
from app.ontology.classes.datos_sinteticos.ds_10_validacion_archivos_humedad_relativa import HRInputFinder


class HRPipeline:
    """
    Pipeline HR con rutas controladas:
    - input_root: donde están los insumos (CSV HR, ndvi/ndbi/tair, dist_agua, urban, dem, slope...)
    - output_root: donde se crean TODAS las salidas (csv mensual y rasters sintéticos)
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        year: int,
        overwrite: bool = False,
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.year = int(year)
        self.overwrite = bool(overwrite)

    @staticmethod
    def _ensure_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)

    def _should_skip(self, path: Path, label: str = "") -> bool:
        if path.exists() and not self.overwrite:
            print(f"⏭️  {label or path.name} existe → omitido")
            return True
        return False

    def run(self):
        # --------------------------------------------------
        # 0) Validar insumos
        # --------------------------------------------------
        print("\n[0] Validando insumos de Humedad Relativa")

        finder = HRInputFinder(base_root=self.input_root, year=self.year)
        ins = finder.find_all()

        # --------------------------------------------------
        # 1) Directorios de resultados (AHORA salen de output_root)
        # --------------------------------------------------
        OUT_BASE = self.output_root / str(self.year)
        RESULTS_BASE = OUT_BASE / "00_resultados_humedad_relativa_sintetica"
        CSV_OUT_DIR = OUT_BASE / "03_Insumos_datos_sinteticos" / "01_archivo_csv_promedio_humedad_relativa"

        self._ensure_dir(RESULTS_BASE)
        self._ensure_dir(CSV_OUT_DIR)

        hr_out_csv = CSV_OUT_DIR / f"humedad_relativa_mensual_{self.year}.csv"

        # --------------------------------------------------
        # 2) HR mensual (CSV → dict)
        # --------------------------------------------------
        print("\n[1] HR mensual (CSV → dict)")

        hr_reader = RelativeHumidityMonthly(str(ins.hr_csv))
        hr_reader.load()
        hr_reader.compute_monthly()
        hr_dict = hr_reader.to_dict("yyyymm")

        if not self._should_skip(hr_out_csv, "CSV HR mensual"):
            hr_reader.save_csv(str(hr_out_csv))
            print("✓ CSV mensual HR generado")

        # --------------------------------------------------
        # 3) HR sintética (3 m)
        # --------------------------------------------------
        print("\n[2] HR sintética (3 m)")

        synthetic_hr = SyntheticRelativeHumidity3m(
            year=self.year,
            hr_monthly_dict=hr_dict,
            ndvi_dir=str(ins.ndvi_dir),
            ndbi_dir=str(ins.ndbi_dir),
            tair_dir=str(ins.tair_dir),
            dist_water_path=str(ins.dist_agua),
            urban_path=str(ins.urban),
            dem_path=str(ins.dem_3m),
            slope_path=str(ins.slope_3m),
            output_dir=str(RESULTS_BASE),
        )

        synthetic_hr.process_all()

        print("\n✅ PROCESO DE HUMEDAD RELATIVA FINALIZADO")
        print(f"📌 Salidas en: {OUT_BASE}")
