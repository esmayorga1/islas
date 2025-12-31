from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from app.ontology.classes.UHI.UHI_00_normalizar_raster import RasterNormalizer01
from app.ontology.classes.UHI.UHI_00_normalizar_por_archivo import RasterNormalizer01File
from app.ontology.classes.UHI.UHI_01_Alinear import FolderGridAligner
from app.ontology.classes.UHI.UHI_01_Alinear_por_archivo_o_folder import GridAlignerFile
from app.ontology.classes.UHI.UHI_01_Alinear_Y_Remuestrar import FolderTo3mResampler4326

from app.ontology.classes.UHI.UHI_03_SOM_input_builder import SOMInputBuilder
from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner
from app.ontology.classes.UHI.UHI_06_cluster_stats import ClusterStatsCalculator
from app.ontology.classes.UHI.UHI_07_icu_cluster_labeler import ICUClusterLabeler
from app.ontology.classes.UHI.UHI_08_ICUPostProcessor import ICUPostProcessor
from app.ontology.classes.UHI.UHI_09_Icu_score_agrupado import ICUIsobandPolygonGenerator


# =========================
# CONFIG DTO
# =========================
@dataclass(frozen=True)
class UHIFullPipelineConfig:
    year: str
    overwrite: bool
    root: Path
    ref_align: Path

    # SOM
    som_rows: int = 10
    som_cols: int = 10
    sample_max_pixels: int = 200_000
    sample_per_month: int = 200_000

    # Regex
    month_regex: str = r"(20\d{2})[_-](0?[1-9]|1[0-2])(?=[_-]|$)"

    # ICU Post + Isobandas
    icu_min_class: int = 3
    icu_kernel_size: int = 9
    icu_close_iters: int = 2
    icu_min_area_m2: float = 9000.0
    icu_connectivity: int = 8
    decay_m: float = 60.0
    smooth_sigma_m: float = 18.0
    band_edges: tuple = (0.25, 0.40, 0.55, 0.70, 0.85)

    # ICU weights
    weights: dict | None = None


# =========================
# PIPELINE CLASS
# =========================
class UHIFullPipeline:
    """
    Pipeline completo UHI:
    1) Normaliza TAIR/HR (folder->folder)
    2) Alinea normalizados (folder->folder)
    3) Normaliza distancias (file->file)
    4) Alinea estáticos categóricos (file->file)
    5) Remuestrea+alinea índices a 3m (folder->folder)
    6) Genera NPZ SOM input
    7) Entrena SOM anual + exporta modelo
    8) Calcula stats por cluster (CSV)
    9) Calcula ICU_SCORE (TIF)
    10) Postproceso ICU + Isobandas (GeoJSON)
    """

    def __init__(self, cfg: UHIFullPipelineConfig):
        self.cfg = cfg

        self.UHI = self.cfg.root / "06_UHI"
        self.NORM_DIR = self.UHI / "01_datos_normalizados"
        self.ALIGNED_DIR = self.UHI / "02_datos_normalizados_y_aliados"
        self.RESAMPLED_DIR = self.UHI / "03_alinear_resampliar"
        self.NPZ_DIR = self.UHI / "04_SOM_input_builder"
        self.SOM_OUT = self.UHI / "06_salidas_SOM" / self.cfg.year

        for d in [self.NORM_DIR, self.ALIGNED_DIR, self.RESAMPLED_DIR, self.NPZ_DIR, self.SOM_OUT]:
            self._ensure_dir(d)

        # Fuentes sintéticas
        self.TAIR_SYN_DIR = self.cfg.root / "04_Temperatura_Aire" / self.cfg.year / "02_resultados" / "03_resultado_temperatura_sintetica"
        self.HR_SYN_DIR = self.cfg.root / "05_humedad_relativa" / self.cfg.year / "00_resultados_humedad_relativa_sintetica"

        # Fuentes estáticas (TAIR pipeline)
        self.TAIR_RAS_DIR = self.cfg.root / "04_Temperatura_Aire" / self.cfg.year / "02_resultados" / "01_resultado_raster"
        self.TAIR_ALI_DIR = self.TAIR_RAS_DIR / "alineados"

        # Outputs clave
        self.MODEL_OUT = self.SOM_OUT / f"SOM_MODEL_ANNUAL_{self.cfg.year}_{self.cfg.som_rows}x{self.cfg.som_cols}.npz"
        self.STATS_DIR = self.SOM_OUT / "STATS"
        self.CLUSTERS_DIR = self.SOM_OUT / "CLUSTERS_TIF"
        self.ICU_DIR = self.SOM_OUT / "ICU_SCORE"
        self.ICU_AGR_DIR = self.SOM_OUT / "ICU_SCORE_AGRUPADO"
        self.ICU_POLY_DIR = self.SOM_OUT / "ICU_ISOBANDS_POLY"

        for d in [self.STATS_DIR, self.ICU_DIR, self.ICU_AGR_DIR, self.ICU_POLY_DIR]:
            self._ensure_dir(d)

        if self.cfg.weights is None:
            self._weights = {
                "TAIR": 3.0, "LST": 0.5, "HR": 2.0,
                "NDVI": 1.0, "MNDWI": 0.8, "CUERPOS_AGUA_BIN": 0.7, "VEGETACION_BIN": 0.7,
                "NDBI": 1.0, "PORC_URBANO": 1.0, "CONSTRUCCIONES_BIN": 0.9, "VIAS_BIN": 0.7,
                "DIST_AGUA_NORM": 0.9, "DIST_VIAS_NORM": 0.4,
                "ALBEDO": 0.2,
            }
        else:
            self._weights = dict(self.cfg.weights)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _has_any(dir_path: Path, pattern: str) -> bool:
        return dir_path.exists() and any(dir_path.glob(pattern))

    @staticmethod
    def _all_exist(paths) -> bool:
        return all(Path(p).exists() for p in paths)

    def _run_if_needed(self, name: str, done: bool, fn):
        if done and not self.cfg.overwrite:
            print(f"⏭️  {name} ya existe → omitido")
            return None
        print(f"\n▶ {name}")
        return fn()

    # -------------------------
    # Steps
    # -------------------------
    def step_01_norm_tair(self):
        def _do():
            norm = RasterNormalizer01(str(self.TAIR_SYN_DIR), str(self.NORM_DIR), overwrite=True)
            outs = norm.process_folder()
            print("Normalizados TAIR:", len(outs))
            return outs

        done = self._has_any(self.NORM_DIR, f"*TAIR*{self.cfg.year}*_norm_3m*.tif") or \
               self._has_any(self.NORM_DIR, f"*TAIRE*{self.cfg.year}*_norm_3m*.tif")
        return self._run_if_needed("1) Normalizar TAIR", done, _do)

    def step_01b_norm_hr(self):
        def _do():
            norm = RasterNormalizer01(str(self.HR_SYN_DIR), str(self.NORM_DIR), overwrite=True)
            outs = norm.process_folder()
            print("Normalizados HR:", len(outs))
            return outs

        done = self._has_any(self.NORM_DIR, f"*HR*{self.cfg.year}*_norm_3m*.tif")
        return self._run_if_needed("1b) Normalizar HR", done, _do)

    def step_02_align_normalized(self):
        def _do():
            aligner = FolderGridAligner(
                input_dir=str(self.NORM_DIR),
                output_dir=str(self.ALIGNED_DIR),
                align_to_path=str(self.cfg.ref_align),
                overwrite=True,
                suffix="_3m",
            )
            outs = aligner.process_folder(kind_mode="auto", copy_if_already_aligned=True)
            print("Alineados:", len(outs))
            return outs

        done = self._has_any(self.ALIGNED_DIR, f"*{self.cfg.year}*_3m.tif")
        return self._run_if_needed("2) Alinear normalizados", done, _do)

    def step_03_norm_distances(self):
        dist_agua_out = self.ALIGNED_DIR / "distancia_agua_3m_ALINEADO_NORMALIZADO.tif"
        dist_vias_out = self.ALIGNED_DIR / "distancia_vias_3m_ALINEADO_NORMALIZADO.tif"

        def _do():
            jobs = [
                ("distancia_agua_3m_ALINEADO.tif", dist_agua_out),
                ("distancia_vias_3m_ALINEADO.tif", dist_vias_out),
            ]
            for src, outp in jobs:
                norm_file = RasterNormalizer01File(
                    input_path=str(self.TAIR_ALI_DIR / src),
                    output_path=str(outp),
                    overwrite=True
                )
                print(norm_file.process_file())
            return [dist_agua_out, dist_vias_out]

        done = dist_agua_out.exists() and dist_vias_out.exists()
        return self._run_if_needed("3) Normalizar distancias (agua/vías)", done, _do)

    def step_04_align_static(self):
        static_outs = {
            "construcciones.tif": self.ALIGNED_DIR / "construcciones_ALINEADO.tif",
            "cuerpos_de_agua.tif": self.ALIGNED_DIR / "cuerpos_de_agua_ALINEADO.tif",
            "vias.tif": self.ALIGNED_DIR / "vias_ALINEADO.tif",
            "vegetacion.tif": self.ALIGNED_DIR / "vegetacion_ALINEADO.tif",
        }

        def _do():
            aligner = GridAlignerFile(
                align_to_path=str(self.cfg.ref_align),
                overwrite=True,
                suffix="_ALINEADO",
            )
            for src, dst in static_outs.items():
                out = aligner.process_file(
                    input_path=str(self.TAIR_RAS_DIR / src),
                    output_path=str(dst),
                    kind_mode="categorical",
                    copy_if_already_aligned=True,
                )
                print("Archivo alineado:", out)
            return list(static_outs.values())

        done = self._all_exist(static_outs.values())
        return self._run_if_needed("4) Alinear estáticos (categorical)", done, _do)

    def step_05_resample_indices(self):
        def _do():
            res = FolderTo3mResampler4326(
                input_dir=str(self.cfg.root / "02_Calculos"),
                output_dir=str(self.RESAMPLED_DIR),
                align_to_path=str(self.cfg.ref_align),
                mode="direct",
                overwrite=True
            )
            outs = res.process_folder(apply_clip=True, clip_range=(0.0, 1.0), kind_mode="auto")
            print("Generados:", len(outs))
            return outs

        done = self._has_any(self.RESAMPLED_DIR, f"*_{self.cfg.year}_*_NDVI_3m*.tif") or \
               self._has_any(self.RESAMPLED_DIR, f"*S2_{self.cfg.year}_*_PatioBonito_NDVI_3m*.tif")
        return self._run_if_needed("5) Resample+align índices 3m", done, _do)

    def step_06_build_npz(self):
        def _do():
            builder = SOMInputBuilder(
                input_dir=str(self.cfg.root),
                month_regex=self.cfg.month_regex,
                monthly_patterns={
                    "NDVI":   r"_NDVI_3m",
                    "NDBI":   r"_NDBI_3m",
                    "ALBEDO": r"_Albedo_3m",
                    "LST":    r"_LST_3m",
                    "MNDWI":  r"_MNDWI_3m|_MNWI_3m",
                    "TAIR":   r"TAIR(E)?_SINTETICA_3M_.*_norm_3m",
                    "HR":     r"HR_SINTETICA_3M_.*_norm_3m",
                },
                static_patterns={
                    "CONSTRUCCIONES_BIN": r"construcciones_ALINEADO",
                    "CUERPOS_AGUA_BIN":   r"cuerpos_de_agua_ALINEADO",
                    "VEGETACION_BIN":     r"vegetacion_ALINEADO",
                    "VIAS_BIN":           r"vias_ALINEADO",
                    "PORC_URBANO":        r"porcentaje_urbano_3m_ALINEADO",
                    "DIST_AGUA_NORM":     r"^distancia_agua_3m_ALINEADO_NORMALIZADO",
                    "DIST_VIAS_NORM":     r"distancia_vias_3m_ALINEADO_NORMALIZADO",
                },
                mask_mode="all_valid",
                sample_max_pixels=self.cfg.sample_max_pixels,
                allow_missing_monthly_vars=False,
                require_static=True,
            )

            print("DIAG:", builder.diagnose(self.cfg.year))

            packs = builder.build()
            packs_year = {k: v for k, v in packs.items() if k.startswith(f"{self.cfg.year}_")}

            for month_key, pack in sorted(packs_year.items()):
                out_path = self.NPZ_DIR / f"SOM_INPUT_{month_key}.npz"
                if out_path.exists() and not self.cfg.overwrite:
                    continue
                builder.save_npz(str(out_path), pack["X"], pack["coords"], pack["meta"])
                print(f"✅ {month_key} → X shape: {pack['X'].shape}")

            total = len(list(self.NPZ_DIR.glob(f"SOM_INPUT_{self.cfg.year}_??.npz")))
            print("🎉 NPZ generados:", total)
            return True

        done = self._has_any(self.NPZ_DIR, f"SOM_INPUT_{self.cfg.year}_??.npz")
        return self._run_if_needed("6) Generar NPZ (SOM inputs)", done, _do)

    def step_07_som_annual(self):
        def _do():
            runner = SOMYearRunner(
                npz_dir=str(self.NPZ_DIR),
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
                standardize=True,
                sample_per_month=self.cfg.sample_per_month,
            )
            packs = runner.load_year(self.cfg.year)
            runner.fit_annual(packs)
            results = runner.predict_year(packs)

            model = runner.export_model()
            np.savez_compressed(self.MODEL_OUT, **model)
            print("✅ Modelo anual guardado:", self.MODEL_OUT)
            return True

        done = self.MODEL_OUT.exists()
        return self._run_if_needed("7) SOM anual + export modelo", done, _do)

    def step_08_cluster_stats(self):
        def _do():
            runner = SOMYearRunner(
                npz_dir=str(self.NPZ_DIR),
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
                standardize=False,
                sample_per_month=self.cfg.sample_per_month,
            )
            packs = runner.load_year(self.cfg.year)
            runner.fit_annual(packs)
            results = runner.predict_year(packs)

            calc = ClusterStatsCalculator(min_pixels_per_cluster=50)

            for month_key, pack in results.items():
                out_csv = self.STATS_DIR / f"CLUSTER_STATS_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.csv"
                if out_csv.exists() and not self.cfg.overwrite:
                    continue
                rows = calc.compute(
                    X=pack["X"],
                    cluster_ids=pack["cluster_ids"],
                    variables=pack["meta"]["variables"],
                    extra={"year": self.cfg.year, "month": month_key},
                )
                calc.save_csv(rows, str(out_csv))
                print("✅", out_csv.name, "| clusters:", len(rows))

            return True

        done = self._has_any(self.STATS_DIR, f"CLUSTER_STATS_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.csv")
        return self._run_if_needed("8) Cluster stats CSV", done, _do)

    def step_09_icu_score(self):
        def _do():
            ensure_dir(self.ICU_DIR)

            labeler = ICUClusterLabeler(
                p_hot=92, p_cool=8,
                score_p1=80, score_p2=90, score_p3=97,
                min_pixels_per_cluster=400,
                weights=self._weights
            )

            cluster_files = sorted(self.CLUSTERS_DIR.glob(f"SOM_CLUSTER_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"))
            if not cluster_files:
                raise FileNotFoundError(f"No se encontraron clusters en:\n{self.CLUSTERS_DIR}")

            for cpath in cluster_files:
                parts = cpath.stem.split("_")
                month_key = f"{parts[2]}_{parts[3]}"

                stats_csv = self.STATS_DIR / f"CLUSTER_STATS_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.csv"
                out_tif = self.ICU_DIR / f"ICU_SCORE_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"

                if out_tif.exists() and not self.cfg.overwrite:
                    continue
                if not stats_csv.exists():
                    print(f"⚠️  SKIP {month_key} → falta CSV {stats_csv.name}")
                    continue

                labeler.run(cluster_tif=str(cpath), stats_csv=str(stats_csv), out_tif=str(out_tif))
                print("✅", out_tif.name)

            return True

        done = self._has_any(self.ICU_DIR, f"ICU_SCORE_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif")
        return self._run_if_needed("9) ICU_SCORE (tif)", done, _do)

    def step_10_post_isobands(self):
        def _do_post():
            post = ICUPostProcessor(
                nodata=-1,
                min_class=self.cfg.icu_min_class,
                kernel_size=self.cfg.icu_kernel_size,
                close_iters=self.cfg.icu_close_iters,
                min_area_m2=self.cfg.icu_min_area_m2,
                connectivity=self.cfg.icu_connectivity,
                keep_low_class=False,
                fill_nodata_to_zero=True
            )

            for in_tif in sorted(self.ICU_DIR.glob(f"ICU_SCORE_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif")):
                out_tif = self.ICU_AGR_DIR / f"{in_tif.stem}_AGRUPADO.tif"
                if out_tif.exists() and not self.cfg.overwrite:
                    continue
                post.run(str(in_tif), str(out_tif))
                print("✅", out_tif.name)

            return True

        def _do_isobands():
            gen = ICUIsobandPolygonGenerator(
                nodata=-1,
                min_class=self.cfg.icu_min_class,
                decay_m=self.cfg.decay_m,
                smooth_sigma_m=self.cfg.smooth_sigma_m,
                band_edges=list(self.cfg.band_edges),
                dissolve=True
            )

            for f in sorted(self.ICU_AGR_DIR.glob(f"ICU_SCORE_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif")):
                out_geojson = self.ICU_POLY_DIR / f"{f.stem}_ISOBANDS_DISSOLVE.geojson"
                if out_geojson.exists() and not self.cfg.overwrite:
                    continue
                gen.run(str(f), str(out_geojson))
                print("✅", out_geojson.name)

            return True

        post_done = self._has_any(self.ICU_AGR_DIR, f"ICU_SCORE_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif")
        self._run_if_needed("10a) ICU postproceso", post_done, _do_post)

        iso_done = self._has_any(self.ICU_POLY_DIR, f"ICU_SCORE_{self.cfg.year}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO_ISOBANDS_DISSOLVE.geojson")
        self._run_if_needed("10b) Isobandas (geojson)", iso_done, _do_isobands)

        return True

    # -------------------------
    # Run all
    # -------------------------
    def run(self):
        self.step_01_norm_tair()
        self.step_01b_norm_hr()
        self.step_02_align_normalized()
        self.step_03_norm_distances()
        self.step_04_align_static()
        self.step_05_resample_indices()
        self.step_06_build_npz()
        self.step_07_som_annual()
        self.step_08_cluster_stats()
        self.step_09_icu_score()
        self.step_10_post_isobands()
        print("\n🎉 PIPELINE COMPLETO OK. (Con overwrite=False no reescribe si ya hay salidas)")


# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    cfg = UHIFullPipelineConfig(
        year="2022",
        overwrite=True,
        root=Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\Pruebas1"),
        ref_align=Path(r"D:\002trabajos\21_islas_de_calor\CAPAS RASTER\IMAGENES SENTINEL 2\2021\Muestra\porcentaje_urbano_3m_ALINEADO.tif"),
        som_rows=10,
        som_cols=10,
    )
    UHIFullPipeline(cfg).run()
