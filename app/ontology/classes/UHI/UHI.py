from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from app.ontology.classes.UHI.UHI_00_normalizar_raster import RasterNormalizer01
from app.ontology.classes.UHI.UHI_00_normalizar_por_archivo import RasterNormalizer01File
from app.ontology.classes.UHI.UHI_01_Alinear import FolderGridAligner
from app.ontology.classes.UHI.UHI_01_Alinear_por_archivo_o_folder import GridAlignerFile
from app.ontology.classes.UHI.UHI_01_Alinear_Y_Remuestrar import FolderTo3mResampler4326

from app.ontology.classes.UHI.UHI_03_SOM_input_builder import SOMInputBuilder
from app.ontology.classes.UHI.UHI_04_SOM_year_runner import SOMYearRunner
from app.ontology.classes.UHI.UHI_05_cluster_geotiff_writer import ClusterGeoTIFFWriter
from app.ontology.classes.UHI.UHI_06_cluster_stats import ClusterStatsCalculator
from app.ontology.classes.UHI.UHI_07_icu_cluster_labeler import ICUClusterLabeler
from app.ontology.classes.UHI.UHI_08_ICUPostProcessor import ICUPostProcessor

# ✅ V11 (UPL mask + clip + anual by-month)
from app.ontology.classes.UHI.UHI_11_Icu_score_agrupado_v2 import ICUIsobandPolygonGenerator


# ======================================================
# CONFIG (inputs necesarios)
# ======================================================
@dataclass
class UHIPipelineConfig:
    # obligatorios
    year: str
    root: str                  # carpeta raíz del proyecto (donde está 02_Calculos, 04_Temperatura_Aire, etc.)
    ref_align: str             # raster referencia para alinear (ej: porcentaje_urbano_3m_ALINEADO.tif)
    upl_shp: str               # SHP/GeoJSON de UPL para máscara+clip final en isobandas

    # opcionales
    overwrite: bool = False

    # SOM
    som_rows: int = 10
    som_cols: int = 10
    sample_per_month: int = 200_000

    # ICU score
    icu_min_pixels_per_cluster: int = 400
    icu_weights: Optional[Dict[str, float]] = None

    # Postproceso ICU agrupado
    post_min_class: int = 2
    post_kernel_size: int = 9
    post_close_iters: int = 2
    post_min_area_m2: float = 9000.0

    # Isobandas
    isob_min_class: int = 2
    isob_band_edges: Tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)
    isob_decay_m: float = 60.0
    isob_smooth_sigma_m: float = 18.0
    isob_pixel_size_m: float = 3.0
    isob_dissolve: bool = True
    isob_drop_class0: bool = False


# ======================================================
# PIPELINE
# ======================================================
class UHIFullYearPipeline:
    """
    Pipeline anual 2021 (aplicable a otros años) que ejecuta:
    1) Normalizar TAIR/HR
    2) Alinear normalizados
    3) Normalizar distancias (agua/vías) por archivo
    4) Alinear estáticos categóricos
    5) Resample+align índices 3m
    6) Construir NPZ SOM
    7) SOM anual + export modelo
    8) Raster clusters FULL
    9) Stats clusters CSV
    10) ICU_SCORE tif
    11) Postproceso ICU agrupado (AGRUPADO)
    12) Isobandas V11: mensuales + anual_by_month + anual_unified
    """

    def __init__(self, cfg: UHIPipelineConfig):
        self.cfg = cfg

        self.YEAR = str(cfg.year)
        self.OVERWRITE = bool(cfg.overwrite)

        self.ROOT = Path(cfg.root)
        self.REF_ALIGN = Path(cfg.ref_align)
        self.UPL_SHP = Path(cfg.upl_shp)

        self._assert_exists(self.ROOT, "ROOT")
        self._assert_exists(self.REF_ALIGN, "REF_ALIGN")
        self._assert_exists(self.UPL_SHP, "UPL_SHP")

        # -------------------------
        # Rutas
        # -------------------------
        self.UHI = self.ROOT / "06_UHI"

        self.NORM_DIR = self.UHI / "01_datos_normalizados"
        self.ALIGNED_DIR = self.UHI / "02_datos_normalizados_y_aliados"
        self.RESAMP_DIR = self.UHI / "03_alinear_resampliar"
        self.NPZ_DIR = self.UHI / "04_SOM_input_builder"

        self.SOM_YEAR_DIR = self.UHI / "06_salidas_SOM" / self.YEAR
        self.CLUSTERS_DIR = self.SOM_YEAR_DIR / "CLUSTERS_TIF"
        self.STATS_DIR = self.SOM_YEAR_DIR / "STATS"
        self.ICU_DIR = self.SOM_YEAR_DIR / "ICU_SCORE"
        self.ICU_AGR_DIR = self.SOM_YEAR_DIR / "ICU_SCORE_AGRUPADO"
        self.ICU_POLY_DIR = self.SOM_YEAR_DIR / "ICU_ISOBANDS_POLY"

        for d in [
            self.NORM_DIR,
            self.ALIGNED_DIR,
            self.RESAMP_DIR,
            self.NPZ_DIR,
            self.SOM_YEAR_DIR,
            self.CLUSTERS_DIR,
            self.STATS_DIR,
            self.ICU_DIR,
            self.ICU_AGR_DIR,
            self.ICU_POLY_DIR,
        ]:
            self._ensure_dir(d)

        # -------------------------
        # Fuentes
        # -------------------------
        self.TAIR_SYN_DIR = self.ROOT / "04_Temperatura_Aire" / self.YEAR / "02_resultados" / "03_resultado_temperatura_sintetica"
        self.HR_SYN_DIR = self.ROOT / "05_humedad_relativa" / self.YEAR / "00_resultados_humedad_relativa_sintetica"

        self.TAIR_RAS_DIR = self.ROOT / "04_Temperatura_Aire" / self.YEAR / "02_resultados" / "01_resultado_raster"
        self.TAIR_ALI_DIR = self.TAIR_RAS_DIR / "alineados"

        self._assert_exists(self.TAIR_SYN_DIR, "TAIR_SYN_DIR")
        self._assert_exists(self.HR_SYN_DIR, "HR_SYN_DIR")
        self._assert_exists(self.TAIR_RAS_DIR, "TAIR_RAS_DIR")
        self._assert_exists(self.TAIR_ALI_DIR, "TAIR_ALI_DIR")

        # Defaults ICU weights
        if self.cfg.icu_weights is None:
            self.cfg.icu_weights = {
                "TAIR": 3.0, "LST": 0.5, "HR": 2.0,
                "NDVI": 1.0, "MNDWI": 0.8, "CUERPOS_AGUA_BIN": 0.7, "VEGETACION_BIN": 0.7,
                "NDBI": 1.0, "PORC_URBANO": 1.0, "CONSTRUCCIONES_BIN": 0.9, "VIAS_BIN": 0.7,
                "DIST_AGUA_NORM": 0.9, "DIST_VIAS_NORM": 0.4,
                "ALBEDO": 0.2,
            }

    # ======================================================
    # Helpers
    # ======================================================
    @staticmethod
    def _ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _has_any(dir_path: Path, pattern: str) -> bool:
        return dir_path.exists() and any(dir_path.glob(pattern))

    @staticmethod
    def _assert_exists(p: Path, label: str):
        if not p.exists():
            raise FileNotFoundError(f"{label} no existe:\n{p}")

    def _run_if_needed(self, name: str, done: bool, fn: Callable[[], object]):
        if done and not self.OVERWRITE:
            print(f"⏭️  {name} ya existe → omitido")
            return None
        print(f"\n▶ {name}")
        return fn()

    # ======================================================
    # Steps
    # ======================================================
    def step_01_normalize_tair_hr(self):
        def step_norm(src_dir: Path):
            self._assert_exists(src_dir, f"Fuente {src_dir.name}")
            norm = RasterNormalizer01(input_dir=str(src_dir), output_dir=str(self.NORM_DIR), overwrite=self.OVERWRITE)
            outs = norm.process_folder()
            print(f"Normalizados {src_dir.name}:", len(outs))
            return outs

        self._run_if_needed(
            "1a) Normalizar TAIR",
            done=self._has_any(self.NORM_DIR, f"*TAIR*{self.YEAR}*_norm_3m*.tif") or self._has_any(self.NORM_DIR, f"*TAIRE*{self.YEAR}*_norm_3m*.tif"),
            fn=lambda: step_norm(self.TAIR_SYN_DIR),
        )
        self._run_if_needed(
            "1b) Normalizar HR",
            done=self._has_any(self.NORM_DIR, f"*HR*{self.YEAR}*_norm_3m*.tif"),
            fn=lambda: step_norm(self.HR_SYN_DIR),
        )

    def step_02_align_normalized(self):
        def step_align_norm():
            aligner = FolderGridAligner(
                input_dir=str(self.NORM_DIR),
                output_dir=str(self.ALIGNED_DIR),
                align_to_path=str(self.REF_ALIGN),
                overwrite=self.OVERWRITE,
                suffix="_3m",
            )
            outs = aligner.process_folder(kind_mode="auto", copy_if_already_aligned=True)
            print("Alineados:", len(outs))
            return outs

        self._run_if_needed(
            "2) Alinear normalizados (TAIR/HR)",
            done=self._has_any(self.ALIGNED_DIR, f"*{self.YEAR}*_3m.tif"),
            fn=step_align_norm,
        )

    def step_03_norm_distances(self):
        dist_jobs = [
            ("distancia_agua_3m_ALINEADO.tif", self.ALIGNED_DIR / "distancia_agua_3m_ALINEADO_NORMALIZADO.tif"),
            ("distancia_vias_3m_ALINEADO.tif", self.ALIGNED_DIR / "distancia_vias_3m_ALINEADO_NORMALIZADO.tif"),
        ]

        def step_norm_dist():
            outs = []
            for src_name, dst in dist_jobs:
                src = self.TAIR_ALI_DIR / src_name
                self._assert_exists(src, f"Distancia {src_name}")
                norm_file = RasterNormalizer01File(input_path=str(src), output_path=str(dst), overwrite=self.OVERWRITE)
                print(norm_file.process_file())
                outs.append(dst)
            return outs

        self._run_if_needed(
            "3) Normalizar distancias (agua/vías)",
            done=all(dst.exists() for _, dst in dist_jobs),
            fn=step_norm_dist,
        )

    def step_04_align_static_categorical(self):
        static_jobs = [
            ("construcciones.tif", "construcciones_ALINEADO.tif"),
            ("cuerpos_de_agua.tif", "cuerpos_de_agua_ALINEADO.tif"),
            ("vias.tif", "vias_ALINEADO.tif"),
            ("vegetacion.tif", "vegetacion_ALINEADO.tif"),
        ]

        def step_align_static():
            aligner = GridAlignerFile(align_to_path=str(self.REF_ALIGN), overwrite=self.OVERWRITE, suffix="_ALINEADO")
            outs = []
            for src_name, out_name in static_jobs:
                src = self.TAIR_RAS_DIR / src_name
                self._assert_exists(src, f"Static {src_name}")
                out = aligner.process_file(
                    input_path=str(src),
                    output_path=str(self.ALIGNED_DIR / out_name),
                    kind_mode="categorical",
                    copy_if_already_aligned=True,
                )
                print("Archivo alineado:", out)
                outs.append(self.ALIGNED_DIR / out_name)
            return outs

        self._run_if_needed(
            "4) Alinear estáticos (categorical)",
            done=all((self.ALIGNED_DIR / out_name).exists() for _, out_name in static_jobs),
            fn=step_align_static,
        )

    def step_05_resample_indices(self):
        def step_resample():
            in_dir = self.ROOT / "02_Calculos"
            self._assert_exists(in_dir, "02_Calculos")
            res = FolderTo3mResampler4326(
                input_dir=str(in_dir),
                output_dir=str(self.RESAMP_DIR),
                align_to_path=str(self.REF_ALIGN),
                mode="direct",
                overwrite=self.OVERWRITE,
            )
            salidas = res.process_folder(apply_clip=True, clip_range=(0.0, 1.0), kind_mode="auto")
            print("Generados:", len(salidas))
            return salidas

        self._run_if_needed(
            "5) Resample+align índices 3m",
            done=self._has_any(self.RESAMP_DIR, f"*_{self.YEAR}_*_NDVI_3m*.tif"),
            fn=step_resample,
        )

    def step_06_build_npz(self):
        def step_build():
            builder = SOMInputBuilder(
                input_dir=str(self.ROOT),
                month_regex=r"(20\d{2})[_-](0?[1-9]|1[0-2])(?=[_-]|$)",
                monthly_patterns={
                    "NDVI": r"_NDVI_3m",
                    "NDBI": r"_NDBI_3m",
                    "ALBEDO": r"_Albedo_3m",
                    "LST": r"_LST_3m",
                    "MNDWI": r"_MNDWI_3m|_MNWI_3m",
                    "TAIR": r"TAIRE_SINTETICA_3M_.*_norm_3m",
                    "HR": r"HR_SINTETICA_3M_.*_norm_3m",
                },
                static_patterns={
                    "CONSTRUCCIONES_BIN": r"construcciones_ALINEADO",
                    "CUERPOS_AGUA_BIN": r"cuerpos_de_agua_ALINEADO",
                    "VEGETACION_BIN": r"vegetacion_ALINEADO",
                    "VIAS_BIN": r"vias_ALINEADO",
                    "PORC_URBANO": r"porcentaje_urbano_3m_ALINEADO",
                    "DIST_AGUA_NORM": r"^distancia_agua_3m_ALINEADO_NORMALIZADO",
                    "DIST_VIAS_NORM": r"distancia_vias_3m_ALINEADO_NORMALIZADO",
                },
                mask_mode="all_valid",
                sample_max_pixels=200_000,
                allow_missing_monthly_vars=False,
                require_static=True,
            )

            print("DIAG:", builder.diagnose(self.YEAR))

            packs = builder.build()
            packs_year = {k: v for k, v in packs.items() if k.startswith(f"{self.YEAR}_")}

            ok = 0
            for month_key, pack in sorted(packs_year.items()):
                out_path = self.NPZ_DIR / f"SOM_INPUT_{month_key}.npz"
                if out_path.exists() and not self.OVERWRITE:
                    continue
                builder.save_npz(str(out_path), pack["X"], pack["coords"], pack["meta"])
                print(f"✅ {month_key} → X shape: {pack['X'].shape}")
                ok += 1

            print("🎉 NPZ generados/actualizados:", ok)
            return True

        self._run_if_needed(
            "6) Generar NPZ (SOM inputs)",
            done=self._has_any(self.NPZ_DIR, f"SOM_INPUT_{self.YEAR}_??.npz"),
            fn=step_build,
        )

    def step_07_som_annual_and_model(self):
        model_out = self.SOM_YEAR_DIR / f"SOM_MODEL_ANNUAL_{self.YEAR}_{self.cfg.som_rows}x{self.cfg.som_cols}.npz"

        def step_som():
            runner = SOMYearRunner(
                npz_dir=str(self.NPZ_DIR),
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
                standardize=True,
                sample_per_month=self.cfg.sample_per_month,
            )
            packs = runner.load_year(self.YEAR)
            print("Meses cargados:", [p.month_key for p in packs])

            runner.fit_annual(packs)
            results = runner.predict_year(packs)

            model = runner.export_model()
            np.savez_compressed(model_out, **model)
            print("✅ Modelo anual guardado:", model_out)

            for month_key, pack in results.items():
                print("✅", month_key, "N:", pack["cluster_ids"].shape[0], "clusters únicos:", len(np.unique(pack["cluster_ids"])))
            return True

        self._run_if_needed(
            "7) SOM anual + export modelo",
            done=model_out.exists(),
            fn=step_som,
        )

    def step_08_clusters_full_tif(self):
        def step_clusters():
            runner = SOMYearRunner(
                npz_dir=str(self.NPZ_DIR),
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
                standardize=False,
                sample_per_month=self.cfg.sample_per_month,
            )

            packs = runner.load_year(self.YEAR)
            runner.fit_annual(packs)
            results = runner.predict_year(packs)
            som_model = runner.export_model()

            writer = ClusterGeoTIFFWriter(nodata=-1, tile_size=256, mask_mode="all_valid")

            ok = 0
            for month_key, pack in results.items():
                out_tif = self.CLUSTERS_DIR / f"SOM_CLUSTER_{month_key}_{runner.som_rows}x{runner.som_cols}.tif"
                if out_tif.exists() and not self.OVERWRITE:
                    continue

                writer.save(out_path=str(out_tif), meta=pack["meta"], mode="full", som_model=som_model)
                print("✅", out_tif.name, "| FULL raster clusters")
                ok += 1

            print(f"🎉 Listo: {ok} GeoTIFF de clusters FULL en {self.CLUSTERS_DIR}")
            return True

        self._run_if_needed(
            "8) Rasterizar clusters FULL (CLUSTERS_TIF)",
            done=self._has_any(self.CLUSTERS_DIR, f"SOM_CLUSTER_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"),
            fn=step_clusters,
        )

    def step_09_cluster_stats(self):
        def step_stats():
            runner = SOMYearRunner(
                npz_dir=str(self.NPZ_DIR),
                som_rows=self.cfg.som_rows,
                som_cols=self.cfg.som_cols,
                standardize=False,
                sample_per_month=self.cfg.sample_per_month,
            )
            packs = runner.load_year(self.YEAR)
            runner.fit_annual(packs)
            results = runner.predict_year(packs)

            calc = ClusterStatsCalculator(min_pixels_per_cluster=50)

            for month_key, pack in results.items():
                out_csv = self.STATS_DIR / f"CLUSTER_STATS_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.csv"
                if out_csv.exists() and not self.OVERWRITE:
                    continue

                rows = calc.compute(
                    X=pack["X"],
                    cluster_ids=pack["cluster_ids"],
                    variables=pack["meta"]["variables"],
                    extra={"year": self.YEAR, "month": month_key},
                )
                calc.save_csv(rows, str(out_csv))
                print("✅", out_csv.name, "| clusters en CSV:", len(rows))
            return True

        self._run_if_needed(
            "9) Cluster stats CSV",
            done=self._has_any(self.STATS_DIR, f"CLUSTER_STATS_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.csv"),
            fn=step_stats,
        )

    def step_10_icu_score(self):
        def step_icu():
            labeler = ICUClusterLabeler(
                p_hot=92, p_cool=8,
                score_p1=80, score_p2=90, score_p3=97,
                min_pixels_per_cluster=self.cfg.icu_min_pixels_per_cluster,
                weights=dict(self.cfg.icu_weights or {}),
            )

            cluster_files = sorted(self.CLUSTERS_DIR.glob(f"SOM_CLUSTER_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"))
            if not cluster_files:
                raise FileNotFoundError(f"No se encontraron clusters en:\n{self.CLUSTERS_DIR}")

            ok, skipped = 0, 0
            for cpath in cluster_files:
                parts = cpath.stem.split("_")
                month_key = f"{parts[2]}_{parts[3]}"

                stats_csv = self.STATS_DIR / f"CLUSTER_STATS_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.csv"
                out_tif = self.ICU_DIR / f"ICU_SCORE_{month_key}_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"

                if not stats_csv.exists():
                    print(f"⚠️  SKIP {month_key} → falta CSV {stats_csv.name}")
                    skipped += 1
                    continue

                if out_tif.exists() and not self.OVERWRITE:
                    continue

                labeler.run(cluster_tif=str(cpath), stats_csv=str(stats_csv), out_tif=str(out_tif))
                print(f"✅ {month_key} → {out_tif.name}")
                ok += 1

            print("\n📊 RESUMEN ICU_SCORE")
            print("✅ Generados:", ok)
            print("⚠️  Omitidos:", skipped)
            print("📁 Salida:", self.ICU_DIR)
            return True

        self._run_if_needed(
            "10) ICU_SCORE (tif)",
            done=self._has_any(self.ICU_DIR, f"ICU_SCORE_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"),
            fn=step_icu,
        )

    def step_11_postprocess_icu_agrupado(self):
        def step_post():
            post = ICUPostProcessor(
                nodata=-1,
                min_class=self.cfg.post_min_class,
                kernel_size=self.cfg.post_kernel_size,
                close_iters=self.cfg.post_close_iters,
                min_area_m2=float(self.cfg.post_min_area_m2),
                connectivity=8,
                keep_low_class=False,
                fill_nodata_to_zero=True,
            )

            icu_files = sorted(self.ICU_DIR.glob(f"ICU_SCORE_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}.tif"))
            if not icu_files:
                raise FileNotFoundError(f"No se encontraron ICU_SCORE en: {self.ICU_DIR}")

            ok = 0
            for in_tif in icu_files:
                out_tif = self.ICU_AGR_DIR / f"{in_tif.stem}_AGRUPADO.tif"
                if out_tif.exists() and not self.OVERWRITE:
                    continue
                post.run(str(in_tif), str(out_tif))
                print(f"✅ {out_tif.name}")
                ok += 1

            print("\n🎉 Postproceso terminado")
            print("✅ Generados:", ok)
            print("📁 Resultados en:", self.ICU_AGR_DIR)
            return True

        self._run_if_needed(
            "11) ICU postproceso (AGRUPADO)",
            done=self._has_any(self.ICU_AGR_DIR, f"ICU_SCORE_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif"),
            fn=step_post,
        )

    def step_12_isobands_v11(self):
        def step_isobands():
            gen = ICUIsobandPolygonGenerator(
                nodata=-1,
                min_class=int(self.cfg.isob_min_class),
                pixel_size_m=float(self.cfg.isob_pixel_size_m),
                decay_m=float(self.cfg.isob_decay_m),
                smooth_sigma_m=float(self.cfg.isob_smooth_sigma_m),
                band_edges=list(self.cfg.isob_band_edges),
                upl_path=str(self.UPL_SHP),
                clip_polygons_to_upl=True,
                dissolve=bool(self.cfg.isob_dissolve),
                drop_class0=bool(self.cfg.isob_drop_class0),
            )

            # 12A mensuales
            outs = gen.run_folder(
                in_dir=str(self.ICU_AGR_DIR),
                out_dir=str(self.ICU_POLY_DIR),
                pattern=f"ICU_SCORE_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif",
            )
            print("✅ GeoJSON mensuales:", len(outs))

            # 12B anual MES A MES
            out_year_by_month = self.ICU_POLY_DIR / f"ICU_ISOBANDS_{self.YEAR}_ANUAL_BY_MONTH.geojson"
            gen.run_year_by_month(
                in_dir=str(self.ICU_AGR_DIR),
                year=self.YEAR,
                out_geojson_year=str(out_year_by_month),
                pattern=f"ICU_SCORE_{{year}}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif",
                dissolve_per_month=True,
            )
            print("✅ GeoJSON anual MES A MES:", out_year_by_month.name)

            # 12C anual UNIFICADO
            out_year_unified = self.ICU_POLY_DIR / f"ICU_ISOBANDS_{self.YEAR}_ANUAL_UNIFIED.geojson"
            gen.run_year_unified(
                in_dir=str(self.ICU_AGR_DIR),
                year=self.YEAR,
                out_geojson_year=str(out_year_unified),
                pattern=f"ICU_SCORE_{{year}}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO.tif",
            )
            print("✅ GeoJSON anual UNIFICADO:", out_year_unified.name)
            return True

        done = (
            self._has_any(self.ICU_POLY_DIR, f"ICU_SCORE_{self.YEAR}_??_{self.cfg.som_rows}x{self.cfg.som_cols}_AGRUPADO_ISOBANDS_DISSOLVE.geojson")
            and (self.ICU_POLY_DIR / f"ICU_ISOBANDS_{self.YEAR}_ANUAL_BY_MONTH.geojson").exists()
            and (self.ICU_POLY_DIR / f"ICU_ISOBANDS_{self.YEAR}_ANUAL_UNIFIED.geojson").exists()
        )

        self._run_if_needed(
            "12) Isobandas V11 (mensuales + anual mes a mes + anual unificado)",
            done=done,
            fn=step_isobands,
        )

    # ======================================================
    # Runner
    # ======================================================
    def run_all(self):
        self.step_01_normalize_tair_hr()
        self.step_02_align_normalized()
        self.step_03_norm_distances()
        self.step_04_align_static_categorical()
        self.step_05_resample_indices()
        self.step_06_build_npz()
        self.step_07_som_annual_and_model()
        self.step_08_clusters_full_tif()
        self.step_09_cluster_stats()
        self.step_10_icu_score()
        self.step_11_postprocess_icu_agrupado()
        self.step_12_isobands_v11()
        print("\n✅ Proceso COMPLETO OK.")
