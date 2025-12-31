from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import geopandas as gpd
from sqlalchemy import create_engine, text

# ✅ OJO: en tu proyecto settings viene de app.core.config (no app.config.settings)
from app.core.config import settings


IfExists = Literal["replace", "append", "fail"]


@dataclass
class SubirGeoJSONIslasCalor:
    """
    Sube un GeoJSON a PostGIS y, si lo pides, REEMPLAZA registros existentes por año o por (año, mes).

    - replace_year=True  => DELETE WHERE year = :y  (y luego INSERT)
    - replace_month=True => DELETE WHERE year = :y AND month = :m (y luego INSERT)

    Nota:
    - NO hace upsert (porque class_id NO es único).
    - Asume que tu tabla ya existe con columnas:
      class_id, strength, range_min, range_max, label, year, month, source, geom
    """

    tabla_destino: str = "islas_de_calor"
    schema: Optional[str] = None
    if_exists: IfExists = "append"
    srid: int = 4326

    columnas: Sequence[str] = (
        "class_id",
        "strength",
        "range_min",
        "range_max",
        "label",
        "year",
        "month",
        "source",
        "geom",
    )

    def _engine(self):
        # ✅ en tu app suele ser settings.DATABASE_URL o settings.database_url.
        # Dejamos fallback robusto:
        db_url = getattr(settings, "database_url", None) or getattr(settings, "DATABASE_URL", None)
        if not db_url:
            raise AttributeError(
                "No encuentro database_url en settings. Define settings.database_url o settings.DATABASE_URL."
            )
        return create_engine(db_url)

    def _infer_single_value(self, series) -> Optional[str]:
        vals = sorted({str(v) for v in series.dropna().unique()})
        return vals[0] if len(vals) == 1 else None

    def _safe_schema(self, schema: str) -> str:
        # Evita schema raro; no “escapa” nombres, pero reduce riesgos básicos
        s = (schema or "public").strip().replace('"', "")
        return s or "public"

    def _delete_year(self, engine, schema: str, year: str):
        schema = self._safe_schema(schema)
        with engine.begin() as conn:
            conn.execute(
                text(f'DELETE FROM "{schema}"."{self.tabla_destino}" WHERE year = :y'),
                {"y": str(year)},
            )

    def _delete_month(self, engine, schema: str, year: str, month: str):
        schema = self._safe_schema(schema)
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    DELETE FROM "{schema}"."{self.tabla_destino}"
                    WHERE year = :y AND month = :m
                """),
                {"y": str(year), "m": str(month).zfill(2)},
            )

    def subir_geojson(
        self,
        ruta_geojson: str | Path,
        replace_year: bool = False,
        replace_month: bool = False,
        year: Optional[str] = None,
        month: Optional[str] = None,
    ) -> int:
        ruta_geojson = Path(ruta_geojson)
        if not ruta_geojson.exists():
            raise FileNotFoundError(f"No existe: {ruta_geojson}")

        if replace_year and replace_month:
            raise ValueError("Usa SOLO uno: replace_year=True o replace_month=True (no ambos).")

        engine = self._engine()
        schema = self.schema or getattr(settings, "DB_SCHEMA", "public")
        schema = self._safe_schema(schema)

        print(f"📂 Cargando GeoJSON: {ruta_geojson}")
        gdf = gpd.read_file(ruta_geojson)

        if gdf.empty:
            raise ValueError(f"El GeoJSON está vacío: {ruta_geojson}")

        # ✅ geometry -> geom (PostGIS)
        if gdf.geometry.name != "geom":
            gdf = gdf.rename_geometry("geom")

        # ✅ CRS
        if gdf.crs is None:
            print("⚠️ GeoJSON sin CRS. Se asigna EPSG:4326.")
            gdf = gdf.set_crs(epsg=self.srid)
        else:
            epsg = gdf.crs.to_epsg()
            if epsg is None or int(epsg) != self.srid:
                print(f"🔄 Reproyectando {gdf.crs.to_string()} -> EPSG:{self.srid}")
                gdf = gdf.to_crs(epsg=self.srid)

        # ✅ Normalizar month a 2 dígitos si existe
        if "month" in gdf.columns:
            gdf["month"] = gdf["month"].astype(str).str.zfill(2)

        # ✅ Validar columnas
        faltantes = [c for c in self.columnas if c not in gdf.columns]
        if faltantes:
            raise ValueError(
                f"❌ Faltan columnas en el GeoJSON: {faltantes}\n"
                f"Columnas encontradas: {list(gdf.columns)}"
            )

        # ✅ Inferir year/month si no los pasas (solo si son únicos)
        if year is None:
            year = self._infer_single_value(gdf["year"])
        if month is None and "month" in gdf.columns:
            month = self._infer_single_value(gdf["month"])
            if month is not None:
                month = str(month).zfill(2)

        # ✅ Reemplazo (DELETE antes del INSERT)
        if replace_year:
            if year is None:
                raise ValueError(
                    "replace_year=True requiere un year único (pásalo o que el GeoJSON tenga un solo year)."
                )
            print(f'🧹 Reemplazando año {year}: DELETE en "{schema}"."{self.tabla_destino}"')
            self._delete_year(engine, schema, year)

        if replace_month:
            if year is None or month is None:
                raise ValueError("replace_month=True requiere year y month únicos.")
            print(f'🧹 Reemplazando {year}-{month}: DELETE en "{schema}"."{self.tabla_destino}"')
            self._delete_month(engine, schema, year, month)

        # ✅ Subir (append recomendado)
        gdf = gdf[list(self.columnas)]

        # si tu tabla usa MultiPolygon y llega Polygon, geopandas normalmente lo maneja.
        # Si falla por tipos, te lo ajusto (convertir a MULTIPOLYGON).

        print(f'🚀 Insertando en PostGIS: {getattr(settings, "DB_NAME", "")} ("{schema}"."{self.tabla_destino}")')
        gdf.to_postgis(
            name=self.tabla_destino,
            con=engine,
            schema=schema,
            if_exists=self.if_exists,  # "append"
            index=False,
        )

        print(f'✅ Insertadas {len(gdf)} filas en "{schema}"."{self.tabla_destino}"')
        return len(gdf)
