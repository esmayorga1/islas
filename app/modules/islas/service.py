# # service.py
# import json
# from sqlalchemy.orm import Session
# from sqlalchemy import text


# # =========================
# # GET: GeoJSON completo
# # =========================
# def get_islas_geojson(db: Session):
#     """
#     Devuelve TODAS las islas (isobandas) en formato GeoJSON (FeatureCollection),
#     desde public.islas_de_calor.
#     """
#     query = text("""
#         SELECT jsonb_build_object(
#             'type', 'FeatureCollection',
#             'features', COALESCE(jsonb_agg(
#                 jsonb_build_object(
#                     'type', 'Feature',
#                     'geometry', ST_AsGeoJSON(i.geom)::jsonb,
#                     'properties', jsonb_build_object(
#                         'gid', i.gid,
#                         'class_id', i.class_id,
#                         'strength', i.strength,
#                         'range_min', i.range_min,
#                         'range_max', i.range_max,
#                         'label', i.label,
#                         'year', i.year,
#                         'month', i.month,
#                         'source', i.source
#                     )
#                 )
#             ), '[]'::jsonb)
#         ) AS geojson
#         FROM public.islas_de_calor i;
#     """)
#     return db.execute(query).scalar()


# # =========================
# # DELETE: por año/mes (para recargar sin duplicar)
# # =========================
# def delete_islas_by_month(db: Session, year: str, month: str):
#     query = text("""
#         DELETE FROM public.islas_de_calor
#         WHERE year = :year AND month = :month;
#     """)
#     db.execute(query, {"year": str(year), "month": str(month).zfill(2)})
#     db.commit()


# # =========================
# # INSERT: una feature (sin upsert)
# # =========================
# def insert_isla_feature(
#     db: Session,
#     year: str,
#     month: str,
#     class_id: int,
#     strength: int,
#     range_min: float,
#     range_max: float,
#     label: str,
#     source: str,
#     geom_geojson: dict,
# ):
#     query = text("""
#         INSERT INTO public.islas_de_calor (
#             class_id, strength, range_min, range_max, label,
#             year, month, source, geom
#         )
#         VALUES (
#             :class_id, :strength, :range_min, :range_max, :label,
#             :year, :month, :source,
#             ST_SetSRID(ST_GeomFromGeoJSON(:geom_geojson), 4326)
#         );
#     """)
#     db.execute(query, {
#         "class_id": int(class_id),
#         "strength": int(strength) if strength is not None else int(class_id),
#         "range_min": float(range_min) if range_min is not None else None,
#         "range_max": float(range_max) if range_max is not None else None,
#         "label": label,
#         "year": str(year),
#         "month": str(month).zfill(2),
#         "source": source,
#         "geom_geojson": json.dumps(geom_geojson),
#     })


# # =========================
# # IMPORT: FeatureCollection (mensual o merge)
# # =========================
# def insert_islas_from_featurecollection(
#     db: Session,
#     featurecollection: dict,
#     delete_before_insert: bool = False,
# ):
#     """
#     Inserta todas las features de un FeatureCollection.
#     - Mantiene atributos: year, month, class_id, strength, range_min, range_max, label, source
#     - Si delete_before_insert=True:
#         borra primero por (year, month) detectados en cada feature y luego inserta.
#       (útil para recargar un mes sin duplicar)
#     """
#     feats = featurecollection.get("features", []) or []
#     if not feats:
#         return {"inserted": 0}

#     # (opcional) borrar por año/mes antes de insertar para evitar duplicados
#     if delete_before_insert:
#         seen = set()
#         for ft in feats:
#             props = ft.get("properties", {}) or {}
#             y = props.get("year")
#             m = props.get("month")
#             if y is None or m is None:
#                 continue
#             key = (str(y), str(m).zfill(2))
#             if key not in seen:
#                 delete_islas_by_month(db, key[0], key[1])
#                 seen.add(key)

#     inserted = 0
#     for ft in feats:
#         props = ft.get("properties", {}) or {}
#         geom = ft.get("geometry")

#         if not geom:
#             continue

#         year = props.get("year")
#         month = props.get("month")
#         class_id = props.get("class_id")

#         if year is None or month is None or class_id is None:
#             continue  # o lanza error si quieres estricto

#         insert_isla_feature(
#             db=db,
#             year=str(year),
#             month=str(month).zfill(2),
#             class_id=int(class_id),
#             strength=int(props.get("strength", class_id)),
#             range_min=props.get("range_min"),
#             range_max=props.get("range_max"),
#             label=props.get("label"),
#             source=props.get("source", ""),
#             geom_geojson=geom,
#         )
#         inserted += 1

#     db.commit()
#     return {"inserted": inserted}
# service.py

import json
from sqlalchemy.orm import Session
from sqlalchemy import text


# =========================
# GET: GeoJSON (FILTRADO)
# =========================
def get_islas_geojson(db: Session, year: str | None = None, month: str | None = None):
    """
    Devuelve islas (isobandas) en formato GeoJSON (FeatureCollection),
    desde public.islas_de_calor.

    ✅ Si year y month vienen -> filtra SOLO ese mes/año.
    ✅ Si no vienen -> devuelve todo (compatibilidad).
    """
    where = []
    params = {}

    if year is not None:
        where.append("i.year = :year")
        params["year"] = str(year)

    if month is not None:
        where.append("i.month = :month")
        params["month"] = str(month).zfill(2)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    query = text(f"""
        SELECT jsonb_build_object(
            'type', 'FeatureCollection',
            'features', COALESCE(jsonb_agg(
                jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(i.geom)::jsonb,
                    'properties', jsonb_build_object(
                        'gid', i.gid,
                        'class_id', i.class_id,
                        'strength', i.strength,
                        'range_min', i.range_min,
                        'range_max', i.range_max,
                        'label', i.label,
                        'year', i.year,
                        'month', i.month,
                        'source', i.source
                    )
                )
            ), '[]'::jsonb)
        ) AS geojson
        FROM public.islas_de_calor i
        {where_sql};
    """)

    return db.execute(query, params).scalar()


# =========================
# DELETE: por año/mes (para recargar sin duplicar)
# =========================
def delete_islas_by_month(db: Session, year: str, month: str):
    query = text("""
        DELETE FROM public.islas_de_calor
        WHERE year = :year AND month = :month;
    """)
    db.execute(query, {"year": str(year), "month": str(month).zfill(2)})
    db.commit()


# =========================
# INSERT: una feature (sin upsert)
# =========================
def insert_isla_feature(
    db: Session,
    year: str,
    month: str,
    class_id: int,
    strength: int,
    range_min: float,
    range_max: float,
    label: str,
    source: str,
    geom_geojson: dict,
):
    query = text("""
        INSERT INTO public.islas_de_calor (
            class_id, strength, range_min, range_max, label,
            year, month, source, geom
        )
        VALUES (
            :class_id, :strength, :range_min, :range_max, :label,
            :year, :month, :source,
            ST_SetSRID(ST_GeomFromGeoJSON(:geom_geojson), 4326)
        );
    """)
    db.execute(query, {
        "class_id": int(class_id),
        "strength": int(strength) if strength is not None else int(class_id),
        "range_min": float(range_min) if range_min is not None else None,
        "range_max": float(range_max) if range_max is not None else None,
        "label": label,
        "year": str(year),
        "month": str(month).zfill(2),
        "source": source,
        "geom_geojson": json.dumps(geom_geojson),
    })


# =========================
# IMPORT: FeatureCollection (mensual o merge)
# =========================
def insert_islas_from_featurecollection(
    db: Session,
    featurecollection: dict,
    delete_before_insert: bool = False,
):
    feats = featurecollection.get("features", []) or []
    if not feats:
        return {"inserted": 0}

    if delete_before_insert:
        seen = set()
        for ft in feats:
            props = ft.get("properties", {}) or {}
            y = props.get("year")
            m = props.get("month")
            if y is None or m is None:
                continue
            key = (str(y), str(m).zfill(2))
            if key not in seen:
                delete_islas_by_month(db, key[0], key[1])
                seen.add(key)

    inserted = 0
    for ft in feats:
        props = ft.get("properties", {}) or {}
        geom = ft.get("geometry")
        if not geom:
            continue

        year = props.get("year")
        month = props.get("month")
        class_id = props.get("class_id")
        if year is None or month is None or class_id is None:
            continue

        insert_isla_feature(
            db=db,
            year=str(year),
            month=str(month).zfill(2),
            class_id=int(class_id),
            strength=int(props.get("strength", class_id)),
            range_min=props.get("range_min"),
            range_max=props.get("range_max"),
            label=props.get("label"),
            source=props.get("source", ""),
            geom_geojson=geom,
        )
        inserted += 1

    db.commit()
    return {"inserted": inserted}
