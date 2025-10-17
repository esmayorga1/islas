# service.py
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.modules.processing.upl.schemas import uplGeoJSON

def get_upl_geojson(db: Session):
    """
    Devuelve todas las upl en formato GeoJSON (FeatureCollection).
    """
    query = text("""
        SELECT jsonb_build_object(
            'type', 'FeatureCollection',
            'features', jsonb_agg(
                jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geom)::jsonb,
                    'properties', to_jsonb(u) - 'geom'
                )
            )
        ) AS geojson
        FROM upl u;
    """)

    result = db.execute(query).scalar()
    return result
