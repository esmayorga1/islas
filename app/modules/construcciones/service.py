# service.py
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.modules.construcciones.schemas import construccionesGeoJSON

def get_construcciones_geojson(db: Session):
    """
    Devuelve todas las construcciones en formato GeoJSON (FeatureCollection).
    """
    query = text("""
        SELECT jsonb_build_object(
            'type', 'FeatureCollection',
            'features', jsonb_agg(
                jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geom)::jsonb,
                    'properties', to_jsonb(c) - 'geom'
                )
            )
        ) AS geojson
        FROM construcciones c;
    """)

    result = db.execute(query).scalar()
    return result
