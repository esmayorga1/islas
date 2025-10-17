# service.py
import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.modules.processing.islas.schemas import islasGeoJSON

def get_islas_geojson(db: Session):
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
                    'properties', to_jsonb(i) - 'geom'
                )
            )
        ) AS geojson
        FROM islasdecalorp i;
    """)

    result = db.execute(query).scalar()
    return result

#

def upsert_islas_calor(db: Session, ano: int, mes: int, geom_wkt: str, propiedades: dict):
    """
    Inserta o actualiza el registro de islas de calor en la tabla islasdecalorp.
    Si ya existe (ano, mes), se actualiza; si no, se inserta.
    """

    propiedades_json = json.dumps(propiedades)

    query = text("""
        INSERT INTO public.islasdecalorp (ano, mes, geom, propiedades, fecha_actualizacion)
        VALUES (:ano, :mes, ST_GeomFromText(:geom_wkt, 4326), :propiedades, NOW())
        ON CONFLICT (ano, mes)
        DO UPDATE
        SET geom = EXCLUDED.geom,
            propiedades = EXCLUDED.propiedades,
            fecha_actualizacion = NOW();
    """)

    db.execute(query, {
        "ano": ano,
        "mes": mes,
        "geom_wkt": geom_wkt,
        "propiedades": propiedades_json
    })

    db.commit()
    print(f"✅ Isla de calor {ano}-{mes} creada o actualizada correctamente.")

def upsert_islas_calor2(db: Session, anio: int, mes: int, geom_wkt: str, propiedades: dict):
    """
    Inserta o actualiza un registro de islas de calor (año + mes).
    Si ya existe => actualiza geom y propiedades.
    Si no => inserta nuevo registro.
    """

    # Convertir propiedades a JSON string
    propiedades_json = json.dumps(propiedades)

    
    query = text("""
        INSERT INTO public.islasdecalorp (ano, mes, geom, propiedades, fecha_actualizacion)
        VALUES (:anio, :mes, ST_GeomFromText(:geom_wkt, 4326), :propiedades, NOW())
        ON CONFLICT (anio, mes)
        DO UPDATE
        SET geom = EXCLUDED.geom,
            propiedades = EXCLUDED.propiedades,
            fecha_actualizacion = NOW();
    """)

    # Ejecutar con binding limpio
    db.execute(query, {
        "anio": anio,
        "mes": mes,
        "geom_wkt": geom_wkt,
        "propiedades": propiedades_json
    })

    db.commit()
    print(f"✅ Isla de calor {anio}-{mes} creada o actualizada correctamente.")