from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.db.connection import get_db
from app.modules.islas import service

router = APIRouter(prefix="/islas", tags=["islas"])

@router.get("/", summary="Obtener islas de calor en formato GeoJSON")
def obtener_islas(
    year: str = Query(None, description="Año (YYYY), ej: 2022"),
    month: str = Query(None, description="Mes (MM), ej: 01"),
    db: Session = Depends(get_db),
):
    # ✅ ahora sí filtra cuando llegan params
    return service.get_islas_geojson(db, year=year, month=month)
