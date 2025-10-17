from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db.connection import get_db
from app.modules.processing.islas import service

router = APIRouter(prefix="/islas", tags=["islas"])

@router.get("/", summary="Obtener islas de calor en formato GeoJSON")
def obtener_upl(db: Session = Depends(get_db)):
    return service.get_islas_geojson(db)
