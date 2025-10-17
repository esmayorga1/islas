from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db.connection import get_db
from app.modules.processing.construcciones import service

router = APIRouter(prefix="/construcciones", tags=["construcciones"])

@router.get("/", summary="Obtener construcciones en formato GeoJSON")
def obtener_upl(db: Session = Depends(get_db)):
    return service.get_construcciones_geojson(db)
