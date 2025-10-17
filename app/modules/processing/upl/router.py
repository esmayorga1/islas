from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db.connection import get_db
from app.modules.processing.upl import service

router = APIRouter(prefix="/upl", tags=["UPL"])

@router.get("/", summary="Obtener UPL en formato GeoJSON")
def obtener_upl(db: Session = Depends(get_db)):
    return service.get_upl_geojson(db)
