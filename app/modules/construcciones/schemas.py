# schemas.py
from pydantic import BaseModel
from typing import Any, Dict

class construccionesGeoJSON(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]
