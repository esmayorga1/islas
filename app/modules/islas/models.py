# models.py
from sqlalchemy import Column, Integer, String
from geoalchemy2 import Geometry
from app.db.connection import Base

class Manzana(Base):
    __tablename__ = "islas_de_calor"  # nombre de la tabla en PostgreSQL

    id = Column(Integer, primary_key=True, index=True)
    codigo = Column(String, unique=True, nullable=False)
    nombre = Column(String, nullable=True)
    geom = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326))

    def __repr__(self):
        return f"<upl {self.codigo}>"