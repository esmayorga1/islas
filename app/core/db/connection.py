# core/db/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config.settings import settings

# === Base de datos: configuración del motor y sesión ===

# Motor de conexión a PostgreSQL
engine = create_engine(
    settings.database_url,
    echo=(settings.APP_ENV == "development"),  # Muestra logs solo en desarrollo
    pool_pre_ping=True
)

# Sesión local para interactuar con la base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clase base para los modelos ORM
Base = declarative_base()

# === Dependencia para FastAPI (inyección de sesión) ===
def get_db():
    """
    Proporciona una sesión de base de datos para cada solicitud.
    Cierra la sesión automáticamente al finalizar.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()