from sqlalchemy import create_engine, text
from app.core.config import settings

def test_connection():
    try:
        db_url = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión exitosa a la base de datos.")
    except Exception as e:
        print("❌ ERROR de conexión a la BD:", e)

if __name__ == "__main__":
    test_connection()









