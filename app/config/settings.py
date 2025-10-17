# app/core/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Configuración general de la aplicación y la base de datos."""

    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_USER: str = Field(default="postgres")
    DB_PASSWORD: str = Field(default="admin")
    DB_NAME: str = Field(default="islasdecalor")
    DB_SCHEMA: str = Field(default="public")
    APP_ENV: str = Field(default="development")

    @property
    def database_url(self) -> str:
        """Construye la URL completa de conexión a PostgreSQL."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"

settings = Settings()
