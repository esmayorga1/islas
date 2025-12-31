# app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Configuración global de la aplicación.
    Todos los valores se cargan desde el archivo .env
    """

    APP_ENV: str = Field(..., description="Entorno de ejecución")

    DB_HOST: str = Field(..., description="Host de la base de datos")
    DB_PORT: int = Field(..., description="Puerto de la base de datos")
    DB_USER: str = Field(..., description="Usuario de la base de datos")
    DB_PASSWORD: str = Field(..., description="Contraseña de la base de datos")
    DB_NAME: str = Field(..., description="Nombre de la base de datos")
    DB_SCHEMA: str = Field(..., description="Esquema de la base de datos")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
