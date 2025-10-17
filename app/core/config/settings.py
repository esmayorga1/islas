from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    DB_HOST: str = Field("localhost", env="DB_HOST")
    DB_PORT: int = Field(5434, env="DB_PORT")
    DB_USER: str = Field("postgres", env="DB_USER")
    DB_PASSWORD: str = Field("admin", env="DB_PASSWORD")
    DB_NAME: str = Field("islasdecalor", env="DB_NAME")
    DB_SCHEMA: str = Field("public", env="DB_SCHEMA")
    APP_ENV: str = Field("development", env="APP_ENV")

    # ✅ Propiedad calculada para generar la URL de conexión
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"

# Instancia global de configuración
settings = Settings()