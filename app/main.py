from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.modules.upl.router import router as upl_router
from app.modules.islas.router import router as islas_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Islas de Calor",
        version="0.1.0",
        description="API para identificación y caracterización de islas de calor (UHI).",
    )

    # Configuración CORS
    origins = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(upl_router)
    app.include_router(islas_router)

    # === Ruta de prueba básica ===
    @app.get("/", tags=["Health"])
    def read_root():
        return {
            "message": "Backend está operativo",
            "environment": settings.APP_ENV,
            "database": settings.DB_NAME,
        }

    return app


app = create_app()