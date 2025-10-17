"""# main.py
from fastapi import FastAPI
from core.config.settings import settings
from core.db.connection import Base, engine
from routers import observaciones, productos, variables, entidades, fuentes, frontend

# === Inicialización de la aplicación FastAPI ===
app = FastAPI(
    title="Islas de Calor",
    description="API del backend para el análisis y visualización de islas de calor urbanas.",
    version="1.0.0",
    contact={
        "name": "",
        "email": "",
    },
)

# === Creación de las tablas (si no existen) ===
Base.metadata.create_all(bind=engine)

# === Registro de routers (módulos funcionales) ===
app.include_router(observaciones.router, prefix="/api/observaciones", tags=["Observaciones"])
app.include_router(productos.router, prefix="/api/productos", tags=["Productos Analíticos"])
app.include_router(variables.router, prefix="/api/variables", tags=["Variables"])
app.include_router(entidades.router, prefix="/api/entidades", tags=["Entidades Espaciales"])
app.include_router(fuentes.router, prefix="/api/fuentes", tags=["Fuentes de Datos"])
app.include_router(frontend.router, prefix="/api/frontend", tags=["Frontend / Visualización"])

# === Ruta de prueba básica ===
@app.get("/")
def read_root():
    return {
        "message": "GeoIA Backend está operativo 🚀",
        "environment": settings.app_env,
        "database": settings.db_name,
    }"""



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from app.modules.processing.upl.router import router as upl_router
from app.modules.processing.construcciones.router import router as construciones_router
from app.modules.processing.islas.router import router as islas_router
from app.config.settings import settings


app = FastAPI(title="Islas de Calor")

# 🌐 Configuración CORS
origins = [
    "http://localhost:4200",  # Frontend Angular en desarrollo
    "http://127.0.0.1:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Permite solicitudes desde Angular
    allow_credentials=True,
    allow_methods=["*"],            # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],            # Permite todos los encabezados
)

# 🔗 Registrar los módulos (routers)
app.include_router(upl_router)
app.include_router(construciones_router)
app.include_router(islas_router)




# === Ruta de prueba básica ===
@app.get("/")
def read_root():
    return {
        "message": "Backend está operativo 🚀",
        "environment": settings.APP_ENV,
        "database": settings.DB_NAME,
    }