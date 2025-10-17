# GeoIA Ontological Backend

Arquitectura modular basada en FastAPI + PostgreSQL/PostGIS.

## Estructura principal
- app/config → Configuración de entorno y base de datos
- app/ontology → Clases y reglas ontológicas
- app/modules → Módulos funcionales (ingestión, análisis, visualización...)
- app/services → Servicios reutilizables
- app/tests → Módulo de testeo

Ejecutar con:
```bash
uvicorn app.main:app --reload
```
