# main.py
"""
Servidor FastAPI para Mizu-Intelligence.
Expone endpoints REST para consultas RAG con soporte de streaming.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain import brain  # Instancia única del cerebro RAG

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Modelos Pydantic (Validación)
# ========================

class QueryRequest(BaseModel):
    """Esquema de solicitud para consultas."""
    question: str = Field(..., min_length=1, max_length=2000, description="Pregunta del usuario")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "¿Qué servicios ofrece Mizu Legends?"
            }
        }
    }


class SourceInfo(BaseModel):
    """Información de fuente recuperada."""
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """Esquema de respuesta para consultas síncronas."""
    answer: str
    sources: list[SourceInfo] = []
    context_used: bool
    error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Mizu Legends ofrece coaching de MOBA y torneos eSports...",
                "sources": [{"content": "Pregunta: Servicios...", "source": "business_questions.json", "score": 0.85}],
                "context_used": True,
                "error": None
            }
        }
    }


class HealthResponse(BaseModel):
    """Esquema de health check."""
    status: str
    vector_store_loaded: bool
    version: str = "1.0.0"


# ========================
# Inicialización de FastAPI
# ========================

app = FastAPI(
    title="Mizu-Intelligence API",
    description="API de orquestación RAG con soporte de voz y streaming",
    version="1.0.0",
)

# Configurar CORS (desde variables de entorno o desarrollo por defecto)
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Endpoints
# ========================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica que el servicio esté funcionando y el vector store cargado."""
    try:
        # Verificar que el vector store esté accesible (brain ya lo inicializa)
        # Podríamos hacer una operación ligera para confirmar, pero por ahora confiamos en que la inicialización fue exitosa
        return HealthResponse(
            status="healthy",
            vector_store_loaded=True,
        )
    except Exception as e:
        logger.exception("Health check falló")
        return HealthResponse(
            status="unhealthy",
            vector_store_loaded=False,
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Endpoint síncrono para realizar una consulta RAG.
    Retorna la respuesta completa junto con metadatos de las fuentes utilizadas.
    """
    logger.info(f"Nueva consulta: {request.question[:100]}...")
    result = brain.generate_response(request.question)

    if result.get("error"):
        # Si hubo error interno, retornar 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {result['error']}"
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceInfo(**src) for src in result["sources"]],
        context_used=result["context_used"],
        error=result.get("error")
    )


@app.post("/query/stream", tags=["Query"])
async def query_stream_endpoint(request: QueryRequest):
    """
    Endpoint streaming que retorna la respuesta token por token.
    Ideal para interfaces de voz que requieren baja latencia.
    """
    logger.info(f"Nuevo streaming query: {request.question[:100]}...")

    async def generate():
        try:
            # brain.stream_response es un generador síncrono; lo envolvemos para async
            for token in brain.stream_response(request.question):
                yield token
        except Exception as e:
            logger.exception("Error en streaming")
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Desactivar buffering para proxies
        }
    )


# ========================
# Manejo de excepciones globales
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejo personalizado de excepciones HTTP para logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return await app.default_http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Manejo de errores no controlados."""
    logger.exception("Error no controlado")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor. Contacte al administrador."}
    )


# ========================
# Startup / Shutdown
# ========================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio del servidor."""
    logger.info("Iniciando servidor Mizu-Intelligence...")
    # Verificar que brain esté correctamente inicializado
    if not hasattr(brain, "vectorstore") or not brain.vectorstore:
        logger.error("Brain no inicializado correctamente. Verificar base de datos vectorial.")
    else:
        logger.info(f"Vector store lista. Colección: {brain.vectorstore._collection.name}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre del servidor."""
    logger.info("Apagando servidor Mizu-Intelligence...")
    # Limpieza si es necesaria (ej. cerrar conexiones)
