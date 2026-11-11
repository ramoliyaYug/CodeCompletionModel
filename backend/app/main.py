import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.router import api_router as api_v1_router
from backend.app.config import settings
from backend.app.services.model_service import model_service
from backend.app.utils.logger import get_logger

logger = get_logger("app_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling startup initialization and shutdown cleanup."""
    logger.info("Initializing FastAPI Code Completion service...")
    logger.info(f"Using base model: {settings.BASE_MODEL_NAME_OR_PATH}")
    logger.info(f"Using adapter dir: {settings.ADAPTER_DIR}")

    # Trigger model initialization on startup
    await model_service.initialize()

    yield

    logger.info("Shutting down FastAPI service...")
    model_service.unload_model()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=(
        "Production-ready FastAPI backend for AI Code Completion using fine-tuned "
        "StarCoder2-3B with LoRA/QLoRA adapters."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Process Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# 3. Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failure on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error processing request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error during inference or request processing",
            "error": str(exc),
        },
    )


# 4. Include Versioned API Router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


# 5. Root & Convenience Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "model": settings.MODEL_NAME,
        "documentation": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
        "endpoints": {
            "completion": f"{settings.API_V1_PREFIX}/completion",
            "top_k": f"{settings.API_V1_PREFIX}/completion/top-k",
            "stream": f"{settings.API_V1_PREFIX}/completion/stream",
            "model_info": f"{settings.API_V1_PREFIX}/model",
            "health": f"{settings.API_V1_PREFIX}/health",
        },
    }


# Convenience aliases for root-level /health and /model as referenced in PRD
@app.get("/health", tags=["Root"], include_in_schema=False)
async def root_health():
    from backend.app.api.v1.endpoints.health import check_health
    return await check_health()


@app.get("/model", tags=["Root"], include_in_schema=False)
async def root_model():
    from backend.app.api.v1.endpoints.model import get_model_info
    return await get_model_info()
