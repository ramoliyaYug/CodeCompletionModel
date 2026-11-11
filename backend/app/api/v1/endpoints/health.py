import time
from fastapi import APIRouter
from backend.app.schemas.health import HealthResponse
from backend.app.services.model_service import model_service, ModelStatus

router = APIRouter()
START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the operational status of the service, model availability, and GPU memory usage.",
)
async def check_health() -> HealthResponse:
    model_loaded = model_service.is_ready()
    status_str = "ok" if model_loaded or model_service.status == ModelStatus.LOADING else (
        "error" if model_service.status == ModelStatus.ERROR else "unloaded"
    )

    return HealthResponse(
        status=status_str,
        modelLoaded=model_loaded,
        device=model_service.device,
        gpuMemory=model_service.get_gpu_memory(),
        uptimeSeconds=round(time.time() - START_TIME, 2),
    )
