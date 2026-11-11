from fastapi import APIRouter, BackgroundTasks, status
from backend.app.schemas.model import ModelInfoResponse, ModelActionResponse
from backend.app.services.model_service import model_service, ModelStatus

router = APIRouter()


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Returns metadata about the active model, LoRA adapter, device, and quantization state.",
)
async def get_model_info() -> ModelInfoResponse:
    info = model_service.get_model_info()
    return ModelInfoResponse(**info)


@router.post(
    "/model/load",
    response_model=ModelActionResponse,
    summary="Load Model",
    description="Asynchronously triggers loading the base model and LoRA adapter weights.",
)
async def load_model(background_tasks: BackgroundTasks) -> ModelActionResponse:
    if model_service.status == ModelStatus.READY:
        return ModelActionResponse(
            status="already_loaded",
            message="Model is already loaded and ready for inference.",
            modelInfo=ModelInfoResponse(**model_service.get_model_info()),
        )

    if model_service.status == ModelStatus.LOADING:
        return ModelActionResponse(
            status="loading",
            message="Model loading is already in progress.",
            modelInfo=ModelInfoResponse(**model_service.get_model_info()),
        )

    # Initiate load in background
    background_tasks.add_task(model_service.load_model)

    return ModelActionResponse(
        status="initiating",
        message="Model loading initiated in background.",
        modelInfo=ModelInfoResponse(**model_service.get_model_info()),
    )


@router.post(
    "/model/unload",
    response_model=ModelActionResponse,
    summary="Unload Model",
    description="Unloads model and clears GPU memory cache.",
)
async def unload_model() -> ModelActionResponse:
    model_service.unload_model()
    return ModelActionResponse(
        status="unloaded",
        message="Model unloaded and GPU cache cleared.",
        modelInfo=ModelInfoResponse(**model_service.get_model_info()),
    )
