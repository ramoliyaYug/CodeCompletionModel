import json
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.config import settings
from backend.app.schemas.completion import (
    CompletionRequest,
    CompletionResponse,
    TopKRequest,
    TopKResponse,
)
from backend.app.services.model_service import model_service
from backend.app.services.syntax_validator import syntax_validator
from backend.app.utils.logger import get_logger

logger = get_logger("completion_endpoint")
router = APIRouter()


@router.post(
    "/completion",
    response_model=CompletionResponse,
    summary="Generate Code Completion",
    description="Predicts the next portion of Python code based on the provided code prefix.",
)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not ready for inference. Status: {model_service.status.value}. {model_service.status_message}",
        )

    try:
        completion_text, latency_ms = model_service.generate_completion(
            code=request.code,
            max_new_tokens=request.max_new_tokens or settings.DEFAULT_MAX_NEW_TOKENS,
            temperature=request.temperature if request.temperature is not None else settings.DEFAULT_TEMPERATURE,
            top_p=request.top_p if request.top_p is not None else settings.DEFAULT_TOP_P,
            stop_tokens=request.stop_tokens,
            fim_suffix=request.fim_suffix,
        )

        syntax_valid = None
        syntax_error = None
        if request.include_syntax_check:
            syntax_valid, syntax_error = syntax_validator.validate_python_code(
                prompt=request.code,
                completion=completion_text,
            )

        return CompletionResponse(
            completion=completion_text,
            model=settings.MODEL_NAME,
            modelVersion=settings.MODEL_VERSION,
            latencyMs=latency_ms,
            syntaxValid=syntax_valid,
            syntaxError=syntax_error,
        )

    except Exception as e:
        logger.exception(f"Error during code completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/completion/top-k",
    response_model=TopKResponse,
    summary="Generate Top-K Code Completions",
    description="Generates multiple candidate completions for a given Python code context.",
)
async def create_top_k_completion(request: TopKRequest) -> TopKResponse:
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not ready for inference. Status: {model_service.status.value}",
        )

    try:
        suggestions, latency_ms = model_service.generate_top_k(
            code=request.code,
            k=request.k,
            max_new_tokens=request.max_new_tokens or settings.DEFAULT_MAX_NEW_TOKENS,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or settings.DEFAULT_TOP_P,
            stop_tokens=request.stop_tokens,
        )

        return TopKResponse(
            suggestions=suggestions,
            model=settings.MODEL_NAME,
            modelVersion=settings.MODEL_VERSION,
            latencyMs=latency_ms,
            k=len(suggestions),
        )

    except Exception as e:
        logger.exception(f"Error during top-k completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/completion/stream",
    summary="Stream Code Completion",
    description="Streams generated completion tokens in real-time using Server-Sent Events (SSE).",
)
async def create_streaming_completion(request: CompletionRequest):
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not ready for inference. Status: {model_service.status.value}",
        )

    async def event_generator():
        try:
            async for token in model_service.generate_stream(
                code=request.code,
                max_new_tokens=request.max_new_tokens or settings.DEFAULT_MAX_NEW_TOKENS,
                temperature=request.temperature if request.temperature is not None else settings.DEFAULT_TEMPERATURE,
                top_p=request.top_p if request.top_p is not None else settings.DEFAULT_TOP_P,
                stop_tokens=request.stop_tokens,
            ):
                chunk = json.dumps({"token": token})
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            err = json.dumps({"error": str(e)})
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
