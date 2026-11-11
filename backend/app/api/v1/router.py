from fastapi import APIRouter
from backend.app.api.v1.endpoints import completion, model, health

api_router = APIRouter()

api_router.include_router(completion.router, tags=["Code Completion"])
api_router.include_router(model.router, tags=["Model Management"])
api_router.include_router(health.router, tags=["Health & Diagnostics"])
