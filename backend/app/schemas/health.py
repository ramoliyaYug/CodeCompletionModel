from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Overall service status (ok, degraded, error).")
    modelLoaded: bool = Field(..., description="Whether the fine-tuned model is loaded and ready.")
    device: Optional[str] = Field(default=None, description="Current execution device.")
    gpuMemory: Optional[Dict[str, Any]] = Field(default=None, description="GPU memory statistics if CUDA is available.")
    uptimeSeconds: Optional[float] = Field(default=None, description="API server uptime in seconds.")
