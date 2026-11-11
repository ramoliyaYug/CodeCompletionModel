from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    model: str = Field(default="starcoder2-3b", description="Base model name.")
    adapter: str = Field(default="code-completion-lora-v1", description="Adapter name.")
    version: str = Field(default="v1", description="Service/model version.")
    language: str = Field(default="python", description="Target programming language.")
    device: Optional[str] = Field(default=None, description="Compute device (e.g. cuda:0 or cpu).")
    quantization: Optional[str] = Field(default=None, description="Quantization mode (e.g. 4-bit NF4).")
    status: str = Field(..., description="Current status: ready, loading, unloaded, or error.")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Extra model details.")


class ModelActionResponse(BaseModel):
    status: str = Field(..., description="Action outcome status.")
    message: str = Field(..., description="Informational message.")
    modelInfo: Optional[ModelInfoResponse] = Field(default=None, description="Updated model information.")
