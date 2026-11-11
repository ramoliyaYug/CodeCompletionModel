from typing import List, Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    code: str = Field(
        ...,
        description="The source code prefix/prompt to complete.",
        min_length=1,
        examples=["def calculate_factorial(n):\n    \"\"\"Calculates factorial\"\"\"\n"],
    )
    max_new_tokens: Optional[int] = Field(
        default=32,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic; 0 disables sampling.",
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (top_p).",
    )
    stop_tokens: Optional[List[str]] = Field(
        default=None,
        description="Optional list of string sequences where generation should stop.",
    )
    include_syntax_check: Optional[bool] = Field(
        default=True,
        description="Whether to run AST syntax validation on the completed code.",
    )
    fim_suffix: Optional[str] = Field(
        default=None,
        description="Optional code suffix for Fill-In-The-Middle (FIM) generation.",
    )


class CompletionResponse(BaseModel):
    completion: str = Field(..., description="The predicted code completion.")
    model: str = Field(default="starcoder2-3b", description="Model identifier.")
    modelVersion: str = Field(default="v1", description="Model and adapter version.")
    latencyMs: int = Field(..., description="Generation latency in milliseconds.")
    syntaxValid: Optional[bool] = Field(
        default=None,
        description="Whether the code (prompt + completion) passes Python AST syntax checking.",
    )
    syntaxError: Optional[str] = Field(
        default=None,
        description="Syntax error message if syntaxValid is False.",
    )


class TopKRequest(BaseModel):
    code: str = Field(
        ...,
        description="The source code prefix/prompt to complete.",
        min_length=1,
        examples=["def add(a, b):\n    return"],
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidate completions to generate.",
    )
    max_new_tokens: Optional[int] = Field(
        default=32,
        ge=1,
        le=512,
        description="Maximum number of new tokens per candidate.",
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature for diversity across candidates.",
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    stop_tokens: Optional[List[str]] = Field(
        default=None,
        description="Optional list of stopping token sequences.",
    )


class TopKResponse(BaseModel):
    suggestions: List[str] = Field(
        ...,
        description="List of top-k unique predicted code completions.",
    )
    model: str = Field(default="starcoder2-3b", description="Model identifier.")
    modelVersion: str = Field(default="v1", description="Model and adapter version.")
    latencyMs: int = Field(..., description="Total generation latency in milliseconds.")
    k: int = Field(..., description="Number of suggestions returned.")
