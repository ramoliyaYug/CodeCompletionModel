import os
from pathlib import Path
from typing import List

# Determine base paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "saved_models" / "starcoder2-3b-qlora-code-completion"

# Try loading .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


class Settings:
    """Application and Model settings loaded from environment variables with sensible defaults."""
    
    # Project info
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "StarCoder2-3B Code Completion API")
    PROJECT_VERSION: str = os.getenv("PROJECT_VERSION", "1.0.0")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "starcoder2-3b")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1")
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    
    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    
    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    
    # Model & Tokenizer Paths
    BASE_MODEL_NAME_OR_PATH: str = os.getenv(
        "BASE_MODEL_NAME_OR_PATH", "bigcode/starcoder2-3b"
    )
    ADAPTER_DIR: str = os.getenv("ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR))
    
    # Model Loading Options
    AUTO_LOAD_MODEL: bool = os.getenv("AUTO_LOAD_MODEL", "true").lower() in ("true", "1", "yes")
    USE_4BIT_QUANTIZATION: bool = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() in ("true", "1", "yes")
    DEVICE: str = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto")
    
    # Inference generation defaults
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "1024"))
    DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "32"))
    MAX_NEW_TOKENS_LIMIT: int = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "512"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.95"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))
    REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY", "1.05"))
    
    # Simulation / Mock mode for unit tests and local testing without GPU/model weights
    MOCK_INFERENCE: bool = os.getenv("MOCK_INFERENCE", "false").lower() in ("true", "1", "yes")


settings = Settings()
