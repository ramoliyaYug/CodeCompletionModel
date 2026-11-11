import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force mock mode for fast and isolated test execution
os.environ["MOCK_INFERENCE"] = "true"
os.environ["AUTO_LOAD_MODEL"] = "true"

from backend.app.main import app
from backend.app.services.model_service import model_service


@pytest.fixture(scope="session", autouse=True)
def setup_mock_model():
    """Initializes model service in mock mode for test suite."""
    model_service._load_model_sync()
    yield
    model_service.unload_model()


@pytest.fixture
def client():
    """Provides a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client
