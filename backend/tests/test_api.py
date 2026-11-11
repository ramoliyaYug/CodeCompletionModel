from fastapi.testclient import TestClient
from backend.app.services.syntax_validator import syntax_validator


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data
    assert "completion" in data["endpoints"]


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelLoaded"] is True
    assert "device" in data


def test_root_health_alias(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelLoaded"] is True


def test_model_info_endpoint(client: TestClient):
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "starcoder2-3b"
    assert data["language"] == "python"
    assert data["version"] == "v1"
    assert data["status"] == "ready"


def test_root_model_alias(client: TestClient):
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "starcoder2-3b"


def test_completion_endpoint_success(client: TestClient):
    payload = {
        "code": "def add(a, b):\n    return",
        "max_new_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.95,
        "include_syntax_check": True,
    }
    response = client.post("/api/v1/completion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "completion" in data
    assert data["model"] == "starcoder2-3b"
    assert data["modelVersion"] == "v1"
    assert isinstance(data["latencyMs"], int)
    assert data["syntaxValid"] is True


def test_completion_validation_error_empty_code(client: TestClient):
    payload = {
        "code": "",
        "max_new_tokens": 32,
    }
    response = client.post("/api/v1/completion", json=payload)
    assert response.status_code == 422


def test_completion_validation_error_invalid_tokens(client: TestClient):
    payload = {
        "code": "print('hello')",
        "max_new_tokens": -5,  # Must be >= 1
    }
    response = client.post("/api/v1/completion", json=payload)
    assert response.status_code == 422


def test_top_k_completion_endpoint(client: TestClient):
    payload = {
        "code": "def add(a, b):\n    return",
        "k": 3,
        "max_new_tokens": 32,
    }
    response = client.post("/api/v1/completion/top-k", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)
    assert len(data["suggestions"]) > 0
    assert data["model"] == "starcoder2-3b"
    assert isinstance(data["latencyMs"], int)


def test_streaming_completion_endpoint(client: TestClient):
    payload = {
        "code": "def multiply(x, y):\n    return",
        "max_new_tokens": 16,
    }
    response = client.post("/api/v1/completion/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "data:" in content
    assert "[DONE]" in content


def test_syntax_validator_logic():
    # Valid Python full code
    is_valid, err = syntax_validator.validate_python_code(
        prompt="def add(a, b):\n    return",
        completion=" a + b",
    )
    assert is_valid is True
    assert err is None

    # Valid standalone expression
    is_valid, err = syntax_validator.validate_python_code(
        prompt="x = ",
        completion="1 + 2",
    )
    assert is_valid is True

    # Invalid syntax
    is_valid, err = syntax_validator.validate_python_code(
        prompt="def bad_syntax(",
        completion=":::(",
    )
    assert is_valid is False
    assert err is not None
