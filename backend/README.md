# StarCoder2-3B Code Completion FastAPI Backend

This directory contains the production-ready FastAPI backend for serving the fine-tuned **StarCoder2-3B** code-completion model with the QLoRA adapter saved in `saved_models/starcoder2-3b-qlora-code-completion`.

---

## 🏗️ Architecture & Features

- **FastAPI Framework**: High-performance asynchronous REST API with automatic OpenAPI / Swagger documentation (`/docs`).
- **QLoRA / PEFT Integration**: Efficiently loads `bigcode/starcoder2-3b` with 4-bit NF4 BitsAndBytes quantization and attaches the fine-tuned LoRA adapter (`r=32`, `alpha=64`).
- **Low-Latency Inference**: Uses KV-caching (`use_cache=True`), left-truncation context management, and FP16/BF16 compute dtypes.
- **Top-K & Greedy Generation**: Supports both deterministic single completions and diverse candidate generation.
- **Server-Sent Events (SSE) Streaming**: Token-by-token streaming via `POST /api/v1/completion/stream` for code editors.
- **AST Syntax Validation**: Validates generated code using Python's static AST parser without ever executing untrusted code (`eval`/`exec` are strictly prohibited).
- **GPU & VRAM Monitoring**: Tracks allocated VRAM, reserved VRAM, and device availability via the `/health` endpoint.
- **Dev/Mock Mode**: Built-in mock mode (`MOCK_INFERENCE=true`) for fast unit tests and development on machines without GPUs.

---

## 📁 Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, CORS, middleware, routers
│   ├── config.py                # Environment configuration & default parameters
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py        # Central API v1 router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── completion.py # /completion, /completion/top-k, /completion/stream
│   │           ├── model.py      # /model, /model/load, /model/unload
│   │           └── health.py     # /health
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── completion.py        # Request & Response Pydantic schemas
│   │   ├── model.py             # Model metadata schemas
│   │   └── health.py            # Health status and GPU memory schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py     # Model loading, QLoRA adapter inference & streaming
│   │   └── syntax_validator.py  # Python AST syntax verification
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Structured logging and safe code sanitization
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Test fixtures and mock model setup
│   └── test_api.py              # Unit & integration test suite
├── .env.example                 # Sample environment variables
├── requirements.txt             # Python dependencies
├── run.py                       # CLI application launcher
└── README.md                    # Documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

Ensure your Python environment (e.g. `~/myenv`) has the required packages installed:

```bash
pip install -r backend/requirements.txt
```

### 2. Configure Environment (Optional)

Copy `.env.example` to `.env` if you want to customize settings:

```bash
cp backend/.env.example backend/.env
```

Key environment variables:
- `BASE_MODEL_NAME_OR_PATH`: Base Hugging Face model (`bigcode/starcoder2-3b`)
- `ADAPTER_DIR`: Path to LoRA adapter (default: `../saved_models/starcoder2-3b-qlora-code-completion`)
- `USE_4BIT_QUANTIZATION`: `true` for 4-bit NF4 BitsAndBytes quantization (recommended for 8GB GPUs)
- `MOCK_INFERENCE`: `true` to run without loading heavy model weights

### 3. Start the Server

#### Method A: Using the CLI Launcher

```bash
python backend/run.py --host 0.0.0.0 --port 8000 --reload
```

#### Method B: Using Uvicorn Directly

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Method C: Fast Mock / Testing Mode (No GPU needed)

```bash
python backend/run.py --mock --port 8000
```

Once running, interactive API documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Reference & Examples

### 1. Single Code Completion

**Endpoint:** `POST /api/v1/completion`

Predicts the next portion of code based on a prompt context.

```bash
curl -X POST "http://localhost:8000/api/v1/completion" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calculate_factorial(n):\n    \"\"\"Calculates factorial\"\"\"\n",
    "max_new_tokens": 64,
    "temperature": 0.2,
    "top_p": 0.95,
    "include_syntax_check": true
  }'
```

**Response (200 OK):**
```json
{
  "completion": "    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    elif n == 0:\n        return 1\n    else:\n        return n * calculate_factorial(n - 1)",
  "model": "starcoder2-3b",
  "modelVersion": "v1",
  "latencyMs": 218,
  "syntaxValid": true,
  "syntaxError": null
}
```

---

### 2. Top-K Candidate Completions

**Endpoint:** `POST /api/v1/completion/top-k`

Generates multiple diverse suggestions for a given code prefix.

```bash
curl -X POST "http://localhost:8000/api/v1/completion/top-k" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return",
    "k": 3,
    "max_new_tokens": 32,
    "temperature": 0.7
  }'
```

**Response (200 OK):**
```json
{
  "suggestions": [
    " a + b",
    " b + a",
    " sum([a, b])"
  ],
  "model": "starcoder2-3b",
  "modelVersion": "v1",
  "latencyMs": 312,
  "k": 3
}
```

---

### 3. Real-Time Token Streaming (SSE)

**Endpoint:** `POST /api/v1/completion/stream`

Streams tokens in real time using Server-Sent Events (SSE).

```bash
curl -N -X POST "http://localhost:8000/api/v1/completion/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def is_prime(n):\n",
    "max_new_tokens": 64
  }'
```

**Response Stream:**
```text
data: {"token": "    "}
data: {"token": "if"}
data: {"token": " n"}
data: {"token": " <="}
data: {"token": " 1"}
data: {"token": ":"}
...
data: [DONE]
```

---

### 4. Health & GPU Diagnostics

**Endpoint:** `GET /api/v1/health` (or `GET /health`)

Checks service status, model availability, and VRAM usage.

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "modelLoaded": true,
  "device": "cuda:0",
  "gpuMemory": {
    "allocatedGB": 3.63,
    "reservedGB": 4.12,
    "totalGB": 7.62,
    "gpuName": "NVIDIA GeForce RTX 4060 Laptop GPU"
  },
  "uptimeSeconds": 142.5
}
```

---

### 5. Model Information & Control

**Endpoint:** `GET /api/v1/model` (or `GET /model`)

Returns metadata about the active model, adapter, and device configuration.

```bash
curl -X GET "http://localhost:8000/api/v1/model"
```

**Response (200 OK):**
```json
{
  "model": "starcoder2-3b",
  "adapter": "starcoder2-3b-qlora-code-completion",
  "version": "v1",
  "language": "python",
  "device": "cuda:0",
  "quantization": "4-bit NF4",
  "status": "ready",
  "details": {
    "baseModel": "bigcode/starcoder2-3b",
    "adapterDir": "/home/mastermind/Desktop/extras/CodeCompletionModel/saved_models/starcoder2-3b-qlora-code-completion",
    "statusMessage": "Fine-tuned StarCoder2-3B model is loaded and ready.",
    "maxContextLength": 1024
  }
}
```

- `POST /api/v1/model/unload`: Free GPU VRAM on demand.
- `POST /api/v1/model/load`: Reload the model and LoRA adapter into VRAM.

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest backend/tests -v
```

All tests execute in mock mode without requiring GPU hardware or model downloads.
