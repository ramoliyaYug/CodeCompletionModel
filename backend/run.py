import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run StarCoder2-3B Code Completion FastAPI Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"),
        help="Enable auto-reload on code change",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=os.getenv("MOCK_INFERENCE", "false").lower() in ("true", "1", "yes"),
        help="Run in mock inference mode (no GPU/model weights required)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1 for GPU models)",
    )

    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_INFERENCE"] = "true"

    import uvicorn

    print("=" * 70)
    print("🚀 Starting StarCoder2-3B Code Completion FastAPI Server")
    print(f"📍 Host: {args.host}:{args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"🧪 Mock Inference: {os.getenv('MOCK_INFERENCE', 'false')}")
    print(f"📖 Docs: http://{args.host}:{args.port}/docs")
    print("=" * 70)

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
