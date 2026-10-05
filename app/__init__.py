"""Lab 19 app package.

Loads `<repo>/.env` on import so every entry point (FastAPI, scripts,
notebooks, Feast CLI via `app.*` imports) sees the same QDRANT_MODE /
EMBEDDING_BACKEND / Feast store settings. Real environment variables win
over `.env` (override=False), so CI can still inject its own values.
"""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # python-dotenv is optional; fall back to the process env
    pass
