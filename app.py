#!/usr/bin/env python3
"""VaultPay Guard web entrypoint — what Render runs.

    python3 app.py

Serves the web UI (New Payment / AG Grid audit ledger / Policy) on
0.0.0.0:$PORT (default 8000). Standard library only — there is no
build step and nothing to pip-install. Configuration comes from the
environment; see .env.example and DEPLOY.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.web import serve  # noqa: E402

if __name__ == "__main__":
    serve()
