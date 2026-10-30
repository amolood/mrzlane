"""FastAPI service exposing the engine over HTTP.

Endpoints:
  POST /api/verify_document   multipart 'image'  -> legacy verify_document JSON
                              (drop-in for common commercial ID-OCR services)
  POST /api/v1/passport       multipart 'image'  -> clean v1 JSON
  GET  /health

Auth: optional X-API-Token header checked against the MRZLANE_TOKEN env var.
Engines are loaded lazily on the first request (and then cached).

Run: PYTHONPATH=src uvicorn server.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse                            # noqa: E402

from mrzlane import pipeline                                        # noqa: E402

import hmac
import logging

log = logging.getLogger("mrzlane")

app = FastAPI(title="mrzlane", version="0.1.0")

_TOKEN = os.environ.get("MRZLANE_TOKEN", "")
# Opt-in escape hatch for local development only. Without a token AND without
# this flag, the service fails closed (refuses every request) rather than
# serving unauthenticated — safe by default when deployed.
_ALLOW_NO_AUTH = os.environ.get("MRZLANE_ALLOW_NO_AUTH", "").lower() in ("1", "true", "yes")
# Reject oversized uploads before decoding (resource-exhaustion guard).
_MAX_UPLOAD_BYTES = int(os.environ.get("MRZLANE_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


def _check_auth(token: str | None):
    if not _TOKEN:
        if _ALLOW_NO_AUTH:
            return
        # Fail closed: no token configured and dev mode not enabled.
        raise HTTPException(status_code=503,
                            detail="Service not configured for authentication.")
    if token is None or not hmac.compare_digest(token, _TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token.")


async def _read_capped(image: UploadFile) -> bytes:
    data = await image.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "engine": "mrzlane", "version": "0.1.0"}


@app.post("/api/verify_document")
async def verify_document(image: UploadFile = File(...),
                          x_api_token: str | None = Header(default=None)):
    """Legacy verify_document-style endpoint (drop-in replacement)."""
    _check_auth(x_api_token)
    data = await _read_capped(image)
    try:
        return pipeline.process(data, with_portrait=True, compat=True)
    except Exception:
        log.exception("verify_document failed")
        return JSONResponse(status_code=200, content={
            "status": "error",
            "data": {"errorCode": 1, "message": "Document could not be processed."},
        })


@app.post("/api/v1/passport")
async def passport_v1(image: UploadFile = File(...),
                      x_api_token: str | None = Header(default=None)):
    """Clean v1 response with per-field confidence and review flags."""
    _check_auth(x_api_token)
    data = await _read_capped(image)
    try:
        return pipeline.process(data, with_portrait=True, compat=False)
    except Exception:
        log.exception("passport_v1 failed")
        raise HTTPException(status_code=422, detail="Document could not be processed.")
