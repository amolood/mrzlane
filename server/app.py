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

app = FastAPI(title="mrzlane", version="0.1.0")

_TOKEN = os.environ.get("MRZLANE_TOKEN", "")


def _check_auth(token: str | None):
    if _TOKEN and token != _TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token.")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "mrzlane", "version": "0.1.0"}


@app.post("/api/verify_document")
async def verify_document(image: UploadFile = File(...),
                          x_api_token: str | None = Header(default=None)):
    """Legacy verify_document-style endpoint (drop-in replacement)."""
    _check_auth(x_api_token)
    data = await image.read()
    try:
        result = pipeline.process(data, with_portrait=True, compat=True)
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "status": "error",
            "data": {"errorCode": 1, "message": f"OCR failed: {e}"},
        })
    return result


@app.post("/api/v1/passport")
async def passport_v1(image: UploadFile = File(...),
                      x_api_token: str | None = Header(default=None)):
    """Clean v1 response with per-field confidence and review flags."""
    _check_auth(x_api_token)
    data = await image.read()
    try:
        return pipeline.process(data, with_portrait=True, compat=False)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed: {e}")
