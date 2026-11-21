#!/usr/bin/env python3
"""Temporary FastAPI inference endpoint for a remotely hosted ASR checkpoint.

Set ASR_API_KEY before starting. The endpoint is intended for short-lived
experiments behind a temporary tunnel, not production deployment.
"""

from __future__ import annotations

import hmac
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from .inference import load_asr_model, transcribe_audio_array

ARTIFACT_DIR = Path(os.getenv("ASR_ARTIFACT_DIR", "artifacts/asr"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 180
MODEL_BUNDLE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE
    MODEL_BUNDLE = load_asr_model(
        ARTIFACT_DIR,
        device=os.getenv("ASR_DEVICE") or None,
    )
    yield
    MODEL_BUNDLE = None


app = FastAPI(
    title="Car Service ASR",
    version="0.1.0",
    description="Temporary inference API for the project's own trained ASR model.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    if MODEL_BUNDLE is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model_epoch": MODEL_BUNDLE["epoch"],
        "device": str(MODEL_BUNDLE["device"]),
    }


@app.post("/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    expected_key = os.getenv("ASR_API_KEY", "")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="ASR_API_KEY is not configured on the inference server.",
        )
    if not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded.")

    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds the 25 MB upload limit.")

    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(content), dtype="float32", always_2d=True
        )
    except Exception as error:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio. Upload WAV, FLAC or another soundfile-readable format.",
        ) from error

    duration = waveform.shape[0] / float(sample_rate)
    if duration <= 0 or duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"Audio must be between 0 and {MAX_AUDIO_SECONDS} seconds.",
        )
    try:
        result = transcribe_audio_array(waveform, sample_rate, MODEL_BUNDLE)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {error}") from error
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "training.asr.server:app",
        host=os.getenv("ASR_HOST", "0.0.0.0"),
        port=int(os.getenv("ASR_PORT", "8000")),
        reload=False,
    )
