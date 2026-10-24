"""ScamShield v2 — FastAPI service for multimodal predictions with uncertainty.

Run::

    SCAMSHIELD_CHECKPOINT=artifacts/student.pt uvicorn app:app --port 8000

Environment
    SCAMSHIELD_CHECKPOINT   path to a teacher.pt / student.pt from train_distill (default artifacts/student.pt)
    SCAMSHIELD_DEVICE       cpu | cuda (default: cuda if available)
    SCAMSHIELD_WHISPER      optional Whisper model name (e.g. "base") to transcribe audio when no text is sent
    SCAMSHIELD_MC_SAMPLES   default number of MC-dropout passes (default 30)

Endpoints
    GET  /health              liveness + whether a model is loaded
    GET  /v1/model            checkpoint kind, taxonomy, OOD threshold, parameter count
    GET  /telemetry           scan counters, latency history, live scan feed (for the dashboard)
    POST /v1/predict          JSON {text?, url?, acoustic?, mc_samples?, explain?}
    POST /v1/predict/batch    JSON {texts: [...], mc_samples?}  (up to 500 messages)
    POST /v1/predict/audio    multipart: file (.wav/.mp3/.flac/.ogg) + optional text, url form fields

The service refuses to predict (503) when no trained checkpoint is found,
rather than serving an untrained network.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from src.config import ACOUSTIC_DIM, TAXONOMY
from src.models.multitask_net import count_parameters

log = logging.getLogger("scamshield.api")
logging.basicConfig(level=logging.INFO)

CHECKPOINT = os.getenv("SCAMSHIELD_CHECKPOINT", "artifacts/student.pt")
DEVICE = os.getenv("SCAMSHIELD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
WHISPER = os.getenv("SCAMSHIELD_WHISPER") or None
DEFAULT_MC = int(os.getenv("SCAMSHIELD_MC_SAMPLES", "30"))
MAX_AUDIO_BYTES = 25 * 1024 * 1024
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}

STATE: dict = {"pipeline": None, "error": None}

try:  # optional: CPU load for the dashboard
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class Telemetry:
    """In-memory counters for the dashboard. Only real scans served by this process are recorded."""

    def __init__(self, maxlen: int = 40):
        self.started = time.time()
        self.lock = threading.Lock()
        self.latencies: deque[float] = deque(maxlen=maxlen)
        self.feed: deque[dict] = deque(maxlen=maxlen)
        self.total = 0

    def record(self, source: str, result: dict, latency_ms: float) -> None:
        risk = result["risk_score"]
        severity = "CRITICAL" if risk >= 0.7 else "ELEVATED" if risk >= 0.35 else "LOW"
        detail = f"risk {risk:.0%}" + (" · needs review (OOD)" if result.get("needs_review") else "")
        with self.lock:
            self.total += 1
            self.latencies.append(latency_ms)
            self.feed.appendleft({
                "ts": datetime.now().strftime("%H:%M:%S"), "severity": severity,
                "vector": f"{source.upper()} · {result['category']}", "detail": detail, "source": "LIVE",
            })

    def snapshot(self) -> dict:
        with self.lock:
            lat = list(self.latencies)
            return {
                "scans_total": self.total,
                "latency_ms_avg": sum(lat) / len(lat) if lat else None,
                "latency_history": lat,
                "feed": list(self.feed),
                "uptime_sec": int(time.time() - self.started),
            }


TELEMETRY = Telemetry()


def _timed(source: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    latency = (time.perf_counter() - start) * 1000
    for r in result if isinstance(result, list) else [result]:
        r["latency_ms"] = round(latency / (len(result) if isinstance(result, list) else 1), 1)
        TELEMETRY.record(source, r, r["latency_ms"])
    return result


def load_pipeline(path: str = CHECKPOINT, text_encoder=None):
    from src.inference.pipeline import ScamShieldPipeline

    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint not found at {path}; train one with src/training/train_distill.py")
    return ScamShieldPipeline.from_checkpoint(path, device=DEVICE, text_encoder=text_encoder, whisper_model=WHISPER)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if STATE["pipeline"] is None:
        try:
            STATE["pipeline"] = load_pipeline()
            log.info("loaded %s checkpoint from %s", STATE["pipeline"].checkpoint_kind, CHECKPOINT)
        except Exception as exc:  # keep serving /health with the reason
            STATE["error"] = str(exc)
            log.error("model not loaded: %s", exc)
    yield


app = FastAPI(title="ScamShield v2", version="2.0.0", lifespan=lifespan)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    mc_samples: int = Field(DEFAULT_MC, ge=1, le=200)
    explain: bool = False


class PredictRequest(BaseModel):
    text: str | None = Field(None, max_length=20_000)
    url: str | None = Field(None, max_length=2_048)
    acoustic: list[float] | None = Field(None, description=f"precomputed {ACOUSTIC_DIM}-d Librosa feature vector")
    mc_samples: int = Field(DEFAULT_MC, ge=1, le=200)
    explain: bool = False
    top_k: int = Field(8, ge=1, le=50)

    @field_validator("acoustic")
    @classmethod
    def _check_acoustic(cls, v):
        if v is not None and len(v) != ACOUSTIC_DIM:
            raise ValueError(f"acoustic must have {ACOUSTIC_DIM} values")
        return v


def _pipeline():
    if STATE["pipeline"] is None:
        raise HTTPException(503, f"model not loaded: {STATE['error']}")
    return STATE["pipeline"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["pipeline"] is not None, "error": STATE["error"], "device": DEVICE}


@app.get("/v1/model")
def model_info():
    p = _pipeline()
    return {
        "checkpoint": CHECKPOINT,
        "kind": getattr(p, "checkpoint_kind", None),
        "parameters": count_parameters(p.model),
        "taxonomy": list(TAXONOMY),
        "ood_threshold": p.ood_threshold,
        "text_encoder": getattr(p.text_encoder, "model_name", type(p.text_encoder).__name__),
        "config": p.model.cfg.to_dict(),
    }


@app.get("/telemetry")
def telemetry():
    p = STATE["pipeline"]
    gpu = torch.cuda.is_available()
    return {
        **TELEMETRY.snapshot(),
        "model_status": "ready" if p is not None else "error",
        "model_error": STATE["error"],
        "model_kind": getattr(p, "checkpoint_kind", None),
        "parameters": count_parameters(p.model) if p is not None else None,
        "ood_threshold": p.ood_threshold if p is not None else None,
        "cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
        "gpu": {"available": gpu, "name": torch.cuda.get_device_name(0) if gpu else None, "load_percent": None},
        "whisper": {"status": "ready" if (p is not None and p.transcriber) else "idle", "model": WHISPER or "off"},
    }


@app.post("/v1/predict")
async def predict(req: PredictRequest):
    p = _pipeline()
    if not (req.text or req.url or req.acoustic):
        raise HTTPException(422, "provide at least one of text, url, acoustic")
    import numpy as np

    acoustic = np.asarray(req.acoustic, dtype=np.float32) if req.acoustic else None
    source = "text" if req.text else "url" if req.url else "audio"
    try:
        return await run_in_threadpool(
            _timed, source, p.predict, text=req.text, url=req.url, acoustic=acoustic,
            mc_samples=req.mc_samples, explain=req.explain, top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/predict/batch")
async def predict_batch(req: BatchRequest):
    """Scores many messages in one vectorised MC-dropout pass. Empty rows come back as {"error": ...}."""
    p = _pipeline()
    from src.inference.pipeline import extract_first_url

    keep = [i for i, t in enumerate(req.texts) if t and t.strip()]
    results: list[dict] = [{"error": "empty text"} for _ in req.texts]
    if keep:
        texts = [req.texts[i].strip() for i in keep]
        scored = await run_in_threadpool(
            _timed, "batch", p.predict_batch, texts, [extract_first_url(t) for t in texts], [None] * len(texts),
            req.mc_samples, req.explain,
        )
        for i, r in zip(keep, scored):
            results[i] = r
    return {"results": results, "engine": getattr(p, "checkpoint_kind", None)}


@app.post("/v1/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    text: str | None = Form(None),
    url: str | None = Form(None),
    mc_samples: int = Form(DEFAULT_MC),
    explain: bool = Form(False),
):
    p = _pipeline()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXT:
        raise HTTPException(415, f"unsupported audio type {suffix!r}; allowed {sorted(AUDIO_EXT)}")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio file larger than 25 MB")
    if not 1 <= mc_samples <= 200:
        raise HTTPException(422, "mc_samples must be in [1, 200]")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return await run_in_threadpool(
            _timed, "combined" if (text or url) else "audio", p.predict, text=text, url=url, audio_path=path, mc_samples=mc_samples, explain=explain,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        os.unlink(path)
