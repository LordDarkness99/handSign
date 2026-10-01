"""FastAPI backend: deteksi abjad ASL dari webcam + sintesis suara SpeechT5.

Endpoint:
  GET  /api/status    -> status model ASL & TTS + daftar kelas
  POST /api/detect    -> 1 frame (base64 JPEG) -> prediksi huruf + hasil ejaan (+ audio)
  POST /api/speak     -> teks -> audio WAV (base64)
  POST /api/settings  -> atur kestabilan huruf & speaker TTS
  POST /api/reset     -> hapus teks ejaan sesi
  GET  /api/text      -> teks ejaan terkini
  GET  /              -> UI (static/)
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asl_model import CLASS_NAMES, get_recognizer
from .spell_engine import SPEAK_MODES, SpellingEngine
from .tts_engine import SPEAKER_PRESETS, get_tts, normalize_for_tts

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="SignSpeak - ASL Alphabet to Speech", version="1.0.0")

# Sesi ejaan per klien (dikunci lewat header X-Client-Id)
_engines: Dict[str, SpellingEngine] = {}


def get_engine(request: Request) -> SpellingEngine:
    cid = request.headers.get("x-client-id") or "default"
    if cid not in _engines:
        _engines[cid] = SpellingEngine()
    return _engines[cid]


class DetectRequest(BaseModel):
    image: str = Field(..., description="data URL / base64 JPEG dari webcam")
    timestamp_ms: Optional[int] = None
    speak: bool = Field(True, description="kirim audio otomatis ketika kata selesai")


class SpeakRequest(BaseModel):
    text: str
    speaker: Optional[str] = None


class SettingsRequest(BaseModel):
    window: Optional[int] = Field(None, ge=2, le=20)
    agreement: Optional[float] = Field(None, ge=0.2, le=1.0)
    cooldown: Optional[int] = Field(None, ge=0, le=30)
    repeat_hold_ms: Optional[int] = Field(None, ge=0, le=5000)
    speak_mode: Optional[str] = None
    word_idle_ms: Optional[int] = Field(None, ge=500, le=10000)
    speaker: Optional[str] = None


@app.get("/api/status")
def api_status():
    recognizer = get_recognizer()
    return {
        "asl": recognizer.status(),
        "tts": get_tts().status(),
        "classes": CLASS_NAMES,
        "special_classes": {
            "del": "hapus satu huruf",
            "space": "pemisah kata",
            "nothing": "abaikan (tidak ada isyarat)",
        },
        "speakers": list(SPEAKER_PRESETS),
        "speak_modes": list(SPEAK_MODES),
    }


@app.post("/api/detect")
def api_detect(payload: DetectRequest, request: Request):
    recognizer = get_recognizer()
    if not recognizer.ensure_loaded():
        raise HTTPException(status_code=503, detail=f"model ASL gagal dimuat: {recognizer.error}")
    try:
        frame = recognizer.decode_image(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ts = payload.timestamp_ms or int(time.time() * 1000)
    pred = recognizer.predict_frame(frame, ts)
    h, w = frame.shape[:2]
    pred["frame_size"] = {"w": int(w), "h": int(h)}

    engine = get_engine(request)
    spell = engine.observe(
        pred.get("label"),
        pred.get("confidence", 0.0),
        ts,
        bbox=pred.get("bbox"),
        frame_size=(h, w),
    )
    result = {**pred, "spell": spell}

    if payload.speak and spell["should_speak"] and spell["speak_text"]:
        audio_b64, err = _synthesize(spell["speak_text"])
        if audio_b64:
            result["audio"] = audio_b64
            result["spoken_text"] = spell["speak_text"]
    return result


def _synthesize(text: str, speaker: Optional[str] = None):
    tts = get_tts()
    if speaker:
        tts.set_speaker(speaker)
    try:
        wav = tts.synthesize(text)
        return base64.b64encode(wav).decode("ascii"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@app.post("/api/speak")
def api_speak(payload: SpeakRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="teks kosong")
    audio, err = _synthesize(text, payload.speaker)
    if not audio:
        raise HTTPException(status_code=503, detail=f"TTS gagal: {err}")
    return {
        "text": text,
        "spoken_text": normalize_for_tts(text),
        "audio": audio,
        "format": "wav",
        "sample_rate": get_tts().status()["sample_rate"],
    }


@app.post("/api/settings")
def api_settings(payload: SettingsRequest, request: Request):
    engine = get_engine(request)
    engine.configure(
        window=payload.window,
        agreement=payload.agreement,
        cooldown=payload.cooldown,
        repeat_hold_ms=payload.repeat_hold_ms,
        speak_mode=payload.speak_mode,
        word_idle_ms=payload.word_idle_ms,
    )
    if payload.speaker:
        get_tts().set_speaker(payload.speaker)
    return {
        "window": engine.stabilizer.window,
        "agreement": engine.stabilizer.agreement,
        "cooldown": engine.stabilizer.cooldown,
        "repeat_hold_ms": engine.stabilizer.repeat_hold_ms,
        "speak_mode": engine.speak_mode,
        "word_idle_ms": engine.word_idle_ms,
        "speaker": get_tts().speaker,
        "text": engine.buffer.text,
    }


@app.post("/api/reset")
def api_reset(request: Request):
    get_engine(request).reset()
    return {"text": "", "words": []}


@app.get("/api/text")
def api_text(request: Request):
    return get_engine(request).buffer.snapshot()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return {"app": "SignSpeak", "docs": "/docs"}
    return FileResponse(str(page))
