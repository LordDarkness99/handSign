"""Mesin TTS: SpeechT5 (fine-tune LJSpeech) + vocoder HiFi-GAN -> WAV 16 kHz.

Model TTS dipakai dari `model_speecht5_ljspeech/` (fine-tune LJSpeech).
Vocoder diambil dari `microsoft/speecht5_hifigan` (diunduh sekali, lalu di-cache).
"""

from __future__ import annotations

import io
import re
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
TTS_MODEL_DIR = BASE_DIR / "model_speecht5_ljspeech"
TOKENIZER_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
BASE_TTS_ID = "microsoft/speecht5_tts"   # fallback bila model fine-tune lokal tidak ada
SAMPLE_RATE = 16000
SPEAKER_DIM = 512

# Preset speaker embedding (512-dim, seed tetap -> suara konsisten per preset)


def normalize_for_tts(text: str) -> str:
    """LJSpeech dilatih pada huruf kecil + tanda baca.

    Hasil ejaan ASL berupa huruf kapital (mis. `HALO`), jadi:
      - huruf ALL-CAPS dieja per huruf ("H A L O") agar TTS mengeja, bukan membacakan akronim,
      - huruf kecil/angka lain tetap apa adanya,
      - spasi berlebih dirapikan, teks dipotong agar tidak terlalu panjang.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    words = []
    for word in text.split(" "):
        letters = [c for c in word if c.isalpha()]
        if len(letters) > 1 and all(c.isupper() for c in letters):
            words.append(" ".join(letters))
        else:
            words.append(word)
    out = " ".join(words)
    if len(out) > MAX_TTS_CHARS:
        out = out[:MAX_TTS_CHARS].rsplit(" ", 1)[0]
    if out and out[-1] not in ".?!":
        out += "."
    return out


class TTSEngine:
    """Lazy-load SpeechT5 + HiFi-GAN, thread-safe (satu model dipakai bersama)."""

    def __init__(self, model_dir: Path = TTS_MODEL_DIR, speaker: str = "netral"):
        self.model_dir = Path(model_dir)
        self.speaker = speaker if speaker in SPEAKER_PRESETS else "netral"
        self._tts = None
        self._vocoder = None
        self._tokenizer = None
        self._speaker_cache = {}
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self.loaded = False
        self.error: Optional[str] = None
        # Dicek sejak init (bukan hanya saat load) supaya /api/status langsung akurat
        self.using_finetuned = self._has_local_model()

    def ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        with self._lock:
            if self.loaded:
                return True
            try:
                self._load()
                self.loaded = True
                self.error = None
            except Exception as exc:  # dependensi lingkungan
                self.error = f"{type(exc).__name__}: {exc}"
                return False
        return True

    def _has_local_model(self) -> bool:
        """True bila bobot fine-tune lokal tersedia dan bisa dipakai."""
        if not (self.model_dir / "config.json").exists():
            return False
        return any(
            (self.model_dir / f).exists()
            for f in ("model.safetensors", "pytorch_model.bin")
        )

    def _load(self) -> None:
        """Muat model TTS.

        Prioritas: model fine-tune lokal `model_speecht5_ljspeech/`. Bila folder itu
        tidak ada (mis. repo di-clone tanpa bobot, karena >1.7 GB), fall back ke model
        dasar `microsoft/speecht5_tts` supaya aplikasi tetap bisa berjalan.
        """
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Tokenizer,
        )

        self._tokenizer = SpeechT5Tokenizer.from_pretrained(TOKENIZER_ID)
        local_ok = (self.model_dir / "config.json").exists() and any(
            (self.model_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
        )
        if local_ok:
            source = str(self.model_dir)
            self.using_finetuned = True
        else:
            source = BASE_TTS_ID
            self.using_finetuned = False

        self._tts = SpeechT5ForTextToSpeech.from_pretrained(source)
        self._tts.eval()
        self._vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
        self._vocoder.eval()

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "error": self.error,
            "using_finetuned": self.using_finetuned,
            "model_source": str(self.model_dir) if self.using_finetuned else BASE_TTS_ID,
            "model_dir": str(self.model_dir),
            "tokenizer": TOKENIZER_ID,
            "vocoder": VOCODER_ID,
            "speakers": list(SPEAKER_PRESETS),
            "speaker": self.speaker,
            "sample_rate": SAMPLE_RATE,
        }

    def set_speaker(self, speaker: str) -> None:
        self.speaker = speaker if speaker in SPEAKER_PRESETS else "netral"

    def _speaker_embedding(self, name: str) -> torch.Tensor:
        if name not in self._speaker_cache:
            seed = SPEAKER_PRESETS[name]
            g = torch.Generator().manual_seed(seed)
            emb = torch.randn(SPEAKER_DIM, generator=g) * 0.1
            self._speaker_cache[name] = emb.unsqueeze(0)
        return self._speaker_cache[name]

    def synthesize(self, text: str, speaker: Optional[str] = None) -> bytes:
        """Teks -> bytes WAV 16 kHz mono (16-bit PCM)."""
        if not self.ensure_loaded():
            raise RuntimeError(self.error or "model TTS gagal dimuat")
        spoken = normalize_for_tts(text)
        if not spoken:
            raise ValueError("teks kosong")
        ids = self._tokenizer(spoken)["input_ids"]
        emb = self._speaker_embedding(speaker or self.speaker)
        with self._gen_lock, torch.inference_mode():
            wav = self._tts.generate_speech(
                torch.tensor([ids]), emb, vocoder=self._vocoder
            )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

SPEAKER_PRESETS = {
    "netral": 0,
    "pria": 1,
    "wanita": 2,
    "lambat": 3,
}

# Teks yang terlalu panjang dipotong agar TTS CPU tidak menggantung
MAX_TTS_CHARS = int(220)


_tts_engine: Optional[TTSEngine] = None
_tts_lock = threading.Lock()


def get_tts() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        with _tts_lock:
            if _tts_engine is None:
                _tts_engine = TTSEngine()
    return _tts_engine
