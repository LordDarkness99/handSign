"""Mesin eja v2: segmentasi gestur -> stabilisasi huruf -> teks -> pemicu suara.

Model ASL berbasis **abjad** hanya mengenali satu pose tangan per huruf, sehingga
berpindah A -> I selalu menghasilkan frame "pose campuran" di tengah jalan.
Mesin ini karena itu memakai 4 mekanisme agar multi-huruf bisa dieja dengan lancar:

1. **Segmentasi gestur (motion gate)** - frame saat tangan sedang bergerak (bbox
   bergeser / ukur berubah) TIDAK dipakai untuk voting; buffer dikosongkan.
   Hanya frame "diam" yang dihitung -> huruf baru punya start yang bersih.
2. **Edge-triggered commit** - sebuah huruf hanya di-commit bila label BERUBAH dari
   huruf terakhir, sehingga tidak ada huruf dobel (= tangan diam tidak dieja
   ribuan kali). Huruf kembar (LL, OO) dapat lewat mode "tahan repeat".
3. **Repeat-by-hold** - menahan pose yang sama selama `repeat_hold_ms` tetap
   menghasilkan huruf kembar (mis. "HELLO" butuh LL).
4. **Kebijakan bicara sadar kata** - auto-speak hanya terjadi setelah selesai satu kata
   (setelah `space` atau jeda panjang), tidak lagi memotong di tengah kata.

Alur: frame -> (bbox) -> motion gate -> LetterStabilizer -> TextBuffer -> TTS
"""

from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

# --- segmentasi gestur ---------------------------------------------------------
MOTION_SHIFT = float(os.getenv("HANDSIGN_MOTION_SHIFT", "0.055"))
"""Batas pergeseran bbox (fraksi diagonal frame) yang masih dianggap 'diam'."""
MOTION_SCALE = float(os.getenv("HANDSIGN_MOTION_SCALE", "0.22"))
"""Batas perubahan luas bbox (fraksi) yang masih dianggap 'diam'."""

# --- kestabilan huruf ----------------------------------------------------------
WINDOW = int(os.getenv("HANDSIGN_WINDOW", "5"))            # jumlah frame buffer
AGREEMENT = float(os.getenv("HANDSIGN_AGREEMENT", "0.7"))  # minimum proporsi huruf sama
COOLDOWN = int(os.getenv("HANDSIGN_COOLDOWN", "2"))         # jeda antar commit (frame)
MIN_CONF = float(os.getenv("HANDSIGN_MIN_CONF", "0.6"))    # keyakinan minimum
REPEAT_HOLD_MS = int(os.getenv("HANDSIGN_REPEAT_HOLD_MS", "1100"))
"""Tahan pose sama selama ini -> huruf kembar (LL, OO, EE...). 0 = matikan."""

# --- teks ----------------------------------------------------------------------
WORD_IDLE_MS = int(os.getenv("HANDSIGN_WORD_IDLE_MS", "2000"))  # jeda -> kata selesai
MAX_TEXT_LEN = int(os.getenv("HANDSIGN_MAX_TEXT", "60"))
BACKSPACE_CLS = "del"
SPACE_CLS = "space"
NOTHING_CLS = "nothing"

# --- kebijakan bicara ----------------------------------------------------------
SPEAK_MODES = ("space", "space+idle", "idle", "manual")
DEFAULT_SPEAK_MODE = os.getenv("HANDSIGN_SPEAK_MODE", "space+idle")


def bbox_motion(bbox, prev_bbox, width: int, height: int) -> Dict:
    """Nilai 'diam atau bergerak' antara dua bbox tangan (x0,y0,x1,y1).

    Digunakan sebagai gerbang: frame yang tangan-nya sedang bergerak diabaikan
    supaya pose campuran tidak mengotori voting huruf.
    """
    if bbox is None or prev_bbox is None or width <= 0 or height <= 0:
        return {"settled": False, "shift": 1.0, "scale_change": 1.0}
    diag = (width * width + height * height) ** 0.5
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    px = (prev_bbox[0] + prev_bbox[2]) / 2.0
    py = (prev_bbox[1] + prev_bbox[3]) / 2.0
    shift = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / diag
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    prev_area = max(1.0, (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]))
    scale_change = abs(area - prev_area) / prev_area
    settled = shift <= MOTION_SHIFT and scale_change <= MOTION_SCALE
    return {
        "settled": settled,
        "shift": round(shift, 4),
        "scale_change": round(scale_change, 4),
    }


@dataclass
class LetterStabilizer:
    """Commit huruf hanya bila (a) pose diam, (b) stabil, (c) label BERUBAH.

    Compared to the naive version this fixes the "only 1 letter is read" problem:
    frames taken while the hand is *moving* between two letters are discarded, so a
    new letter always starts from a clean buffer, and a held pose commits once (not
    thousands of times). Holding the same pose longer than `repeat_hold_ms`
    intentionally re-commits it, which is how double letters (LL, OO) are typed.
    """

    window: int = WINDOW
    agreement: float = AGREEMENT
    cooldown: int = COOLDOWN
    min_conf: float = MIN_CONF
    repeat_hold_ms: int = REPEAT_HOLD_MS
    buffer: Deque[Optional[str]] = field(default_factory=lambda: deque(maxlen=WINDOW))
    cooldown_left: int = 0
    last_committed: Optional[str] = None
    hold_started_ms: Optional[int] = None
    moving_until_ms: int = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.cooldown_left = 0
        self.hold_started_ms = None

    def notify_motion(self, now_ms: int, settled: bool) -> None:
        """Tangan bergerak -> abort voting & jangan ada repeat yg menyesatkan."""
        if settled:
            return
        self.buffer.clear()
        self.cooldown_left = 0
        self.hold_started_ms = None
        self.moving_until_ms = now_ms + 250  # jeda singkat setelah gerak berhenti

    def update(
        self,
        label: Optional[str],
        conf: float,
        now_ms: int,
        settled: bool = True,
    ) -> Optional[str]:
        """Masukkan satu prediksi; kembalikan huruf yang siap di-commit (atau None)."""
        self.notify_motion(now_ms, settled)
        if not settled or now_ms < self.moving_until_ms:
            return None
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return None

        valid = label if (label and conf >= self.min_conf) else None
        self.buffer.append(valid)
        if len(self.buffer) < self.window:
            return None

        counts = Counter(x for x in self.buffer if x)
        if not counts:
            self.hold_started_ms = None
            return None
        top, n = counts.most_common(1)[0]
        if n / self.window < self.agreement:
            self.hold_started_ms = None
            return None

        # Edge-trigger: jangan ulang huruf yang sama (kecuali repeat-by-hold).
        if top == self.last_committed:
            if self.repeat_hold_ms and self.hold_started_ms is None:
                self.hold_started_ms = now_ms
            if not self.repeat_hold_ms or self.hold_started_ms is None:
                return None
            if now_ms - self.hold_started_ms < self.repeat_hold_ms:
                return None

        self.buffer.clear()
        self.cooldown_left = self.cooldown
        self.last_committed = top
        self.hold_started_ms = None
        return top


@dataclass
class TextBuffer:
    """Akumulator huruf menjadi kalimat, plus daftar kata yang sudah selesai."""

    max_len: int = MAX_TEXT_LEN
    text: str = ""
    words: List[str] = field(default_factory=list)
    spoken_upto: int = 0          # karakter terakhir yang sudah diucapkan
    last_commit_ms: int = 0

    def clear(self) -> None:
        self.text = ""
        self.words = []
        self.spoken_upto = 0
        self.last_commit_ms = 0

    def backspace(self) -> bool:
        if not self.text:
            return False
        self.text = self.text[:-1]
        self.spoken_upto = min(self.spoken_upto, len(self.text))
        if self.words and (not self.text or self.text.endswith(" ")):
            self.words = self.words[:-1]
        return True

    def apply(self, label: str, now_ms: int) -> str:
        """Terapkan satu label stabil.

        Return aksi: 'char' | 'space' | 'del' | 'nothing' | 'full'.
        """
        if label == NOTHING_CLS:
            return "nothing"
        if label == SPACE_CLS:
            if self.text and not self.text.endswith(" "):
                self.text += " "
                # hanya kata TERAKHIR yang selesai, bukan seluruh teks
                word = self.text.strip().split(" ")[-1]
                if word and (not self.words or self.words[-1] != word):
                    self.words.append(word)
            return "space"
        if label == BACKSPACE_CLS:
            self.backspace()
            return "del"
        if len(self.text) >= self.max_len:
            return "full"
        self.text += label
        self.last_commit_ms = now_ms
        return "char"

    def idle_ms(self, now_ms: int) -> int:
        return now_ms - self.last_commit_ms if self.last_commit_ms else 0

    def pending(self) -> str:
        """Teks yang belum diucapkan (setelah `spoken_upto`)."""
        return self.text[self.spoken_upto:].strip()

    def pending_word(self) -> str:
        """Kata terakhir yang belum diucapkan (tanpa spasi di ujung)."""
        pending = self.text[self.spoken_upto:].strip()
        if not pending:
            return ""
        return pending.split(" ")[-1] if self.text.endswith(" ") else pending

    def mark_spoken(self, upto_chars: int) -> None:
        self.spoken_upto = min(len(self.text), max(self.spoken_upto, upto_chars))

    def snapshot(self) -> Dict:
        return {
            "text": self.text,
            "words": list(self.words),
            "length": len(self.text),
            "max_len": self.max_len,
            "pending": self.pending(),
        }


class SpellingEngine:
    """Mesin eja satu sesi: motion gate -> stabilizer -> text buffer -> kebijakan bicara."""

    def __init__(
        self,
        window: int = WINDOW,
        agreement: float = AGREEMENT,
        cooldown: int = COOLDOWN,
        min_conf: float = MIN_CONF,
        word_idle_ms: int = WORD_IDLE_MS,
        repeat_hold_ms: int = REPEAT_HOLD_MS,
        speak_mode: str = DEFAULT_SPEAK_MODE,
    ):
        self.stabilizer = LetterStabilizer(
            window, agreement, cooldown, min_conf, repeat_hold_ms
        )
        self.buffer = TextBuffer()
        self.word_idle_ms = word_idle_ms
        self.speak_mode = speak_mode if speak_mode in SPEAK_MODES else DEFAULT_SPEAK_MODE
        self.prev_bbox = None
        self.last_seen_ms = 0

    # ------------------------------------------------------------------ input
    def observe(
        self,
        label: Optional[str],
        conf: float,
        now_ms: Optional[int] = None,
        bbox=None,
        frame_size=(480, 640),
    ) -> Dict:
        """Satu frame webcam -> status ejaan (+ indication untuk bicara)."""
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        width, height = frame_size
        motion = bbox_motion(bbox, self.prev_bbox, width, height)
        self.prev_bbox = bbox
        if bbox is not None:
            self.last_seen_ms = now_ms

        committed = self.stabilizer.update(label, conf, now_ms, motion["settled"])
        action = "idle"
        if committed:
            action = self.buffer.apply(committed, now_ms)

        result = {
            "action": action,
            "committed": committed,
            "letter": self.buffer.text[-1] if self.buffer.text else None,
            "motion": motion,
            "progress": self._progress(),
            "last_committed": self.stabilizer.last_committed,
            "should_speak": False,
            "speak_text": None,
            **self.buffer.snapshot(),
        }
        speak = self._speak_decision(action, now_ms)
        if speak:
            result.update(should_speak=True, speak_text=speak)
        return result

    def _progress(self) -> Dict:
        """Progres kestabilan huruf untuk indikator UI."""
        stab = self.stabilizer
        counts = Counter(x for x in stab.buffer if x)
        top, n = counts.most_common(1)[0] if counts else (None, 0)
        return {
            "filled": len(stab.buffer),
            "window": stab.window,
            "candidate": top,
            "votes": n,
            "ratio": round(n / stab.window, 3) if stab.window else 0.0,
            "ready": bool(top and n / max(1, stab.window) >= stab.agreement),
        }

    # ------------------------------------------------------------------ bicara
    def _speak_decision(self, action: str, now_ms: int) -> Optional[str]:
        """Tentukan teks yang perlu diucapkan.

        Penting: jeda idle HANYA dianggap "selesai mengetik" bila tangan memang
        tidak terlihat (bbox None). Kalau hanya "tidak ada huruf baru" sementara
        tangan masih di kamera, sistem tidak boleh bicara karena itu memotong
        kata di tengah (mis. sedang mengetik "LOVE" lalu tiba-tiba diucapkan).
        """
        if self.speak_mode == "manual":
            return None
        if action == "space" and self.speak_mode in ("space", "space+idle"):
            word = self.buffer.pending_word()
            if word and " " not in word:
                self.buffer.mark_spoken(len(self.buffer.text))
                return word
            return None
        if self.speak_mode in ("idle", "space+idle"):
            hand_away = self.prev_bbox is None
            idle = now_ms - self.last_seen_ms if self.last_seen_ms else 0
            if self.buffer.pending() and hand_away and idle >= self.word_idle_ms:
                text = self.buffer.pending()
                self.buffer.mark_spoken(len(self.buffer.text))
                return text
        return None

    def mark_spoken(self) -> None:
        self.buffer.mark_spoken(len(self.buffer.text))

    def flush(self) -> str:
        """Ambil teks lengkap dan bersihkan buffer (tombol 'Ucapkan')."""
        text = self.buffer.text.strip()
        self.buffer.clear()
        self.stabilizer.reset()
        return text

    def reset(self) -> None:
        self.buffer.clear()
        self.stabilizer.reset()
        self.prev_bbox = None

    # -------------------------------------------------------------- pengaturan
    def configure(
        self,
        window: Optional[int] = None,
        agreement: Optional[float] = None,
        cooldown: Optional[int] = None,
        repeat_hold_ms: Optional[int] = None,
        speak_mode: Optional[str] = None,
        word_idle_ms: Optional[int] = None,
    ) -> None:
        stab = self.stabilizer
        if window:
            stab.window = max(2, int(window))
            stab.buffer = deque(stab.buffer, maxlen=stab.window)
        if agreement is not None:
            stab.agreement = min(1.0, max(0.1, float(agreement)))
        if cooldown is not None:
            stab.cooldown = max(0, int(cooldown))
        if repeat_hold_ms is not None:
            stab.repeat_hold_ms = max(0, int(repeat_hold_ms))
        if speak_mode in SPEAK_MODES:
            self.speak_mode = speak_mode
        if word_idle_ms:
            self.word_idle_ms = max(500, int(word_idle_ms))
