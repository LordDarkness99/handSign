"""Uji mandiri tanpa kamera: segmentasi gestur, ejaan multi-huruf, dan normalisasi TTS.

Jalankan:  .venv/bin/python tools/selftest_engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.spell_engine import SpellingEngine, bbox_motion  # noqa: E402
from backend.tts_engine import normalize_for_tts  # noqa: E402

FAILED = []
W, H = 480, 640
BOX_A = (200, 150, 300, 250)
BOX_B = (200, 152, 300, 252)      # hampir sama -> "diam"
BOX_MOVED = (340, 150, 440, 250)  # bergeser jauh -> "bergerak"


def check(name, got, expected):
    if got != expected:
        FAILED.append(f"{name}: got {got!r}, expected {expected!r}")
        print(f"  x {name}: {got!r} != {expected!r}")
    else:
        print(f"  v {name}: {got!r}")


def hold(engine, label, box, frames=8, conf=0.95, t0=1000, step=100):
    """Tahan pose `label` pada bbox `box` (diam) selama `frames` frame."""
    t = t0
    out = None
    for i in range(frames):
        t += step
        out = engine.observe(label, conf, t, bbox=box, frame_size=(H, W))
    return out, t


print("1) bbox_motion: bedakan diam vs bergerak")
check("diam", bbox_motion(BOX_B, BOX_A, W, H)["settled"], True)
check("bergeser", bbox_motion(BOX_MOVED, BOX_A, W, H)["settled"], False)
check("tanpa bbox", bbox_motion(None, BOX_A, W, H)["settled"], False)

print("2) Satu huruf tetap, gestur diam -> commit")
e = SpellingEngine(window=5, agreement=0.7, cooldown=1, min_conf=0.6, repeat_hold_ms=0)
r, t = hold(e, "A", BOX_A)
check("teks A", r["text"], "A")

print("3) Tahan pose SAMA tidak mengulang huruf")
r2, t2 = hold(e, "A", BOX_B, frames=10, t0=t)
check("tidak dobel", r2["text"], "A")

print("4) Perpindahan A -> I (dengan frame transisi) -> 'AI'")
# 4 frame pose campuran (bbox bergerak + label ragu), lalu pose I stabil
t3 = t2
for i in range(4):
    t3 += 100
    r3 = e.observe("B", 0.35, t3, bbox=(BOX_A[0] + i * 30, BOX_A[1], BOX_A[2] + i * 30, BOX_A[3]),
                   frame_size=(H, W))
check("transisi tidak commit", r3["text"], "A")
r4, t4 = hold(e, "I", BOX_MOVED, frames=8, t0=t3)
check("huruf I masuk", r4["text"], "AI")

print("5) Kata: A-I + space + L-O-V-E + space")
e = SpellingEngine(window=5, agreement=0.7, cooldown=1, min_conf=0.6,
                   repeat_hold_ms=0, speak_mode="space")
t = 50_000
for ch in "AI":
    _, t = hold(e, ch, BOX_A if ch == "A" else BOX_B, frames=8, t0=t)
r5a, t = hold(e, "space", BOX_B, frames=8, t0=t)
check("kata pertama", r5a["text"].strip(), "AI")
for ch in "LOVE":
    _, t = hold(e, ch, BOX_A if ch == "L" else BOX_B, frames=8, t0=t)
r5, t = hold(e, "space", BOX_B, frames=8, t0=t)
check("kata kedua", r5["text"].strip(), "AI LOVE")
check("daftar kata", r5["words"], ["AI", "LOVE"])

print("6) Repeat-by-hold untuk huruf kembar (HELLO)")
e2 = SpellingEngine(window=4, agreement=0.75, cooldown=1, min_conf=0.6,
                    repeat_hold_ms=800, speak_mode="manual")
t = 1000
for ch in "HE":
    _, t = hold(e2, ch, BOX_A, frames=6, t0=t)
_, t = hold(e2, "L", BOX_B, frames=6, t0=t)
check("sampai L", e2.buffer.text, "HEL")
# tahan pose L lagi (> repeat_hold_ms) -> huruf L kembar
r6, t = hold(e2, "L", BOX_B, frames=20, t0=t + 900, step=100)
check("huruf dobel LL", r6["text"], "HELL")
# lalu huruf baru langsung setelahnya (uji tidak terkunci)
r6b, t = hold(e2, "O", BOX_A, frames=8, t0=t)
check("huruf lanjutan O", r6b["text"], "HELLO")

print("7) Backspace (kelas 'del') dan 'nothing' diabaikan")
e3 = SpellingEngine(window=4, agreement=0.75, cooldown=1, min_conf=0.6, speak_mode="manual")
t = 1000
for ch in "ABC":
    _, t = hold(e3, ch, BOX_A, frames=6, t0=t)
r7, t = hold(e3, "nothing", BOX_B, frames=6, t0=t)
check("nothing diabaikan", r7["text"], "ABC")
r8, t = hold(e3, "del", BOX_B, frames=6, t0=t)
check("backspace", r8["text"], "AB")

print("8) Auto-speak tidak memotong kata di tengah")
e4 = SpellingEngine(window=4, agreement=0.75, cooldown=1, min_conf=0.6,
                    word_idle_ms=1500, speak_mode="space+idle")
t = 10_000
for ch in "HI":
    _, t = hold(e4, ch, BOX_A, frames=6, t0=t)
# tangan MASIH di kamera, jeda panjang -> tidak boleh bicara (memotong kata)
t += 2000
r9 = e4.observe("nothing", 0.0, t, bbox=BOX_B, frame_size=(H, W))
check("tidak ucap saat tangan masih di kamera", r9["should_speak"], False)
# tangan dikeluarkan (bbox None) lalu jeda -> selesai mengetik, boleh bicara
t += 100
r9b = e4.observe(None, 0.0, t, bbox=None, frame_size=(H, W))
check("tidak ucap sebelum jeda", r9b["should_speak"], False)
t += 1600
r9c = e4.observe(None, 0.0, t, bbox=None, frame_size=(H, W))
check("ucap setelah tangan ditarik", r9c["should_speak"], True)
check("teks yang diucapkan", r9c["speak_text"], "HI")
check("teks tetap tampil di UI", r9c["text"].strip(), "HI")

print("9) Normalisasi teks untuk TTS (LJSpeech huruf kecil)")
check("ALO", normalize_for_tts("HALO"), "H A L O.")
check("campur", normalize_for_tts("OK 123"), "O K 123.")
check("sudah titik", normalize_for_tts("hi!"), "hi!")

print("10) Batas panjang teks")
e5 = SpellingEngine(window=3, agreement=0.6, cooldown=0, repeat_hold_ms=0, speak_mode="manual")
e5.buffer.max_len = 5
t = 1000
for ch in "ABCDEFGH":
    _, t = hold(e5, ch, BOX_A, frames=5, t0=t)
check("dipotong di max_len", len(e5.buffer.text) <= 5, True)

print()
if FAILED:
    print(f"GAGAL {len(FAILED)} pemeriksaan:")
    for f in FAILED:
        print(" -", f)
    sys.exit(1)
print("SEMUA PEMERIKSAAN LULUS")
