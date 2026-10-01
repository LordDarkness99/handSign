"""Uji klasifikasi abjad ASL pada gambar contoh (butuh model ASL + MediaPipe).

Jalankan:  .venv/bin/python tools/selftest_images.py [folder_gambar]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from backend.asl_model import get_recognizer  # noqa: E402

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "test_images"


def main(folder: Path):
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not images:
        print(f"Tidak ada gambar di {folder}")
        return 1
    rec = get_recognizer()
    if not rec.ensure_loaded():
        print("Model gagal dimuat:", rec.error)
        return 1

    detected = correct = 0
    ts = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        ts += 50  # timestamp harus monoton naik untuk mode VIDEO
        out = rec.predict_frame(frame, ts)
        expected = path.stem.split("_")[0].upper()
        if out["hand_detected"]:
            detected += 1
            ok = out["label"] == expected
            correct += ok
            top = ", ".join(f"{t['label']}({t['confidence']:.2f})" for t in out["top"])
            print(f"{'✓' if ok else '✗'} {path.name:14s} diharapkan={expected:3s} "
                  f"prediksi={out['label']:>7s} conf={out['confidence']:.2f} | {top}")
        else:
            print(f"- {path.name:14s} tangan tidak terdeteksi")
    print(f"\nterdeteksi {detected}/{len(images)} gambar, benar {correct}/{detected}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    sys.exit(main(target))
