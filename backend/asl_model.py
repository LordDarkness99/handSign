"""Model klasifikasi abjad ASL (EfficientNet-B0, 29 kelas) + deteksi tangan MediaPipe.

Mengikuti proyek referensi `ASL-Alphabet-Detection-main`:
  - model: `EfficientNetB0(num_classes=29)` (A-Z + 'del', 'nothing', 'space')
  - bobot: `model/efficientnet_model.pth` (state_dict dengan prefix `model.`)
  - preprocessing: resize 128x128 -> ToTensor (tanpa normalisasi ImageNet)
  - hand crop dari bounding box landmark tangan + padding 0.05
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "efficientnet_model.pth"
LANDMARKER_PATH = BASE_DIR / "model-asl" / "hand_landmarker.task"

# 29 kelas: A-Z + 3 kelas khusus (identik demo/main.py proyek referensi)
CLASS_NAMES: List[str] = [chr(i) for i in range(ord("A"), ord("Z") + 1)] + [
    "del",
    "nothing",
    "space",
]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 128
BBOX_PADDING = 0.05


class EfficientNetB0(nn.Module):
    """Wrapper sesuai bobot tersimpan: kunci state_dict diawali `model.`."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        self.model = EfficientNet.from_name("efficientnet-b0", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)


class ASLAlphabetRecognizer:
    """Pembungkus model + MediaPipe HandLandmarker (thread-safe, lazy load)."""

    def __init__(self, model_path: Path = MODEL_PATH, landmarker_path: Path = LANDMARKER_PATH):
        self.model_path = Path(model_path)
        self.landmarker_path = Path(landmarker_path)
        self.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.landmarker = None
        self._lock = threading.Lock()
        self._loaded = False
        self.error: Optional[str] = None

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            try:
                self._load_model()
                self._load_landmarker()
                self._loaded = True
                self.error = None
            except Exception as exc:  # dependensi lingkungan
                self.error = f"{type(exc).__name__}: {exc}"
                return False
        return True

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"bobot model ASL tidak ditemukan: {self.model_path}")
        model = EfficientNetB0(NUM_CLASSES)
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.model = model.to(self.device)

    def _load_landmarker(self) -> None:
        if not self.landmarker_path.exists():
            raise FileNotFoundError(f"model MediaPipe tidak ditemukan: {self.landmarker_path}")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.landmarker_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "error": self.error,
            "model_path": str(self.model_path),
            "landmarker_path": str(self.landmarker_path),
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
        }

    @staticmethod
    def decode_image(payload: str) -> np.ndarray:
        """Terima data-URL / base64 JPEG dari browser -> BGR ndarray."""
        raw = payload.split(",", 1)[1] if payload.startswith("data:") else payload
        buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("gambar tidak dapat didekode (base64 JPEG tidak valid)")
        return img

    def hand_bbox(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Deteksi tangan pertama; return (bbox_xyxy, landmarks) atau (None, None)."""
        if not self.ensure_loaded():
            return None, None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._to_mp_image(rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        landmarks = result.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        pad = BBOX_PADDING
        x0 = max(0, int((min(xs) - pad) * w))
        y0 = max(0, int((min(ys) - pad) * h))
        x1 = min(w, int((max(xs) + pad) * w))
        y1 = min(h, int((max(ys) + pad) * h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, landmarks
        return (x0, y0, x1, y1), landmarks

    @staticmethod
    def _to_mp_image(rgb: np.ndarray):
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    @staticmethod
    def crop_hand(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        return frame_bgr[y0:y1, x0:x1]

    def classify_crop(self, crop_bgr: np.ndarray):
        """Klasifikasikan crop tangan -> (label, confidence, probabilitas 29 kelas)."""
        if not self.ensure_loaded() or self.model is None:
            return None, 0.0, np.zeros(NUM_CLASSES, dtype=np.float32)
        resized = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return CLASS_NAMES[idx], float(probs[idx]), probs

    def predict_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        """Pipeline lengkap satu frame webcam."""
        bbox, landmarks = self.hand_bbox(frame_bgr, timestamp_ms)
        out = {
            "hand_detected": bbox is not None,
            "bbox": list(bbox) if bbox else None,
            "label": None,
            "confidence": 0.0,
            "top": [],
        }
        if bbox is None:
            out["landmarks"] = self.landmarks_for_frontend(landmarks, frame_bgr.shape[1], frame_bgr.shape[0])
            return out
        label, conf, probs = self.classify_crop(self.crop_hand(frame_bgr, bbox))
        out["label"] = label
        out["confidence"] = round(conf, 4)
        order = np.argsort(probs)[::-1][:3]
        out["top"] = [{"label": CLASS_NAMES[i], "confidence": round(float(probs[i]), 4)} for i in order]
        out["landmarks"] = self.landmarks_for_frontend(landmarks, frame_bgr.shape[1], frame_bgr.shape[0])
        return out

    @staticmethod
    def landmarks_for_frontend(landmarks, width: int, height: int) -> list:
        if landmarks is None:
            return []
        return [
            {"x": round(lm.x * width, 1), "y": round(lm.y * height, 1), "z": round(lm.z, 4)}
            for lm in landmarks
        ]


_recognizer: Optional[ASLAlphabetRecognizer] = None
_recognizer_lock = threading.Lock()


def get_recognizer() -> ASLAlphabetRecognizer:
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = ASLAlphabetRecognizer()
    return _recognizer
