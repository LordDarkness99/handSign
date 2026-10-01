#!/bin/bash
# ==============================================================================
# SignSpeak - Abjad ASL menjadi Suara (EfficientNet-B0 + MediaPipe + SpeechT5)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🤟 SignSpeak — Abjad ASL ➜ Suara"
echo "=========================================================="

# Buat virtual environment bila belum ada
if [ ! -d ".venv" ]; then
    echo "⚙️  Membuat virtual environment Python 3.11..."
    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/pip install -r requirements.txt
fi

source .venv/bin/activate

# Cek bobot model ASL abjad (EfficientNet-B0, 29 kelas)
if [ ! -f "model/efficientnet_model.pth" ]; then
    echo "❌ Error: bobot model ASL tidak ditemukan di $DIR/model/efficientnet_model.pth"
    echo "   Salin dari proyek ASL-Alphabet-Detection-main/model/efficientnet_model.pth"
    exit 1
fi

# Cek model landmarker tangan MediaPipe
if [ ! -f "model-asl/hand_landmarker.task" ]; then
    echo "❌ Error: hand_landmarker.task tidak ditemukan di $DIR/model-asl/"
    echo "   Unduh: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    exit 1
fi

# Cek model TTS SpeechT5 hasil fine-tuning LJSpeech
if [ ! -f "model_speecht5_ljspeech/config.json" ]; then
    echo "❌ Error: model TTS tidak ditemukan di $DIR/model_speecht5_ljspeech/"
    exit 1
fi

PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

echo "🚀 Menjalankan SignSpeak..."
echo "🌐 Buka browser di: http://localhost:$PORT"
echo "📖 Swagger API Docs: http://localhost:$PORT/docs"
echo "ℹ️  Camera hanya bisa diakses lewat localhost atau HTTPS."
echo "ℹ️  Model TTS (SpeechT5) diunduh saat evaluasi pertama — butuh koneksi internet."
echo "Tekan Ctrl + C untuk menghentikan server."
echo "=========================================================="

exec uvicorn backend.main:app --host "$HOST" --port "$PORT"

