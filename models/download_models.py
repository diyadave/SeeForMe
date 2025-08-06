import urllib.request
import os
import zipfile
from pathlib import Path

VOSK_MODELS = {
    "vosk-en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-hi": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "vosk-gu": "https://github.com/DiyaDaveAI/models/releases/download/v1/vosk-model-small-gujarati-0.4.zip"  # you can upload Gujarati model to your own GitHub Releases
}

def download_and_extract(url, extract_to):
    zip_path = extract_to + ".zip"
    if not os.path.exists(extract_to):
        print(f"📥 Downloading: {url}")
        urllib.request.urlretrieve(url, zip_path)
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        print(f"✅ Downloaded and extracted to {extract_to}")
    else:
        print(f"✔️ Already exists: {extract_to}")

def download_yolo_model():
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n model downloaded")
    except Exception as e:
        print(f"❌ YOLO download failed: {e}")

def download_vosk_models():
    os.makedirs("models/vosk", exist_ok=True)
    for lang, url in VOSK_MODELS.items():
        path = os.path.join("models/vosk", lang)
        download_and_extract(url, path)

def create_basic_emotion_model():
    print("✅ Using OpenCV for basic emotion analysis")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    download_yolo_model()
    download_vosk_models()
    create_basic_emotion_model()
