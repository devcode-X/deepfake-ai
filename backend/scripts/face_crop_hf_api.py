"""Detect the largest face locally, crop it, and call a Hugging Face model."""

from __future__ import annotations

from pathlib import Path
import os
import sys

from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.image_utils import NoFaceDetectedError, prepare_face_image_for_inference

load_dotenv(ROOT / "backend" / ".env", override=True)

# Replace MODEL_NAME or API_URL if you want to target a different HF image model.
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
HF_API_TOKEN = (
    os.environ.get("HF_API_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_API_TOKEN")
)


def classify_face_image(image_path: str) -> list[dict]:
    """Prepare a face crop locally and submit it to the HF Inference API."""
    image_bytes = Path(image_path).read_bytes()

    try:
        prepared_face_bytes = prepare_face_image_for_inference(image_bytes)
    except NoFaceDetectedError:
        print("No face detected")
        return []

    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN or HF_TOKEN must be set before calling the API.")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "image/jpeg",
    }
    response = requests.post(API_URL, headers=headers, data=prepared_face_bytes, timeout=120)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python backend/scripts/face_crop_hf_api.py <image_path>")
        raise SystemExit(1)

    result = classify_face_image(sys.argv[1])
    if result:
        print(result)
