# DeepFake Detector

This project runs a FastAPI backend and Streamlit frontend that classify uploaded images as `REAL` or `FAKE` through the Hugging Face Inference API using `prithivMLmods/deepfake-detector-model-v1`. Before the API call, the backend detects the largest face locally with OpenCV, crops it with a margin, resizes it to `224x224`, and sends that face crop to Hugging Face.

## Run

```powershell
.\venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
.\venv\Scripts\python -m streamlit run frontend/app.py
```

## Environment

- `HF_TOKEN`: Hugging Face API token
- `HUGGINGFACE_API_TOKEN`: optional compatibility fallback
- `MODEL_NAME`: Hugging Face deepfake model id

## Standalone Script

Run this helper to test the local face-crop + HF API path outside the web app:

```powershell
.\venv\Scripts\python backend\scripts\face_crop_hf_api.py path\to\image.jpg
```
