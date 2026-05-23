# DeepFake App Documentation

## Owner
Made by Devansh

## Project Summary
This project is a DeepFake image detection web application with:

- a FastAPI backend
- a Streamlit frontend
- local CPU-based face preprocessing
- Hugging Face API-based image classification
- a Hugging Face vision-language model as the final judge

The app accepts an uploaded image, detects the largest face locally, creates forensic image variants, sends the processed face to multiple Hugging Face image-classification models, and then uses a Hugging Face VLM to produce the final `REAL` or `FAKE` judgment with reasoning.

## What Was Built

### 1. Backend
- FastAPI app with `/health` and `/predict`
- image upload validation for `jpg`, `jpeg`, `png`, and `webp`
- centralized service layer for inference
- structured response models with Pydantic
- clear HTTP error handling

### 2. Frontend
- Streamlit upload interface
- image preview
- analyze button
- result card with final verdict, confidence, reasoning, and ensemble evidence
- persistent UI credit: `Made by Devansh`

### 3. Local Image Processing
- image validation with Pillow
- largest-face detection with OpenCV Haar cascade
- face crop with margin
- resize to `224x224`
- ELA artifact generation
- Laplacian / high-pass residual generation

### 4. Hugging Face Integration
- classifier ensemble through the Hugging Face Inference API
- final reasoning through a Hugging Face chat-completions VLM
- shared env-token loading from `backend/.env`

### 5. Local Run Support
- `venv`-based setup
- root-level launcher: [run_deepfake_app.bat](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\run_deepfake_app.bat)
- PowerShell helpers in [tools/run_backend.ps1](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\tools\run_backend.ps1) and [tools/run_frontend.ps1](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\tools\run_frontend.ps1)

## Earlier Requested Structure
This was the earlier target structure for the project:

```text
deepfake-detector/
├── backend/
│   ├── main.py
│   ├── routes/
│   │   └── predict.py
│   ├── services/
│   │   └── model_service.py
│   ├── schemas/
│   │   └── response.py
│   ├── utils/
│   │   └── image_utils.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── uploader.py
│   │   └── result_card.py
│   ├── utils/
│   │   └── api_client.py
│   └── requirements.txt
├── .gitignore
└── README.md
```

## Current Important Files

### Root
- [run_deepfake_app.bat](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\run_deepfake_app.bat)
- [README.md](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\README.md)
- [PROJECT_DOCS.md](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\PROJECT_DOCS.md)

### Backend
- [backend/main.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\main.py)
- [backend/routes/predict.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\routes\predict.py)
- [backend/services/model_service.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\services\model_service.py)
- [backend/utils/image_utils.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\utils\image_utils.py)
- [backend/scripts/ensemble_vlm_judge.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\scripts\ensemble_vlm_judge.py)
- [backend/scripts/face_crop_hf_api.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\scripts\face_crop_hf_api.py)
- [backend/.env](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\backend\.env)

### Frontend
- [frontend/app.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\frontend\app.py)
- [frontend/components/result_card.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\frontend\components\result_card.py)
- [frontend/components/uploader.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\frontend\components\uploader.py)
- [frontend/utils/api_client.py](C:\Users\devan\OneDrive\Desktop\deepfake%20ai\frontend\utils\api_client.py)

## Current Processing Flow
1. User uploads an image from Streamlit.
2. Streamlit sends the file to FastAPI `/predict`.
3. Backend validates the image bytes.
4. Backend detects the largest face locally with OpenCV.
5. Backend crops and resizes the face.
6. Backend creates forensic variants:
   - ELA image
   - high-pass / Laplacian residual image
7. Backend sends the cropped face to multiple Hugging Face classifier endpoints.
8. Backend sends the cropped face, forensic images, and ensemble evidence to a Hugging Face VLM.
9. VLM returns final JSON reasoning.
10. Backend returns the final structured response to Streamlit.
11. Streamlit renders the verdict and supporting evidence.

## Response Shape Used by the Frontend
The main API result includes:

- `label`
- `confidence`
- `all_scores`
- `model`
- `justification`
- `artifact_observations`
- `ensemble_consideration`
- `ensemble_results`
- `preprocessing`

## How To Run

### Recommended
Run:

```powershell
.\run_deepfake_app.bat
```

### Manual
Activate `venv`:

```powershell
.\venv\Scripts\Activate.ps1
```

Run backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run frontend in another terminal:

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

## Default Local URLs
- Streamlit: `http://127.0.0.1:8501`
- FastAPI: `http://127.0.0.1:8000`
- FastAPI docs: `http://127.0.0.1:8000/docs`

## Notes
- The backend depends on a valid Hugging Face token in `backend/.env`.
- The local machine only handles preprocessing; model inference is API-based.
- The frontend now hides model names and shows a persistent `Made by Devansh` credit.
