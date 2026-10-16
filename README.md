# 🕵️ DeepFake Detector — Real-Time Face Authenticity Classifier via HuggingFace + FastAPI

> Upload any image → the system detects faces, crops them precisely, and classifies **REAL vs FAKE** using a fine-tuned deepfake detection model served via HuggingFace Inference API.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Face_Detection-5C3EE8?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-FFD21E?style=flat-square)

---

## 🎯 Overview

DeepFake Detector is a full-stack AI app with a **FastAPI backend** and **Streamlit frontend**. Before calling the model, it locally detects and crops the largest face using OpenCV (with margin padding), resizes to `224×224`, then forwards the face crop to `prithivMLmods/deepfake-detector-model-v1` on HuggingFace for classification.

---

## ✨ Features

- 👤 **Local Face Detection** — OpenCV Haar cascade detects largest face before API call
- ✂️ **Smart Face Cropping** — crops with margin, resizes to model-expected `224×224`
- 🤖 **HuggingFace Inference** — lightweight, no GPU needed on local machine
- ⚡ **FastAPI Backend** — async endpoints with full validation
- 🖼️ **Streamlit UI** — drag-and-drop image upload with confidence score display

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Face Detection | OpenCV (Haar Cascade) |
| Model | prithivMLmods/deepfake-detector-model-v1 (HuggingFace) |
| Language | Python 3.10+ |

---

## 🚀 Quick Start

```bash
git clone https://github.com/devcode-X/deepfake-ai
cd deepfake-ai

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Set environment variable
set HF_TOKEN=your_huggingface_token_here

# Run backend
.\venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000

# Run frontend (new terminal)
.\venv\Scripts\python -m streamlit run frontend/app.py
```

Then open `http://localhost:8501` and upload an image.

---

## 📁 Structure

```
deepfake-ai/
├── backend/
│   └── main.py          # FastAPI app, face detection, HF API call
├── frontend/
│   └── app.py           # Streamlit UI
├── requirements.txt
└── README.md
```
