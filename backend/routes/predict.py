"""Prediction routes for image deepfake detection."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.schemas.response import PredictionResponse
from backend.services.model_service import ModelServiceError, predict
from backend.utils.image_utils import NoFaceDetectedError, validate_image_bytes

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify an uploaded image as REAL or FAKE",
    responses={
        400: {"description": "Invalid file type or corrupt image"},
        500: {"description": "Backend configuration error"},
        502: {"description": "Upstream Hugging Face API error"},
        503: {"description": "Upstream model loading or timeout"},
    },
)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    """Validate an uploaded image and return the deepfake prediction."""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if suffix not in ALLOWED_EXTENSIONS or content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only jpg, jpeg, png, and webp images are supported.",
        )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read the uploaded file: {exc}",
        ) from exc

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        validate_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        return await run_in_threadpool(predict, image_bytes)
    except NoFaceDetectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ModelServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected inference error: {exc}",
        ) from exc
