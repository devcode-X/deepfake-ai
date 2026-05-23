"""Utilities for validating and preparing uploaded images."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class NoFaceDetectedError(ValueError):
    """Raised when no usable face can be detected in the uploaded image."""


@dataclass(frozen=True)
class FaceBoundingBox:
    """Represents a detected face bounding box."""

    x: int
    y: int
    width: int
    height: int


def validate_image_bytes(data: bytes) -> Image.Image:
    """Open raw bytes as an RGB PIL image or raise a clear validation error."""
    if not data:
        raise ValueError("No image data was provided.")

    try:
        probe = Image.open(BytesIO(data))
        probe.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise ValueError(f"Unable to process the image: {exc}") from exc

    return image


def detect_largest_face(image: Image.Image) -> FaceBoundingBox | None:
    """Detect the largest frontal face in the given image."""
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: int(face[2] * face[3]))
    return FaceBoundingBox(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
    )


def crop_face_with_margin(
    image: Image.Image,
    face_box: FaceBoundingBox,
    margin_ratio: float = 0.18,
) -> Image.Image:
    """Crop the detected face with a small margin while staying inside image bounds."""
    margin_x = int(face_box.width * margin_ratio)
    margin_y = int(face_box.height * margin_ratio)

    left = max(face_box.x - margin_x, 0)
    top = max(face_box.y - margin_y, 0)
    right = min(face_box.x + face_box.width + margin_x, image.width)
    bottom = min(face_box.y + face_box.height + margin_y, image.height)

    return image.crop((left, top, right, bottom))


def resize_image(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize an image to the model's expected input resolution."""
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_face_image_for_inference(
    data: bytes,
    size: tuple[int, int] = (224, 224),
) -> bytes:
    """Validate an image, extract the largest face, resize it, and encode as JPEG bytes."""
    image = validate_image_bytes(data)
    face_box = detect_largest_face(image)
    if face_box is None:
        raise NoFaceDetectedError("No face detected.")

    face_crop = crop_face_with_margin(image, face_box)
    resized_face = resize_image(face_crop, size=size)

    buffer = BytesIO()
    resized_face.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()
