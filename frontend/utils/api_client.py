"""HTTP client helpers for calling the FastAPI backend from Streamlit."""

import os
from typing import Any

import requests

# Allow overriding via env-var; default to localhost.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 45
HEALTH_TIMEOUT_SECONDS = 5


class BackendError(Exception):
    """Raised when the backend returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Backend returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class BackendConnectionError(BackendError):
    """Raised when the Streamlit app cannot reach the backend service."""


class BackendTimeoutError(BackendError):
    """Raised when the backend request exceeds the configured timeout."""


def predict(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """Send an image to the backend prediction endpoint and parse the response."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    files = {
        "file": (
            filename,
            image_bytes,
            mime_types.get(extension, "image/jpeg"),
        )
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise BackendTimeoutError(
            f"The backend request timed out after {TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendConnectionError(
            f"Could not connect to the backend at {BACKEND_URL}."
        ) from exc

    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise BackendError(status_code=response.status_code, detail=detail)

    return response.json()


def health_check() -> bool:
    """Return True when the backend health endpoint is reachable."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=HEALTH_TIMEOUT_SECONDS)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _extract_error_detail(response: requests.Response) -> str:
    """Extract a readable error detail from the backend response body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unknown backend error."
    return str(payload.get("detail", "Unknown backend error."))
