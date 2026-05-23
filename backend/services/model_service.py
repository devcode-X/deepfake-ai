"""Service layer for the ensemble deepfake pipeline."""

from __future__ import annotations

from os import environ
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from backend.schemas.response import (
    EnsembleResultEntry,
    PredictionResponse,
    PreprocessingResponse,
    ScoreEntry,
)
from backend.scripts.ensemble_vlm_judge import (
    DEFAULT_CLASSIFIER_TIMEOUT,
    DEFAULT_VLM_TIMEOUT,
    PLACEHOLDER_TOKENS,
    run_pipeline_from_bytes,
)

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, override=True)

DEFAULT_CLASSIFIER_URLS = [
    "https://router.huggingface.co/hf-inference/models/prithivMLmods/deepfake-detector-model-v1",
    "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model",
    "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deepfake-Detect-Siglip2",
]
DEFAULT_VLM_MODEL = "CohereLabs/aya-vision-32b:cohere"

_config_lock = Lock()
_runtime_config: dict[str, object] | None = None


class ModelServiceError(RuntimeError):
    """Raised when the ensemble or VLM pipeline cannot satisfy a request."""

    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _get_token() -> str:
    """Return the configured Hugging Face token or raise a config error."""
    token = (
        environ.get("HF_TOKEN")
        or environ.get("HF_API_TOKEN")
        or environ.get("HUGGINGFACE_API_TOKEN")
        or ""
    )
    if not token:
        raise ModelServiceError(
            "HF_TOKEN is not configured for the backend.",
            status_code=500,
        )
    return token


def _read_classifier_urls() -> list[str]:
    """Load up to three classifier URLs from the environment, with sane defaults."""
    configured = [
        environ.get("HF_MODEL_URL_1", "").strip(),
        environ.get("HF_MODEL_URL_2", "").strip(),
        environ.get("HF_MODEL_URL_3", "").strip(),
    ]

    if any(url and url not in PLACEHOLDER_TOKENS for url in configured):
        return configured

    return DEFAULT_CLASSIFIER_URLS


def _aggregate_scores(ensemble_results: list[dict[str, object]]) -> list[ScoreEntry]:
    """Create a compact REAL/FAKE score view from the successful ensemble calls."""
    fake_probabilities = [
        float(item["fake_probability"])
        for item in ensemble_results
        if item.get("status") == "ok" and item.get("fake_probability") is not None
    ]

    if not fake_probabilities:
        return []

    average_fake_probability = round(sum(fake_probabilities) / len(fake_probabilities), 6)
    average_real_probability = round(1.0 - average_fake_probability, 6)
    return sorted(
        [
            ScoreEntry(label="REAL", score=average_real_probability),
            ScoreEntry(label="FAKE", score=average_fake_probability),
        ],
        key=lambda entry: entry.score,
        reverse=True,
    )


def load_model() -> dict[str, object]:
    """Initialize and cache the runtime configuration once."""
    global _runtime_config

    if _runtime_config is None:
        with _config_lock:
            if _runtime_config is None:
                _runtime_config = {
                    "token": _get_token(),
                    "classifier_urls": _read_classifier_urls(),
                    "vlm_model": environ.get("HF_VLM_MODEL", DEFAULT_VLM_MODEL),
                    "classifier_timeout": int(
                        environ.get("HF_CLASSIFIER_TIMEOUT", DEFAULT_CLASSIFIER_TIMEOUT)
                    ),
                    "vlm_timeout": int(environ.get("HF_VLM_TIMEOUT", DEFAULT_VLM_TIMEOUT)),
                }

    return _runtime_config


def _normalize_vlm_label(raw_label: str) -> str:
    """Normalize the VLM classification into REAL or FAKE."""
    lowered = raw_label.strip().lower()
    if lowered == "real":
        return "REAL"
    if lowered == "fake":
        return "FAKE"
    raise ModelServiceError(
        f"VLM returned an unsupported classification label '{raw_label}'.",
        status_code=502,
    )


def predict(image_bytes: bytes) -> PredictionResponse:
    """Run the full ensemble + VLM workflow and return an API response."""
    config = load_model()

    try:
        pipeline_result = run_pipeline_from_bytes(
            image_bytes=image_bytes,
            classifier_urls=list(config["classifier_urls"]),
            vlm_model_name=str(config["vlm_model"]),
            classifier_timeout=int(config["classifier_timeout"]),
            vlm_timeout=int(config["vlm_timeout"]),
        )
    except FileNotFoundError as exc:
        raise ModelServiceError(
            f"Local preprocessing dependency is missing: {exc}",
            status_code=500,
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        if "timed out" in lowered:
            raise ModelServiceError(message, status_code=503) from exc
        if "http 401" in lowered or "http 403" in lowered or "authentication" in lowered:
            raise ModelServiceError(message, status_code=502) from exc
        if "http 429" in lowered:
            raise ModelServiceError(message, status_code=503) from exc
        if "http 503" in lowered or "loading" in lowered:
            raise ModelServiceError(message, status_code=503) from exc
        raise ModelServiceError(message, status_code=502) from exc
    except Exception as exc:
        raise ModelServiceError(
            f"Unexpected pipeline failure: {exc}",
            status_code=500,
        ) from exc

    final_judgment = dict(pipeline_result["final_judgment"])
    ensemble_results = list(pipeline_result["ensemble_results"])
    preprocessing = dict(pipeline_result["preprocessing"])
    all_scores = _aggregate_scores(ensemble_results)

    confidence = round(float(final_judgment.get("confidence", 0.0)), 6)
    label = _normalize_vlm_label(str(final_judgment.get("classification", "")))

    return PredictionResponse(
        label=label,
        confidence=confidence,
        all_scores=all_scores,
        model=str(pipeline_result["vlm_model"]),
        justification=str(final_judgment.get("justification", "")),
        artifact_observations=[
            str(item) for item in final_judgment.get("artifact_observations", [])
        ],
        ensemble_consideration=str(
            final_judgment.get("ensemble_consideration", "")
        ),
        ensemble_results=[
            EnsembleResultEntry(
                model_name=str(item.get("model_name", "")),
                api_url=str(item.get("api_url", "")),
                status=str(item.get("status", "failed")),
                fake_probability=(
                    round(float(item["fake_probability"]), 6)
                    if item.get("fake_probability") is not None
                    else None
                ),
                scores=[
                    ScoreEntry(
                        label=str(score.get("label", "")),
                        score=round(float(score.get("score", 0.0)), 6),
                    )
                    for score in item.get("scores", [])
                ],
                error=str(item["error"]) if item.get("error") is not None else None,
            )
            for item in ensemble_results
        ],
        preprocessing=PreprocessingResponse(
            face_width=int(preprocessing["face_size"]["width"]),
            face_height=int(preprocessing["face_size"]["height"]),
            laplacian_variance=round(float(preprocessing["laplacian_variance"]), 6),
        ),
    )
