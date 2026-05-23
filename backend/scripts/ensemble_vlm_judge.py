"""Run a face-first deepfake workflow with local artifacts, HF ensembles, and a VLM judge.

This script is CPU friendly locally:
1. Detect the largest face with OpenCV.
2. Crop with a margin and resize to 224x224.
3. Generate two forensic views:
   - Error Level Analysis (ELA)
   - High-pass / Laplacian residual image
4. Send the cropped face to up to three Hugging Face image-classification endpoints.
5. Send the cropped face, forensic views, and ensemble scores to a Hugging Face VLM
   that acts as the final judge.

Heavy inference stays remote on Hugging Face. The local machine only does image
preprocessing and artifact generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import argparse
import base64
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageChops
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.image_utils import (  # noqa: E402
    NoFaceDetectedError,
    crop_face_with_margin,
    detect_largest_face,
    resize_image,
    validate_image_bytes,
)

load_dotenv(ROOT / "backend" / ".env", override=True)

VLM_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_VLM_MODEL = os.environ.get("HF_VLM_MODEL", "CohereLabs/aya-vision-32b:cohere")
DEFAULT_CLASSIFIER_URLS = [
    os.environ.get("HF_MODEL_URL_1", "").strip(),
    os.environ.get("HF_MODEL_URL_2", "").strip(),
    os.environ.get("HF_MODEL_URL_3", "").strip(),
]
DEFAULT_CLASSIFIER_TIMEOUT = 60
DEFAULT_VLM_TIMEOUT = 90
PLACEHOLDER_TOKENS = {
    "",
    "HF_MODEL_URL_1",
    "HF_MODEL_URL_2",
    "HF_MODEL_URL_3",
    "https://router.huggingface.co/hf-inference/models/your-model-1",
    "https://router.huggingface.co/hf-inference/models/your-model-2",
    "https://router.huggingface.co/hf-inference/models/your-model-3",
}
FAKE_LABEL_HINTS = {
    "fake",
    "deepfake",
    "ai-generated",
    "synthetic",
    "manipulated",
    "forged",
}
REAL_LABEL_HINTS = {
    "real",
    "realism",
    "authentic",
    "genuine",
    "original",
    "human",
}


@dataclass
class EnsembleModelResult:
    """Represents the outcome of one HF classifier call."""

    model_name: str
    api_url: str
    status: str
    fake_probability: float | None
    scores: list[dict[str, float]]
    error: str | None = None


def get_hf_token() -> str:
    """Load the Hugging Face token from the shared backend env config."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN, HF_API_TOKEN, or HUGGINGFACE_API_TOKEN before running the script."
        )
    return token


def extract_face_crop(image_bytes: bytes, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Validate the input image, detect the largest face, and return a resized face crop."""
    image = validate_image_bytes(image_bytes)
    face_box = detect_largest_face(image)
    if face_box is None:
        raise NoFaceDetectedError("No face detected.")

    face_crop = crop_face_with_margin(image, face_box, margin_ratio=0.18)
    return resize_image(face_crop, size=size)


def create_ela_image(image: Image.Image, jpeg_quality: int = 90) -> Image.Image:
    """Create an Error Level Analysis image that amplifies JPEG recompression artifacts."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    recompressed = Image.open(BytesIO(buffer.getvalue())).convert("RGB")
    difference = ImageChops.difference(image.convert("RGB"), recompressed)

    max_diff = max(channel_max for _, channel_max in difference.getextrema()) or 1
    scale = 255.0 / max_diff
    return difference.point(lambda value: int(value * scale))


def create_high_pass_image(image: Image.Image) -> tuple[Image.Image, float]:
    """Create a normalized Laplacian residual image and return its variance."""
    image_array = np.array(image.convert("RGB"))
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())

    residual = cv2.convertScaleAbs(laplacian)
    normalized = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(normalized), laplacian_variance


def pil_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """Serialize a PIL image into JPEG bytes for API payloads."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_to_data_url(image: Image.Image, mime_type: str = "image/jpeg") -> str:
    """Convert a PIL image into a base64 data URL for multimodal chat payloads."""
    encoded = base64.b64encode(pil_to_jpeg_bytes(image)).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def maybe_save_debug_images(
    output_dir: Path | None,
    face_image: Image.Image,
    ela_image: Image.Image,
    high_pass_image: Image.Image,
) -> dict[str, str]:
    """Optionally write the generated forensic views to disk for inspection."""
    if output_dir is None:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    face_path = output_dir / "face_crop.jpg"
    ela_path = output_dir / "face_ela.jpg"
    high_pass_path = output_dir / "face_high_pass.jpg"

    face_image.save(face_path, format="JPEG", quality=95)
    ela_image.save(ela_path, format="JPEG", quality=95)
    high_pass_image.save(high_pass_path, format="JPEG", quality=95)

    return {
        "face_crop": str(face_path),
        "ela": str(ela_path),
        "high_pass": str(high_pass_path),
    }


def model_name_from_url(api_url: str) -> str:
    """Extract a readable model identifier from a HF inference URL."""
    return api_url.rstrip("/").split("/")[-1] or "unknown-model"


def extract_fake_probability(payload: Any) -> tuple[float, list[dict[str, float]]]:
    """Normalize image-classification scores and extract a fake probability."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("Model response did not contain a score list.")

    normalized_scores: list[dict[str, float]] = []
    fake_score: float | None = None
    real_score: float | None = None

    for item in payload:
        label = str(item.get("label", "")).strip()
        score = float(item.get("score", 0.0))
        normalized_scores.append({"label": label, "score": round(score, 6)})

        lowered_label = label.lower()
        if any(hint in lowered_label for hint in FAKE_LABEL_HINTS):
            fake_score = max(fake_score or 0.0, score)
        if any(hint in lowered_label for hint in REAL_LABEL_HINTS):
            real_score = max(real_score or 0.0, score)

    normalized_scores.sort(key=lambda item: item["score"], reverse=True)

    if fake_score is not None:
        return round(fake_score, 6), normalized_scores
    if real_score is not None:
        return round(1.0 - real_score, 6), normalized_scores

    raise ValueError(
        "The model response did not include a recognizable fake/real label."
    )


def call_classifier_api(
    api_url: str,
    image_bytes: bytes,
    token: str,
    timeout_seconds: int,
) -> EnsembleModelResult:
    """Submit the cropped face to a HF image-classification endpoint."""
    model_name = model_name_from_url(api_url) if api_url else "unconfigured-model"

    if api_url in PLACEHOLDER_TOKENS:
        return EnsembleModelResult(
            model_name=model_name,
            api_url=api_url,
            status="skipped",
            fake_probability=None,
            scores=[],
            error="Model URL is not configured. Replace the HF_MODEL_URL_* placeholder.",
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "image/jpeg",
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=image_bytes,
            timeout=timeout_seconds,
        )
    except requests.Timeout:
        return EnsembleModelResult(
            model_name=model_name,
            api_url=api_url,
            status="failed",
            fake_probability=None,
            scores=[],
            error=f"Timed out after {timeout_seconds}s.",
        )
    except requests.RequestException as exc:
        return EnsembleModelResult(
            model_name=model_name,
            api_url=api_url,
            status="failed",
            fake_probability=None,
            scores=[],
            error=f"Request failed: {exc}",
        )

    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    if response.status_code != 200:
        error_message = payload if isinstance(payload, str) else json.dumps(payload)
        return EnsembleModelResult(
            model_name=model_name,
            api_url=api_url,
            status="failed",
            fake_probability=None,
            scores=[],
            error=f"HTTP {response.status_code}: {error_message}",
        )

    try:
        fake_probability, normalized_scores = extract_fake_probability(payload)
    except Exception as exc:
        return EnsembleModelResult(
            model_name=model_name,
            api_url=api_url,
            status="failed",
            fake_probability=None,
            scores=[],
            error=f"Could not parse classifier response: {exc}",
        )

    return EnsembleModelResult(
        model_name=model_name,
        api_url=api_url,
        status="ok",
        fake_probability=fake_probability,
        scores=normalized_scores,
    )


def summarize_ensemble_results(results: list[EnsembleModelResult]) -> str:
    """Create a plain-text summary that can be injected into the VLM prompt."""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        if result.status == "ok":
            lines.append(
                f"Model {index} ({result.model_name}) fake_probability={result.fake_probability:.6f}; "
                f"scores={json.dumps(result.scores)}"
            )
        else:
            lines.append(
                f"Model {index} ({result.model_name}) status={result.status}; error={result.error}"
            )

    successful_scores = [
        result.fake_probability for result in results if result.fake_probability is not None
    ]
    if successful_scores:
        lines.append(
            f"Successful ensemble fake_probability mean={statistics.mean(successful_scores):.6f}"
        )
    else:
        lines.append("No classifier returned a usable fake probability.")

    return "\n".join(lines)


def build_vlm_messages(
    face_data_url: str,
    ela_data_url: str,
    high_pass_data_url: str,
    ensemble_results_text: str,
    laplacian_variance: float,
) -> list[dict[str, Any]]:
    """Build the multimodal chat payload for the final Hugging Face VLM."""
    system_prompt = (
        "You are a forensic image analyst acting as the final judge in a deepfake "
        "screening pipeline. Weigh visible facial artifacts, ELA inconsistencies, "
        "high-frequency residual patterns, and the ensemble classifier scores. "
        "Return only JSON."
    )

    user_text = f"""
You are given three views of the same detected face:
1. The original cropped face.
2. An Error Level Analysis (ELA) view that exposes compression inconsistencies.
3. A high-pass / Laplacian residual view that highlights fine-grained texture artifacts.

The local preprocessing resized the face to 224x224. The Laplacian variance for the
residual image is {laplacian_variance:.6f}.

Ensemble classifier evidence:
{ensemble_results_text}

Instructions:
- Inspect the original crop for unnatural blending, symmetry artifacts, warped facial regions,
  unrealistic skin texture, mismatched lighting, or inconsistent boundaries.
- Inspect the ELA view for localized compression spikes or patchy regions around the eyes,
  mouth, hairline, cheeks, or jaw.
- Inspect the high-pass residual for suspicious sharpening halos, missing natural pores,
  or inconsistent frequency structure.
- Use the ensemble scores as supporting evidence, not as the only reason.
- If classifier calls failed, still make a best-effort judgment from the images.

Return JSON with exactly these fields:
{{
  "classification": "real" | "fake",
  "confidence": 0.0 to 1.0,
  "justification": "short but specific explanation grounded in the visible evidence",
  "artifact_observations": ["observation 1", "observation 2"],
  "ensemble_consideration": "how the model scores influenced or did not influence the final decision"
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": "Image A: original cropped face."},
                {"type": "image_url", "image_url": {"url": face_data_url}},
                {"type": "text", "text": "Image B: ELA artifact visualization."},
                {"type": "image_url", "image_url": {"url": ela_data_url}},
                {"type": "text", "text": "Image C: high-pass / Laplacian residual view."},
                {"type": "image_url", "image_url": {"url": high_pass_data_url}},
            ],
        },
    ]


def build_vlm_payload(
    model_name: str,
    messages: list[dict[str, Any]],
    use_schema: bool = True,
) -> dict[str, Any]:
    """Build the OpenAI-compatible Hugging Face chat-completions payload."""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 700,
    }

    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "deepfake_verdict",
                "schema": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["real", "fake"],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "justification": {"type": "string"},
                        "artifact_observations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "ensemble_consideration": {"type": "string"},
                    },
                    "required": [
                        "classification",
                        "confidence",
                        "justification",
                        "artifact_observations",
                        "ensemble_consideration",
                    ],
                    "additionalProperties": False,
                },
            },
        }

    return payload


def extract_text_content(message_content: Any) -> str:
    """Flatten text content from a chat response message."""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return "\n".join(
            part.get("text", "")
            for part in message_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message_content)


def extract_first_json_object(text: str) -> str:
    """Extract the first JSON object from a text blob."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rstrip("`").strip()

    try:
        json.loads(stripped)
        return stripped
    except Exception:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object was found in the VLM response.")

    depth = 0
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    raise ValueError("Could not find a complete JSON object in the VLM response.")


def call_vlm_judge(
    face_data_url: str,
    ela_data_url: str,
    high_pass_data_url: str,
    ensemble_results: list[EnsembleModelResult],
    laplacian_variance: float,
    token: str,
    model_name: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Ask a Hugging Face VLM to make the final judgment from images and scores."""
    messages = build_vlm_messages(
        face_data_url=face_data_url,
        ela_data_url=ela_data_url,
        high_pass_data_url=high_pass_data_url,
        ensemble_results_text=summarize_ensemble_results(ensemble_results),
        laplacian_variance=laplacian_variance,
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = build_vlm_payload(model_name=model_name, messages=messages, use_schema=True)

    try:
        response = requests.post(
            VLM_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise RuntimeError(f"VLM call timed out after {timeout_seconds}s.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM request failed: {exc}") from exc

    # Some providers reject structured output even when the chat endpoint itself works.
    if response.status_code >= 400:
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = response.text

        error_text = (
            response_payload if isinstance(response_payload, str) else json.dumps(response_payload)
        )
        if "response_format" in error_text.lower() or "json_schema" in error_text.lower():
            fallback_payload = build_vlm_payload(
                model_name=model_name,
                messages=messages,
                use_schema=False,
            )
            response = requests.post(
                VLM_API_URL,
                headers=headers,
                json=fallback_payload,
                timeout=timeout_seconds,
            )
        else:
            raise RuntimeError(f"VLM HTTP {response.status_code}: {error_text}")

    if response.status_code != 200:
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = response.text
        raise RuntimeError(
            f"VLM HTTP {response.status_code}: "
            f"{response_payload if isinstance(response_payload, str) else json.dumps(response_payload)}"
        )

    try:
        response_payload = response.json()
        message = response_payload["choices"][0]["message"]
    except Exception as exc:
        raise RuntimeError("VLM response payload was missing the expected chat structure.") from exc

    raw_text = extract_text_content(message.get("content", ""))

    try:
        return json.loads(extract_first_json_object(raw_text))
    except Exception as exc:
        raise RuntimeError(
            "VLM response could not be parsed as JSON. Raw content:\n"
            f"{raw_text}"
        ) from exc


def run_pipeline(
    image_path: Path,
    classifier_urls: list[str],
    vlm_model_name: str,
    classifier_timeout: int,
    vlm_timeout: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full local-preprocess -> ensemble -> VLM flow and return JSON-ready data."""
    token = get_hf_token()
    image_bytes = image_path.read_bytes()
    face_image = extract_face_crop(image_bytes)
    ela_image = create_ela_image(face_image)
    high_pass_image, laplacian_variance = create_high_pass_image(face_image)

    saved_files = maybe_save_debug_images(
        output_dir=output_dir,
        face_image=face_image,
        ela_image=ela_image,
        high_pass_image=high_pass_image,
    )

    face_bytes = pil_to_jpeg_bytes(face_image)
    ensemble_results = [
        call_classifier_api(
            api_url=api_url,
            image_bytes=face_bytes,
            token=token,
            timeout_seconds=classifier_timeout,
        )
        for api_url in classifier_urls
    ]

    vlm_result = call_vlm_judge(
        face_data_url=image_to_data_url(face_image),
        ela_data_url=image_to_data_url(ela_image),
        high_pass_data_url=image_to_data_url(high_pass_image),
        ensemble_results=ensemble_results,
        laplacian_variance=laplacian_variance,
        token=token,
        model_name=vlm_model_name,
        timeout_seconds=vlm_timeout,
    )

    return {
        "input_image": str(image_path),
        "preprocessing": {
            "face_size": {"width": face_image.width, "height": face_image.height},
            "laplacian_variance": round(laplacian_variance, 6),
            "saved_debug_files": saved_files,
        },
        "ensemble_results": [asdict(result) for result in ensemble_results],
        "vlm_model": vlm_model_name,
        "final_judgment": vlm_result,
    }


def run_pipeline_from_bytes(
    image_bytes: bytes,
    classifier_urls: list[str],
    vlm_model_name: str,
    classifier_timeout: int,
    vlm_timeout: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the same workflow directly from in-memory image bytes."""
    token = get_hf_token()
    face_image = extract_face_crop(image_bytes)
    ela_image = create_ela_image(face_image)
    high_pass_image, laplacian_variance = create_high_pass_image(face_image)

    saved_files = maybe_save_debug_images(
        output_dir=output_dir,
        face_image=face_image,
        ela_image=ela_image,
        high_pass_image=high_pass_image,
    )

    face_bytes = pil_to_jpeg_bytes(face_image)
    ensemble_results = [
        call_classifier_api(
            api_url=api_url,
            image_bytes=face_bytes,
            token=token,
            timeout_seconds=classifier_timeout,
        )
        for api_url in classifier_urls
    ]

    vlm_result = call_vlm_judge(
        face_data_url=image_to_data_url(face_image),
        ela_data_url=image_to_data_url(ela_image),
        high_pass_data_url=image_to_data_url(high_pass_image),
        ensemble_results=ensemble_results,
        laplacian_variance=laplacian_variance,
        token=token,
        model_name=vlm_model_name,
        timeout_seconds=vlm_timeout,
    )

    return {
        "preprocessing": {
            "face_size": {"width": face_image.width, "height": face_image.height},
            "laplacian_variance": round(laplacian_variance, 6),
            "saved_debug_files": saved_files,
        },
        "ensemble_results": [asdict(result) for result in ensemble_results],
        "vlm_model": vlm_model_name,
        "final_judgment": vlm_result,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run local forensic preprocessing, a HF classifier ensemble, and a HF VLM judge."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--save-debug-dir",
        type=Path,
        default=None,
        help="Optional directory to save the cropped face, ELA image, and high-pass image.",
    )
    parser.add_argument(
        "--classifier-url",
        action="append",
        dest="classifier_urls",
        default=[],
        help=(
            "HF image-classification API URL. Supply up to three of these, or set "
            "HF_MODEL_URL_1/HF_MODEL_URL_2/HF_MODEL_URL_3 in the environment."
        ),
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help=(
            "HF chat-completions VLM model name. Default: "
            f"{DEFAULT_VLM_MODEL}"
        ),
    )
    parser.add_argument(
        "--classifier-timeout",
        type=int,
        default=DEFAULT_CLASSIFIER_TIMEOUT,
        help=f"Timeout in seconds for each classifier call. Default: {DEFAULT_CLASSIFIER_TIMEOUT}.",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=int,
        default=DEFAULT_VLM_TIMEOUT,
        help=f"Timeout in seconds for the VLM call. Default: {DEFAULT_VLM_TIMEOUT}.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    classifier_urls = args.classifier_urls[:3] if args.classifier_urls else DEFAULT_CLASSIFIER_URLS

    try:
        result = run_pipeline(
            image_path=args.image_path,
            classifier_urls=classifier_urls,
            vlm_model_name=args.vlm_model,
            classifier_timeout=args.classifier_timeout,
            vlm_timeout=args.vlm_timeout,
            output_dir=args.save_debug_dir,
        )
    except NoFaceDetectedError:
        print("No face detected")
        raise SystemExit(0)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
