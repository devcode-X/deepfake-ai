"""Pydantic response models returned by the API."""

from pydantic import BaseModel, Field


class ScoreEntry(BaseModel):
    """A normalized label and its probability score."""

    label: str = Field(..., description="Returned class label.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability score between 0 and 1.",
    )


class EnsembleResultEntry(BaseModel):
    """One classifier call inside the ensemble stage."""

    model_name: str = Field(..., description="Short model identifier.")
    api_url: str = Field(..., description="Inference endpoint used for this model.")
    status: str = Field(..., description="ok, failed, or skipped.")
    fake_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Parsed fake probability from this model when available.",
    )
    scores: list[ScoreEntry] = Field(
        default_factory=list,
        description="Raw normalized scores returned by the classifier.",
    )
    error: str | None = Field(
        default=None,
        description="Failure detail when the classifier did not return a usable result.",
    )


class PreprocessingResponse(BaseModel):
    """Useful local preprocessing metadata from the face-first pipeline."""

    face_width: int = Field(..., ge=1, description="Width of the final cropped face image.")
    face_height: int = Field(..., ge=1, description="Height of the final cropped face image.")
    laplacian_variance: float = Field(
        ...,
        ge=0.0,
        description="Variance of the Laplacian residual image.",
    )


class PredictionResponse(BaseModel):
    """Structured response for the ensemble deepfake pipeline."""

    label: str = Field(..., description="Final REAL or FAKE label from the VLM judge.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final confidence score between 0 and 1.",
    )
    all_scores: list[ScoreEntry] = Field(
        default_factory=list,
        description="Aggregate REAL/FAKE score view derived from the successful ensemble calls.",
    )
    model: str = Field(..., description="Hugging Face VLM used for the final judgment.")
    justification: str = Field(
        ...,
        description="Short explanation describing the final verdict.",
    )
    artifact_observations: list[str] = Field(
        default_factory=list,
        description="Observed visual artifacts called out by the VLM.",
    )
    ensemble_consideration: str = Field(
        ...,
        description="How the VLM used the ensemble scores in the final decision.",
    )
    ensemble_results: list[EnsembleResultEntry] = Field(
        default_factory=list,
        description="Per-model classifier outcomes from the ensemble stage.",
    )
    preprocessing: PreprocessingResponse = Field(
        ...,
        description="Metadata from the local face preprocessing pipeline.",
    )
