"""Presentation helpers for model inference results."""

from typing import Any

import streamlit as st


def render_result_card(result: dict[str, Any]) -> None:
    """Display the final verdict, reasoning, and ensemble evidence."""
    label = str(result.get("label", "UNKNOWN")).upper()
    confidence = float(result.get("confidence", 0.0))
    all_scores = result.get("all_scores", [])
    model = str(result.get("model", "Unknown model"))
    justification = str(result.get("justification", "")).strip()
    artifact_observations = [
        str(item) for item in result.get("artifact_observations", []) if str(item).strip()
    ]
    ensemble_consideration = str(result.get("ensemble_consideration", "")).strip()
    ensemble_results = result.get("ensemble_results", [])
    preprocessing = result.get("preprocessing", {})

    st.subheader("Analysis Result")
    st.caption(f"Final judge model: `{model}`")

    if label == "REAL":
        st.success("\u2705 This image appears to be REAL")
    elif label == "FAKE":
        st.error("\u26A0\uFE0F This image appears to be a DEEPFAKE")
    else:
        st.warning(f"Unexpected model label: {label}")

    st.progress(min(max(confidence, 0.0), 1.0), text=f"{confidence * 100:.1f}% confidence")

    if justification:
        st.markdown("**Reasoning**")
        st.write(justification)

    if artifact_observations:
        st.markdown("**Artifact observations**")
        for observation in artifact_observations:
            st.write(f"- {observation}")

    if ensemble_consideration:
        st.markdown("**Ensemble consideration**")
        st.write(ensemble_consideration)

    if all_scores:
        rows = []
        for entry in all_scores:
            score = float(entry.get("score", 0.0))
            rows.append(
                {
                    "Label": str(entry.get("label", "UNKNOWN")),
                    "Probability": f"{score * 100:.2f}%",
                }
            )
        st.caption("All label scores")
        st.table(rows)

    if preprocessing:
        face_width = preprocessing.get("face_width")
        face_height = preprocessing.get("face_height")
        laplacian_variance = preprocessing.get("laplacian_variance")
        st.caption(
            f"Face crop: {face_width}x{face_height} | Laplacian variance: {laplacian_variance}"
        )

    if ensemble_results:
        st.markdown("**Ensemble model results**")
        rows = []
        for entry in ensemble_results:
            fake_probability = entry.get("fake_probability")
            rows.append(
                {
                    "Model": str(entry.get("model_name", "Unknown")),
                    "Status": str(entry.get("status", "unknown")),
                    "Fake Probability": (
                        f"{float(fake_probability) * 100:.2f}%"
                        if fake_probability is not None
                        else "-"
                    ),
                    "Error": str(entry.get("error", "")) if entry.get("error") else "-",
                }
            )
        st.table(rows)
