"""Reusable uploader widgets for the Streamlit interface."""

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def render_uploader() -> UploadedFile | None:
    """Render the upload control and return the selected file if present."""
    st.subheader("Upload an Image")
    st.caption("Supported formats: jpg, jpeg, png, webp")
    return st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )


def render_image_preview(uploaded_file: UploadedFile, max_width: int = 400) -> None:
    """Render a constrained preview for the current uploaded image."""
    st.image(uploaded_file, width=max_width, caption=uploaded_file.name)
