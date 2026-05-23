"""Streamlit user interface for the DeepFake detector."""

import streamlit as st

from frontend.components.result_card import render_result_card
from frontend.components.uploader import render_image_preview, render_uploader
from frontend.utils.api_client import (
    BackendConnectionError,
    BackendError,
    BackendTimeoutError,
    health_check,
    predict,
)

st.set_page_config(
    page_title="\U0001F50D DeepFake Detector",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg-main: #08111f;
        --bg-panel: #0f1d33;
        --bg-soft: #132743;
        --text-main: #edf2ff;
        --text-soft: #9fb3d1;
        --accent: #3aa3ff;
        --accent-strong: #71d8ff;
        --border: rgba(113, 216, 255, 0.18);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(58, 163, 255, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(113, 216, 255, 0.12), transparent 28%),
            linear-gradient(180deg, var(--bg-main) 0%, #050b14 100%);
        color: var(--text-main);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1425 0%, #08101d 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(113, 216, 255, 0.35);
        border-radius: 18px;
        background: rgba(19, 39, 67, 0.55);
        padding: 0.75rem;
    }

    .hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid var(--border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15, 29, 51, 0.96), rgba(8, 17, 31, 0.92));
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.8rem;
        line-height: 1.05;
    }

    .hero p {
        margin: 0.75rem 0 0 0;
        color: var(--text-soft);
        font-size: 1.02rem;
    }

    .signature-card {
        margin: 1rem 0 1.25rem 0;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(113, 216, 255, 0.28);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(58, 163, 255, 0.18), rgba(15, 29, 51, 0.92));
        color: var(--text-main);
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        text-align: center;
    }

    .devansh-watermark {
        position: fixed;
        right: 18px;
        bottom: 14px;
        z-index: 99999;
        padding: 0.65rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(113, 216, 255, 0.35);
        background: rgba(8, 17, 31, 0.88);
        color: #edf2ff;
        font-weight: 700;
        letter-spacing: 0.04em;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("\U0001F50D DeepFake Detector")
    st.markdown("### About")
    st.write(
        "Upload an image and check whether its deepfake or real, made with love for your security."
    )
    st.markdown(
        '<div class="signature-card">Made by Devansh</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Backend Status")
    if health_check():
        st.success("Backend is reachable at `:8000`.")
    else:
        st.error(
            "Backend is offline. Start `venv\\Scripts\\python -m uvicorn backend.main:app --reload --port 8000`."
        )

st.markdown(
    """
    <div class="hero">
        <h1>\U0001F50D DeepFake Detector</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="devansh-watermark">Made by Devansh</div>',
    unsafe_allow_html=True,
)

left_gutter, main_column, right_gutter = st.columns([0.6, 2.4, 0.6])

with main_column:
    uploaded_file = render_uploader()

    if uploaded_file is not None:
        render_image_preview(uploaded_file, max_width=400)

    analyze_clicked = st.button("Analyze Image", use_container_width=True)

    if analyze_clicked and uploaded_file is None:
        st.warning("Upload an image before starting the analysis.")
    elif analyze_clicked and uploaded_file is not None:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        with st.spinner("Analyzing image..."):
            try:
                result = predict(image_bytes=image_bytes, filename=uploaded_file.name)
                render_result_card(result)
            except BackendTimeoutError as exc:
                st.warning(str(exc))
            except BackendConnectionError as exc:
                st.error(str(exc))
            except BackendError as exc:
                st.error(exc.detail)
            except Exception as exc:
                st.error(f"Unexpected frontend error: {exc}")
