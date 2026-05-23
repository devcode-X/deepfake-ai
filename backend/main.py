"""FastAPI entry point for the DeepFake detector backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.predict import router as predict_router
from backend.services.model_service import load_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the Hugging Face client once before serving requests."""
    load_model()
    yield


app = FastAPI(
    title="DeepFake Detector API",
    description=(
        "Upload an image and classify it as REAL or FAKE with a Hugging Face "
        "deepfake model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    """Return a simple status payload for monitoring and UI checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
