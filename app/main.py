"""
Chest X-ray caption drafting API.

Research prototype. Serves a BLIP image-captioning model adapted with a LoRA
adapter fine-tuned on a filtered, chest/thorax subset of ROCOv2-radiology.

This tool drafts free-text captions from chest X-ray images. It does not
diagnose, triage, or replace clinical judgment. See README.md and
model/roco_chest_xray_lora/README.md for training data, evaluation loss,
and limitations.
"""

from __future__ import annotations

import io
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.model import ADAPTER_DIR, BASE_MODEL_ID, caption_model
from app.schemas import CaptionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Attempt to load the model at startup. Failure does not crash the app."""
    caption_model.load()
    yield


app = FastAPI(
    title="Chest X-ray Caption Drafting API",
    description=(
        "Research prototype for drafting chest X-ray captions with a "
        "BLIP + LoRA model. NOT for clinical use."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app.mount("/static", StaticFiles(directory="static"), name="static")


def require_api_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require the production bearer token when BLIP_API_TOKEN is configured."""
    api_token = os.environ.get("BLIP_API_TOKEN", "").strip()
    if not api_token:
        return

    expected = f"Bearer {api_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get(
    "/health",
    response_model=HealthResponse,
    dependencies=[Depends(require_api_token)],
)
async def health() -> HealthResponse:
    """Report whether the model actually loaded, so failures are visible, not silent."""
    return HealthResponse(
        status="ok" if caption_model.is_loaded else "degraded",
        model_loaded=caption_model.is_loaded,
        base_model=BASE_MODEL_ID,
        adapter_path=str(ADAPTER_DIR),
        device=caption_model.device,
        detail=None if caption_model.is_loaded else caption_model.load_error,
    )


@app.post(
    "/caption",
    response_model=CaptionResponse,
    dependencies=[Depends(require_api_token)],
)
async def caption_xray(file: UploadFile = File(...)) -> CaptionResponse:
    """
    Generate a draft caption for an uploaded chest X-ray image.

    This is a research prototype: the output is an unreviewed AI draft,
    not a radiology report.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not read image file.") from exc

    if not caption_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Caption model is not loaded. Check GET /health for details. "
                f"Last error: {caption_model.load_error}"
            ),
        )

    try:
        caption_text, elapsed_ms = caption_model.caption(image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {exc}") from exc

    return CaptionResponse(
        caption=caption_text,
        model_id=f"{BASE_MODEL_ID}+lora:roco_chest_xray_lora",
        generation_time_ms=round(elapsed_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
