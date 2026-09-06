"""Modal deployment entrypoint for the private BLIP caption API."""

from __future__ import annotations

import modal

APP_NAME = "chest-xray-blip-api"
API_SECRET_NAME = "chest-xray-blip-api-auth"
BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"


def download_base_model() -> None:
    """Bake the public BLIP base weights into the image to reduce cold starts."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    BlipProcessor.from_pretrained(BASE_MODEL_ID)
    BlipForConditionalGeneration.from_pretrained(BASE_MODEL_ID)


runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .uv_pip_install(
        "fastapi>=0.110,<1",
        "python-multipart>=0.0.9,<1",
        "pillow>=10,<13",
        "torch>=2.2,<3",
        "transformers>=4.40,<6",
        "peft>=0.19,<1",
    )
    .run_function(download_base_model, cpu=2.0, memory=4096, timeout=1200)
    .add_local_dir("app", "/root/app")
    .add_local_dir("static", "/root/static")
    .add_local_dir("model", "/root/model")
)

app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    secrets=[modal.Secret.from_name(API_SECRET_NAME)],
    cpu=2.0,
    memory=4096,
    timeout=600,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(label="chest-xray-blip-api")
def fastapi_app():
    """Expose the existing FastAPI application as a Modal Web Function."""
    import os
    import sys

    os.chdir("/root")
    sys.path.insert(0, "/root")

    from app.main import app as api

    return api
