"""
API tests. The captioning model itself is mocked so these tests run without
downloading BLIP weights or needing a GPU.
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.model import caption_model


def _fake_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


client = TestClient(app)


def test_health_reports_unloaded_model_gracefully():
    caption_model.model = None
    caption_model._load_error = "model not loaded in test"  # noqa: SLF001
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_caption_rejects_non_image_upload():
    res = client.post(
        "/caption",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert res.status_code == 400


def test_caption_returns_503_when_model_not_loaded():
    caption_model.model = None
    res = client.post(
        "/caption",
        files={"file": ("xray.png", _fake_png_bytes(), "image/png")},
    )
    assert res.status_code == 503


def test_caption_returns_prediction_when_model_mocked(monkeypatch):
    monkeypatch.setattr(caption_model, "model", object())  # any non-None sentinel
    monkeypatch.setattr(
        caption_model,
        "caption",
        lambda image, max_new_tokens=60: ("chest x-ray showing no acute findings", 12.3),
    )

    res = client.post(
        "/caption",
        files={"file": ("xray.png", _fake_png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["caption"] == "chest x-ray showing no acute findings"
    assert "disclaimer" in body
    assert "not" in body["disclaimer"].lower()
