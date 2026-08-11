"""
Loads Salesforce/blip-image-captioning-base plus the roco_chest_xray_lora
adapter, and exposes a single `caption()` method used by the API.

Model loading is lazy and defensive: if the base model can't be downloaded
(e.g. no internet access, no HF cache) the app should still start so that
/health can report a clear status instead of crashing the whole server.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "model" / "roco_chest_xray_lora"


class CaptionModel:
    """Thin wrapper around BLIP + the ROCOv2 chest X-ray LoRA adapter."""

    def __init__(self, base_model_id: str = BASE_MODEL_ID, adapter_path: Path = ADAPTER_DIR):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> None:
        """Load the base model and apply the LoRA adapter. Safe to call once at startup."""
        try:
            import torch
            from peft import PeftModel
            from transformers import BlipForConditionalGeneration, BlipProcessor

            if not self.adapter_path.exists():
                raise FileNotFoundError(f"Adapter directory not found: {self.adapter_path}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading base model %s ...", self.base_model_id)
            self.processor = BlipProcessor.from_pretrained(self.base_model_id)
            base_model = BlipForConditionalGeneration.from_pretrained(self.base_model_id)

            logger.info("Applying LoRA adapter from %s ...", self.adapter_path)
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model ready on device=%s", self.device)
            self._load_error = None
        except Exception as exc:  # noqa: BLE001 - we want to report *any* load failure
            self._load_error = str(exc)
            self.model = None
            self.processor = None
            logger.exception("Failed to load caption model: %s", exc)

    def caption(self, image: Image.Image, max_new_tokens: int = 60) -> tuple[str, float]:
        """Generate a caption for a PIL image. Returns (caption, generation_time_ms)."""
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Check GET /health for details "
                f"(last error: {self._load_error})."
            )

        import torch

        start = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return caption, elapsed_ms


# Module-level singleton used by the FastAPI app.
caption_model = CaptionModel()
