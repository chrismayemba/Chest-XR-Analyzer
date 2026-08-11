---
base_model: Salesforce/blip-image-captioning-base
library_name: peft
tags:
  - base_model:adapter:Salesforce/blip-image-captioning-base
  - lora
  - transformers
  - medical-imaging
  - radiology
---

# roco-chest-xray-lora

A LoRA adapter fine-tuning BLIP (`Salesforce/blip-image-captioning-base`) to draft
descriptive captions for chest/thorax X-ray images. **This is a research prototype
for caption *drafting*, not a diagnostic tool.**

## Model Details

- **Base model:** `Salesforce/blip-image-captioning-base` (BLIP image captioning)
- **Adapter type:** LoRA (PEFT 0.19.1)
  - `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`
  - Target modules: `query`, `value`
  - Bias: none, no DoRA / rsLoRA / QLoRA
- **Task:** Conditional image captioning (free-text caption generation from a chest X-ray image)
- **Trainable footprint:** LoRA adapter only, ~4.7 MB of weights on top of the frozen BLIP base

## Training Data

- **Source:** [`eltorio/ROCOv2-radiology`](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)
  (Radiology Objects in COntext v2 — image/caption pairs harvested from open-access
  PubMed Central articles)
- **Filter applied:** captions containing a chest/thorax term (e.g. "chest", "thorax")
  **and** an X-ray/radiograph term (e.g. "x-ray", "radiograph", "radiography"), to
  restrict training to chest radiographs rather than the full multi-modality ROCOv2 corpus
- **Important caveat:** ROCOv2 captions are figure legends extracted from published case
  reports and articles — written by article authors for a scholarly audience, often
  describing a specific abnormal finding the article is about. They are **not**
  standardized radiology reports and are not representative of the general distribution
  of normal-vs-abnormal findings seen in routine clinical practice.

## Training Results

- **Best validation loss:** 5.90

This is a high validation loss for a captioning model and should be read as: the
adapter is an early-stage research checkpoint, not a converged, production-grade
model. Generated captions should be expected to be inconsistent, occasionally
fluent-but-wrong, and not infrequently unrelated to the actual image content.
Quantitative caption-quality metrics (BLEU/ROUGE/CIDEr, or clinical accuracy of
generated findings) have not yet been computed — only training/validation loss.

## Intended Use

- **Direct use:** Research on vision-language captioning applied to a low-resource
  radiology captioning setting; a starting checkpoint for further fine-tuning or
  evaluation experiments.
- **Downstream use:** As a caption *drafting* assistant inside a human-in-the-loop
  research tool, where every generated caption is explicitly labeled as
  AI-drafted and reviewed by a person before being treated as a description of
  the image.

## Out-of-Scope Use

- **Not for clinical diagnosis, triage, or any use that influences real patient care.**
- Not validated against radiologist-written reports or clinical ground truth.
- Not evaluated for calibration, bias across patient subgroups, image acquisition
  protocols, or scanner/vendor differences.
- Not a substitute for a radiologist's or clinician's interpretation of any image.

## Bias, Risks, and Limitations

- Trained on a filtered slice of ROCOv2, which itself is drawn from published case
  reports — these skew towards notable/abnormal findings and towards whatever
  imaging is submitted to open-access journals, not a representative clinical
  population.
- No demographic, ethnic, or geographic metadata is available in ROCOv2, so
  performance across populations (including African patient populations, relevant
  to the author's broader research context) is unknown and unverified.
- High validation loss means outputs can be fluent but factually wrong — this is
  a classic risk of captioning models and is amplified in a medical context.

## How to Get Started

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image

base_model_id = "Salesforce/blip-image-captioning-base"
adapter_path = "model/roco_chest_xray_lora"

processor = BlipProcessor.from_pretrained(base_model_id)
base_model = BlipForConditionalGeneration.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

image = Image.open("chest_xray.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=60)
caption = processor.decode(output_ids[0], skip_special_tokens=True)
print(caption)
```

## Framework Versions

- PEFT 0.19.1
