# Training pipeline

Reproduces `model/roco_chest_xray_lora`.

```bash
# 1. Filter ROCOv2-radiology down to chest/thorax X-ray captions
python training/prepare_dataset.py --output data/roco_chest_xray

# 2. Train the LoRA adapter
python training/train_lora.py \
  --data data/roco_chest_xray \
  --output model/roco_chest_xray_lora \
  --epochs 5 \
  --batch-size 8
```

Both scripts require the extra `training` dependency group
(`pip install -r requirements.txt -r training/requirements.txt`, or
`pip install datasets`, since `datasets` isn't needed at inference time).

`prepare_dataset.py` needs network access to the Hugging Face Hub the first
time it runs (to download `eltorio/ROCOv2-radiology`). The images are cached
locally afterward.

See `model/roco_chest_xray_lora/README.md` for what the resulting adapter is
(and isn't) validated for.
