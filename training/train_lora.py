"""
Fine-tunes Salesforce/blip-image-captioning-base with a LoRA adapter on a
filtered, chest/thorax subset of ROCOv2-radiology (see prepare_dataset.py).

This reproduces the configuration recorded in
model/roco_chest_xray_lora/adapter_config.json:
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["query", "value"]

Usage:
    python training/prepare_dataset.py --output data/roco_chest_xray
    python training/train_lora.py --data data/roco_chest_xray --output model/roco_chest_xray_lora
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"


def build_collate_fn(processor: BlipProcessor):
    def collate(batch):
        images = [example["image"].convert("RGB") for example in batch]
        captions = [example["caption"] for example in batch]
        inputs = processor(images=images, text=captions, padding=True, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return collate


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to filtered dataset (from prepare_dataset.py).")
    parser.add_argument("--output", required=True, help="Directory to save the LoRA adapter to.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading base model {BASE_MODEL_ID} ...")
    processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    dataset = load_from_disk(args.data)
    collate_fn = build_collate_fn(processor)
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_split = "validation" if "validation" in dataset else "test"
    val_loader = DataLoader(dataset[val_split], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            running_loss += outputs.loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"  -> new best (val_loss={val_loss:.4f}), saved to {output_dir}")

    metadata = {
        "purpose": "research_only_caption_drafting",
        "dataset": "eltorio/ROCOv2-radiology",
        "base_model": BASE_MODEL_ID,
        "filter": "caption contains chest/thorax term and x-ray/radiograph term",
        "warning": "ROCOv2 captions are not clinical reports; generated text requires expert review.",
        "best_validation_loss": best_val_loss,
        "epochs": args.epochs,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
