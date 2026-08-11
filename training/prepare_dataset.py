"""
Filters eltorio/ROCOv2-radiology down to chest/thorax X-ray image-caption
pairs, matching the filter recorded in model/roco_chest_xray_lora/training_metadata.json.

Usage:
    python training/prepare_dataset.py --output data/roco_chest_xray
"""

from __future__ import annotations

import argparse
import re

from datasets import DatasetDict, load_dataset

CHEST_TERMS = re.compile(r"\b(chest|thorax|thoracic)\b", re.IGNORECASE)
XRAY_TERMS = re.compile(r"\b(x-?ray|radiograph|radiography)\b", re.IGNORECASE)


def is_chest_xray_caption(example: dict) -> bool:
    caption = example.get("caption") or ""
    return bool(CHEST_TERMS.search(caption) and XRAY_TERMS.search(caption))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="Directory to save the filtered dataset to (via save_to_disk)."
    )
    parser.add_argument(
        "--dataset", default="eltorio/ROCOv2-radiology", help="Source HF dataset id."
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    raw = load_dataset(args.dataset)

    filtered = DatasetDict()
    for split_name, split in raw.items():
        before = len(split)
        filtered[split_name] = split.filter(is_chest_xray_caption)
        after = len(filtered[split_name])
        print(f"[{split_name}] kept {after}/{before} examples ({after / max(before, 1):.1%})")

    filtered.save_to_disk(args.output)
    print(f"Saved filtered dataset to {args.output}")


if __name__ == "__main__":
    main()
