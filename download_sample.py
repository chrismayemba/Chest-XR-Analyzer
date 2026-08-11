"""Download a sample chest X-ray image to try the API against."""

from pathlib import Path

import requests


def download_sample_xray() -> None:
    """Download a sample X-ray image from a public chest X-ray dataset."""
    save_dir = Path("test_images")
    save_dir.mkdir(exist_ok=True)

    image_url = "https://production-media.paperswithcode.com/datasets/NIH-Chest-X-ray-0000000002-6552ea86_A5PEvtR.jpg"
    save_path = save_dir / "sample_chest_xray.jpg"

    print(f"Downloading sample chest X-ray image to {save_path}...")
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded sample image to: {save_path}")
        print("\nWith the API running (uvicorn app.main:app --reload), try:")
        print(f'  curl -X POST "http://localhost:8000/caption" -F "file=@{save_path}"')
    except Exception as exc:  # noqa: BLE001
        print(f"Error downloading image: {exc}")


if __name__ == "__main__":
    download_sample_xray()
