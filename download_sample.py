import os
import requests
from pathlib import Path

def download_sample_xray():
    """Download a sample X-ray image from the NIH Chest X-ray Dataset"""
    
    # Create test_images directory if it doesn't exist
    save_dir = Path("test_images")
    save_dir.mkdir(exist_ok=True)
    
    # Sample image URL from a public chest X-ray dataset
    image_url = "https://production-media.paperswithcode.com/datasets/NIH-Chest-X-ray-0000000002-6552ea86_A5PEvtR.jpg"
    save_path = save_dir / "sample_chest_xray.jpg"
    
    print(f"Downloading sample chest X-ray image to {save_path}...")
    
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Successfully downloaded sample image to: {save_path}")
        print("\nYou can now analyze this image using:")
        print(f"python test_model.py {save_path}")
        
    except Exception as e:
        print(f"Error downloading image: {str(e)}")

if __name__ == "__main__":
    download_sample_xray()
