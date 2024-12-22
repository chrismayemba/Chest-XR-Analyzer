import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

class ROCODataset(Dataset):
    def __init__(self, data_dir: str, processor: AutoProcessor, transform=None):
        """
        Args:
            data_dir (str): Directory with all the images and captions.
            processor (AutoProcessor): Processor for preparing inputs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform
        self.image_paths, self.captions = self._load_data()

    def _load_data(self) -> Tuple[List[str], List[str]]:
        # Implement logic to load image paths and captions from a CSV file
        csv_file = os.path.join(self.data_dir, 'annotations.csv')
        df = pd.read_csv(csv_file)
        image_paths = df['image_path'].tolist()
        captions = df['caption'].tolist()
        return image_paths, captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        return inputs

# Example usage
# processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
# dataset = ROCODataset(data_dir='path/to/roco', processor=processor)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
