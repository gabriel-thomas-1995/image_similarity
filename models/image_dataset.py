from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms._presets import ImageClassification


class ImageDataset(Dataset):
    def __init__(self, image_file_paths: list[Path],
                 preprocessor: ImageClassification) -> None:
        self.image_file_paths = image_file_paths
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        image_file_path = self.image_file_paths[idx]
        image = Image.open(fp=image_file_path, mode="r").convert("RGB")
        preprocessed_image = self.preprocessor(image)
        return preprocessed_image
