from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms._presets import ImageClassification
from tqdm import tqdm

from models import ImageDataset
from settings import settings


class ImageVectorizer:
    def __init__(self) -> None:
        self.model, self.preprocessor = self.__load_model_and_preprocessor()
        self.batch_size = settings.image_vectorizer_batch_size
        self.vector_dimension = self.__get_vector_dimension()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def __load_model_and_preprocessor(self) -> tuple[nn.Module, ImageClassification]:
        match settings.image_vectorizer_model_name:
            case "efficientnet_v2_s":
                weights = models.EfficientNet_V2_S_Weights.DEFAULT
                model = models.efficientnet_v2_s(weights=weights)
            case "efficientnet_v2_m":
                weights = models.EfficientNet_V2_M_Weights.DEFAULT
                model = models.efficientnet_v2_m(weights=weights)
            case "efficientnet_v2_l":
                weights = models.EfficientNet_V2_L_Weights.DEFAULT
                model = models.efficientnet_v2_l(weights=weights)
            case _:
                raise ValueError(
                    f"Unsupported model name: {settings.image_vectorizer_model_name}"
                )

        model.classifier = torch.nn.Identity()
        preprocessor = weights.transforms()
        return model.eval(), preprocessor

    def __extract_vector(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vectors = self.model(image_tensor)
            normalized_vectors = nn.functional.normalize(vectors, p=2, dim=1)
        return normalized_vectors

    def __get_vector_dimension(self) -> int:
        dummy_input_image = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            output = self.model(dummy_input_image)
        vector_dimension = output.shape[1]
        return vector_dimension

    def get_image_vectors(
        self, image_file_paths: list[Path], num_workers: int = 4
    ) -> np.ndarray:
        dataset = ImageDataset(image_file_paths=image_file_paths,
                               preprocessor=self.preprocessor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                num_workers=num_workers)
        number_images = len(dataset)
        number_batches = number_images // self.batch_size
        if number_images % self.batch_size > 0:
            number_batches += 1
        vectors_list = [None] * number_batches
        for index_batch, batch in enumerate(tqdm(dataloader, total=number_batches,
                                                 desc="Processing Batches")):
            batch_on_device = batch.to(self.device)
            vectors = self.__extract_vector(batch_on_device)
            vectors_list[index_batch] = vectors.cpu()
        vectors_tensor = torch.cat(vectors_list, dim=0)
        vectors_numpy = vectors_tensor.numpy().astype(np.float32)
        return vectors_numpy
