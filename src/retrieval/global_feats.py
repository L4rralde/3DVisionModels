
from typing import List
import os
from abc import abstractmethod

import torch
import numpy as np
from torchvision.transforms import v2


from .datasets import ImageDataset


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



class GlobalFeaturesExtractor:
    def __init__(self) -> None:
        self.model = self.get_model().eval().to(device)

    @abstractmethod
    def get_model() -> torch.nn.Module:
        raise NotImplementedError("Abstract method")

    def forward(self, img_batch: torch.Tensor) -> List[np.ndarray]:
        with torch.no_grad():
            img_feats = self.model(img_batch.to(device))
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
            img_feats_np = img_feats.half().cpu().numpy()
        
        return img_feats_np

    def __call__(self, image_path_list: List[os.PathLike]) -> List[np.ndarray]:
        return self.forward(image_path_list)

    @abstractmethod
    def get_dataset(self) -> torch.utils.data.DataLoader:
        raise NotImplementedError("Abstract method")


class MegaLoc(GlobalFeaturesExtractor):
    def get_model(self) -> torch.nn.Module:
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        return model

    def get_dataset(self, path: os.PathLike, **kwargs) -> torch.utils.data.DataLoader:
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        img_size = kwargs.get('resize_size', 322)
        resize = v2.Resize((img_size, img_size), antialias=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        transform = v2.Compose([to_tensor, resize, normalize])
        dataset = ImageDataset(path, transform)

        return dataset
    

#Shouldn't be used. Features are already normalized.
def cosine_similarity(a: np.array, b: np.array) -> float:
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


