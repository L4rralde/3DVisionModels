import os
from PIL import Image

import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    def __init__(self, input_dir: os.PathLike, transform: object=None) -> None:
        self.input_dir = input_dir
        self.paths = glob.glob(f"{input_dir}/**/*.jpg", recursive=True)
        self.transform = transform or ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        path = self.paths[idx]
        img = self.transform(Image.open(path))

        return os.path.relpath(path, self.input_dir), img
