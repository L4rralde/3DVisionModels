import os

from PIL import Image
import pandas as pd
import numpy as np


class FeaturesDf:
    def __init__(self, df: pd.DataFrame, dir_path: os.PathLike=""):
        self.df = df
        self.feats_mat = np.vstack(self.df['global_descriptors'])
        self.sims_mat = np.dot(self.feats_mat, self.feats_mat.T)
        self.dir_path = dir_path
        self.image_paths = [
            os.path.join(self.dir_path, path)
            for path in self.df['image_paths']
        ]

    @staticmethod
    def from_json(json_path: os.PathLike, dir_path: os.PathLike="") -> "FeaturesDf":
        df = pd.read_json(json_path)
        return FeaturesDf(df, dir_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> object:
        return self.df.iloc[idx]

    def get_image(self, idx: int) -> Image:
        return Image.open(self.image_paths[idx])

    def get_similarity(self, i: int, j: int) -> float:
        return self.sims_mat[i, j]

    def find_most_similar_photos(self) -> tuple:
        triu = np.triu(self.sims_mat, 1)
        weight = triu.max()
        i, j = np.where(triu == weight)
        return i.item(), j.item(), weight

    def save(self, path: os.PathLike):
        self.df.to_json(path)


class ClusteredFeaturesDf(FeaturesDf):
    def __init__(self, df: pd.DataFrame, dir_path: os.PathLike = ""):
        super().__init__(df, dir_path)
        if not 'clusters' in self.df:
            raise ValueError(f"Dataframe does not include 'clusters' column")

        self.clusters = {
            label: self.df[self.df['clusters'] == label]
            for label in self.df['clusters'].unique()
        }


class HierarchicalTree:
    def __init__(self) -> None:
        pass
