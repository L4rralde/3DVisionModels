import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.retrieval import MegaLoc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_path', default='global_features.json')

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.realpath(args.input_dir)
    output_path = os.path.realpath(args.output_path)

    img_paths = []
    feats = []

    model = MegaLoc()
    dataset = model.get_dataset(input_dir)
    dataloader = DataLoader(dataset, 8)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            path, imgs = batch
            f = model(imgs)
            feats.append(f)
            img_paths += path

    d_for_df = {
        'image_paths': img_paths,
        'global_descriptors': [f for f in np.vstack(feats)]
    }

    df = pd.DataFrame(d_for_df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path)


if __name__ == '__main__':
    main()
