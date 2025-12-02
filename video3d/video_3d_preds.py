import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from glob import glob
import numpy as np
from tqdm import tqdm

from src.recons3d import Client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_dir', type=str)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    files = sorted(glob('*.jpg', root_dir=args.frames_dir))

    model = Client()

    print("Predicting individual frames")
    for file in tqdm(files):
        preds = model.infer([os.path.join(args.frames_dir, file)])
        npz_path = os.path.join(args.frames_dir, f"{os.path.splitext(file)[0]}.npz")
        np.savez(npz_path, **preds)

    print("Predicting scene")
    npz_path = os.path.join(args.frames_dir, "predicts.npz")
    all_predicts = model.infer([os.path.join(args.frames_dir, f) for f in files])
    print("Saving predictions")
    np.savez(npz_path, **all_predicts)


if __name__ == '__main__':
    main()
