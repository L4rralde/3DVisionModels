import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from video3d.utils import to_pointcloud


def scene_pointcloud(scene: dict):
    return to_pointcloud(
        scene['conf'],
        scene['images'],
        scene['world_points']
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', type=str, nargs='+')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    preds = (np.load(p, allow_pickle=True) for p in args.paths)

    print("Generating point clouds for Open3D")
    pcds = [
        scene_pointcloud(pred)
        for pred in tqdm(preds)
    ]

    o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    main()
