import os
import argparse

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.glueing import (
    ConnectedClusterTree,
    SceneTree,
    est_scenes_transform,
    transform_scene
)


def child_align(src_scene: dict, dst_scene: dict) -> dict:
    s, T = est_scenes_transform(src_scene, dst_scene)
    t_scene = transform_scene(src_scene, T, s)
    return t_scene


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str)
    parser.add_argument('--clusters', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    clusters = ConnectedClusterTree.from_dir_output(args.clusters)
    root = '0' #TODO. Use best root

    connections = [
        (str(i), str(j), 1-w)
        for i, j, w in clusters.edges
    ]
    scene_tree = SceneTree.from_preds_folder(
        root,
        args.preds,
        connections
    )

    print("Aligning predictions")
    scene_tree.align(child_align)

    output_path = os.path.join(args.preds, 'aligned')
    os.makedirs(output_path, exist_ok=True)
    print(f"Writing back aligned predictions to {output_path}")
    for name, pred in tqdm(scene_tree.predictions.items()):
        npz_path = os.path.join(output_path, f"{name}.npz")
        np.savez(npz_path, **pred)


if __name__ == '__main__':
    main()
