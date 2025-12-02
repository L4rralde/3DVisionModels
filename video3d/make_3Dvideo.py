import os
import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm

from utils import to_pointcloud, align


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help="Path of input directory"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_path = args.input
    scene_predictions_p = os.path.join(input_path, 'predicts.npz')
    scene_predictions = np.load(scene_predictions_p, allow_pickle=True)

    frame_predictions = []
    for img_p in scene_predictions['image_paths']:
        frame = os.path.splitext(img_p)[0] 
        frame_prediction_p = os.path.join(input_path, f"{frame}.npz")
        frame_predictions.append(np.load(frame_prediction_p, allow_pickle=True))
        

    for idx in tqdm(range(len(frame_predictions))):
        frame_conf = frame_predictions[idx]['conf'].squeeze(0)
        scene_conf = scene_predictions['conf'][idx]
        
        frame_depth = frame_predictions[idx]['depth'].squeeze(0)
        scene_depth = scene_predictions['depth'][idx]
        
        scene_extrinsic = scene_predictions['extrinsic'][idx]
        scene_intrinsic = scene_predictions['intrinsic'][idx]
        
        frame_images = frame_predictions[idx]['images'].squeeze(0)
        
        
        frame_world_points = align(
            frame_conf,
            scene_conf,
            frame_depth,
            scene_depth,
            scene_extrinsic,
            scene_intrinsic
        )
        
        scene_pcd = to_pointcloud(
            frame_conf,
            frame_images,
            frame_world_points
        )

        frame = os.path.splitext(frame_predictions[idx]['image_paths'][0])[0]

        pcd_path = os.path.join(input_path, f"{frame}.ply")
        tqdm.write(f"Writing point cloud to {pcd_path}")
        o3d.io.write_point_cloud(pcd_path, scene_pcd)


if __name__ == '__main__':
    main()
