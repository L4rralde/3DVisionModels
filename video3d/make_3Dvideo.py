import os
import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm

from utils import to_pointcloud
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.glueing.align import est_scale_factor, depth_to_frame


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
    for img_p in scene_predictions['image_names']:
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

        scale = est_scale_factor(
            frame_depth, frame_conf,
            scene_depth, scene_conf
        )

        frame_world_points = depth_to_frame(
            frame_depth, scene_intrinsic, scene_extrinsic, scale
        )
        
        scene_pcd = to_pointcloud(
            frame_conf,
            frame_images,
            frame_world_points
        )

        frame = os.path.splitext(frame_predictions[idx]['image_names'][0])[0]

        pcd_path = os.path.join(input_path, f"{frame}.ply")
        tqdm.write(f"Writing point cloud to {pcd_path}")
        o3d.io.write_point_cloud(pcd_path, scene_pcd)


if __name__ == '__main__':
    main()
