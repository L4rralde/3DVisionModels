import os
from typing import List
from collections import defaultdict

import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame


DEVICE = "cuda"


class MapanythingInference:
    def __init__(self, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available to use this model")
        self.model = MapanythingInference.load_model_for_inference()

    @staticmethod
    def load_model_for_inference():
        model = MapAnything.from_pretrained("facebook/map-anything").to(DEVICE)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def infer(self, img_path_list: List[os.PathLike]) -> dict:
        views = load_images(img_path_list)
        outputs = self.model.infer(
            views,
            memory_efficient_inference = False #Experiment with True
        )
        img_names = [os.path.basename(path) for path in img_path_list]

        #Add 3D points in world frame
        for pred in outputs:
            depth_torch = pred["depth_z"][0].squeeze(-1)
            intrinsics_torch = pred["intrinsics"][0]
            camera_pose_torch = pred["camera_poses"][0]

            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depth_torch,
                intrinsics_torch,
                camera_pose_torch
            )

            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            mask = mask & valid_mask.cpu().numpy()
            pts3d_np = pts3d_computed.cpu().numpy()
            pred["pts3d_computed"] = pts3d_np
            pred["pts3d_computed_mask"] = mask

        #Convert tensors to np array
        for pred in outputs:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.squeeze(0).cpu().numpy()

        #Converting list of dicts to dict of lists
        to_keep = [
            'metric_scaling_factor',
            'conf',
            'intrinsics',
            'camera_poses',
            'depth_z',
            'pts3d_computed',
            'pts3d_computed_mask',
            'img_no_norm'
        ]
        predictions_to_keep = defaultdict(list)
        for pred in outputs:
            for k in to_keep:
                predictions_to_keep[k].append(pred[k])
        for k, v in predictions_to_keep.items():
            predictions_to_keep[k] = np.stack(v)

        renamed_predictions = {
            'model': 'map-anything',
            'depth': predictions_to_keep['depth_z'].squeeze(-1),
            'world_points': predictions_to_keep['pts3d_computed'],
            'images': predictions_to_keep['img_no_norm'],
            'image_names': img_names,
            'extrinsic': predictions_to_keep['camera_poses'],
            'intrinsic': predictions_to_keep['intrinsics'],
            'conf': predictions_to_keep['conf'],
            'mask': predictions_to_keep['pts3d_computed_mask'],
            'scale': predictions_to_keep['metric_scaling_factor']
        }
        
        return renamed_predictions
