import os
from typing import List

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame


DEVICE = "cuda"


class MapanythingInference:
    def __init__(self):
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
        world_points_list = []
        masks_list = []

        for view_idx, pred in enumerate(outputs):
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

            world_points_list.append(pts3d_np)
            masks_list.append(mask)

        return outputs
