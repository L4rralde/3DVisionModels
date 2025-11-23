import sys
import os
from typing import List

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors, get_conf_thresh, _as_homogeneous44
from depth_anything_3.specs import Prediction


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class Da3Inference:
    MODEL_NAMES = (
        "DA3NESTED-GIANT-LARGE",
        "DA3-GIANT",
        "DA3-LARGE",
        "DA3-BASE",
        "DA3-SMALL"
    )
    def __init__(self, model_name: str="DA3-GIANT", **kwargs) -> None:
        #Gaussian splatting not supported yet. gsplat nor e3nn are installed.
        # Both are only required for gaussian splatting features.
        if not model_name in Da3Inference.MODEL_NAMES:
            raise ValueError(f"Model name {model_name} not supported")
        self.model_name = model_name
        self.model = Da3Inference.load_model_for_inference(model_name)

    @staticmethod
    def load_model_for_inference(model_name: str):
        model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}").to(DEVICE)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def infer(self, img_path_list: List[os.PathLike], **kwargs) -> dict:
        """
        kwargs:
            conf_thresh: Base confidence threshold used before percentile adjustments.
            conf_thresh_percentile: Lower percentile used when adapting the confidence threshold.
            ensure_thresh_percentile: Upper percentile clamp for the adaptive threshold.
        """
        prediction: Prediction = self.model.inference(img_path_list)
        img_names = [os.path.basename(path) for path in img_path_list]

        points, _ = _depths_to_world_points_with_colors( #Warning, this returns a flatten array with valid depth points. You want all.
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,
            prediction.processed_images,
            prediction.conf,
            conf_thr=0.0 #This value must ensure no point is discarded.
        )

        points = points.reshape((*prediction.depth.shape, 3)) #Keep image shape. Now is a point map

        predictions_dict = {
            'model': self.model_name,
            'depth': prediction.depth,
            'world_points': points,
            'is_metric': prediction.is_metric,
            'images': prediction.processed_images/255,
            'image_paths': img_names,
            'extrinsic': prediction.extrinsics,
            'intrinsic': prediction.intrinsics,
            'conf': prediction.conf,
            'is_metric': prediction.is_metric,
            'scale': prediction.scale_factor
        }

        return predictions_dict

