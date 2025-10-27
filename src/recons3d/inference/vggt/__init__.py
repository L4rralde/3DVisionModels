import os
from typing import List
import gc

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


DEVICE = "cuda"


class VggtInference:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available to use this model")
        self.model = VggtInference.load_model_for_inference()

    @staticmethod
    def load_model_for_inference():
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(DEVICE)
        return model

    def infer(self, img_path_list: List[os.PathLike]) -> dict:
        gc.collect() #Collect garbage
        torch.cuda.empty_cache()

        images = load_and_preprocess_images(img_path_list).to(DEVICE)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(DEVICE, dtype=dtype):
                predictions = self.model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        torch.cuda.empty_cache()

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"]
        )

        return predictions
