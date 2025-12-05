from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber


def as_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    homo = np.eye(4)
    homo[:3, :] = extrinsic[:3, :]
    return homo


def get_conf_mask(conf):
    lower = np.percentile(conf, 20)
    upper = np.percentile(conf, 80)
    conf_thresh = min(max(1.00, lower), upper)
    mask = (conf > conf_thresh)

    return mask


def est_scale_factor(
    src_depth: np.ndarray,
    src_conf: np.ndarray,
    dst_depth: np.ndarray,
    dst_conf: np.ndarray
) -> float:
    src_mask = get_conf_mask(src_conf)
    dst_mask = get_conf_mask(dst_conf)
    
    common_mask = src_mask & dst_mask

    dst_conf_depth = dst_depth[common_mask]
    src_conf_depth = src_depth[common_mask]

    loss = lambda s: huber(1e-3, dst_conf_depth - s*src_conf_depth).mean()
    scale = minimize(loss, 1.0).x[0]

    return scale


def depth_to_frame(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    scale: float=1.0
) -> np.ndarray:
    ext_w2c = as_homogeneous(extrinsic)
    K = intrinsic
    K_inv = np.linalg.inv(K)

    H, W = depth.shape

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)

    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3) # (H*W, 3)
    scaled_depth = scale * depth
    d_flat = scaled_depth.reshape(-1)

    rays = K_inv @ pix.T

    Xc = rays * d_flat[None, :]
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    c2w = np.linalg.inv(ext_w2c)
    Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)
    Xw = Xw.reshape(H, W, 3)

    return Xw


def relative_transform(
    src_extrinsic: np.ndarray,
    dst_extrinsic: np.ndarray
) -> np.ndarray:
    src_pose = as_homogeneous(src_extrinsic)
    dst_pose = as_homogeneous(dst_extrinsic)

    #dst = T @ src
    transform = np.linalg.inv(src_pose) @ dst_pose

    return transform[:3, :]


def est_scenes_transform(
    src_scene: dict,
    dst_scene: dict,
) -> Tuple[float, np.ndarray]:
    src_names = list(src_scene['image_names'])
    dst_names = list(dst_scene['image_names'])
    common_images = set(src_names).intersection(set(dst_names))
    if not common_images:
        raise ValueError("Non-overlapping scenes")
    
    #By the moment use first appearance only
    link_name = list(common_images)[0]
    print(link_name)

    src_idx = src_names.index(link_name)
    dst_idx = dst_names.index(link_name)

    src_depth = src_scene["depth"][src_idx]
    src_conf = src_scene["conf"][src_idx]
    src_extrinsic = src_scene["extrinsic"][src_idx]

    dst_depth = dst_scene["depth"][dst_idx]
    dst_conf = dst_scene["conf"][dst_idx]
    dst_extrinsic = dst_scene["extrinsic"][dst_idx]

    scale = est_scale_factor(
        src_depth, src_conf, dst_depth, dst_conf
    )

    src_extrinsic_cp = np.copy(src_extrinsic)
    src_extrinsic_cp[:, -1] *= scale

    transform = relative_transform(src_extrinsic_cp, dst_extrinsic)

    return scale, transform


def transform_scene(
    scene: dict,
    transform: np.ndarray,
    scale: float = 1.0,
    inplace: bool = False
) -> dict:
    transform = as_homogeneous(transform)

    new_scene = scene if inplace else scene.copy()

    new_scene['depth'] *= scale
    for i, extrinsic in enumerate(scene['extrinsic']):
        h_extrinsic = as_homogeneous(extrinsic)
        h_extrinsic[:, -1] *= scale
        new_scene['extrinsic'][i] = (h_extrinsic @ transform)[:3, :]

    iterator = zip(
        new_scene['depth'],
        new_scene['intrinsic'],
        new_scene['extrinsic']
    )

    for i, (depth, intrinsic, extrinsic) in enumerate(iterator):
        points = depth_to_frame(depth, intrinsic, extrinsic)
        new_scene['world_points'][i] = points

    return new_scene
