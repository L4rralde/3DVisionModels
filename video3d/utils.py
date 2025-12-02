import numpy as np
import open3d as o3d


def get_conf_mask(conf):
    lower = np.percentile(conf, 20)
    upper = np.percentile(conf, 80)
    conf_thresh = min(max(1.01, lower), upper)
    mask = (conf > conf_thresh)

    return mask


def to_pointcloud(conf, images, world_points):
    mask = get_conf_mask(conf)
    #mask = np.ones_like(conf, dtype=np.bool_)

    points = world_points[mask].reshape(-1, 3)
    colors = images[mask].reshape(-1, 3)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


from scipy.optimize import minimize
from scipy.special import huber

def align(
    frame_conf,
    scene_conf,
    frame_depth,
    scene_depth,
    scene_extrinsic,
    scene_intrinsic
):
    frame_mask = get_conf_mask(frame_conf)
    scene_mask = get_conf_mask(scene_conf)
    
    common_mask = frame_mask & scene_mask

    scene_conf_depth = scene_depth[common_mask]
    frame_conf_depth = frame_depth[common_mask]

    loss = lambda s: huber(1e-3, scene_conf_depth - s*frame_conf_depth).mean()
    scale = minimize(loss, 1.0).x[0]

    ext_w2c = np.eye(4)
    ext_w2c[:3, :] = scene_extrinsic
    
    K = scene_intrinsic
    K_inv = np.linalg.inv(K)
    
    H, W = frame_depth.shape
    
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    scaled_frame_depth = scale*frame_depth
    d_flat = scaled_frame_depth.reshape(-1)
    rays = K_inv @ pix.T
    
    Xc = rays * d_flat[None, :]
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
    c2w = np.linalg.inv(ext_w2c)
    Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)
    Xw = Xw.reshape(H, W, 3)

    return Xw