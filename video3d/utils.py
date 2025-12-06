import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.special import huber


def get_conf_mask(conf):
    lower = np.percentile(conf, 20)
    upper = np.percentile(conf, 80)
    conf_thresh = min(max(1.01, lower), upper)
    mask = (conf > conf_thresh)

    return mask


def to_pointcloud(conf, images, world_points):
    mask = get_conf_mask(conf)

    points = world_points[mask].reshape(-1, 3)
    colors = images[mask].reshape(-1, 3)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud
