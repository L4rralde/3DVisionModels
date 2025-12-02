import os
import sys
import select
import time
import argparse

import numpy as np
import open3d as o3d
from glob import glob


def user_pressed_key():
    """Check non-blocking terminal input."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def play_animation(vis, pcd, frames, m=3, angle_step=5):
    view_ctl = vis.get_view_control()
    # -------------------------
    # Apply rotation around Y
    # -------------------------
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.eye(4)
    view_ctl.convert_from_pinhole_camera_parameters(cam)


    for loop in range(m):                     # repeat animation m times
        for frame in frames:                  # loop over pointcloud frames

            # Load frame points
            new_pcd = o3d.io.read_point_cloud(frame)
            pcd.points = new_pcd.points
            if new_pcd.has_colors():
                pcd.colors = new_pcd.colors

            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            # Exit the animation when user types anything
            if user_pressed_key():
                print("Key pressed → stopping animation.")
                return

            time.sleep(0.05)   # smooth animation delay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Path to input directory"
    )
    args = parser.parse_args()
    return args


def visualize_dynamic_point_cloud(root_dir):
    # Your usual setup
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    frames = [
        os.path.join(root_dir, frame)
        for frame in sorted(glob("frame_*.ply", root_dir=root_dir))
    ]
    pcd = o3d.io.read_point_cloud(frames[0])
    vis.add_geometry(pcd)

    print("Animation running. Type anything + ENTER in this terminal to exit.")
    play_animation(vis, pcd, frames, m=50, angle_step=3)

    vis.destroy_window()


if __name__ == "__main__":
    args = parse_args()
    visualize_dynamic_point_cloud(args.input)
