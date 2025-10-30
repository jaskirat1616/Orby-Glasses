#!/bin/bash
# Run pySLAM main_vo.py with OpenCV windows (no Rerun)

echo "🚀 Starting pySLAM main_vo.py with OpenCV windows..."

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

# Copy config to pySLAM directory
cp ../../config_pyslam_live.yaml config_live.yaml

# Create a modified main_vo.py that disables Rerun
echo "Creating modified main_vo.py with OpenCV windows..."
cat > main_vo_opencv.py << 'EOF'
#!/usr/bin/env -S python3 -O
"""
Modified main_vo.py to use OpenCV windows instead of Rerun
"""

import numpy as np
import cv2
import os
import math
import time
import platform

from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam.camera import PinholeCamera

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType

from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
from pyslam.viz.qplot_thread import Qplot2d

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.utils_sys import Printer, force_kill_all_and_exit
from pyslam.utilities.utils_eval import eval_ate

import argparse

# Disable Rerun and use OpenCV windows
kUseRerun = False
kUsePangolin = False
kUseQplot2d = False

def factory_plot2d(*args, **kwargs):
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=None,
        help="Optional path for custom configuration file",
    )
    args = parser.parse_args()

    if args.config_path:
        config = Config(args.config_path)
    else:
        config = Config()

    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    cam = PinholeCamera(config)

    num_features = 2000
    if config.num_features_to_extract > 0:
        num_features = config.num_features_to_extract

    # Use ORB tracker for better performance
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)
        print("Using VisualOdometryRgbdTensor")
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        print("Using VisualOdometryEducational")
    
    time.sleep(1)

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5 * traj_img_size)
    draw_scale = 1

    is_draw_3d = True
    is_draw_with_rerun = False

    is_draw_err = True
    err_plt = factory_plot2d(xlabel="img id", ylabel="m", title="error")

    is_draw_matched_points = True
    matched_points_plt = factory_plot2d(xlabel="img id", ylabel="# matches", title="# matches")

    img_id = 0
    print("Starting Visual Odometry with OpenCV windows...")
    print("Press 'q' to quit")
    
    while True:
        img = None

        if dataset.is_ok:
            timestamp = dataset.getTimestamp()
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            img_right = (
                dataset.getImageColorRight(img_id)
                if dataset.sensor_type == SensorType.STEREO
                else None
            )

        if img is not None:
            vo.track(img, img_right, depth, img_id, timestamp)

            if len(vo.traj3d_est) > 1:
                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]

                if is_draw_traj_img:
                    draw_x, draw_y = int(draw_scale * x) + half_traj_img_size, half_traj_img_size - int(draw_scale * z)
                    draw_gt_x, draw_gt_y = int(draw_scale * gt_x) + half_traj_img_size, half_traj_img_size - int(draw_scale * gt_z)
                    cv2.circle(traj_img, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
                    cv2.circle(traj_img, (draw_gt_x, draw_gt_y), 1, (0, 0, 255), 1)

                cv2.imshow("Trajectory", traj_img)

                if is_draw_matched_points and hasattr(vo, 'num_matches'):
                    inliers_signal = vo.num_matches
                    if matched_points_plt:
                        matched_points_plt.draw(inliers_signal, "# inliers", color="g")

            # draw camera image
            cv2.imshow("Camera", vo.draw_img)
            
            # Print trajectory info
            if img_id % 30 == 0 and len(vo.traj3d_est) > 1:  # Print every 30 frames
                x, y, z = vo.traj3d_est[-1]
                print(f"Frame {img_id}: Position ({x:.2f}, {y:.2f}, {z:.2f})")

        else:
            time.sleep(0.1)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        img_id += 1

    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
EOF

# Run the modified main_vo.py
echo "Starting pySLAM Visual Odometry with OpenCV windows..."
python main_vo_opencv.py --config_path=config_live.yaml

echo "pySLAM Visual Odometry finished."
