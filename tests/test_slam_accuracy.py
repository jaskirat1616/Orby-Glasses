"""
Unit tests for SLAM system accuracy.
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slam_system import SLAMSystem
from utils import ConfigManager

class MockConfig(ConfigManager):
    def __init__(self, config_dict):
        self.config = config_dict

def create_test_video(output_path, width=640, height=480, num_frames=100, motion_type='linear'):
    """Creates a synthetic test video with known camera motion."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    background = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    for i in range(num_frames):
        frame = background.copy()
        if motion_type == 'linear':
            # Simulate linear motion by shifting the background
            shift = i * 2
            frame = np.roll(background, shift, axis=1)
        elif motion_type == 'rotation':
            # Simulate rotation
            angle = i * 1.0
            center = (width // 2, height // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(background, rot_mat, (width, height))

        out.write(frame)
    out.release()

def calculate_trajectory_rmse(ground_truth, estimated_trajectory):
    """Calculates the Root Mean Squared Error of the trajectory."""
    if len(ground_truth) != len(estimated_trajectory):
        raise ValueError("Ground truth and estimated trajectories must have the same length.")

    errors = np.array(ground_truth) - np.array(estimated_trajectory)
    squared_errors = np.sum(errors**2, axis=1)
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

@pytest.fixture(scope="module")
def test_video_path():
    video_path = "test_video.mp4"
    create_test_video(video_path)
    yield video_path
    os.remove(video_path)

def run_slam_on_video(config, video_path):
    slam = SLAMSystem(config)
    cap = cv2.VideoCapture(video_path)
    trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = slam.process_frame(frame)
        trajectory.append(result['position'])

    cap.release()
    return trajectory

def test_slam_accuracy_linear_motion(test_video_path):
    """Tests the SLAM system's accuracy with linear motion."""
    config_dict = {
        'slam.enabled': True,
        'slam.orb_features': 3000,
        'slam.fast_threshold': 10,
        'slam.min_matches': 15,
        'slam.reprojection_threshold': 3.0,
        'slam.pose_alpha': 0.8,
        'slam.loop_closure_threshold': 0.6,
        'mapping3d.fx': 500,
        'mapping3d.fy': 500,
        'camera.width': 640,
        'camera.height': 480,
    }
    config = MockConfig(config_dict)

    estimated_trajectory = run_slam_on_video(config, test_video_path)

    # Generate ground truth trajectory (this is an approximation)
    num_frames = len(estimated_trajectory)
    ground_truth = [[i * 0.02, 0, 0] for i in range(num_frames)]

    rmse = calculate_trajectory_rmse(ground_truth, estimated_trajectory)
    print(f"Linear Motion RMSE: {rmse}")
    assert rmse < 0.5

@pytest.mark.parametrize("params", [
    {'orb_features': 2000, 'fast_threshold': 20},
    {'orb_features': 4000, 'fast_threshold': 5},
    {'min_matches': 10, 'reprojection_threshold': 5.0},
    {'min_matches': 20, 'reprojection_threshold': 2.0},
    {'pose_alpha': 0.5},
    {'pose_alpha': 0.9},
])
def test_slam_accuracy_with_different_parameters(test_video_path, params):
    """Tests the SLAM system's accuracy with different parameters."""
    config_dict = {
        'slam.enabled': True,
        'slam.orb_features': 3000,
        'slam.fast_threshold': 10,
        'slam.min_matches': 15,
        'slam.reprojection_threshold': 3.0,
        'slam.pose_alpha': 0.8,
        'slam.loop_closure_threshold': 0.6,
        'mapping3d.fx': 500,
        'mapping3d.fy': 500,
        'camera.width': 640,
        'camera.height': 480,
    }
    config_dict.update(params)
    config = MockConfig(config_dict)

    estimated_trajectory = run_slam_on_video(config, test_video_path)
    num_frames = len(estimated_trajectory)
    ground_truth = [[i * 0.02, 0, 0] for i in range(num_frames)]

    rmse = calculate_trajectory_rmse(ground_truth, estimated_trajectory)
    print(f"Parameters: {params}, RMSE: {rmse}")
    assert rmse < 1.0 # Looser threshold for non-optimal parameters
