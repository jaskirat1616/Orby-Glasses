"""
Coordinate Transformation System for OrbyGlasses
Handles proper coordinate system transformations for SLAM and 3D mapping
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging


class CoordinateTransformer:
    """
    Proper coordinate transformation system for SLAM and 3D mapping.
    Handles transformations between camera, world, and various coordinate systems.
    """
    
    def __init__(self, config):
        """
        Initialize coordinate transformation system.
        
        Args:
            config: Configuration object containing camera parameters
        """
        self.config = config
        
        # Camera intrinsics
        self.fx = config.get('mapping3d.fx', 500.0)
        self.fy = config.get('mapping3d.fy', 500.0)
        self.cx = config.get('camera.width', 320) / 2  # Principal point x
        self.cy = config.get('camera.height', 320) / 2  # Principal point y
        
        # Form camera matrix and its inverse
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.K_inv = np.linalg.inv(self.K)
        
        # Define coordinate system conventions
        # Default: X-right, Y-down, Z-forward (image coordinates)
        # World: X-east, Y-north, Z-up
        self.camera_to_world_rotation = np.array([
            [0, 0, 1],  # Z_forward -> X_world
            [-1, 0, 0], # X_right -> Y_world  
            [0, -1, 0]  # Y_down -> Z_world
        ], dtype=np.float32)
        
        # Initialize transformation history for smoothing
        self.pose_history = []
        self.max_history = 10
        
        logging.info("Coordinate Transformer initialized")
        logging.info(f"Camera matrix: [[{self.fx}, 0, {self.cx}], [0, {self.fy}, {self.cy}], [0, 0, 1]]")

    def pixel_to_camera_frame(self, pixel_coords: np.ndarray, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates to camera frame coordinates.
        
        Args:
            pixel_coords: 2D pixel coordinates (u, v) or array of (u, v)
            depth: Depth value for the pixel(s)
            
        Returns:
            3D coordinates in camera frame (x, y, z) or array of coordinates
        """
        # Handle both single point and array of points
        single_point = pixel_coords.ndim == 1
        if single_point:
            pixel_coords = pixel_coords.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        pixels_hom = np.hstack([pixel_coords, np.ones((pixel_coords.shape[0], 1))])
        
        # Back-project to camera frame
        points_camera = (self.K_inv @ pixels_hom.T).T * depth
        
        if single_point:
            return points_camera[0]
        return points_camera

    def camera_frame_to_world(self, camera_coords: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from camera frame to world frame using camera pose.
        
        Args:
            camera_coords: 3D coordinates in camera frame (x, y, z)
            camera_pose: 4x4 transformation matrix from world to camera frame
            
        Returns:
            3D coordinates in world frame (x, y, z)
        """
        # Ensure input is 3D point(s)
        single_point = camera_coords.ndim == 1
        if single_point:
            camera_coords = camera_coords.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        points_hom = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])
        
        # Transform to world coordinates using the inverse of the pose
        # camera_pose transforms world to camera, so we need its inverse
        world_to_camera = camera_pose  # This is the extrinsic matrix
        camera_to_world = np.linalg.inv(world_to_camera)
        
        world_coords = (camera_to_world @ points_hom.T).T
        
        # Return 3D coordinates only
        result = world_coords[:, :3]
        
        if single_point:
            return result[0]
        return result

    def world_to_camera_frame(self, world_coords: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from world frame to camera frame.
        
        Args:
            world_coords: 3D coordinates in world frame (x, y, z)
            camera_pose: 4x4 transformation matrix from world to camera frame
            
        Returns:
            3D coordinates in camera frame (x, y, z)
        """
        # Ensure input is 3D point(s)
        single_point = world_coords.ndim == 1
        if single_point:
            world_coords = world_coords.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        points_hom = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
        
        # Transform to camera coordinates (world to camera using the pose matrix)
        camera_coords = (camera_pose @ points_hom.T).T
        
        # Return 3D coordinates only
        result = camera_coords[:, :3]
        
        if single_point:
            return result[0]
        return result

    def camera_frame_to_pixel(self, camera_coords: np.ndarray) -> np.ndarray:
        """
        Project 3D camera frame coordinates to 2D pixel coordinates.
        
        Args:
            camera_coords: 3D coordinates in camera frame (x, y, z)
            
        Returns:
            2D pixel coordinates (u, v)
        """
        # Handle both single point and array of points
        single_point = camera_coords.ndim == 1
        if single_point:
            camera_coords = camera_coords.reshape(1, -1)
        
        # Filter points behind camera
        valid_mask = camera_coords[:, 2] > 0  # Only points in front of camera
        result = np.full((camera_coords.shape[0], 2), -1.0)  # Invalid coordinates
        
        # Project valid points
        valid_coords = camera_coords[valid_mask]
        if len(valid_coords) > 0:
            # Normalize by depth (z-coordinate)
            normalized = valid_coords[:, :2] / valid_coords[:, 2:3]  # Divide x, y by z
            
            # Apply camera intrinsics
            pixels = (self.K[:2, :2] @ normalized.T).T + self.K[:2, 2]  # Add principal point
            
            result[valid_mask] = pixels
        
        if single_point:
            return result[0]
        return result

    def pixel_to_world(self, pixel_coords: np.ndarray, depth: float, camera_pose: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates directly to world coordinates.
        
        Args:
            pixel_coords: 2D pixel coordinates (u, v)
            depth: Depth value at the pixel
            camera_pose: 4x4 transformation matrix from world to camera frame
            
        Returns:
            3D coordinates in world frame (x, y, z)
        """
        # First convert pixel to camera frame
        camera_coords = self.pixel_to_camera_frame(pixel_coords, depth)
        
        # Then convert camera frame to world
        world_coords = self.camera_frame_to_world(camera_coords, camera_pose)
        
        return world_coords

    def world_to_pixel(self, world_coords: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """
        Project 3D world coordinates to 2D pixel coordinates.
        
        Args:
            world_coords: 3D coordinates in world frame (x, y, z)
            camera_pose: 4x4 transformation matrix from world to camera frame
            
        Returns:
            2D pixel coordinates (u, v)
        """
        # First convert world to camera frame
        camera_coords = self.world_to_camera_frame(world_coords, camera_pose)
        
        # Then project to pixel coordinates
        pixel_coords = self.camera_frame_to_pixel(camera_coords)
        
        return pixel_coords

    def get_relative_transform(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """
        Calculate the relative transformation between two poses.
        
        Args:
            pose1: First 4x4 pose matrix
            pose2: Second 4x4 pose matrix
            
        Returns:
            Relative transformation matrix (pose2 relative to pose1)
        """
        # T_relative = T1^(-1) * T2
        # This gives the transformation from pose1 frame to pose2 frame
        pose1_inv = np.linalg.inv(pose1)
        relative_transform = pose1_inv @ pose2
        return relative_transform

    def decompose_transform(self, transform_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose a 4x4 transformation matrix into translation and rotation.
        
        Args:
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            Tuple of (translation_vector, rotation_matrix)
        """
        translation = transform_matrix[:3, 3]
        rotation = transform_matrix[:3, :3]
        return translation, rotation

    def compose_transform(self, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """
        Compose a 4x4 transformation matrix from translation and rotation.
        
        Args:
            translation: 3-element translation vector
            rotation: 3x3 rotation matrix
            
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a transformation matrix to a set of 3D points.
        
        Args:
            points: Array of 3D points (Nx3 or single point)
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            Transformed points
        """
        # Ensure input is 3D point(s)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation
        transformed_hom = (transform_matrix @ points_hom.T).T
        
        # Extract 3D coordinates
        result = transformed_hom[:, :3]
        
        if single_point:
            return result[0]
        return result

    def rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (XYZ convention).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        # Handle singularity for pitch = ±90 degrees
        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                     rotation_matrix[1, 0] * rotation_matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])

    def euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (XYZ convention).
        
        Args:
            euler_angles: [roll, pitch, yaw] in radians
            
        Returns:
            3x3 rotation matrix
        """
        rx, ry, rz = euler_angles
        
        # Rotation around X axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Rotation around Y axis
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Rotation around Z axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (XYZ convention: R = Rz * Ry * Rx)
        return Rz @ Ry @ Rx

    def smooth_pose(self, current_pose: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        Apply smoothing to pose based on history.
        
        Args:
            current_pose: Current 4x4 pose matrix
            alpha: Smoothing factor (higher = more responsive)
            
        Returns:
            Smoothed pose matrix
        """
        # Add current pose to history
        self.pose_history.append(current_pose.copy())
        
        # Keep only recent poses
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        if len(self.pose_history) == 1:
            return current_pose.copy()
        
        # Calculate average translation
        translations = [pose[:3, 3] for pose in self.pose_history]
        avg_translation = np.mean(translations, axis=0)
        
        # Calculate average rotation using quaternion averaging
        rotations = [pose[:3, :3] for pose in self.pose_history]
        avg_rotation = self._average_rotations(rotations)
        
        # Compose the smoothed transform
        smoothed_pose = np.eye(4)
        smoothed_pose[:3, :3] = avg_rotation
        smoothed_pose[:3, 3] = avg_translation
        
        return smoothed_pose

    def _average_rotations(self, rotations: list) -> np.ndarray:
        """
        Average a list of rotation matrices using quaternion averaging.
        
        Args:
            rotations: List of 3x3 rotation matrices
            
        Returns:
            Average rotation matrix
        """
        # Convert rotation matrices to quaternions
        quaternions = []
        for R in rotations:
            q = self.rotation_matrix_to_quaternion(R)
            quaternions.append(q)
        
        # Average quaternions (simple averaging, then normalize)
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        
        # Convert back to rotation matrix
        return self.quaternion_to_rotation_matrix(avg_quat)

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        # Method from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        x2, y2, z2 = x * 2, y * 2, z * 2
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        
        R = np.array([
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)]
        ])
        
        return R

    def transform_vector(self, vector: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Transform a 3D vector using a rotation matrix only (no translation).
        
        Args:
            vector: 3D vector to transform
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Transformed 3D vector
        """
        return rotation_matrix @ vector


# Example usage function
def example_usage():
    """Example of how to use the CoordinateTransformer."""
    # Create a mock config
    config = {
        'mapping3d.fx': 500.0,
        'mapping3d.fy': 500.0,
        'camera.width': 640,
        'camera.height': 480
    }
    
    # Create transformer
    transformer = CoordinateTransformer(config)
    
    # Example 1: Pixel to world transformation
    pixel = np.array([320, 240])  # Center of image
    depth = 2.0  # 2 meters away
    
    # Create a sample camera pose (camera at [0, 0, 0] looking along +Z)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 1.0  # Camera is 1m above ground
    
    world_point = transformer.pixel_to_world(pixel, depth, camera_pose)
    print(f"Pixel {pixel} at depth {depth}m -> World coordinates: {world_point}")
    
    # Example 2: World to pixel transformation
    world_point2 = np.array([1.0, 0.5, 0.0])  # Point in world coordinates
    pixel_coords = transformer.world_to_pixel(world_point2, camera_pose)
    print(f"World point {world_point2} -> Pixel coordinates: {pixel_coords}")
    
    # Example 3: Transform a point using a pose
    point_in_camera = np.array([0.5, 0.3, 1.5])  # Point in camera coordinates
    transformed_point = transformer.camera_frame_to_world(point_in_camera, camera_pose)
    print(f"Point {point_in_camera} in camera frame -> World frame: {transformed_point}")
    
    print("Coordinate transformation examples completed")


if __name__ == "__main__":
    example_usage()