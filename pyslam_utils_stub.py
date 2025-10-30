"""
Stub module for pySLAM utils compatibility
Copy this to third_party/pyslam/pyslam_utils.py if needed
"""

def compare_poses(*args, **kwargs):
    """Stub for compare_poses"""
    return 0.0

def compute_orientation(*args, **kwargs):
    """Stub for compute_orientation"""
    return 0.0

def compute_fundamental_matrix(*args, **kwargs):
    """Stub for compute_fundamental_matrix"""
    import numpy as np
    return np.eye(3)

def triangulate_points(*args, **kwargs):
    """Stub for triangulate_points"""
    import numpy as np
    return np.zeros((3, 1))

def solve_pnp(*args, **kwargs):
    """Stub for solve_pnp"""
    import numpy as np
    return True, np.zeros((3, 1)), np.zeros((3, 1))

__all__ = [
    'compare_poses',
    'compute_orientation', 
    'compute_fundamental_matrix',
    'triangulate_points',
    'solve_pnp'
]
