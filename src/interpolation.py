"""Camera pose interpolation utilities for smooth transitions."""

import numpy as np
from typing import Tuple


def lerp_translation(t_start: np.ndarray, t_end: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolation between two translation vectors.
    
    Args:
        t_start: 3x1 starting translation
        t_end: 3x1 ending translation
        alpha: interpolation parameter in [0, 1]
    
    Returns:
        interpolated 3x1 translation
    """
    return (1 - alpha) * t_start + alpha * t_end


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    
    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R


def slerp_rotation(R_start: np.ndarray, R_end: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical linear interpolation between two rotation matrices.
    
    Args:
        R_start: 3x3 starting rotation
        R_end: 3x3 ending rotation
        alpha: interpolation parameter in [0, 1]
    
    Returns:
        interpolated 3x3 rotation matrix
    """
    q_start = rotation_to_quaternion(R_start)
    q_end = rotation_to_quaternion(R_end)
    
    # Ensure shortest path (quaternion double cover)
    dot = np.dot(q_start, q_end)
    if dot < 0:
        q_end = -q_end
        dot = -dot
    
    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        # Quaternions are very close, use linear interpolation
        q_interp = (1 - alpha) * q_start + alpha * q_end
    else:
        # Standard slerp formula
        sin_theta = np.sin(theta)
        q_interp = (np.sin((1 - alpha) * theta) / sin_theta) * q_start + \
                   (np.sin(alpha * theta) / sin_theta) * q_end
    
    q_interp /= np.linalg.norm(q_interp)
    return quaternion_to_rotation(q_interp)


def interpolate_pose(
    R_start: np.ndarray,
    t_start: np.ndarray,
    R_end: np.ndarray,
    t_end: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate camera pose (R, t) using slerp + lerp.
    
    Args:
        R_start: 3x3 starting rotation
        t_start: 3x1 starting translation
        R_end: 3x3 ending rotation
        t_end: 3x1 ending translation
        alpha: interpolation parameter in [0, 1]
    
    Returns:
        (R_interp, t_interp) at parameter alpha in [0, 1]
    """
    R_interp = slerp_rotation(R_start, R_end, alpha)
    t_interp = lerp_translation(t_start, t_end, alpha)
    return R_interp, t_interp