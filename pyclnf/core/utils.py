"""
Utility functions for CLNF - similarity transforms and coordinate transformations.
"""

import numpy as np
from typing import Tuple


def align_shapes_with_scale(src_shape: np.ndarray, dst_shape: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform (scale + rotation + translation) from src to dst.

    This matches OpenFace's Utilities::AlignShapesWithScale function, which computes
    a 2D similarity transform using Kabsch algorithm (SVD-based rotation estimation).

    The transform is represented as a 2x3 matrix:
        [a  -b  tx]
        [b   a  ty]

    where (a,b) encode scale and rotation, and (tx,ty) is translation.

    Args:
        src_shape: Source landmarks, shape (n_points, 2)
        dst_shape: Destination landmarks, shape (n_points, 2)

    Returns:
        transform: 2x3 similarity transform matrix
                  Applying this to src_shape aligns it with dst_shape
    """
    assert src_shape.shape == dst_shape.shape, "Shapes must have same dimensions"
    assert src_shape.shape[1] == 2, "Shapes must be 2D"

    n = src_shape.shape[0]

    # Center shapes (remove translation) - matches C++ mean normalization
    src_mean = src_shape.mean(axis=0)
    dst_mean = dst_shape.mean(axis=0)

    src_centered = src_shape - src_mean
    dst_centered = dst_shape - dst_mean

    # Compute RMS scales (matches C++ exactly: sqrt(sum/n))
    # C++ line 221-222: s_src = sqrt(cv::sum(src_sq)[0] / n)
    src_scale = np.sqrt((src_centered ** 2).sum() / n)
    dst_scale = np.sqrt((dst_centered ** 2).sum() / n)

    # Normalize shapes to unit RMS scale
    src_norm = src_centered / (src_scale + 1e-10)
    dst_norm = dst_centered / (dst_scale + 1e-10)

    # Kabsch algorithm: compute rotation using SVD (matches C++ AlignShapesKabsch2D)
    # C++ line 171: svd(align_from.t() * align_to)
    # Note: src_norm is (n, 2), so src_norm.T @ dst_norm gives (2, 2) cross-covariance
    H = src_norm.T @ dst_norm  # (2, 2) cross-covariance matrix

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (no reflection)
    # C++ line 175: d = determinant(vt.t() * u.t())
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d < 0:
        corr[1, 1] = -1

    # Rotation matrix: R = V * corr * U^T
    # C++ line 188: R = vt.t() * corr * u.t()
    R = Vt.T @ corr @ U.T

    # Overall scale factor
    scale = dst_scale / (src_scale + 1e-10)

    # Build 2x3 similarity transform matrix
    # A = s * R (matches C++ line 233)
    # For a proper rotation R = [[cos, -sin], [sin, cos]]
    # So A = [[s*cos, -s*sin], [s*sin, s*cos]]
    A = scale * R

    # Translation: dst_mean = A @ src_mean + t  =>  t = dst_mean - A @ src_mean
    t = dst_mean - A @ src_mean

    # The transform matrix directly uses A (which already has correct signs)
    # Standard form: [[a, -b, tx], [b, a, ty]] where a=s*cos, b=s*sin
    # A[0,0]=s*cos, A[0,1]=-s*sin, A[1,0]=s*sin, A[1,1]=s*cos
    transform = np.array([
        [A[0, 0], A[0, 1], t[0]],
        [A[1, 0], A[1, 1], t[1]]
    ], dtype=np.float64)

    return transform


def apply_similarity_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply 2x3 similarity transform to points.

    Args:
        points: Points to transform, shape (n_points, 2) or (2,)
        transform: 2x3 similarity transform matrix

    Returns:
        transformed_points: Transformed points, same shape as input
    """
    if points.ndim == 1:
        # Single point (x, y)
        assert points.shape[0] == 2
        homogeneous = np.array([points[0], points[1], 1.0])
        result = transform @ homogeneous
        return result
    else:
        # Multiple points (n_points, 2)
        assert points.shape[1] == 2
        n_points = points.shape[0]

        # Convert to homogeneous coordinates (n_points, 3)
        homogeneous = np.column_stack([points, np.ones(n_points)])

        # Apply transform (2x3) @ (3xn) -> (2xn) -> transpose -> (n, 2)
        result = (transform @ homogeneous.T).T

        return result


def invert_similarity_transform(transform: np.ndarray) -> np.ndarray:
    """
    Invert a 2x3 similarity transform.

    Args:
        transform: 2x3 similarity transform matrix [[a -b tx][b a ty]]

    Returns:
        inv_transform: 2x3 inverse transform
    """
    # Extract components
    a = transform[0, 0]
    b = transform[1, 0]
    tx = transform[0, 2]
    ty = transform[1, 2]

    # Determinant of rotation+scale part: a^2 + b^2
    det = a*a + b*b

    # Inverse rotation+scale: [[a b][-b a]] / det
    inv_a = a / det
    inv_b = -b / det

    # Inverse translation: -R^(-1) * t
    inv_tx = -(inv_a * tx - inv_b * ty)
    inv_ty = -(inv_b * tx + inv_a * ty)

    inv_transform = np.array([
        [inv_a, -inv_b, inv_tx],
        [inv_b,  inv_a, inv_ty]
    ], dtype=np.float64)  # Use float64 for precision (matches C++ double)

    return inv_transform
