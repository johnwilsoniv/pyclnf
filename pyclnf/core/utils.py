"""
Utility functions for CLNF - similarity transforms and coordinate transformations.
"""

import numpy as np
from typing import Tuple


def align_shapes_with_scale(src_shape: np.ndarray, dst_shape: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform (scale + rotation + translation) from src to dst.

    This matches OpenFace's Utilities::AlignShapesWithScale function, which computes
    a 2D similarity transform that best aligns source landmarks to destination landmarks.

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

    # Center shapes (remove translation)
    src_mean = src_shape.mean(axis=0)
    dst_mean = dst_shape.mean(axis=0)

    src_centered = src_shape - src_mean
    dst_centered = dst_shape - dst_mean

    # Compute scales (for normalization)
    src_scale = np.sqrt((src_centered ** 2).sum())
    dst_scale = np.sqrt((dst_centered ** 2).sum())

    # Normalize shapes to unit scale
    src_norm = src_centered / (src_scale + 1e-8)
    dst_norm = dst_centered / (dst_scale + 1e-8)

    # Compute rotation parameters (a, b)
    # a = cos(θ), b = sin(θ) in the rotation matrix
    a = (src_norm * dst_norm).sum()
    b = (src_norm[:, 0] * dst_norm[:, 1] - src_norm[:, 1] * dst_norm[:, 0]).sum()

    # Overall scale factor
    scale = dst_scale / (src_scale + 1e-8)

    # Build 2x3 similarity transform matrix
    # Apply: dst = [[a -b][b a]] * scale * (src - src_mean) + dst_mean
    #           = [[sa -sb][sb sa]] * src + [tx, ty]
    # where tx = dst_mean_x - sa*src_mean_x + sb*src_mean_y
    #       ty = dst_mean_y - sb*src_mean_x - sa*src_mean_y

    sa = scale * a
    sb = scale * b

    tx = dst_mean[0] - (sa * src_mean[0] - sb * src_mean[1])
    ty = dst_mean[1] - (sb * src_mean[0] + sa * src_mean[1])

    transform = np.array([
        [sa, -sb, tx],
        [sb,  sa, ty]
    ], dtype=np.float32)

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
    ], dtype=np.float32)

    return inv_transform
