"""
Eye Point Distribution Model (EyePDM) - Shape model for 28-point eye landmarks

This implements the hierarchical eye model from OpenFace for eye landmark refinement.
The eye PDM uses 28 landmarks around each eye with 10 shape modes.

Parameter vector: p = [s, wx, wy, wz, tx, ty, q0, q1, ..., q9]
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class EyePDM:
    """Point Distribution Model for 28-point eye landmarks."""

    def __init__(self, model_dir: str):
        """
        Load Eye PDM from exported NumPy files.

        Args:
            model_dir: Directory containing mean_shape.npy, eigenvectors.npy, eigenvalues.npy
        """
        self.model_dir = Path(model_dir)

        # Load PDM components
        self.mean_shape = np.load(self.model_dir / 'mean_shape.npy', allow_pickle=True)  # (3n, 1) = (84, 1)
        self.princ_comp = np.load(self.model_dir / 'eigenvectors.npy', allow_pickle=True)  # (3n, m) = (84, 10)
        self.eigen_values = np.load(self.model_dir / 'eigenvalues.npy', allow_pickle=True)  # (1, m) = (1, 10)

        # Extract dimensions
        self.n_points = self.mean_shape.shape[0] // 3  # 28 landmarks
        self.n_modes = self.princ_comp.shape[1]  # 10 PCA modes

        # Parameter vector size: scale(1) + rotation(3) + translation(2) + shape(n_modes)
        self.n_params = 6 + self.n_modes

        # Pre-slice principal components for vectorized Jacobian computation
        n = self.n_points
        self._princ_comp_x = self.princ_comp[:n, :]      # (28, 10)
        self._princ_comp_y = self.princ_comp[n:2*n, :]   # (28, 10)
        self._princ_comp_z = self.princ_comp[2*n:3*n, :] # (28, 10)

    def params_to_landmarks_3d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 3D landmark positions.

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]
                   (OpenFace order)

        Returns:
            landmarks_3d: 3D landmark positions, shape (n_points, 3)
        """
        params = params.flatten()

        # Extract parameters (OpenFace order)
        s = params[0]  # Scale
        wx, wy, wz = params[1], params[2], params[3]  # Rotation (Euler angles)
        tx, ty = params[4], params[5]  # Translation
        q = params[6:]  # Shape parameters

        # Apply PCA: shape = mean + principal_components @ shape_params
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q  # (3n,)

        # Reshape to (n, 3) - OpenFace stores as [x1,...,xn, y1,...,yn, z1,...,zn]
        n = self.n_points
        shape_3d = np.column_stack([
            shape_3d[:n],      # x coordinates
            shape_3d[n:2*n],   # y coordinates
            shape_3d[2*n:3*n]  # z coordinates
        ])  # (n_points, 3)

        # Compute rotation matrix from Euler angles
        R = self._euler_to_rotation_matrix(np.array([wx, wy, wz]))

        # Apply similarity transform: landmarks = s * R @ shape + t
        landmarks_3d = s * (shape_3d @ R.T)

        # Add translation (only to x and y)
        landmarks_3d[:, 0] += tx
        landmarks_3d[:, 1] += ty

        return landmarks_3d

    def params_to_landmarks_2d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 2D landmark positions.

        Args:
            params: Parameter vector

        Returns:
            landmarks_2d: 2D landmark positions, shape (n_points, 2)
        """
        landmarks_3d = self.params_to_landmarks_3d(params)
        return landmarks_3d[:, :2]

    def get_reference_shape(self, patch_scaling: float, params_local: np.ndarray = None) -> np.ndarray:
        """
        Generate reference shape at fixed scale for patch evaluation.

        Args:
            patch_scaling: Fixed scale for reference shape (1.0 or 1.5 for eyes)
            params_local: Local shape parameters (default: zeros = mean shape)

        Returns:
            reference_shape: 2D landmarks at reference scale, shape (n_points, 2)
        """
        if params_local is None:
            params_local = np.zeros(self.n_modes)

        # Create reference global params
        global_ref = np.array([patch_scaling, 0.0, 0.0, 0.0, 0.0, 0.0])
        ref_params = np.concatenate([global_ref, params_local])

        return self.params_to_landmarks_2d(ref_params)

    def compute_jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of 2D landmarks with respect to parameters.

        Uses STACKED format matching OpenFace C++:
            J[i, j] = ∂(landmark_i.x) / ∂param_j      for i = 0 to n-1
            J[i+n, j] = ∂(landmark_i.y) / ∂param_j    for i = 0 to n-1

        Args:
            params: Parameter vector

        Returns:
            jacobian: Jacobian matrix, shape (2*n_points, n_params)
        """
        params = params.flatten()

        # Extract parameters
        s = params[0]
        wx, wy, wz = params[1], params[2], params[3]
        q = params[6:]

        # Compute 3D shape before rotation
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q
        n = self.n_points

        # Extract coordinates
        X = shape_3d[:n]
        Y = shape_3d[n:2*n]
        Z = shape_3d[2*n:3*n]

        # Compute rotation matrix
        R = self._euler_to_rotation_matrix(np.array([wx, wy, wz]))

        # Extract rotation matrix elements
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]

        # Initialize Jacobian
        J = np.zeros((2 * self.n_points, self.n_params))

        # 1. Derivative w.r.t. scale (column 0) - STACKED format
        J[:n, 0] = X * r11 + Y * r12 + Z * r13
        J[n:, 0] = X * r21 + Y * r22 + Z * r23

        # 2. Derivative w.r.t. rotation (columns 1-3)
        J[:n, 1] = s * (Y * r13 - Z * r12)
        J[n:, 1] = s * (Y * r23 - Z * r22)

        J[:n, 2] = -s * (X * r13 - Z * r11)
        J[n:, 2] = -s * (X * r23 - Z * r21)

        J[:n, 3] = s * (X * r12 - Y * r11)
        J[n:, 3] = s * (X * r22 - Y * r21)

        # 3. Derivative w.r.t. translation (columns 4-5)
        J[:n, 4] = 1.0
        J[n:, 5] = 1.0

        # 4. Derivative w.r.t. shape parameters (columns 6:) - VECTORIZED
        # Uses pre-sliced principal components for ~2x speedup
        J[:n, 6:] = s * (r11 * self._princ_comp_x + r12 * self._princ_comp_y + r13 * self._princ_comp_z)
        J[n:, 6:] = s * (r21 * self._princ_comp_x + r22 * self._princ_comp_y + r23 * self._princ_comp_z)

        return J

    def init_params_from_eye_landmarks(self, eye_landmarks: np.ndarray) -> np.ndarray:
        """
        Initialize eye PDM parameters from eye landmarks in image coordinates.

        This estimates the rigid transformation (scale, rotation, translation)
        from the provided eye landmarks to match the mean shape.

        Args:
            eye_landmarks: 2D eye landmarks, shape (n_points, 2) or subset

        Returns:
            params: Initial parameter vector
        """
        params = np.zeros(self.n_params)

        if eye_landmarks is None or len(eye_landmarks) == 0:
            params[0] = 1.0  # Default scale
            return params

        # Get mean shape in 2D
        mean_2d = self.params_to_landmarks_2d(np.zeros(self.n_params))

        # Compute scale from bounding box ratio
        eye_bbox = self._compute_bbox(eye_landmarks)
        mean_bbox = self._compute_bbox(mean_2d)

        eye_width = eye_bbox[2] - eye_bbox[0]
        eye_height = eye_bbox[3] - eye_bbox[1]
        mean_width = mean_bbox[2] - mean_bbox[0]
        mean_height = mean_bbox[3] - mean_bbox[1]

        if mean_width > 0 and mean_height > 0:
            scale = ((eye_width / mean_width) + (eye_height / mean_height)) / 2.0
        else:
            scale = 1.0

        # Compute translation
        eye_center = np.mean(eye_landmarks, axis=0)
        mean_center = np.mean(mean_2d, axis=0)
        tx = eye_center[0] - scale * mean_center[0]
        ty = eye_center[1] - scale * mean_center[1]

        params[0] = scale
        params[4] = tx
        params[5] = ty

        return params

    def _compute_bbox(self, landmarks: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute bounding box [xmin, ymin, xmax, ymax] from landmarks."""
        return (
            landmarks[:, 0].min(),
            landmarks[:, 1].min(),
            landmarks[:, 0].max(),
            landmarks[:, 1].max()
        )

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (XYZ convention).

        Args:
            euler: [pitch, yaw, roll] in radians

        Returns:
            R: 3x3 rotation matrix
        """
        s1, s2, s3 = np.sin(euler[0]), np.sin(euler[1]), np.sin(euler[2])
        c1, c2, c3 = np.cos(euler[0]), np.cos(euler[1]), np.cos(euler[2])

        R = np.array([
            [c2 * c3,              -c2 * s3,             s2],
            [c1 * s3 + c3 * s1 * s2,  c1 * c3 - s1 * s2 * s3,  -c2 * s1],
            [s1 * s3 - c1 * c3 * s2,  c3 * s1 + c1 * s2 * s3,   c1 * c2]
        ], dtype=np.float32)

        return R

    def clamp_params(self, params: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Clamp parameters to valid ranges.

        Args:
            params: Parameter vector
            n_std: Number of standard deviations for shape clamping

        Returns:
            Clamped parameters
        """
        params = params.copy()

        # Clamp scale to be positive (minimum 0.1 to avoid degenerate cases)
        # C++ always has positive scale from bounding box initialization
        params[0] = max(params[0], 0.1)

        # Clamp rotation
        params[1:4] = np.clip(params[1:4], -np.pi, np.pi)

        # Clamp shape parameters
        std_devs = np.sqrt(self.eigen_values.flatten())
        lower = -n_std * std_devs
        upper = n_std * std_devs
        params[6:] = np.clip(params[6:], lower, upper)

        return params

    def update_params(self, params: np.ndarray, delta_p: np.ndarray) -> np.ndarray:
        """
        Update parameters with delta, using proper rotation composition.

        Args:
            params: Current parameters
            delta_p: Parameter update

        Returns:
            Updated parameters
        """
        params = params.copy().flatten()
        delta_p = delta_p.flatten()

        # Scale and translation: simple addition
        params[0] += delta_p[0]
        params[4] += delta_p[4]
        params[5] += delta_p[5]

        # Rotation: compose on SO(3) manifold
        euler = np.array([params[1], params[2], params[3]])
        R1 = self._euler_to_rotation_matrix(euler)

        # Build incremental rotation from small-angle approximation
        R2 = np.eye(3, dtype=np.float32)
        R2[0, 1] = -delta_p[3]  # -wz
        R2[1, 0] = delta_p[3]   # wz
        R2[0, 2] = delta_p[2]   # wy
        R2[2, 0] = -delta_p[2]  # -wy
        R2[1, 2] = -delta_p[1]  # -wx
        R2[2, 1] = delta_p[1]   # wx

        # Orthonormalize and compose
        U, S, Vt = np.linalg.svd(R2)
        R2 = U @ Vt
        R3 = R1 @ R2

        # Convert back to Euler angles
        euler_new = self._rotation_matrix_to_euler(R3)
        params[1:4] = euler_new

        # Shape parameters: simple addition
        if len(delta_p) > 6:
            params[6:] += delta_p[6:]

        return params

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles."""
        # Using quaternion intermediate for stability
        trace = np.trace(R)
        q0 = np.sqrt(max(1e-10, 1 + trace)) / 2.0

        q1 = (R[2, 1] - R[1, 2]) / (4.0 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4.0 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4.0 * q0)

        # Quaternion to Euler
        t1 = np.clip(2.0 * (q0 * q2 + q1 * q3), -1.0, 1.0)
        yaw = np.arcsin(t1)
        pitch = np.arctan2(2.0 * (q0 * q1 - q2 * q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        roll = np.arctan2(2.0 * (q0 * q3 - q1 * q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def get_info(self) -> dict:
        """Get EyePDM information."""
        return {
            'n_points': self.n_points,
            'n_modes': self.n_modes,
            'n_params': self.n_params,
            'mean_shape_shape': self.mean_shape.shape,
            'princ_comp_shape': self.princ_comp.shape,
            'eigen_values_shape': self.eigen_values.shape,
        }
