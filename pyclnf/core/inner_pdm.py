"""
Inner Point Distribution Model (PDM) for hierarchical face refinement.

51-point PDM for the inner face region (landmarks 17-67 of the main 68-point model).
This implements the same algorithm as the main 68-point PDM but with different dimensions:
    - 51 points (vs 68)
    - 32 shape modes (vs 34)

Used by the hierarchical inner model refinement for eyebrows, nose, eyes, and mouth.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class InnerPDM:
    """51-point PDM for inner face model refinement."""

    def __init__(self, model_dir: str):
        """
        Load inner PDM from exported NumPy files.

        Args:
            model_dir: Directory containing mean_shape.npy, eigenvectors.npy, eigenvalues.npy
        """
        self.model_dir = Path(model_dir)

        # Load PDM components
        self.mean_shape = np.load(self.model_dir / 'mean_shape.npy')  # (51, 3)
        self.eigenvectors = np.load(self.model_dir / 'eigenvectors.npy')  # (153, 32)
        self.eigenvalues = np.load(self.model_dir / 'eigenvalues.npy')  # (32,)

        # Reshape mean_shape to match main PDM format [x1..xn, y1..yn, z1..zn]
        # Input is (51, 3) row-major, need (153,) column-major
        mean_3d = self.mean_shape  # (51, 3)
        self.mean_shape_flat = np.concatenate([
            mean_3d[:, 0],  # x coords
            mean_3d[:, 1],  # y coords
            mean_3d[:, 2]   # z coords
        ])  # (153,)

        # eigenvectors is already (153, 32) in the right format
        self.princ_comp = self.eigenvectors

        # Extract dimensions
        self.n_points = 51  # 51 inner landmarks
        self.n_modes = self.eigenvectors.shape[1]  # 32 shape modes

        # Parameter vector size: scale(1) + rotation(3) + translation(2) + shape(n_modes)
        self.n_params = 6 + self.n_modes  # 38

        # Store eigenvalues in format expected by optimizer (1, n_modes)
        self.eigen_values = self.eigenvalues.reshape(1, -1)

    def params_to_landmarks_3d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 3D landmark positions.

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]

        Returns:
            landmarks_3d: 3D landmark positions, shape (n_points, 3)
        """
        params = params.flatten()

        # Extract parameters (OpenFace order)
        s = params[0]  # Scale
        wx, wy, wz = params[1], params[2], params[3]  # Rotation (Euler)
        tx, ty = params[4], params[5]  # Translation
        q = params[6:]  # Shape parameters

        # Apply PCA: shape = mean + eigenvectors @ shape_params
        shape_3d = self.mean_shape_flat + self.princ_comp @ q  # (153,)

        # Reshape to (n, 3) by extracting x, y, z blocks
        n = self.n_points
        shape_3d = np.column_stack([
            shape_3d[:n],      # x coordinates
            shape_3d[n:2*n],   # y coordinates
            shape_3d[2*n:3*n]  # z coordinates
        ])  # (51, 3)

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
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]

        Returns:
            landmarks_2d: 2D landmark positions, shape (n_points, 2)
        """
        landmarks_3d = self.params_to_landmarks_3d(params)
        return landmarks_3d[:, :2]

    def get_reference_shape(self, patch_scaling: float, params_local: np.ndarray = None) -> np.ndarray:
        """
        Generate reference shape at fixed scale for patch evaluation.

        Args:
            patch_scaling: Fixed scale (1.0 for inner model)
            params_local: Local shape parameters (default: zeros)

        Returns:
            reference_shape: 2D landmarks at reference scale, shape (n_points, 2)
        """
        if params_local is None:
            params_local = np.zeros(self.n_modes)

        # Reference params: [scale, 0, 0, 0, 0, 0, local...]
        global_ref = np.array([patch_scaling, 0.0, 0.0, 0.0, 0.0, 0.0])
        ref_params = np.concatenate([global_ref, params_local])

        return self.params_to_landmarks_2d(ref_params)

    def compute_jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of 2D landmarks with respect to parameters.

        The Jacobian J has shape (2*n_points, n_params) in STACKED format:
            J[i, j] = d(landmark_i.x) / d(param_j)      for i = 0 to n-1
            J[i+n, j] = d(landmark_i.y) / d(param_j)   for i = 0 to n-1

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]

        Returns:
            jacobian: Jacobian matrix, shape (2*n_points, n_params) = (102, 38)
        """
        params = params.flatten()

        # Extract parameters
        s = params[0]
        wx, wy, wz = params[1], params[2], params[3]
        q = params[6:]

        # Compute 3D shape before rotation
        shape_3d = self.mean_shape_flat + self.princ_comp @ q
        n = self.n_points

        # Extract X, Y, Z coordinates
        X = shape_3d[:n]
        Y = shape_3d[n:2*n]
        Z = shape_3d[2*n:3*n]

        # Compute rotation matrix
        R = self._euler_to_rotation_matrix(np.array([wx, wy, wz]))

        # Extract rotation matrix elements
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]

        # Initialize Jacobian
        J = np.zeros((2 * self.n_points, self.n_params), dtype=np.float64)

        # 1. d/ds (scale) - column 0
        J[:n, 0] = X * r11 + Y * r12 + Z * r13
        J[n:, 0] = X * r21 + Y * r22 + Z * r23

        # 2. d/dw (rotation) - columns 1-3
        # wx (pitch)
        J[:n, 1] = s * (Y * r13 - Z * r12)
        J[n:, 1] = s * (Y * r23 - Z * r22)

        # wy (yaw)
        J[:n, 2] = -s * (X * r13 - Z * r11)
        J[n:, 2] = -s * (X * r23 - Z * r21)

        # wz (roll)
        J[:n, 3] = s * (X * r12 - Y * r11)
        J[n:, 3] = s * (X * r22 - Y * r21)

        # 3. d/dt (translation) - columns 4-5
        J[:n, 4] = 1.0  # dx/dtx
        J[n:, 5] = 1.0  # dy/dty

        # 4. d/dq (shape) - columns 6+
        for i in range(self.n_modes):
            phi_i = self.princ_comp[:, i]
            phi_x = phi_i[:n]
            phi_y = phi_i[n:2*n]
            phi_z = phi_i[2*n:3*n]

            J[:n, 6 + i] = s * (r11 * phi_x + r12 * phi_y + r13 * phi_z)
            J[n:, 6 + i] = s * (r21 * phi_x + r22 * phi_y + r23 * phi_z)

        return J

    def compute_jacobian_rigid(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian for rigid parameters only.

        Args:
            params: Parameter vector

        Returns:
            jacobian: Jacobian matrix, shape (2*n_points, 6)
        """
        J_full = self.compute_jacobian(params)
        return J_full[:, :6]

    def update_params(self, params: np.ndarray, delta_p: np.ndarray) -> np.ndarray:
        """
        Update parameters with proper manifold update for rotations.

        Args:
            params: Current parameters
            delta_p: Parameter update

        Returns:
            updated_params: New parameters
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

        # Build incremental rotation from delta (small-angle approximation)
        R2 = np.eye(3, dtype=np.float32)
        R2[0, 1] = -delta_p[3]  # -wz
        R2[1, 0] = delta_p[3]   # wz
        R2[0, 2] = delta_p[2]   # wy
        R2[2, 0] = -delta_p[2]  # -wy
        R2[1, 2] = -delta_p[1]  # -wx
        R2[2, 1] = delta_p[1]   # wx

        R2 = self._orthonormalize(R2)
        R3 = R1 @ R2
        R3 = self._orthonormalize(R3)

        # Convert back to Euler
        axis_angle = self._rotation_matrix_to_axis_angle(R3)
        euler_new = self._axis_angle_to_euler(axis_angle)

        if np.any(np.isnan(euler_new)):
            euler_new = np.array([0.0, 0.0, 0.0])

        params[1] = euler_new[0]
        params[2] = euler_new[1]
        params[3] = euler_new[2]

        # Shape parameters: simple addition
        if len(delta_p) > 6:
            params[6:] += delta_p[6:]

        return params

    def clamp_params(self, params: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Clamp parameters to valid ranges.

        Args:
            params: Parameter vector
            n_std: Number of standard deviations for shape clamping

        Returns:
            clamped_params: Clamped parameters
        """
        params = params.copy()

        # Clamp rotation to [-pi, pi]
        params[1:4] = np.clip(params[1:4], -np.pi, np.pi)

        # Clamp shape parameters to +/- n_std * sqrt(eigenvalue)
        std_devs = np.sqrt(self.eigenvalues.flatten())
        lower = -n_std * std_devs
        upper = n_std * std_devs
        params[6:] = np.clip(params[6:], lower, upper)

        return params

    def calc_params(self, landmarks_2d: np.ndarray) -> np.ndarray:
        """
        Fit PDM parameters to 2D landmarks using Procrustes + least squares.

        This is used to initialize inner PDM parameters from extracted landmarks.

        Args:
            landmarks_2d: Target 2D landmarks, shape (51, 2)

        Returns:
            params: Fitted parameter vector
        """
        # Initialize with neutral parameters
        params = np.zeros(self.n_params)
        params[0] = 1.0  # scale = 1

        # Estimate rigid parameters using Procrustes
        # Target: landmarks_2d (51, 2)
        # Source: mean shape projected to 2D
        mean_2d = self.get_reference_shape(1.0)  # (51, 2)

        # Compute centroids
        target_center = landmarks_2d.mean(axis=0)
        source_center = mean_2d.mean(axis=0)

        # Center both shapes
        target_centered = landmarks_2d - target_center
        source_centered = mean_2d - source_center

        # Compute scale from RMS distances
        target_rms = np.sqrt(np.mean(target_centered ** 2))
        source_rms = np.sqrt(np.mean(source_centered ** 2))
        scale = target_rms / (source_rms + 1e-10)

        # Translation
        tx = target_center[0] - scale * source_center[0]
        ty = target_center[1] - scale * source_center[1]

        # Set initial parameters
        params[0] = scale
        params[4] = tx
        params[5] = ty
        params[6:] = 0.0  # Start with mean shape

        # Refine with a few Gauss-Newton iterations
        params = self._refine_params(landmarks_2d, params, max_iter=20)

        return params

    def _refine_params(self, landmarks_2d: np.ndarray, params: np.ndarray,
                       max_iter: int = 20, damping: float = 0.75) -> np.ndarray:
        """
        Refine parameters using Gauss-Newton optimization.

        Args:
            landmarks_2d: Target landmarks (51, 2)
            params: Initial parameters
            max_iter: Maximum iterations
            damping: Update damping factor

        Returns:
            refined_params: Optimized parameters
        """
        params = params.copy()
        n = self.n_points

        # Target in stacked format
        target = np.concatenate([landmarks_2d[:, 0], landmarks_2d[:, 1]])

        # Regularization for shape parameters
        reg_diag = np.zeros(self.n_params)
        reg_diag[6:] = 1.0 / (self.eigenvalues.flatten() + 1e-10)

        prev_error = np.inf
        for _ in range(max_iter):
            # Current landmarks
            current_2d = self.params_to_landmarks_2d(params)
            current = np.concatenate([current_2d[:, 0], current_2d[:, 1]])

            # Residual and error
            residual = target - current
            error = np.linalg.norm(residual)

            # Check convergence
            if error < 1e-6 or (prev_error - error) / (prev_error + 1e-10) < 0.001:
                break
            prev_error = error

            # Jacobian
            J = self.compute_jacobian(params)

            # Hessian with regularization
            H = J.T @ J + np.diag(reg_diag)

            # Gradient
            g = J.T @ residual - reg_diag * params

            # Solve
            try:
                delta_p = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta_p = np.linalg.lstsq(H, g, rcond=None)[0]

            # Update
            params = self.update_params(params, damping * delta_p)
            params = self.clamp_params(params)

        return params

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix (XYZ convention)."""
        s1, s2, s3 = np.sin(euler)
        c1, c2, c3 = np.cos(euler)

        R = np.array([
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]
        ], dtype=np.float32)

        return R

    def _orthonormalize(self, R: np.ndarray) -> np.ndarray:
        """Orthonormalize matrix using SVD."""
        U, S, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle."""
        trace = np.trace(R)
        theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

        if theta < 1e-10:
            return np.zeros(3, dtype=np.float32)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=np.float32)

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.zeros(3, dtype=np.float32)

        return theta * axis / axis_norm

    def _axis_angle_to_euler(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to Euler angles."""
        theta = np.linalg.norm(axis_angle)
        if theta < 1e-10:
            return np.zeros(3, dtype=np.float32)

        axis = axis_angle / theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)

        R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Convert to quaternion then Euler
        q0 = np.sqrt(max(1e-10, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        q1 = (R[2, 1] - R[1, 2]) / (4.0 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4.0 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4.0 * q0)

        t1 = np.clip(2.0 * (q0 * q2 + q1 * q3), -1.0, 1.0)
        yaw = np.arcsin(t1)
        pitch = np.arctan2(2.0 * (q0 * q1 - q2 * q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        roll = np.arctan2(2.0 * (q0 * q3 - q1 * q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

        return np.array([pitch, yaw, roll], dtype=np.float32)


def test_inner_pdm():
    """Test inner PDM implementation."""
    print("=" * 60)
    print("Testing Inner PDM (51-point)")
    print("=" * 60)

    model_dir = "pyclnf/pyclnf/models/exported_inner_pdm"
    pdm = InnerPDM(model_dir)

    print(f"\nInner PDM Info:")
    print(f"  n_points: {pdm.n_points}")
    print(f"  n_modes: {pdm.n_modes}")
    print(f"  n_params: {pdm.n_params}")
    print(f"  mean_shape_flat shape: {pdm.mean_shape_flat.shape}")
    print(f"  princ_comp shape: {pdm.princ_comp.shape}")
    print(f"  eigenvalues shape: {pdm.eigenvalues.shape}")

    # Test neutral pose
    params = np.zeros(pdm.n_params)
    params[0] = 1.0  # scale

    landmarks_2d = pdm.params_to_landmarks_2d(params)
    print(f"\nNeutral pose landmarks shape: {landmarks_2d.shape}")
    print(f"  First 3 landmarks: {landmarks_2d[:3]}")

    # Test Jacobian
    J = pdm.compute_jacobian(params)
    print(f"\nJacobian shape: {J.shape}")
    print(f"  Expected: (102, 38)")

    # Verify Jacobian numerically
    h = 1e-6
    for param_idx in [0, 1, 6]:
        params_plus = params.copy()
        params_plus[param_idx] += h
        lm_plus = pdm.params_to_landmarks_2d(params_plus)

        params_minus = params.copy()
        params_minus[param_idx] -= h
        lm_minus = pdm.params_to_landmarks_2d(params_minus)

        numerical = np.concatenate([
            (lm_plus[:, 0] - lm_minus[:, 0]) / (2 * h),
            (lm_plus[:, 1] - lm_minus[:, 1]) / (2 * h)
        ])
        analytical = J[:, param_idx]
        error = np.linalg.norm(numerical - analytical)
        print(f"  Param {param_idx} Jacobian error: {error:.2e}")

    print("\n" + "=" * 60)
    print("Inner PDM tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_inner_pdm()
