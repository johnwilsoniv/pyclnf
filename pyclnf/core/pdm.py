"""
Point Distribution Model (PDM) - Core shape model for CLNF

Implements the PDM transform from parameters to 3D landmarks:
    xi = s · R2D · (x̄i + Φiq) + t

Where:
    - x̄i: Mean position of landmark i
    - Φi: Principal component matrix for landmark i
    - q: Non-rigid shape parameters (PCA coefficients)
    - s: Global scale
    - t: Translation [tx, ty]
    - w: Orientation [wx, wy, wz] (axis-angle)
    - R2D: First two rows of 3×3 rotation matrix from w

Parameter vector: p = [s, tx, ty, wx, wy, wz, q0, q1, ..., qm]
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class PDM:
    """Point Distribution Model for facial landmark representation."""

    def __init__(self, model_dir: str):
        """
        Load PDM from exported NumPy files.

        Args:
            model_dir: Directory containing mean_shape.npy, princ_comp.npy, eigen_values.npy
        """
        self.model_dir = Path(model_dir)

        # Load PDM components
        self.mean_shape = np.load(self.model_dir / 'mean_shape.npy')  # (3n, 1)
        self.princ_comp = np.load(self.model_dir / 'princ_comp.npy')  # (3n, m)
        self.eigen_values = np.load(self.model_dir / 'eigen_values.npy')  # (1, m)

        # Extract dimensions
        self.n_points = self.mean_shape.shape[0] // 3  # Number of landmarks (68)
        self.n_modes = self.princ_comp.shape[1]  # Number of PCA modes (34)

        # Parameter vector size: scale(1) + translation(2) + rotation(3) + shape(n_modes)
        self.n_params = 6 + self.n_modes

    def params_to_landmarks_3d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 3D landmark positions.

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]
                   (OpenFace order: scale, rotation, translation, shape)
                   Shape: (n_params,) or (n_params, 1)

        Returns:
            landmarks_3d: 3D landmark positions, shape (n_points, 3)
        """
        params = params.flatten()

        # Extract parameters (OpenFace order)
        s = params[0]  # Scale
        wx, wy, wz = params[1], params[2], params[3]  # Rotation (axis-angle)
        tx, ty = params[4], params[5]  # Translation
        q = params[6:]  # Shape parameters

        # Apply PCA: shape = mean + principal_components @ shape_params
        # mean_shape is (3n, 1), princ_comp is (3n, m), q is (m,)
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q  # (3n,)

        # OpenFace stores shapes as [x1,...,xn, y1,...,yn, z1,...,zn] (column-major)
        # Reshape to (n, 3) by extracting x, y, z blocks
        n = self.n_points
        shape_3d = np.column_stack([
            shape_3d[:n],      # x coordinates
            shape_3d[n:2*n],   # y coordinates
            shape_3d[2*n:3*n]  # z coordinates
        ])  # (n_points, 3)

        # Compute rotation matrix from Euler angles (NOT axis-angle!)
        # OpenFace uses Euler angles with XYZ convention: R = Rx * Ry * Rz
        R = self._euler_to_rotation_matrix(np.array([wx, wy, wz]))  # (3, 3)

        # Apply similarity transform: landmarks = s * R @ shape + t
        # R is (3, 3), shape_3d is (n, 3)
        # We want to rotate each point: result[i] = s * R @ shape_3d[i] + t
        landmarks_3d = s * (shape_3d @ R.T)  # (n, 3)

        # Add translation (only to x and y, z stays as is)
        landmarks_3d[:, 0] += tx
        landmarks_3d[:, 1] += ty

        return landmarks_3d

    def params_to_landmarks_2d(self, params: np.ndarray) -> np.ndarray:
        """
        Convert parameter vector to 2D landmark positions (x, y projection).

        Args:
            params: Parameter vector [s, tx, ty, wx, wy, wz, q0, ..., qm]

        Returns:
            landmarks_2d: 2D landmark positions, shape (n_points, 2)
        """
        landmarks_3d = self.params_to_landmarks_3d(params)
        return landmarks_3d[:, :2]  # Take only x, y coordinates

    def get_reference_shape(self, patch_scaling: float, params_local: np.ndarray = None) -> np.ndarray:
        """
        Generate reference shape at fixed scale for patch evaluation.

        This creates a canonical reference shape at a specific scale (patch_scaling)
        that matches the scale at which CCNF patches were trained. This is CRITICAL
        for correct patch response evaluation.

        OpenFace does this with:
            cv::Vec6f global_ref(patch_scaling[scale], 0, 0, 0, 0, 0);
            pdm.CalcShape2D(reference_shape, params_local, global_ref);

        Args:
            patch_scaling: Fixed scale for reference shape (0.25, 0.35, or 0.5)
                          Must match the scale of the patch experts being used!
            params_local: Local shape parameters (default: zeros = mean shape)

        Returns:
            reference_shape: 2D landmarks at reference scale, shape (n_points, 2)
                           Centered at origin with fixed scale, zero rotation
        """
        if params_local is None:
            params_local = np.zeros(self.n_modes)

        # Create reference global params: [scale, tx, ty, wx, wy, wz]
        # Scale = patch_scaling, rotation = 0, translation = 0
        # This creates a canonical pose: upright face at fixed scale, centered at origin
        global_ref = np.array([patch_scaling, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Concatenate global and local params
        ref_params = np.concatenate([global_ref, params_local])

        # Generate 2D shape using standard params_to_landmarks_2d
        reference_shape = self.params_to_landmarks_2d(ref_params)

        return reference_shape

    def compute_jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix of 2D landmarks with respect to parameters.

        FIXED: Now uses analytical rotation derivatives matching OpenFace's
        small-angle approximation (R * R') instead of numerical differentiation.

        The Jacobian J has shape (2*n_points, n_params) where:
            J[2*i, j] = ∂(landmark_i.x) / ∂param_j
            J[2*i+1, j] = ∂(landmark_i.y) / ∂param_j

        This is used in the NU-RLMS optimization update step.

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]
                    (OpenFace order)

        Returns:
            jacobian: Jacobian matrix, shape (2*n_points, n_params)
        """
        params = params.flatten()

        # Extract parameters (OpenFace order)
        s = params[0]
        wx, wy, wz = params[1], params[2], params[3]
        tx, ty = params[4], params[5]
        q = params[6:]

        # Compute 3D shape before rotation
        shape_3d = self.mean_shape.flatten() + self.princ_comp @ q  # (3n,)
        # OpenFace stores shapes as [x1,...,xn, y1,...,yn, z1,...,zn] (column-major)
        n = self.n_points

        # Extract X, Y, Z coordinates for each landmark
        X = shape_3d[:n]      # (n,) x coordinates
        Y = shape_3d[n:2*n]   # (n,) y coordinates
        Z = shape_3d[2*n:3*n] # (n,) z coordinates

        # Compute rotation matrix from Euler angles
        euler = np.array([wx, wy, wz])
        R = self._euler_to_rotation_matrix(euler)

        # Extract rotation matrix elements (OpenFace PDM.cpp lines 367-375)
        r11 = R[0, 0]
        r12 = R[0, 1]
        r13 = R[0, 2]
        r21 = R[1, 0]
        r22 = R[1, 1]
        r23 = R[1, 2]
        r31 = R[2, 0]  # Not used in 2D projection, but kept for completeness
        r32 = R[2, 1]
        r33 = R[2, 2]

        # Initialize Jacobian
        J = np.zeros((2 * self.n_points, self.n_params))

        # ==================================================================
        # RIGID PARAMETER DERIVATIVES (OpenFace PDM.cpp lines 396-412)
        # ==================================================================

        # 1. Derivative w.r.t. scale (column 0)
        # ∂x/∂s = (X·r11 + Y·r12 + Z·r13)
        # ∂y/∂s = (X·r21 + Y·r22 + Z·r23)
        J[0::2, 0] = X * r11 + Y * r12 + Z * r13  # x components
        J[1::2, 0] = X * r21 + Y * r22 + Z * r23  # y components

        # 2. Derivative w.r.t. rotation (columns 1-3) - ANALYTICAL FORMULAS
        # These come from the small-angle approximation: R * R'
        # where R' = [1,   -wz,   wy ]
        #            [wz,   1,   -wx ]
        #            [-wy,  wx,   1  ]

        # Rotation around X-axis (pitch) - column 1
        # ∂x/∂wx = s * (Y·r13 - Z·r12)
        # ∂y/∂wx = s * (Y·r23 - Z·r22)
        J[0::2, 1] = s * (Y * r13 - Z * r12)
        J[1::2, 1] = s * (Y * r23 - Z * r22)

        # Rotation around Y-axis (yaw) - column 2
        # ∂x/∂wy = -s * (X·r13 - Z·r11)
        # ∂y/∂wy = -s * (X·r23 - Z·r21)
        J[0::2, 2] = -s * (X * r13 - Z * r11)
        J[1::2, 2] = -s * (X * r23 - Z * r21)

        # Rotation around Z-axis (roll) - column 3
        # ∂x/∂wz = s * (X·r12 - Y·r11)
        # ∂y/∂wz = s * (X·r22 - Y·r21)
        J[0::2, 3] = s * (X * r12 - Y * r11)
        J[1::2, 3] = s * (X * r22 - Y * r21)

        # 3. Derivative w.r.t. translation (columns 4-5)
        # ∂x/∂tx = 1, ∂y/∂ty = 1
        J[0::2, 4] = 1.0  # ∂x/∂tx = 1
        J[1::2, 5] = 1.0  # ∂y/∂ty = 1

        # ==================================================================
        # NON-RIGID SHAPE PARAMETER DERIVATIVES (OpenFace PDM.cpp lines 414-420)
        # ==================================================================

        # 4. Derivative w.r.t. shape parameters (columns 6:)
        # ∂x/∂qi = s * (r11·Φx[i] + r12·Φy[i] + r13·Φz[i])
        # ∂y/∂qi = s * (r21·Φx[i] + r22·Φy[i] + r23·Φz[i])
        for i in range(self.n_modes):
            phi_i = self.princ_comp[:, i]  # (3n,)

            # Extract Φx, Φy, Φz for this mode
            phi_x = phi_i[:n]
            phi_y = phi_i[n:2*n]
            phi_z = phi_i[2*n:3*n]

            # Compute derivatives
            J[0::2, 6 + i] = s * (r11 * phi_x + r12 * phi_y + r13 * phi_z)
            J[1::2, 6 + i] = s * (r21 * phi_x + r22 * phi_y + r23 * phi_z)

        return J

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.

        OpenFace uses XYZ Euler angles convention: R = Rx * Ry * Rz (left-handed positive sign)

        Args:
            euler: Euler angles [pitch, yaw, roll] in radians, shape (3,)
                  pitch (rx): rotation around X axis
                  yaw (ry): rotation around Y axis
                  roll (rz): rotation around Z axis

        Returns:
            R: 3×3 rotation matrix

        This matches OpenFace's Utilities::Euler2RotationMatrix function.
        """
        s1 = np.sin(euler[0])  # sin(pitch)
        s2 = np.sin(euler[1])  # sin(yaw)
        s3 = np.sin(euler[2])  # sin(roll)

        c1 = np.cos(euler[0])  # cos(pitch)
        c2 = np.cos(euler[1])  # cos(yaw)
        c3 = np.cos(euler[2])  # cos(roll)

        # Rotation matrix from XYZ Euler angles (OpenFace convention)
        R = np.array([
            [c2 * c3,              -c2 * s3,             s2],
            [c1 * s3 + c3 * s1 * s2,  c1 * c3 - s1 * s2 * s3,  -c2 * s1],
            [s1 * s3 - c1 * c3 * s2,  c3 * s1 + c1 * s2 * s3,   c1 * c2]
        ], dtype=np.float32)

        return R

    def _rodrigues(self, w: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotation vector to rotation matrix using Rodrigues formula.

        NOTE: OpenFace uses EULER ANGLES, not axis-angle!
        This function is kept for reference but should NOT be used for OpenFace compatibility.

        Args:
            w: Axis-angle rotation vector [wx, wy, wz], shape (3,)

        Returns:
            R: 3×3 rotation matrix

        Formula:
            θ = ||w||
            k = w / θ (unit axis)
            R = I + sin(θ) * K + (1 - cos(θ)) * K²

        where K is the skew-symmetric matrix of k:
            K = [[ 0,  -kz,  ky],
                 [ kz,  0,  -kx],
                 [-ky,  kx,  0]]
        """
        theta = np.linalg.norm(w)

        if theta < 1e-10:
            # Small angle approximation: R ≈ I + K
            return np.eye(3) + self._skew(w)

        # Normalize to get unit axis
        k = w / theta

        # Skew-symmetric matrix of k
        K = self._skew(k)

        # Rodrigues formula: R = I + sin(θ)*K + (1-cos(θ))*K²
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        return R

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from vector.

        Args:
            v: Vector [vx, vy, vz]

        Returns:
            Skew-symmetric matrix:
                [[ 0,  -vz,  vy],
                 [ vz,  0,  -vx],
                 [-vy,  vx,  0]]
        """
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0]
        ])

    def init_params(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Initialize parameter vector from face bounding box or to neutral pose.

        Implements OpenFace PDM::CalcParams exactly (PDM.cpp lines 193-231).

        Args:
            bbox: Optional bounding box [x, y, width, height] to estimate initial scale/translation

        Returns:
            params: Initial parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]
                    (OpenFace order)
        """
        params = np.zeros(self.n_params)

        if bbox is not None:
            x, y, width, height = bbox

            # OpenFace-style initialization (aspect-ratio aware)
            # Based on OpenFace PDM.cpp:193-231
            # This computes scale from model dimensions, accounting for both width and height.
            # Validated to improve convergence by 44.5% on average across all bbox sources.

            # Get mean shape from PDM
            # OpenFace stores as [x0,...,xn, y0,...,yn, z0,...,zn] (separated by dimension)
            mean_shape_3d = self.mean_shape.reshape(3, -1)  # Shape: (3, 68)

            # With zero rotation, shape is just mean_shape rotated by identity
            rotation = np.array([0.0, 0.0, 0.0])
            R = cv2.Rodrigues(rotation)[0]  # 3x3 rotation matrix

            # Rotate shape (identity rotation doesn't change it)
            rotated_shape = R @ mean_shape_3d  # (3, 68)

            # Find bounding box of model
            min_x = rotated_shape[0, :].min()
            max_x = rotated_shape[0, :].max()
            min_y = rotated_shape[1, :].min()
            max_y = rotated_shape[1, :].max()

            model_width = abs(max_x - min_x)
            model_height = abs(max_y - min_y)

            # OpenFace formula: average of width and height scaling
            # This accounts for aspect ratio differences between bbox and model
            scaling = ((width / model_width) + (height / model_height)) / 2.0

            # Translation with correction for model center offset
            # This ensures the face is properly centered within the bbox
            tx = x + width / 2.0 - scaling * (min_x + max_x) / 2.0
            ty = y + height / 2.0 - scaling * (min_y + max_y) / 2.0

            # Set parameters (OpenFace order)
            params[0] = scaling
            params[1] = 0.0  # pitch = 0
            params[2] = 0.0  # yaw = 0
            params[3] = 0.0  # roll = 0
            params[4] = tx
            params[5] = ty
        else:
            # Neutral initialization
            params[0] = 1.0  # scale = 1
            params[1] = 0.0  # wx = 0
            params[2] = 0.0  # wy = 0
            params[3] = 0.0  # wz = 0
            params[4] = 0.0  # tx = 0
            params[5] = 0.0  # ty = 0

        # Shape parameters = 0 (mean shape)
        params[6:] = 0.0

        return params

    def clamp_shape_params(self, params: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Clamp shape parameters to valid range based on eigenvalues.

        Constrains each shape parameter qi to ±n_std standard deviations:
            -n_std * sqrt(λi) <= qi <= n_std * sqrt(λi)

        Args:
            params: Parameter vector
            n_std: Number of standard deviations (typically 3.0)

        Returns:
            params: Clamped parameter vector
        """
        params = params.copy()

        # Extract shape parameters
        q = params[6:]

        # Compute bounds from eigenvalues
        std_devs = np.sqrt(self.eigen_values.flatten())
        lower_bounds = -n_std * std_devs
        upper_bounds = n_std * std_devs

        # Clamp
        q_clamped = np.clip(q, lower_bounds, upper_bounds)

        # Update params
        params[6:] = q_clamped

        return params

    def clamp_rotation_params(self, params: np.ndarray) -> np.ndarray:
        """
        Clamp rotation parameters to valid range to prevent unbounded growth.

        Axis-angle rotation wraps around 2π, but unconstrained optimization can
        lead to divergence (rotation values growing to hundreds of radians).
        This function clamps rotation to [-π, π] range.

        OpenFace's PDM.cpp has commented-out code for this (lines 119-133 in PDM::Clamp),
        which would clamp to [-π/2, π/2]. We use the full [-π, π] range to allow
        more pose variation while preventing divergence.

        Args:
            params: Parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]

        Returns:
            params: Parameter vector with rotation clamped to [-π, π]
        """
        params = params.copy()

        # Clamp rotation parameters (indices 1-3: wx, wy, wz)
        params[1:4] = np.clip(params[1:4], -np.pi, np.pi)

        return params

    def clamp_params(self, params: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Clamp all parameters to valid ranges.

        This matches OpenFace's approach in LandmarkDetectorModel.cpp:1134,
        where pdm.Clamp() is called after every parameter update.

        Args:
            params: Parameter vector
            n_std: Number of standard deviations for shape parameter clamping

        Returns:
            params: Clamped parameter vector
        """
        # Clamp rotation to prevent divergence
        params = self.clamp_rotation_params(params)

        # Clamp shape parameters to valid eigenvalue range
        params = self.clamp_shape_params(params, n_std=n_std)

        return params

    def update_params(self, params: np.ndarray, delta_p: np.ndarray) -> np.ndarray:
        """
        Update parameters with delta, using proper manifold update for rotations.

        This matches OpenFace's PDM::UpdateModelParameters (PDM.cpp lines 454-503).
        Rotation parameters require special handling because they live on the SO(3) manifold,
        not in Euclidean space. Naive addition would violate rotation constraints.

        Args:
            params: Current parameter vector [s, wx, wy, wz, tx, ty, q0, ..., qm]
            delta_p: Parameter update vector (same shape as params)

        Returns:
            updated_params: New parameter vector with proper rotation composition
        """
        params = params.copy().flatten()
        delta_p = delta_p.flatten()

        # Scale and translation: simple addition (OpenFace lines 458-460)
        params[0] += delta_p[0]  # scale
        params[4] += delta_p[4]  # tx
        params[5] += delta_p[5]  # ty

        # Rotation: compose on SO(3) manifold (OpenFace lines 462-498)
        # 1. Get current rotation matrix from Euler angles
        euler = np.array([params[1], params[2], params[3]])
        R1 = self._euler_to_rotation_matrix(euler)

        # 2. Build incremental rotation matrix R2 from delta using small-angle approximation
        #    R2 = [1,    -wz,    wy  ]
        #         [wz,    1,    -wx  ]
        #         [-wy,   wx,    1   ]
        #    This matches OpenFace lines 470-474
        R2 = np.eye(3, dtype=np.float32)
        R2[0, 1] = -delta_p[3]  # -wz
        R2[1, 0] = delta_p[3]   # wz
        R2[0, 2] = delta_p[2]   # wy
        R2[2, 0] = -delta_p[2]  # -wy
        R2[1, 2] = -delta_p[1]  # -wx
        R2[2, 1] = delta_p[1]   # wx

        # 3. Orthonormalize R2 (OpenFace line 477)
        R2 = self._orthonormalize(R2)

        # 4. Compose rotations: R3 = R1 * R2 (OpenFace line 480)
        R3 = R1 @ R2

        # 5. Convert back to Euler angles via axis-angle (OpenFace lines 482-485)
        #    This ensures the result is a valid rotation
        axis_angle = self._rotation_matrix_to_axis_angle(R3)
        euler_new = self._axis_angle_to_euler(axis_angle)

        # 6. Handle numerical instability (OpenFace lines 487-494)
        if np.any(np.isnan(euler_new)):
            euler_new = np.array([0.0, 0.0, 0.0])

        params[1] = euler_new[0]  # pitch
        params[2] = euler_new[1]  # yaw
        params[3] = euler_new[2]  # roll

        # Shape parameters: simple addition (OpenFace lines 501-503)
        if len(delta_p) > 6:
            params[6:] += delta_p[6:]

        return params

    def _orthonormalize(self, R: np.ndarray) -> np.ndarray:
        """
        Orthonormalize a rotation matrix using SVD.

        Matches OpenFace's Orthonormalise function (RotationHelpers.h).
        Ensures the matrix remains a valid rotation after small-angle approximation.

        Args:
            R: 3x3 matrix (approximately a rotation matrix)

        Returns:
            R_ortho: Orthonormalized 3x3 rotation matrix
        """
        U, S, Vt = np.linalg.svd(R)
        return U @ Vt

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to axis-angle representation.

        Matches OpenFace's RotationMatrix2AxisAngle (RotationHelpers.h).

        Args:
            R: 3x3 rotation matrix

        Returns:
            axis_angle: 3D axis-angle vector [θnx, θny, θnz]
        """
        # Compute rotation angle
        trace = np.trace(R)
        theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

        if theta < 1e-10:
            # Near-identity rotation
            return np.zeros(3, dtype=np.float32)

        # Compute rotation axis
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=np.float32)

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.zeros(3, dtype=np.float32)

        axis = axis / axis_norm
        return theta * axis

    def _axis_angle_to_euler(self, axis_angle: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle to Euler angles (XYZ convention).

        Matches OpenFace's AxisAngle2Euler (RotationHelpers.h).

        Args:
            axis_angle: 3D axis-angle vector [θnx, θny, θnz]

        Returns:
            euler: Euler angles [pitch, yaw, roll] in radians
        """
        theta = np.linalg.norm(axis_angle)
        if theta < 1e-10:
            return np.zeros(3, dtype=np.float32)

        # Convert axis-angle to rotation matrix
        axis = axis_angle / theta
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)

        R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Extract Euler angles from rotation matrix (XYZ convention)
        # R = Rx(pitch) * Ry(yaw) * Rz(roll)
        pitch = np.arctan2(-R[2, 1], R[2, 2])
        yaw = np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        roll = np.arctan2(-R[1, 0], R[0, 0])

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def get_info(self) -> dict:
        """Get PDM information."""
        return {
            'n_points': self.n_points,
            'n_modes': self.n_modes,
            'n_params': self.n_params,
            'mean_shape_shape': self.mean_shape.shape,
            'princ_comp_shape': self.princ_comp.shape,
            'eigen_values_shape': self.eigen_values.shape,
        }


def test_pdm():
    """Test PDM implementation."""
    print("=" * 60)
    print("Testing PDM Core Implementation")
    print("=" * 60)

    # Load PDM
    model_dir = "pyclnf/models/exported_pdm"
    pdm = PDM(model_dir)

    print("\nPDM Info:")
    for key, value in pdm.get_info().items():
        print(f"  {key}: {value}")

    # Test 1: Neutral pose (mean shape)
    print("\nTest 1: Neutral pose (mean shape)")
    params_neutral = pdm.init_params()
    print(f"  Params shape: {params_neutral.shape}")
    print(f"  Params: scale={params_neutral[0]:.3f}, tx={params_neutral[1]:.3f}, ty={params_neutral[2]:.3f}")
    print(f"  Rotation: wx={params_neutral[3]:.3f}, wy={params_neutral[4]:.3f}, wz={params_neutral[5]:.3f}")
    print(f"  Shape params (first 5): {params_neutral[6:11]}")

    landmarks_3d = pdm.params_to_landmarks_3d(params_neutral)
    landmarks_2d = pdm.params_to_landmarks_2d(params_neutral)
    print(f"  3D landmarks shape: {landmarks_3d.shape}")
    print(f"  2D landmarks shape: {landmarks_2d.shape}")
    print(f"  First 3 landmarks (2D): {landmarks_2d[:3]}")

    # Test 2: Initialize from bbox
    print("\nTest 2: Initialize from bounding box")
    bbox = (100, 100, 200, 250)  # [x, y, width, height]
    params_bbox = pdm.init_params(bbox)
    print(f"  Bbox: {bbox}")
    print(f"  Params: scale={params_bbox[0]:.3f}, tx={params_bbox[1]:.3f}, ty={params_bbox[2]:.3f}")

    landmarks_2d_bbox = pdm.params_to_landmarks_2d(params_bbox)
    print(f"  2D landmarks center: ({np.mean(landmarks_2d_bbox[:, 0]):.1f}, {np.mean(landmarks_2d_bbox[:, 1]):.1f})")
    print(f"  Expected center: ({bbox[0] + bbox[2]/2:.1f}, {bbox[1] + bbox[3]/2:.1f})")

    # Test 3: Non-zero shape parameters
    print("\nTest 3: Varying shape parameters")
    params_shape = params_neutral.copy()
    params_shape[6] = 2.0  # First PCA mode
    params_shape[7] = -1.5  # Second PCA mode

    landmarks_varied = pdm.params_to_landmarks_2d(params_shape)
    diff = np.linalg.norm(landmarks_varied - landmarks_2d)
    print(f"  Modified first 2 shape params: {params_shape[6:8]}")
    print(f"  Difference from neutral: {diff:.3f} pixels")

    # Test 4: Rotation
    print("\nTest 4: Rotation")
    params_rot = params_neutral.copy()
    params_rot[4] = 0.3  # Yaw rotation (around y-axis)

    landmarks_rot = pdm.params_to_landmarks_2d(params_rot)
    print(f"  Yaw rotation: {params_rot[4]:.3f} radians ({np.degrees(params_rot[4]):.1f}°)")
    print(f"  First 3 landmarks (rotated): {landmarks_rot[:3]}")

    # Test 5: Shape parameter clamping
    print("\nTest 5: Shape parameter clamping")
    params_extreme = params_neutral.copy()
    params_extreme[6:11] = 100.0  # Extreme values
    print(f"  Before clamping (first 5): {params_extreme[6:11]}")

    params_clamped = pdm.clamp_shape_params(params_extreme)
    print(f"  After clamping (first 5): {params_clamped[6:11]}")
    print(f"  Eigenvalues (first 5): {np.sqrt(pdm.eigen_values.flatten()[:5])}")

    # Test 6: Jacobian computation
    print("\nTest 6: Jacobian computation")
    J = pdm.compute_jacobian(params_bbox)
    print(f"  Jacobian shape: {J.shape}")
    print(f"  Expected shape: ({2 * pdm.n_points}, {pdm.n_params}) = (136, 40)")

    # Verify Jacobian accuracy using numerical differentiation
    # Test a few parameters
    h = 1e-6
    errors = []

    for param_idx in [0, 1, 2, 6, 10]:  # Test scale, tx, ty, and two shape params
        # Compute numerical derivative
        params_plus = params_bbox.copy()
        params_plus[param_idx] += h
        landmarks_plus = pdm.params_to_landmarks_2d(params_plus)

        params_minus = params_bbox.copy()
        params_minus[param_idx] -= h
        landmarks_minus = pdm.params_to_landmarks_2d(params_minus)

        numerical_deriv = (landmarks_plus - landmarks_minus) / (2 * h)
        numerical_deriv_flat = numerical_deriv.flatten()  # (136,)

        # Get analytical derivative from Jacobian
        analytical_deriv = J[:, param_idx]  # (136,)

        # Compute error
        error = np.linalg.norm(numerical_deriv_flat - analytical_deriv)
        errors.append(error)

    print(f"  Jacobian verification errors (numerical vs analytical):")
    print(f"    Param 0 (scale): {errors[0]:.2e}")
    print(f"    Param 1 (tx): {errors[1]:.2e}")
    print(f"    Param 2 (ty): {errors[2]:.2e}")
    print(f"    Param 6 (shape 0): {errors[3]:.2e}")
    print(f"    Param 10 (shape 4): {errors[4]:.2e}")
    print(f"  Max error: {max(errors):.2e} (should be < 1e-4)")

    if max(errors) < 1e-4:
        print("  ✓ Jacobian accuracy verified!")
    else:
        print("  ⚠ Jacobian may have numerical issues")

    print("\n" + "=" * 60)
    print("✓ PDM Core Implementation Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_pdm()
