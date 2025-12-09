"""
NU-RLMS Optimizer for 51-point Inner Face Model.

Adapted from the main 68-point optimizer with fixed parameters for inner model:
    - reg_factor: 2.5
    - sigma: 1.75
    - weight_factor: 0.0 (video mode)
    - window_size: 9 (single scale)

C++ Reference: OpenFace LandmarkDetectorModel.cpp NU_RLMS function
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from numba import jit

from .inner_pdm import InnerPDM
from .ccnf_patch_expert import CCNFPatchExperts
from .utils import align_shapes_with_scale, invert_similarity_transform


@jit(nopython=True, cache=True)
def _kde_mean_shift_direct(response_map: np.ndarray, dx: float, dy: float,
                           a: float) -> Tuple[float, float]:
    """
    KDE-based mean-shift computation.

    Args:
        response_map: Patch response (window_size, window_size)
        dx, dy: Current position within response map
        a: Gaussian kernel parameter (-0.5 / sigma^2)

    Returns:
        (ms_x, ms_y): Mean-shift in x and y
    """
    resp_size = response_map.shape[0]

    mx = 0.0
    my = 0.0
    total_weight = 0.0

    for ii in range(resp_size):
        for jj in range(resp_size):
            resp_val = response_map[ii, jj]

            # KDE weight
            dist_x = dx - jj
            dist_y = dy - ii
            kde_weight = np.exp(a * (dist_x * dist_x + dist_y * dist_y))

            weight = resp_val * kde_weight
            total_weight += weight
            mx += weight * jj
            my += weight * ii

    if total_weight > 1e-10:
        ms_x = (mx / total_weight) - dx
        ms_y = (my / total_weight) - dy
    else:
        ms_x = 0.0
        ms_y = 0.0

    return ms_x, ms_y


class InnerModelOptimizer:
    """
    NU-RLMS optimizer for 51-point inner face model.

    Uses fixed parameters from C++ OpenFace inner model configuration:
        reg_factor = 2.5
        sigma = 1.75
        weight_factor = 0.0 (video mode)
        window_size = 9
    """

    def __init__(self):
        """Initialize with C++ inner model parameters."""
        # C++ inner model parameters (LandmarkDetectorModel.cpp:541-556)
        self.reg_factor = 2.5
        self.sigma = 1.75
        self.weight_factor = 0.0  # Video mode
        self.window_size = 9  # Single window size for inner model
        self.num_iterations = 5
        self.convergence_threshold = 0.01

        # KDE parameter
        self.a_kde = -0.5 / (self.sigma * self.sigma)

    def optimize(self,
                 pdm: InnerPDM,
                 initial_params: np.ndarray,
                 ccnf_experts: CCNFPatchExperts,
                 image: np.ndarray,
                 window_size: int = None) -> Tuple[np.ndarray, dict]:
        """
        NU-RLMS optimization for inner model.

        Args:
            pdm: InnerPDM instance
            initial_params: Initial parameters [s, wx, wy, wz, tx, ty, q...]
            ccnf_experts: CCNFPatchExperts instance
            image: Grayscale image
            window_size: Override window size (default: 9)

        Returns:
            optimized_params: Optimized parameter vector
            info: Optimization info dict
        """
        if window_size is None:
            window_size = self.window_size

        params = initial_params.copy()
        n_points = pdm.n_points
        n_params = pdm.n_params

        # Get initial landmarks
        landmarks_2d = pdm.params_to_landmarks_2d(params)

        # Compute reference shape and similarity transform
        reference_shape = pdm.get_reference_shape(1.0, params[6:])
        sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
        sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

        # Precompute response maps at initial positions
        response_maps = self._precompute_response_maps(
            landmarks_2d, ccnf_experts, image, window_size,
            sim_img_to_ref, sim_ref_to_img
        )

        # Weight matrix (identity for video mode)
        W = np.eye(n_points * 2)

        # Lambda_inv for regularization
        Lambda_inv = self._compute_lambda_inv(pdm)

        # Base landmarks for offset computation
        base_landmarks = landmarks_2d.copy()

        # Track convergence
        previous_landmarks = None
        total_iterations = 0

        # ========================================
        # RIGID PHASE
        # ========================================
        for rigid_iter in range(self.num_iterations):
            current_landmarks = pdm.params_to_landmarks_2d(params)

            # Check convergence
            if previous_landmarks is not None:
                shape_change = np.linalg.norm(current_landmarks - previous_landmarks)
                if shape_change < self.convergence_threshold:
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute mean-shift
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks, response_maps,
                window_size, sim_img_to_ref, sim_ref_to_img
            )

            # Compute rigid Jacobian
            J_rigid = pdm.compute_jacobian_rigid(params)

            # Solve rigid update (no regularization)
            JtWJ = J_rigid.T @ W @ J_rigid
            JtWv = J_rigid.T @ W @ mean_shift

            try:
                delta_p_rigid = np.linalg.solve(JtWJ, JtWv)
            except np.linalg.LinAlgError:
                delta_p_rigid = np.linalg.lstsq(JtWJ, JtWv, rcond=None)[0]

            # Update rigid params only
            delta_p_full = np.zeros(n_params)
            delta_p_full[:6] = delta_p_rigid
            params = pdm.update_params(params, delta_p_full)
            params = pdm.clamp_params(params)

            total_iterations += 1

        # ========================================
        # NON-RIGID PHASE
        # ========================================
        previous_landmarks = None

        for nonrigid_iter in range(self.num_iterations):
            current_landmarks = pdm.params_to_landmarks_2d(params)

            # Check convergence
            if previous_landmarks is not None:
                shape_change = np.linalg.norm(current_landmarks - previous_landmarks)
                if shape_change < self.convergence_threshold:
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute mean-shift
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks, response_maps,
                window_size, sim_img_to_ref, sim_ref_to_img
            )

            # Compute full Jacobian
            J = pdm.compute_jacobian(params)

            # Solve with regularization
            # (J^T W J + λ Λ^-1) Δp = J^T W v - λ Λ^-1 p
            JtWJ = J.T @ W @ J
            Lambda_inv_diag = np.diag(self.reg_factor * Lambda_inv)
            A = JtWJ + Lambda_inv_diag

            JtWv = J.T @ W @ mean_shift
            reg_term = self.reg_factor * Lambda_inv * params
            b = JtWv - reg_term

            try:
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, b)
                delta_p = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                try:
                    delta_p = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update all params
            params = pdm.update_params(params, delta_p)
            params = pdm.clamp_params(params)

            total_iterations += 1

        info = {
            'converged': True,
            'iterations': total_iterations,
            'final_shape_change': shape_change if previous_landmarks is not None else 0.0
        }

        return params, info

    def _compute_lambda_inv(self, pdm: InnerPDM) -> np.ndarray:
        """Compute inverse regularization matrix."""
        Lambda_inv = np.zeros(pdm.n_params)
        Lambda_inv[:6] = 0.0  # No regularization for rigid params
        Lambda_inv[6:] = 1.0 / pdm.eigenvalues.flatten()  # Shape params
        return Lambda_inv

    def _precompute_response_maps(self,
                                   landmarks_2d: np.ndarray,
                                   ccnf_experts: CCNFPatchExperts,
                                   image: np.ndarray,
                                   window_size: int,
                                   sim_img_to_ref: np.ndarray,
                                   sim_ref_to_img: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Precompute response maps at initial landmark positions.

        Args:
            landmarks_2d: Initial landmarks (51, 2)
            ccnf_experts: CCNF patch experts
            image: Grayscale image
            window_size: Response window size
            sim_img_to_ref: Image to reference transform
            sim_ref_to_img: Reference to image transform

        Returns:
            response_maps: Dict mapping landmark_idx -> response map
        """
        response_maps = {}

        for point_idx in range(len(landmarks_2d)):
            if point_idx not in ccnf_experts.experts:
                continue

            expert = ccnf_experts.experts[point_idx]
            lm_x, lm_y = landmarks_2d[point_idx]

            # Extract area of interest using warping
            aoi = self._extract_aoi(image, lm_x, lm_y, expert.width,
                                    window_size, sim_ref_to_img)

            if aoi is None:
                continue

            # Compute response
            response = ccnf_experts.compute_response(point_idx, aoi, window_size)
            if response is not None:
                response_maps[point_idx] = response

        return response_maps

    def _extract_aoi(self, image: np.ndarray, center_x: float, center_y: float,
                     patch_dim: int, window_size: int,
                     sim_ref_to_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract area of interest around landmark with warping.

        Args:
            image: Input image
            center_x, center_y: Landmark position
            patch_dim: Patch expert size
            window_size: Response window size
            sim_ref_to_img: Similarity transform

        Returns:
            aoi: Warped area of interest
        """
        aoi_size = window_size + patch_dim - 1

        # Extract rotation/scale from similarity transform
        a1 = sim_ref_to_img[0, 0]
        b1 = -sim_ref_to_img[0, 1]

        # Construct warp matrix
        center_offset = (aoi_size - 1.0) / 2.0
        tx = center_x - a1 * center_offset + b1 * center_offset
        ty = center_y - a1 * center_offset - b1 * center_offset

        sim_matrix = np.array([
            [a1, -b1, tx],
            [b1, a1, ty]
        ], dtype=np.float32)

        # Warp
        try:
            aoi = cv2.warpAffine(
                image, sim_matrix, (aoi_size, aoi_size),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )
            return aoi.astype(np.float32)
        except:
            return None

    def _compute_mean_shift(self,
                            landmarks_2d: np.ndarray,
                            base_landmarks_2d: np.ndarray,
                            response_maps: Dict[int, np.ndarray],
                            window_size: int,
                            sim_img_to_ref: np.ndarray,
                            sim_ref_to_img: np.ndarray) -> np.ndarray:
        """
        Compute mean-shift vector from response maps.

        Args:
            landmarks_2d: Current landmarks (51, 2)
            base_landmarks_2d: Base landmarks where responses were computed
            response_maps: Precomputed response maps
            window_size: Response window size
            sim_img_to_ref: Image to reference transform
            sim_ref_to_img: Reference to image transform

        Returns:
            mean_shift: Mean-shift vector (102,) in STACKED format
        """
        n_points = landmarks_2d.shape[0]
        mean_shift = np.zeros(2 * n_points)

        center = (window_size - 1) / 2.0

        # Get transform coefficients
        a_sim = sim_img_to_ref[0, 0]
        b_sim = sim_img_to_ref[1, 0]
        a_mat = sim_ref_to_img[0, 0]
        b_mat = sim_ref_to_img[1, 0]

        for point_idx, response_map in response_maps.items():
            if point_idx >= n_points:
                continue

            # Compute offset from base to current in image coords
            offset_img_x = landmarks_2d[point_idx, 0] - base_landmarks_2d[point_idx, 0]
            offset_img_y = landmarks_2d[point_idx, 1] - base_landmarks_2d[point_idx, 1]

            # Transform offset to reference coords
            offset_ref_x = a_sim * offset_img_x - b_sim * offset_img_y
            offset_ref_y = b_sim * offset_img_x + a_sim * offset_img_y

            # Position in response map
            dx = offset_ref_x + center
            dy = offset_ref_y + center

            # KDE mean-shift
            ms_ref_x, ms_ref_y = _kde_mean_shift_direct(
                response_map, dx, dy, self.a_kde
            )

            # Transform back to image coords
            ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y
            ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y

            # STACKED format
            mean_shift[point_idx] = ms_x
            mean_shift[point_idx + n_points] = ms_y

        return mean_shift


def test_inner_optimizer():
    """Test inner optimizer."""
    print("=" * 60)
    print("Testing Inner Model Optimizer")
    print("=" * 60)

    from .inner_pdm import InnerPDM

    # Load components
    pdm = InnerPDM("pyclnf/pyclnf/models/exported_inner_pdm")
    ccnf = CCNFPatchExperts("pyclnf/pyclnf/models/exported_inner_ccnf")
    optimizer = InnerModelOptimizer()

    print(f"\nOptimizer parameters:")
    print(f"  reg_factor: {optimizer.reg_factor}")
    print(f"  sigma: {optimizer.sigma}")
    print(f"  weight_factor: {optimizer.weight_factor}")
    print(f"  window_size: {optimizer.window_size}")

    # Create synthetic test image
    test_image = np.random.randint(100, 200, (500, 500), dtype=np.uint8)

    # Initialize params
    params = np.zeros(pdm.n_params)
    params[0] = 2.0  # scale
    params[4] = 250  # tx
    params[5] = 250  # ty

    print(f"\nInitial params:")
    print(f"  scale: {params[0]:.2f}")
    print(f"  translation: ({params[4]:.1f}, {params[5]:.1f})")

    # Run optimization
    optimized_params, info = optimizer.optimize(pdm, params, ccnf, test_image)

    print(f"\nOptimization result:")
    print(f"  iterations: {info['iterations']}")
    print(f"  converged: {info['converged']}")
    print(f"  final params scale: {optimized_params[0]:.4f}")
    print(f"  final params translation: ({optimized_params[4]:.1f}, {optimized_params[5]:.1f})")

    print("\n" + "=" * 60)
    print("Inner Optimizer tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_inner_optimizer()
