"""
NU-RLMS Optimizer - Parameter optimization for CLNF

Implements the Normalized Unconstrained Regularized Least Mean Squares optimizer
used in OpenFace CLNF for fitting the Point Distribution Model to detected landmarks.

The optimizer minimizes:
    E(p) = ||v - J·Δp||² + λ||Λ^(-1/2)·Δp||²

Where:
    - p: Current parameter vector [scale, tx, ty, wx, wy, wz, q0, ..., qm]
    - Δp: Parameter update
    - v: Mean-shift vector (from patch expert responses)
    - J: Jacobian matrix (∂landmarks/∂params)
    - λ: Regularization weight
    - Λ: Diagonal matrix of parameter variances (eigenvalues for shape params)

Update rule:
    Δp = (J^T·W·J + λ·Λ^(-1))^(-1) · (J^T·W·v - λ·Λ^(-1)·p)

Where W is a diagonal weight matrix (typically identity for uniform weighting).
"""

import numpy as np
from typing import Tuple, Optional
import cv2

from .utils import align_shapes_with_scale, apply_similarity_transform, invert_similarity_transform


class NURLMSOptimizer:
    """
    NU-RLMS optimizer for CLNF parameter estimation.

    This optimizer iteratively refines the PDM parameters to fit detected landmarks
    using patch expert responses and shape model constraints.
    """

    def __init__(self,
                 regularization: float = 1.0,
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.01,
                 sigma: float = 1.75,
                 weight_multiplier: float = 5.0):
        """
        Initialize NU-RLMS optimizer.

        Args:
            regularization: Regularization weight λ (higher = stronger shape prior)
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for parameter change
            sigma: Gaussian kernel sigma for KDE mean-shift (OpenFace default: 1.75)
            weight_multiplier: Weight multiplier w for patch confidences
                             (OpenFace uses w=7 for Multi-PIE, w=5 for in-the-wild)
                             Controls how much to trust patch responses vs shape prior
        """
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.sigma = sigma
        self.weight_multiplier = weight_multiplier
        self.kde_cache = {}  # Cache for precomputed KDE kernels

    def optimize(self,
                 pdm,
                 initial_params: np.ndarray,
                 patch_experts: dict,
                 image: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 window_size: int = 11,
                 patch_scaling: float = 0.25,
                 sigma_components: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Optimize PDM parameters to fit landmarks to image.

        Args:
            pdm: PDM instance with compute_jacobian and params_to_landmarks methods
            initial_params: Initial parameter guess [s, tx, ty, wx, wy, wz, q...]
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            image: Grayscale image to fit to
            weights: Optional per-landmark weights (default: uniform)
            window_size: Search window size for mean-shift (default: 11)
            patch_scaling: Scale at which patches were trained (0.25, 0.35, or 0.5)
                          Used to create reference shape for warping

        Returns:
            optimized_params: Optimized parameter vector
            info: Dictionary with optimization info (iterations, convergence, etc.)
        """
        params = initial_params.copy()
        n_params = len(params)
        n_landmarks = pdm.n_points

        # Initialize weights (default: uniform)
        if weights is None:
            weights = np.ones(n_landmarks)

        # Create diagonal weight matrix W for 2D landmarks (2n × 2n)
        # OpenFace behavior (see PDM.cpp line 613 and LandmarkDetectorModel.cpp):
        # - weight_factor > 0: W = weight_factor · diag(patch_confidences)  [NU-RLMS mode]
        # - weight_factor = 0: W = Identity  [Video mode - all landmarks weighted equally]
        if self.weight_multiplier > 0:
            # NU-RLMS mode: apply weight multiplier to patch confidences
            W = self.weight_multiplier * np.diag(np.repeat(weights, 2))
        else:
            # Video mode: use identity matrix (all landmarks weighted equally)
            W = np.eye(n_landmarks * 2)

        # Create regularization matrix Λ^(-1)
        Lambda_inv = self._compute_lambda_inv(pdm, n_params)

        # Optimization loop
        iteration_info = []
        converged = False
        previous_landmarks = None  # Track previous shape for convergence check

        for iteration in range(self.max_iterations):
            # 1. Get current landmark positions in IMAGE coordinates
            landmarks_2d = pdm.params_to_landmarks_2d(params)

            # 2. Check shape-based convergence (OpenFace early stopping)
            # OpenFace: if(norm(current_shape, previous_shape) < 0.01) break;
            # (LandmarkDetectorModel.cpp lines 1044-1046)
            if previous_landmarks is not None:
                shape_change = np.linalg.norm(landmarks_2d - previous_landmarks)
                if shape_change < 0.01:  # OpenFace threshold
                    converged = True
                    break

            previous_landmarks = landmarks_2d.copy()

            # 3. Get REFERENCE shape at patch_scaling (canonical pose)
            # This creates the coordinate system in which patches were trained
            reference_shape = pdm.get_reference_shape(patch_scaling, params[6:])

            # 4. Compute similarity transform: IMAGE ↔ REFERENCE
            # This allows us to warp image patches to the coordinate system
            # where patches were trained (at patch_scaling)
            from .utils import align_shapes_with_scale, invert_similarity_transform
            sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
            sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

            # 5. Compute mean-shift vector from patch responses
            # Patches are evaluated on WARPED images at reference scale
            mean_shift = self._compute_mean_shift(
                landmarks_2d, patch_experts, image, pdm, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components
            )

            # 6. Compute Jacobian at current parameters
            J = pdm.compute_jacobian(params)

            # 7. Solve for parameter update: Δp
            delta_p = self._solve_update(J, mean_shift, W, Lambda_inv, params)

            # DEBUG: Print convergence metrics
            if iteration < 3 or iteration % 5 == 0:  # Print first 3 iterations, then every 5th
                ms_mag = np.linalg.norm(mean_shift)
                dp_mag = np.linalg.norm(delta_p)
                w_mean = np.mean(np.diag(W))
                print(f"Iter {iteration:2d} (ws={window_size}): MS={ms_mag:8.4f} DP={dp_mag:8.4f} W_mean={w_mean:6.4f}")

            # 8. Update parameters using manifold-aware update for rotations
            # CRITICAL: Cannot use naive addition (params = params + delta_p) for rotation parameters!
            # Must compose rotations properly on SO(3) manifold using pdm.update_params()
            # This matches OpenFace's PDM::UpdateModelParameters (PDM.cpp lines 454-503)
            params = pdm.update_params(params, delta_p)

            # 9. Clamp ALL parameters to valid range
            # CRITICAL: This prevents rotation divergence!
            # Matches OpenFace LandmarkDetectorModel.cpp:1134 where pdm.Clamp() is called
            # after every parameter update to clamp both rotation and shape parameters.
            params = pdm.clamp_params(params)

            # 10. Check parameter-based convergence (secondary check)
            update_magnitude = np.linalg.norm(delta_p)
            iteration_info.append({
                'iteration': iteration,
                'update_magnitude': update_magnitude,
                'params': params.copy()
            })

            if update_magnitude < self.convergence_threshold:
                converged = True
                break

        # Return optimized parameters and info
        info = {
            'converged': converged,
            'iterations': len(iteration_info),
            'final_update': iteration_info[-1]['update_magnitude'] if iteration_info else 0.0,
            'iteration_history': iteration_info
        }

        return params, info

    def _compute_lambda_inv(self, pdm, n_params: int) -> np.ndarray:
        """
        Compute inverse regularization matrix Λ^(-1).

        For rigid parameters (scale, translation, rotation): no regularization (set to 0)
        For shape parameters: use inverse eigenvalues

        Args:
            pdm: PDM instance
            n_params: Total number of parameters

        Returns:
            Lambda_inv: Diagonal matrix (n_params,)
        """
        Lambda_inv = np.zeros(n_params)

        # No regularization for rigid parameters (indices 0-5)
        # These are: scale, tx, ty, wx, wy, wz
        Lambda_inv[:6] = 0.0

        # Shape parameters (indices 6+): use inverse eigenvalues
        eigenvalues = pdm.eigen_values.flatten()
        Lambda_inv[6:] = 1.0 / (eigenvalues + 1e-8)  # Add small epsilon for stability

        return Lambda_inv

    def _compute_mean_shift(self,
                           landmarks_2d: np.ndarray,
                           patch_experts: dict,
                           image: np.ndarray,
                           pdm,
                           window_size: int = 11,
                           sim_img_to_ref: np.ndarray = None,
                           sim_ref_to_img: np.ndarray = None,
                           sigma_components: dict = None) -> np.ndarray:
        """
        Compute mean-shift vector from patch expert responses using KDE.

        This implements OpenFace's KDE-based mean-shift algorithm with image warping.

        When transforms are provided, patches are evaluated on WARPED image windows
        that have been transformed to the reference coordinate system. This ensures
        patches see features at the scale they were trained on:
        1. Extract response map for each landmark
        2. Apply Gaussian KDE smoothing
        3. Compute weighted mean-shift

        Args:
            landmarks_2d: Current 2D landmark positions (n_points, 2)
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            image: Grayscale image
            pdm: PDM instance
            window_size: Search window size (OpenFace uses [11, 9, 7, 5])

        Returns:
            mean_shift: Mean-shift vector, shape (2 * n_points,)
        """
        n_points = landmarks_2d.shape[0]
        mean_shift = np.zeros(2 * n_points)

        # Precompute KDE kernel for this window size if needed
        kde_kernel = self._get_kde_kernel(window_size)

        # Gaussian kernel parameter: a = -0.5 / sigma^2
        # Use self.sigma which has been adjusted by clnf.py based on patch scale
        # (clnf.py:161: adjusted_sigma = self.sigma + 0.25 * log2(scale_ratio))
        a = -0.5 / (self.sigma * self.sigma)

        # Check if we should use warping (transforms provided)
        use_warping = (sim_img_to_ref is not None and sim_ref_to_img is not None)

        # DEBUG: Track response map statistics and peak locations
        resp_values = []
        peak_locations = []  # Track peak offsets from center

        # For each landmark with a patch expert
        for landmark_idx, patch_expert in patch_experts.items():
            if landmark_idx >= n_points:
                continue

            # Get current landmark position in IMAGE coordinates
            lm_x, lm_y = landmarks_2d[landmark_idx]

            # Compute response map in window around current position
            # If warping enabled, this will warp the window to reference coordinates
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref if use_warping else None,
                sim_ref_to_img if use_warping else None,
                sigma_components
            )

            if response_map is None:
                continue

            # DEBUG: Collect response map statistics and peak location
            resp_values.extend([response_map.min(), response_map.max(), response_map.mean()])
            peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
            peak_y, peak_x = peak_idx  # row, col = y, x
            peak_value = response_map[peak_y, peak_x]
            resp_size = response_map.shape[0]
            center = (resp_size - 1) / 2.0
            # Store peak offset from center (should be near 0 for good convergence)
            peak_locations.append((landmark_idx, peak_x - center, peak_y - center, peak_value))

            # Current offset within response map
            # (center of response map corresponds to current landmark position)
            resp_size = response_map.shape[0]
            center = (resp_size - 1) / 2.0  # Float for sub-pixel precision, matches OpenFace

            # Compute position within response map
            # The response map is centered at the current landmark position
            # For sub-pixel positions, we need the fractional offset
            dx_frac = lm_x - int(lm_x)
            dy_frac = lm_y - int(lm_y)
            dx = dx_frac + center
            dy = dy_frac + center

            # Compute KDE mean-shift using OpenFace's algorithm
            # Result is in REFERENCE coordinates if warping was used
            ms_ref_x, ms_ref_y = self._kde_mean_shift(
                response_map, dx, dy, a
            )

            if use_warping:
                # Transform mean-shift from REFERENCE back to IMAGE coordinates
                # Apply 2x2 rotation/scale matrix: [a -b; b a]
                a_mat = sim_ref_to_img[0, 0]
                b_mat = sim_ref_to_img[1, 0]
                ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y  # Fixed: + to -
                ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y  # Fixed: - to +
            else:
                ms_x = ms_ref_x
                ms_y = ms_ref_y

            mean_shift[2 * landmark_idx] = ms_x
            mean_shift[2 * landmark_idx + 1] = ms_y

        # DEBUG: Print response map statistics (only print occasionally to avoid spam)
        import random
        if resp_values and random.random() < 0.1:  # 10% chance to print
            resp_vals = np.array(resp_values)
            print(f"  Response maps: min={resp_vals.min():.6f}, max={resp_vals.max():.6f}, mean={resp_vals.mean():.6f}")

            # Print worst 3 peak offsets (landmarks where peak is farthest from center)
            if peak_locations:
                peak_offsets = [(lm_idx, np.sqrt(dx**2 + dy**2), dx, dy, val)
                               for lm_idx, dx, dy, val in peak_locations]
                peak_offsets.sort(key=lambda x: x[1], reverse=True)
                print(f"  Worst 3 peak offsets (dist from center):")
                for i in range(min(3, len(peak_offsets))):
                    lm_idx, dist, dx, dy, val = peak_offsets[i]
                    print(f"    Landmark {lm_idx}: offset=({dx:+.1f}, {dy:+.1f}) dist={dist:.1f}px peak_val={val:.6f}")

        return mean_shift

    def _get_kde_kernel(self, window_size: int) -> np.ndarray:
        """
        Get or compute KDE kernel for given window size.

        Args:
            window_size: Size of response map window

        Returns:
            kde_kernel: Precomputed KDE kernel
        """
        if window_size in self.kde_cache:
            return self.kde_cache[window_size]

        # Compute KDE kernel (OpenFace uses step_size=0.1 for sub-pixel precision)
        step_size = 0.1
        a = -0.5 / (self.sigma * self.sigma)

        # Number of discrete positions
        n_steps = int(window_size / step_size)

        # Precompute kernel for all possible (dx, dy) positions
        kernel = np.zeros((n_steps, n_steps, window_size, window_size))

        for i_x in range(n_steps):
            dx = i_x * step_size
            for i_y in range(n_steps):
                dy = i_y * step_size

                # Compute Gaussian kernel centered at (dx, dy)
                for ii in range(window_size):
                    for jj in range(window_size):
                        dist_sq = (dy - ii)**2 + (dx - jj)**2
                        kernel[i_x, i_y, ii, jj] = np.exp(a * dist_sq)

        self.kde_cache[window_size] = kernel
        return kernel

    def _kde_mean_shift(self,
                       response_map: np.ndarray,
                       dx: float,
                       dy: float,
                       a: float) -> Tuple[float, float]:
        """
        Compute KDE-based mean-shift for a single landmark.

        Implements OpenFace's NonVectorisedMeanShift_precalc_kde algorithm.

        Args:
            response_map: Patch expert response map (window_size, window_size)
            dx: Current x offset within response map
            dy: Current y offset within response map
            a: Gaussian kernel parameter (-0.5 / sigma^2)

        Returns:
            (ms_x, ms_y): Mean-shift in x and y directions
        """
        resp_size = response_map.shape[0]

        # Clamp dx, dy to valid range
        dx = np.clip(dx, 0, resp_size - 0.1)
        dy = np.clip(dy, 0, resp_size - 0.1)

        # Compute Gaussian kernel centered at (dx, dy)
        mx = 0.0
        my = 0.0
        total_weight = 0.0

        for ii in range(resp_size):
            for jj in range(resp_size):
                # Distance from (dx, dy) to (jj, ii)
                dist_sq = (dy - ii)**2 + (dx - jj)**2

                # Gaussian weight
                kde_weight = np.exp(a * dist_sq)

                # Combined weight: KDE weight × patch response
                weight = kde_weight * response_map[ii, jj]

                total_weight += weight
                mx += weight * jj
                my += weight * ii

        if total_weight > 1e-10:
            # Mean-shift = weighted mean - current position
            ms_x = (mx / total_weight) - dx
            ms_y = (my / total_weight) - dy
        else:
            ms_x = 0.0
            ms_y = 0.0

        return ms_x, ms_y

    def _compute_response_map(self,
                             image: np.ndarray,
                             center_x: float,
                             center_y: float,
                             patch_expert,
                             window_size: int,
                             sim_img_to_ref: np.ndarray = None,
                             sim_ref_to_img: np.ndarray = None,
                             sigma_components: dict = None) -> Optional[np.ndarray]:
        """
        Compute response map for a landmark in a window around current position.

        When sim_img_to_ref is provided, extracts a larger window around the landmark,
        warps it to reference coordinates using cv2.warpAffine, then evaluates patches
        from the warped window. This ensures patches see features at the scale they
        were trained on.

        Args:
            image: Input image
            center_x, center_y: Current landmark position in IMAGE coordinates
            patch_expert: CCNFPatchExpert for this landmark
            window_size: Size of search window
            sim_img_to_ref: Optional 2x3 similarity transform (IMAGE → REFERENCE)

        Returns:
            response_map: (window_size, window_size) array of patch responses
        """
        response_map = np.zeros((window_size, window_size))

        # Window bounds (centered at current landmark)
        half_window = window_size // 2

        if sim_img_to_ref is not None:
            # WARPING MODE: Mimic OpenFace's exact approach (line 240 in Patch_experts.cpp)
            # Calculate area of interest size
            patch_dim = max(patch_expert.width, patch_expert.height)
            area_of_interest_width = window_size + patch_dim - 1
            area_of_interest_height = window_size + patch_dim - 1

            # Extract rotation/scale components from sim_ref_to_img (the INVERSE transform)
            # OpenFace uses: a1 = sim_ref_to_img(0,0), b1 = -sim_ref_to_img(0,1)
            a1 = sim_ref_to_img[0, 0]
            b1 = -sim_ref_to_img[0, 1]  # Note the NEGATIVE sign!

            # Construct the transform exactly as OpenFace does (line 240)
            # This centers the landmark at (area_of_interest_width-1)/2 in the warped output
            center_offset = (area_of_interest_width - 1.0) / 2.0

            tx = center_x - a1 * center_offset + b1 * center_offset
            ty = center_y - a1 * center_offset - b1 * center_offset

            sim_matrix = np.array([
                [a1, -b1, tx],
                [b1,  a1, ty]
            ], dtype=np.float32)

            # Warp using WARP_INVERSE_MAP (OpenFace line 245)
            # This inverts sim_matrix, effectively applying sim_img_to_ref
            area_of_interest = cv2.warpAffine(
                image,
                sim_matrix,
                (area_of_interest_width, area_of_interest_height),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )

            # Now evaluate patches from the warped area_of_interest
            # The landmark is centered at (area_of_interest_width-1)/2
            center_warped = int((area_of_interest_width - 1) / 2)

            start_x = center_warped - half_window
            start_y = center_warped - half_window

            for i in range(window_size):
                for j in range(window_size):
                    patch_x = start_x + j
                    patch_y = start_y + i

                    # Extract patch from warped area_of_interest
                    patch = self._extract_patch(
                        area_of_interest, patch_x, patch_y,
                        patch_expert.width, patch_expert.height
                    )

                    if patch is not None:
                        response_map[i, j] = patch_expert.compute_response(patch)
                    else:
                        response_map[i, j] = -1e10
        else:
            # NO WARPING: Original direct extraction from image
            start_x = int(center_x) - half_window
            start_y = int(center_y) - half_window

            # Compute response at each position in window
            for i in range(window_size):
                for j in range(window_size):
                    patch_x = start_x + j
                    patch_y = start_y + i

                    # Extract patch at this position
                    patch = self._extract_patch(
                        image, patch_x, patch_y,
                        patch_expert.width, patch_expert.height
                    )

                    if patch is not None:
                        response_map[i, j] = patch_expert.compute_response(patch)
                    else:
                        response_map[i, j] = -1e10  # Very low response for out-of-bounds

        # Apply CCNF Sigma transformation for spatial correlation modeling
        # (OpenFace CCNF_patch_expert.cpp lines 400-404)
        # Use response_map size (window_size × window_size), NOT patch size
        response_window_size = response_map.shape[0]  # Square response map
        if sigma_components is not None and response_window_size in sigma_components:
            try:
                # Get sigma components for this response map window size
                sigma_comps = sigma_components[response_window_size]

                # Compute Sigma covariance matrix with correct window size
                Sigma = patch_expert.compute_sigma(sigma_comps, window_size=response_window_size)

                # Apply transformation: response = Sigma @ response.reshape(-1, 1)
                # This models spatial correlations in the response map
                response_shape = response_map.shape
                response_vec = response_map.reshape(-1, 1)
                response_transformed = Sigma @ response_vec
                response_map = response_transformed.reshape(response_shape)
            except Exception as e:
                # If Sigma transformation fails, continue with untransformed response
                print(f"Warning: Sigma transformation failed: {e}")

        # OpenFace CCNF Response normalization (CCNF_patch_expert.cpp lines 406-413)
        # After computing responses, remove negative values by shifting
        # OpenFace C++ does ONLY this - no [0,1] normalization!
        min_val = response_map.min()
        if min_val < 0:
            response_map = response_map - min_val

        return response_map

    def _extract_patch(self,
                      image: np.ndarray,
                      center_x: int,
                      center_y: int,
                      patch_width: int,
                      patch_height: int) -> Optional[np.ndarray]:
        """
        Extract image patch centered at (center_x, center_y).

        Args:
            image: Source image
            center_x, center_y: Patch center coordinates
            patch_width, patch_height: Patch dimensions

        Returns:
            patch: Extracted patch, or None if out of bounds
        """
        half_w = patch_width // 2
        half_h = patch_height // 2

        # Compute patch bounds
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + patch_width
        y2 = y1 + patch_height

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            return None

        # Extract patch
        patch = image[y1:y2, x1:x2]

        return patch

    def _solve_update(self,
                     J: np.ndarray,
                     v: np.ndarray,
                     W: np.ndarray,
                     Lambda_inv: np.ndarray,
                     params: np.ndarray) -> np.ndarray:
        """
        Solve for parameter update using NU-RLMS equation.

        Solves: (J^T·W·J + λ·Λ^(-1))·Δp = J^T·W·v - λ·Λ^(-1)·p

        Args:
            J: Jacobian matrix (2n, m)
            v: Mean-shift vector (2n,)
            W: Weight matrix (2n, 2n)
            Lambda_inv: Inverse regularization matrix (m,)
            params: Current parameters (m,)

        Returns:
            delta_p: Parameter update (m,)
        """
        # Compute left-hand side: A = J^T·W·J + λ·Λ^(-1)
        JtWJ = J.T @ W @ J  # (m, m)
        Lambda_inv_diag = np.diag(self.regularization * Lambda_inv)  # (m, m)
        A = JtWJ + Lambda_inv_diag

        # Compute right-hand side: b = J^T·W·v - λ·Λ^(-1)·p
        JtWv = J.T @ W @ v  # (m,)
        reg_term = self.regularization * Lambda_inv * params  # (m,)
        b = JtWv - reg_term

        # Solve linear system: A·Δp = b
        try:
            delta_p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

        # Apply learning rate damping (OpenFace PDM.cpp line 660)
        # OpenFace uses 0.75 learning rate to dampen parameter updates
        delta_p = 0.75 * delta_p

        return delta_p


def test_optimizer():
    """Test NU-RLMS optimizer."""
    print("=" * 60)
    print("Testing NU-RLMS Optimizer")
    print("=" * 60)

    # Import dependencies
    from pyclnf.core.pdm import PDM
    from pyclnf.core.patch_expert import CCNFPatchExpert

    # Test 1: Load PDM
    print("\nTest 1: Initialize optimizer and PDM")
    pdm = PDM("pyclnf/models/exported_pdm")
    optimizer = NURLMSOptimizer(
        regularization=1.0,
        max_iterations=5,
        convergence_threshold=0.1
    )
    print(f"  PDM loaded: {pdm.n_points} landmarks, {pdm.n_params} params")
    print(f"  Optimizer: max_iter={optimizer.max_iterations}, λ={optimizer.regularization}")

    # Test 2: Initialize parameters
    print("\nTest 2: Initialize parameters from bbox")
    bbox = (100, 100, 200, 250)
    initial_params = pdm.init_params(bbox)
    print(f"  Initial params shape: {initial_params.shape}")
    print(f"  Scale: {initial_params[0]:.3f}")
    print(f"  Translation: ({initial_params[1]:.1f}, {initial_params[2]:.1f})")

    # Test 3: Create synthetic test image
    print("\nTest 3: Create test scenario")
    test_image = np.random.randint(0, 256, (400, 400), dtype=np.uint8)

    # Load a few patch experts for testing
    from pathlib import Path
    patch_experts = {}
    for landmark_idx in [30, 36, 45]:  # Nose tip, eye corners
        patch_dir = Path(f"pyclnf/models/exported_ccnf_0.25/view_00/patch_{landmark_idx:02d}")
        if patch_dir.exists():
            try:
                patch_experts[landmark_idx] = CCNFPatchExpert(str(patch_dir))
            except:
                pass

    print(f"  Test image: {test_image.shape}")
    print(f"  Loaded {len(patch_experts)} patch experts")

    # Test 4: Test mean-shift computation
    print("\nTest 4: Compute mean-shift vector")
    landmarks_2d = pdm.params_to_landmarks_2d(initial_params)
    mean_shift = optimizer._compute_mean_shift(
        landmarks_2d, patch_experts, test_image, pdm
    )
    print(f"  Mean-shift shape: {mean_shift.shape}")
    print(f"  Mean-shift magnitude: {np.linalg.norm(mean_shift):.3f}")
    print(f"  Non-zero elements: {np.count_nonzero(mean_shift)}")

    # Test 5: Test update computation
    print("\nTest 5: Compute parameter update")
    J = pdm.compute_jacobian(initial_params)
    W = np.eye(2 * pdm.n_points)
    Lambda_inv = optimizer._compute_lambda_inv(pdm, pdm.n_params)

    delta_p = optimizer._solve_update(J, mean_shift, W, Lambda_inv, initial_params)
    print(f"  Delta params shape: {delta_p.shape}")
    print(f"  Update magnitude: {np.linalg.norm(delta_p):.6f}")
    print(f"  Max update component: {np.abs(delta_p).max():.6f}")

    # Test 6: Run full optimization
    print("\nTest 6: Run optimization loop")
    optimized_params, info = optimizer.optimize(
        pdm, initial_params, patch_experts, test_image
    )

    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final update: {info['final_update']:.6f}")
    print(f"  Parameter change: {np.linalg.norm(optimized_params - initial_params):.6f}")

    # Test 7: Verify optimized landmarks
    print("\nTest 7: Compare initial vs optimized landmarks")
    initial_landmarks = pdm.params_to_landmarks_2d(initial_params)
    optimized_landmarks = pdm.params_to_landmarks_2d(optimized_params)

    landmark_shift = np.linalg.norm(optimized_landmarks - initial_landmarks, axis=1)
    print(f"  Mean landmark shift: {landmark_shift.mean():.3f} pixels")
    print(f"  Max landmark shift: {landmark_shift.max():.3f} pixels")
    print(f"  Landmarks moved > 1px: {np.sum(landmark_shift > 1.0)}")

    print("\n" + "=" * 60)
    print("✓ NU-RLMS Optimizer Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_optimizer()
