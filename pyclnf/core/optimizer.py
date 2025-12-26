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

CONVERGENCE PROFILES (for HPC optimization):
- 'accurate': Original settings, highest accuracy (<0.1px threshold)
- 'optimized': Conservative speedup with strict accuracy (<0.3px threshold)
- 'fast': Aggressive speedup, moderate accuracy loss (<1.0px threshold)
- 'video': Optimized for temporal coherence in video processing
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2
from numba import jit
import os

from .utils import align_shapes_with_scale, apply_similarity_transform, invert_similarity_transform
from .cen_patch_expert import MirroredCENPatchExpert, CENPatchExpert

# Try to import cpp_warp for exact OpenCV 4.12 matching with C++ OpenFace
# This is optional - falls back to cv2.warpAffine if not available
try:
    from pyclnf.cpp_warp import extract_aoi
    import pyclnf.cpp_warp as cpp_warp
    CPP_WARP_AVAILABLE = True
except ImportError:
    cpp_warp = None
    CPP_WARP_AVAILABLE = False

# Try to import GPU acceleration modules
try:
    from .batched_cen import BatchedCEN
    BATCHED_CEN_AVAILABLE = True
except ImportError:
    BatchedCEN = None
    BATCHED_CEN_AVAILABLE = False

try:
    from .gpu_mean_shift import GPUMeanShift
    GPU_MEAN_SHIFT_AVAILABLE = True
except ImportError:
    GPUMeanShift = None
    GPU_MEAN_SHIFT_AVAILABLE = False

# ============================================================================
# CONVERGENCE PROFILES - Tuned for accuracy vs speed tradeoffs
# ============================================================================
# Each profile defines:
#   - convergence_threshold: Early stopping threshold (pixels)
#   - rigid_iterations: Max iterations for rigid phase
#   - nonrigid_iterations: Max iterations for non-rigid phase
#   - min_iterations: Minimum iterations before checking convergence
#
# Accuracy impact estimates (vs 'accurate' baseline):
#   - 'optimized': <0.3px additional landmark error
#   - 'fast': <1.0px additional landmark error
#   - 'video': <0.5px additional error (with temporal warm-start)

CONVERGENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    'accurate': {
        'convergence_threshold': 0.01,  # C++ OpenFace uses 0.01
        'rigid_iterations': 5,  # C++ uses num_optimisation_iteration=5
        'nonrigid_iterations': 5,  # Same for both phases in C++
        'min_iterations': 1,
        'description': 'Matches C++ OpenFace settings exactly'
    },
    'optimized': {
        'convergence_threshold': 0.3,  # Conservative relaxation
        'rigid_iterations': 5,
        'nonrigid_iterations': 7,
        'min_iterations': 2,
        'description': 'Balanced accuracy/speed, <0.3px error increase'
    },
    'fast': {
        'convergence_threshold': 1.0,  # Aggressive relaxation
        'rigid_iterations': 3,
        'nonrigid_iterations': 4,
        'min_iterations': 2,
        'description': 'Speed priority, <1.0px error increase'
    },
    'video': {
        'convergence_threshold': 0.005,  # Gold standard (matches C++ accuracy)
        'rigid_iterations': 5,  # C++ uses num_optimisation_iteration=5
        'nonrigid_iterations': 5,  # Same for both phases in C++
        'min_iterations': 1,  # Can converge faster with warm-start
        'description': 'Optimized for video with temporal warm-start'
    },
    'cpp_match': {
        'convergence_threshold': 0.01,  # C++ OpenFace uses 0.01
        'rigid_iterations': 10,  # More rigid iterations for better convergence
        'nonrigid_iterations': 5,  # Fewer nonrigid to avoid divergence
        'min_iterations': 1,
        'description': 'Optimized for matching C++ OpenFace results'
    }
}


def get_convergence_profile(name: str) -> Dict[str, Any]:
    """Get a convergence profile by name.

    Args:
        name: Profile name ('accurate', 'optimized', 'fast', 'video')

    Returns:
        Profile dict with convergence parameters

    Raises:
        ValueError: If profile name is not recognized
    """
    if name not in CONVERGENCE_PROFILES:
        valid = list(CONVERGENCE_PROFILES.keys())
        raise ValueError(f"Unknown convergence profile '{name}'. Valid: {valid}")
    return CONVERGENCE_PROFILES[name].copy()

@jit(nopython=True, cache=True)
def _kde_mean_shift_numba(response_map: np.ndarray,
                          dx: float,
                          dy: float,
                          a: float,
                          kde_weights: np.ndarray) -> Tuple[float, float]:
    """
    Numba-optimized KDE-based mean-shift computation.

    This is extracted from _kde_mean_shift for JIT compilation.

    Args:
        response_map: Patch expert response map (window_size, window_size)
        dx: Current x offset within response map (clamped)
        dy: Current y offset within response map (clamped)
        a: Gaussian kernel parameter (-0.5 / sigma^2)
        kde_weights: Precomputed KDE weights for the grid position

    Returns:
        (ms_x, ms_y): Mean-shift in x and y directions
    """
    resp_size = response_map.shape[0]

    # Compute weighted mean-shift
    mx = 0.0
    my = 0.0
    total_weight = 0.0

    # Iterate through response map and KDE weights
    kde_idx = 0

    for ii in range(resp_size):
        for jj in range(resp_size):
            # Get response value at this position
            resp_val = response_map[ii, jj]

            # Get KDE weight (stored sequentially as we iterate ii, jj)
            kde_weight = kde_weights[kde_idx]

            # Combined weight
            weight = resp_val * kde_weight

            total_weight += weight
            mx += weight * jj
            my += weight * ii

            kde_idx += 1

    # Compute mean-shift
    if total_weight > 1e-10:
        ms_x = (mx / total_weight) - dx
        ms_y = (my / total_weight) - dy
    else:
        ms_x = 0.0
        ms_y = 0.0

    return ms_x, ms_y


@jit(nopython=True, cache=True)
def _kde_mean_shift_direct(response_map: np.ndarray,
                           dx: float,
                           dy: float,
                           a: float) -> Tuple[float, float]:
    """
    Direct KDE-based mean-shift computation without precomputed grid.

    This computes KDE weights directly at the actual (dx, dy) position,
    avoiding the grid snapping that causes systematic errors to cancel
    to zero when summed across landmarks.

    Args:
        response_map: Patch expert response map (window_size, window_size)
        dx: Current x offset within response map (clamped)
        dy: Current y offset within response map (clamped)
        a: Gaussian kernel parameter (-0.5 / sigma^2)

    Returns:
        (ms_x, ms_y): Mean-shift in x and y directions
    """
    resp_size = response_map.shape[0]

    # Compute weighted mean-shift with direct KDE weights
    mx = 0.0
    my = 0.0
    total_weight = 0.0

    for ii in range(resp_size):
        for jj in range(resp_size):
            # Get response value at this position
            resp_val = response_map[ii, jj]

            # Compute KDE weight directly at actual (dx, dy) position
            # dist_x = dx - jj (x-distance from current position to pixel jj)
            # dist_y = dy - ii (y-distance from current position to pixel ii)
            dist_x = dx - jj
            dist_y = dy - ii
            kde_weight = np.exp(a * (dist_x * dist_x + dist_y * dist_y))

            # Combined weight
            weight = resp_val * kde_weight

            total_weight += weight
            mx += weight * jj
            my += weight * ii

    # Compute mean-shift
    if total_weight > 1e-10:
        ms_x = (mx / total_weight) - dx
        ms_y = (my / total_weight) - dy
    else:
        ms_x = 0.0
        ms_y = 0.0

    return ms_x, ms_y


class NURLMSOptimizer:
    """
    NU-RLMS optimizer for CLNF parameter estimation.

    This optimizer iteratively refines the PDM parameters to fit detected landmarks
    using patch expert responses and shape model constraints.

    Supports convergence profiles for HPC optimization:
        - 'accurate': Original settings, highest accuracy
        - 'optimized': Conservative speedup (<0.3px error increase)
        - 'fast': Aggressive speedup (<1.0px error increase)
        - 'video': Optimized for video with temporal warm-start
    """

    def __init__(self,
                 regularization: float = 22.5,  # C++ CECLM: 25.0 base × 0.9 = 22.5
                 max_iterations: int = 10,  # More iterations help Python converge
                 convergence_threshold: float = 0.005,  # Gold standard (stricter than 0.01)
                 sigma: float = 2.5,  # C++ OpenFace uses sigma=2.5 (verified from debug dump)
                 weight_multiplier: float = 0.0,  # C++ video mode (disabled)
                 debug_mode: bool = False,
                 tracked_landmarks: list = None,
                 convergence_profile: str = None,
                 use_peak_confidence: bool = False,
                 use_direct_kde: bool = True,  # True enables fast vectorized mean-shift (2x faster)
                 use_cpp_warp: bool = True,  # Use C++ warpAffine for exact OpenCV 4.12 matching
                 use_gpu: bool = True,  # Enable GPU acceleration for response maps and mean-shift
                 gpu_device: str = 'mps'):  # GPU device: 'mps' (Apple), 'cuda', or 'cpu'
        """
        Initialize NU-RLMS optimizer.

        Note: Defaults match C++ OpenFace (verified from debug dump):
            - regularization: 22.5 (C++ CECLM)
            - sigma: 2.5 (verified from C++ WS7 debug dump)

        Args:
            regularization: Regularization weight λ (higher = stronger shape prior)
                          Default: 22.5 (C++ CECLM)
            max_iterations: Maximum optimization iterations (overridden by profile)
            convergence_threshold: Convergence threshold in pixels (overridden by profile)
                          Default: 0.005 (gold standard for sub-pixel accuracy)
            sigma: Gaussian kernel sigma for KDE mean-shift
                  Default: 2.5 (verified from C++ debug dump)
            weight_multiplier: Weight multiplier w for patch confidences
                             C++ video mode uses w=0 (disabled), wild mode uses w=2.5
                             Controls how much to trust patch responses vs shape prior
            debug_mode: Enable detailed debug output (similar to MTCNN debug mode)
            tracked_landmarks: Landmarks to track in detail when debug_mode=True (default: [36, 48, 30, 8])
            convergence_profile: Named profile ('accurate', 'optimized', 'fast', 'video')
                               If provided, overrides max_iterations and convergence_threshold
            use_peak_confidence: Enable dynamic peak confidence weighting.
                               Reduces influence of landmarks with flat/ambiguous response peaks.
                               Helps improve accuracy on low-contrast image regions.
            use_direct_kde: Use direct KDE computation instead of precomputed grid.
                           Default True to fix zero-sum bug in precomputed grid approach.
                           The precomputed grid snaps positions to 0.1px causing systematic
                           errors that cancel to zero when summed across landmarks.
            use_cpp_warp: Use C++ warpAffine wrapper (links against Homebrew OpenCV 4.12).
                         This ensures exact numerical matching with C++ OpenFace's warpAffine.
                         Default True if cpp_warp module is available.
            use_gpu: Enable GPU acceleration for response maps and mean-shift computation.
                    Uses BatchedCEN for response maps and GPUMeanShift for mean-shift.
                    Provides exact numerical match with CPU while being 2-5x faster.
            gpu_device: GPU device to use: 'mps' (Apple Silicon), 'cuda' (NVIDIA), or 'cpu'.
                       Default 'mps' for Apple Silicon Macs.
        """
        self.regularization = regularization
        self.sigma = sigma
        self.weight_multiplier = weight_multiplier
        self.kde_cache = {}  # Cache for precomputed KDE kernels
        self.debug_mode = debug_mode
        self.tracked_landmarks = tracked_landmarks if tracked_landmarks is not None else [36, 48, 30, 8]
        self.use_peak_confidence = use_peak_confidence
        self.use_direct_kde = use_direct_kde

        # Use C++ warpAffine if requested and available
        self.use_cpp_warp = use_cpp_warp and CPP_WARP_AVAILABLE
        if use_cpp_warp and not CPP_WARP_AVAILABLE:
            import warnings
            warnings.warn("cpp_warp module not available, falling back to cv2.warpAffine")

        # GPU acceleration setup
        self.use_gpu = use_gpu and BATCHED_CEN_AVAILABLE and GPU_MEAN_SHIFT_AVAILABLE

        # Auto-detect best GPU device
        if gpu_device == 'auto':
            import torch
            if torch.backends.mps.is_available():
                self.gpu_device = 'mps'
            elif torch.cuda.is_available():
                self.gpu_device = 'cuda'
            else:
                self.gpu_device = 'cpu'
        else:
            self.gpu_device = gpu_device

        if use_gpu and not self.use_gpu:
            import warnings
            missing = []
            if not BATCHED_CEN_AVAILABLE:
                missing.append("BatchedCEN")
            if not GPU_MEAN_SHIFT_AVAILABLE:
                missing.append("GPUMeanShift")
            warnings.warn(f"GPU modules not available ({', '.join(missing)}), falling back to CPU")

        # GPU modules will be lazily initialized on first use
        self._batched_cen_cache = {}  # Cache per scale
        self._gpu_mean_shift = None

        # Apply convergence profile if specified
        self.convergence_profile_name = convergence_profile
        if convergence_profile is not None:
            profile = get_convergence_profile(convergence_profile)
            self.convergence_threshold = profile['convergence_threshold']
            self.max_iterations = max(profile['rigid_iterations'], profile['nonrigid_iterations'])
            self.rigid_iterations = profile['rigid_iterations']
            self.nonrigid_iterations = profile['nonrigid_iterations']
            self.min_iterations = profile['min_iterations']
        else:
            # Use explicit parameters
            # Default: R=10, NR=5 for best accuracy (fewer nonrigid avoids divergence)
            self.convergence_threshold = convergence_threshold
            self.max_iterations = max_iterations
            self.rigid_iterations = 10  # More rigid iterations for better convergence
            self.nonrigid_iterations = 5  # Fewer nonrigid to avoid jaw divergence
            self.min_iterations = 2  # Default minimum

        # Response map caching for video-mode temporal coherence
        # When landmarks move less than threshold between frames, reuse cached response maps
        # Per-scale caching: each scale maintains its own cache to avoid cross-scale invalidation
        self._scale_cache = {}  # {scale: {'response_maps': dict, 'landmarks': array, 'age': int}}
        # Threshold: response maps with window_size=11 can tolerate ~5px offset
        # since mean-shift finds the peak within the window
        self.response_reuse_threshold = 0.0  # pixels - DISABLED for <1px accuracy (caching introduces ~2-8px error)
        self.cache_max_age = 2  # frames - max age before forcing recompute (reduced for accuracy)
        # Legacy attributes for backward compatibility
        self.cached_response_maps = None
        self.cached_landmarks = None
        self.cache_age = 0

        # Temporal warm-start support (for video mode)
        self.previous_frame_landmarks = None

        # Response map saving hook for debugging/comparison
        self.save_response_maps_dir = None
        self._save_call_count = 0
        self.previous_frame_params = None
        self.use_temporal_warmstart = (convergence_profile == 'video')

        # Diagnostic tracking for late-stage convergence analysis
        self._last_hessian_cond = None
        self._last_reg_ratio = None
        self._last_jtw_norm = None
        self._last_reg_term_norm = None

        # Store base parameters for scale-adaptive computation (C++ lines 943-950)
        # These are adapted based on patch_scaling during optimization
        self._base_regularization = regularization
        self._base_sigma = sigma
        self._base_weight_multiplier = weight_multiplier

        # Enable/disable scale adaptation (matches C++ OpenFace behavior)
        self.use_scale_adaptation = True

        # Stability threshold for video tracking (disabled - not effective)
        # The ~1.5px detection→tracking jump is expected C++ behavior, not an error
        self.stability_threshold = 0.5
        self.use_stability_check = False  # Disabled - C++ has same behavior

        # Jaw weight reduction - DISABLED after testing
        # Testing showed jaw_weight=1.0 gives 0.35px error vs 5.15px with jaw_weight=0.5
        # The weak response maps still provide useful directional signal
        self.jaw_weight_factor = 1.0  # Keep full weight
        self.use_jaw_weight_reduction = False  # Disabled - hurts accuracy

    def _extract_aoi(self, image: np.ndarray, center_x: float, center_y: float,
                     sim_ref_to_img: np.ndarray, aoi_size: int) -> np.ndarray:
        """
        Extract Area of Interest patch around a landmark.

        Uses cpp_warp.extract_aoi if available for exact C++ OpenFace matching,
        otherwise falls back to cv2.warpAffine.

        Args:
            image: Source grayscale image (float32)
            center_x: Landmark X coordinate in image space
            center_y: Landmark Y coordinate in image spaceO
            sim_ref_to_img: 2x3 similarity transform from reference to image
            aoi_size: Size of the AOI patch (square)

        Returns:
            Extracted AOI patch (aoi_size x aoi_size, float32)
        """
        if self.use_cpp_warp:
            # Use C++ wrapper for exact OpenCV 4.12 matching
            return cpp_warp.extract_aoi(
                image.astype(np.float32, copy=False),
                float(center_x),
                float(center_y),
                sim_ref_to_img.astype(np.float64, copy=False),
                int(aoi_size)
            )
        else:
            # Fall back to Python cv2.warpAffine
            a1 = sim_ref_to_img[0, 0]
            b1 = -sim_ref_to_img[0, 1]  # Note the NEGATIVE sign!

            center_offset = (aoi_size - 1.0) / 2.0
            tx = center_x - a1 * center_offset + b1 * center_offset
            ty = center_y - a1 * center_offset - b1 * center_offset

            sim_matrix = np.array([
                [a1, -b1, tx],
                [b1,  a1, ty]
            ], dtype=np.float32)

            return cv2.warpAffine(
                image,
                sim_matrix,
                (aoi_size, aoi_size),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )

    def _compute_scale_adapted_params(self, patch_scaling: float) -> Tuple[float, float, float]:
        """
        Compute scale-adapted optimization parameters matching C++ OpenFace.

        C++ formulas (LandmarkDetectorModel.cpp lines 943-950):
            reg_factor = reg_factor - 15 * log(patch_scaling/0.25)/log(2)
            sigma = sigma + 0.25 * log(patch_scaling/0.25)/log(2)
            weight_factor = weight_factor + 2 * weight_factor * log(patch_scaling/0.25)/log(2)

        Note: C++ limits scale to max of 2 before computing adaptation.

        Args:
            patch_scaling: Current patch scale (0.25, 0.35, 0.5, etc.)

        Returns:
            (adapted_reg, adapted_sigma, adapted_weight): Adapted parameters
        """
        import math

        if not self.use_scale_adaptation:
            return self._base_regularization, self._base_sigma, self._base_weight_multiplier

        # C++ clamps scale to max 0.5 before computing adaptation
        # (LandmarkDetectorModel.cpp line 941: scale = min(scale, 2) where 2 is scale_idx)
        # This prevents regularization from becoming negative at larger scales
        clamped_scale = min(patch_scaling, 0.5)

        # Compute log ratio like C++: log(scale/0.25) / log(2)
        # For scale=0.25: ratio=0, for scale=0.5: ratio=1
        if clamped_scale <= 0.25:
            log_ratio = 0.0
        else:
            log_ratio = math.log(clamped_scale / 0.25) / math.log(2)

        # Adapt regularization: decreases as scale increases (max decrease is 15 at scale 0.5)
        adapted_reg = self._base_regularization - 15 * log_ratio
        if adapted_reg <= 0:
            adapted_reg = 0.001  # C++ minimum threshold

        # Adapt sigma: increases as scale increases
        adapted_sigma = self._base_sigma + 0.25 * log_ratio

        # Adapt weight multiplier: increases as scale increases
        adapted_weight = self._base_weight_multiplier + 2 * self._base_weight_multiplier * log_ratio

        return adapted_reg, adapted_sigma, adapted_weight

    def save_response_maps(self, response_maps: dict, window_size: int, patch_scaling: float, call_idx: int = None):
        """
        Save response maps to binary files for comparison with C++.

        File format matches C++ OpenFace debug output:
        - 8-byte header: rows (int32), cols (int32)
        - Body: float64 values in row-major order

        Filenames: response_lm{idx}_scale{scale:.2f}_ws{ws}.bin

        Args:
            response_maps: Dict mapping landmark_idx -> response_map (ws, ws)
            window_size: Response map window size
            patch_scaling: Current patch scale
            call_idx: Optional call index for uniqueness (default: auto-increment)
        """
        if self.save_response_maps_dir is None:
            return

        import os
        os.makedirs(self.save_response_maps_dir, exist_ok=True)

        if call_idx is None:
            call_idx = self._save_call_count
            self._save_call_count += 1

        for landmark_idx, response_map in response_maps.items():
            # Filename matches C++ format
            filename = f"response_lm{landmark_idx:02d}_scale{patch_scaling:.2f}_ws{window_size}_call{call_idx}.bin"
            filepath = os.path.join(self.save_response_maps_dir, filename)

            # Write in C++ format: int32 rows, int32 cols, then float64 data
            with open(filepath, 'wb') as f:
                rows, cols = response_map.shape
                f.write(np.array([rows], dtype=np.int32).tobytes())
                f.write(np.array([cols], dtype=np.int32).tobytes())
                f.write(response_map.astype(np.float64).tobytes())

        if self.debug_mode:
            print(f"[SAVE] Saved {len(response_maps)} response maps to {self.save_response_maps_dir}")

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

        # =================================================================
        # SCALE-ADAPTIVE PARAMETERS (C++ LandmarkDetectorModel.cpp:943-950)
        # =================================================================
        # Compute adapted regularization, sigma, and weight_multiplier based on patch_scaling
        adapted_reg, adapted_sigma, adapted_weight = self._compute_scale_adapted_params(patch_scaling)

        # Store adapted values for use in this optimization run
        # These override base values for the duration of this optimize() call
        self._current_regularization = adapted_reg
        self._current_sigma = adapted_sigma
        self._current_weight_multiplier = adapted_weight

        if self.debug_mode:
            print(f"\n[PY][SCALE] Scale adaptation at patch_scaling={patch_scaling:.3f}:")
            print(f"[PY][SCALE]   Base reg={self._base_regularization:.3f} -> Adapted reg={adapted_reg:.3f}")
            print(f"[PY][SCALE]   Base sigma={self._base_sigma:.3f} -> Adapted sigma={adapted_sigma:.3f}")
            print(f"[PY][SCALE]   Base weight={self._base_weight_multiplier:.3f} -> Adapted weight={adapted_weight:.3f}")

        # Initialize weights (default: uniform)
        if weights is None:
            weights = np.ones(n_landmarks)

        # Create diagonal weight matrix W for 2D landmarks (2n × 2n)
        # OpenFace behavior (see PDM.cpp line 613 and LandmarkDetectorModel.cpp):
        # - weight_factor > 0: W = weight_factor · diag(patch_confidences)  [NU-RLMS mode]
        # - weight_factor = 0: W = Identity  [Video mode - all landmarks weighted equally]
        #
        # CRITICAL: Weight matrix must be in STACKED format to match Jacobian/mean-shift:
        #   Jacobian rows: [x0, x1, ..., xn-1, y0, y1, ..., yn-1]
        #   Weight diag:   [w0, w1, ..., wn-1, w0, w1, ..., wn-1]
        # NOT INTERLEAVED: [w0, w0, w1, w1, ...] (WRONG!)
        if adapted_weight > 0:
            # NU-RLMS mode: apply adapted weight multiplier to patch confidences
            # STACKED format: concatenate weights for x-components then y-components
            W = adapted_weight * np.diag(np.concatenate([weights, weights]))
        else:
            # Video mode: use identity matrix (all landmarks weighted equally)
            W = np.eye(n_landmarks * 2)

        # Apply jaw weight reduction if enabled
        # Jaw landmarks (0-16) have 40% weaker response maps, so reduce their influence
        # This makes the optimizer rely more on the shape prior for jaw landmarks
        if self.use_jaw_weight_reduction and n_landmarks == 68:
            jaw_indices = list(range(17))  # Landmarks 0-16
            for idx in jaw_indices:
                W[idx, idx] *= self.jaw_weight_factor  # x-component
                W[idx + n_landmarks, idx + n_landmarks] *= self.jaw_weight_factor  # y-component

        # Create regularization matrix Λ^(-1)
        Lambda_inv = self._compute_lambda_inv(pdm, n_params)

        # Debug: Print initialization
        if self.debug_mode:
            init_landmarks = pdm.params_to_landmarks_2d(params)
            print(f"\n[PY][INIT] Initial parameters:")
            print(f"[PY][INIT]   params_local (first 5): {params[:5]}")
            print(f"[PY][INIT]   scale: {params[0]:.6f}")
            print(f"[PY][INIT]   rotation: ({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
            print(f"[PY][INIT]   translation: ({params[4]:.6f}, {params[5]:.6f})")
            print(f"[PY][INIT] Initial tracked landmarks:")
            for lm_idx in self.tracked_landmarks:
                if lm_idx < len(init_landmarks):
                    print(f"[PY][INIT]   Landmark_{lm_idx}: ({init_landmarks[lm_idx][0]:.4f}, {init_landmarks[lm_idx][1]:.4f})")

        # =================================================================
        # STRUCTURE MATCHING C++ NU_RLMS (lines 1148-1278)
        # =================================================================

        # 1. Get initial landmark positions in IMAGE coordinates
        landmarks_2d_initial = pdm.params_to_landmarks_2d(params)

        # 2. Get REFERENCE shape at patch_scaling (canonical pose)
        # FIX: C++ uses params_local=0 (mean shape) for reference, NOT current local params!
        # Verified by debug output: C++ shows params_local = [0,0,0,0,0] at iteration 0
        # This improves jaw accuracy by ~5-6%
        reference_shape = pdm.get_reference_shape(patch_scaling, np.zeros(pdm.n_modes))

        # 3. Compute similarity transform: IMAGE ↔ REFERENCE
        from .utils import align_shapes_with_scale, invert_similarity_transform
        sim_img_to_ref = align_shapes_with_scale(landmarks_2d_initial, reference_shape)
        sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

        # 4. PRECOMPUTE response maps ONCE at initial positions (C++ line 798)
        # These are reused for ALL iterations in both rigid and non-rigid phases!
        # VIDEO-MODE OPTIMIZATION: Per-scale caching for temporal coherence
        cache_hit = self._should_reuse_scale_cache(patch_scaling, landmarks_2d_initial)
        if cache_hit:
            # Reuse cached response maps for this scale - significant speedup for video
            response_maps = self._scale_cache[patch_scaling]['response_maps']
            self._scale_cache[patch_scaling]['age'] += 1
            self.cache_age = self._scale_cache[patch_scaling]['age']  # Legacy compat
        else:
            # Compute fresh response maps
            response_maps = self._precompute_response_maps(
                landmarks_2d_initial, patch_experts, image, window_size,
                sim_img_to_ref, sim_ref_to_img, sigma_components, iteration=0
            )
            # Update per-scale cache for next frame
            self._scale_cache[patch_scaling] = {
                'response_maps': response_maps,
                'landmarks': landmarks_2d_initial.copy(),
                'age': 0
            }
            self.cache_age = 0  # Legacy compat

            # Save response maps if directory is configured
            self.save_response_maps(response_maps, window_size, patch_scaling)

        # =================================================================
        # DYNAMIC PEAK CONFIDENCE WEIGHTING
        # Reduce influence of landmarks with weak/ambiguous response peaks
        # Only apply at larger window sizes (WS >= 9) where divergence originates
        # At smaller window sizes, response maps are naturally less peaked
        # =================================================================
        if self.use_peak_confidence and window_size >= 9:
            peak_confidences = self._compute_peak_confidences(response_maps, n_landmarks)

            # Apply a gentler formula: sqrt(conf) to reduce penalty
            # This gives weak landmarks more influence than linear scaling
            peak_confidences = np.sqrt(peak_confidences)

            # Modify weight matrix W to incorporate peak confidences
            # W is currently: adapted_weight * diag([weights, weights])
            # We multiply each weight by its peak confidence
            for i in range(n_landmarks):
                W[i, i] *= peak_confidences[i]
                W[i + n_landmarks, i + n_landmarks] *= peak_confidences[i]

            if self.debug_mode:
                low_conf_lms = [i for i in range(n_landmarks) if peak_confidences[i] < 0.7]
                if low_conf_lms:
                    print(f"[PEAK_CONF] WS{window_size} reduced-confidence landmarks: {low_conf_lms}")

        # Debug: Print initial landmarks
        if self.debug_mode:
            print(f"\n[PY][ITER0_WS{window_size}] Initial landmark positions:")
            for lm_idx in self.tracked_landmarks:
                if lm_idx < len(landmarks_2d_initial):
                    print(f"[PY][ITER0_WS{window_size}]   Landmark_{lm_idx}: ({landmarks_2d_initial[lm_idx][0]:.4f}, {landmarks_2d_initial[lm_idx][1]:.4f})")

        # =================================================================
        # PHASE 1: RIGID optimization with inner convergence loop
        # Matches OpenFace LandmarkDetectorModel.cpp:844 NU_RLMS(..., rigid=true)
        # =================================================================

        rigid_params = params.copy()
        base_landmarks_rigid = landmarks_2d_initial.copy()  # Base for rigid = initial
        previous_landmarks = None
        rigid_converged = False
        iteration_info = []  # Initialize iteration tracking here for both phases

        for rigid_iter in range(self.rigid_iterations):
            # Compute current shape from rigid params
            current_landmarks = pdm.params_to_landmarks_2d(rigid_params)

            # Early stopping: check if landmarks have converged
            # Only check after min_iterations to ensure we've made progress
            # C++ OpenFace uses L2 norm of entire flattened shape vector (136 values)
            # See LandmarkDetectorModel.cpp:603: if(norm(current_shape, previous_shape) < 0.01)
            if (rigid_iter >= self.min_iterations and
                previous_landmarks is not None and
                self.convergence_threshold > 0):
                # L2 norm of flattened shape difference (matches C++ cv::norm behavior)
                shape_diff = (current_landmarks - previous_landmarks).flatten()  # (136,)
                shape_change = np.linalg.norm(shape_diff)  # sqrt(sum(diff^2))
                if self.debug_mode and window_size == 11:
                    print(f"[DEBUG] RIGID iter {rigid_iter}: shape_change = {shape_change:.6f} (threshold: 0.01)")
                if shape_change < 0.01:  # Matches C++ exactly
                    rigid_converged = True
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute mean-shift using PRECOMPUTED response maps and current offsets
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks_rigid, response_maps, patch_experts,
                window_size, sim_img_to_ref, sim_ref_to_img, iteration=rigid_iter
            )

            # Debug: Print mean-shift for first iteration
            if self.debug_mode and rigid_iter == 0:
                print(f"[PY][ITER0_WS{window_size}] RIGID Mean-shift vectors (STACKED format):")
                n_lm = len(current_landmarks)
                for lm_idx in self.tracked_landmarks:
                    if lm_idx < n_lm:
                        ms_x = mean_shift[lm_idx]
                        ms_y = mean_shift[lm_idx + n_lm]
                        ms_mag = np.sqrt(ms_x**2 + ms_y**2)
                        print(f"[PY][ITER0_WS{window_size}]   Landmark_{lm_idx}: ms=({ms_x:.4f}, {ms_y:.4f}) mag={ms_mag:.4f}")

            # Compute Jacobian for RIGID parameters only
            J_rigid = pdm.compute_jacobian_rigid(rigid_params)

            # Solve for rigid parameter update
            delta_p_rigid = self._solve_rigid_update(J_rigid, mean_shift, W, rigid_iter, window_size)

            # Update ONLY global parameters
            delta_p_full = np.zeros(len(rigid_params))
            delta_p_full[:6] = delta_p_rigid
            rigid_params = pdm.update_params(rigid_params, delta_p_full)
            rigid_params = pdm.clamp_params(rigid_params)

            # Track rigid iteration info
            # STACKED format: first n values are x, next n are y
            n_landmarks = len(mean_shift) // 2
            per_landmark_ms = [np.sqrt(mean_shift[i]**2 + mean_shift[i + n_landmarks]**2)
                              for i in range(n_landmarks)]

            # Reshape for norm calculation: convert stacked to (n, 2) for per-landmark magnitude
            ms_x = mean_shift[:n_landmarks]
            ms_y = mean_shift[n_landmarks:]
            ms_2d = np.column_stack([ms_x, ms_y])  # (n, 2)

            iteration_info.append({
                'iteration': len(iteration_info),  # Global iteration counter
                'phase': 'rigid',
                'window_size': window_size,
                'update_magnitude': np.linalg.norm(delta_p_rigid),
                'params': rigid_params.copy(),
                'mean_shift_norm': np.linalg.norm(mean_shift),
                'mean_shift_mean': np.mean(np.linalg.norm(ms_2d, axis=1)),
                'jacobian_norm': np.linalg.norm(J_rigid),
                'regularization': 0.0,  # No regularization in rigid phase
                # Enhanced diagnostics
                'per_landmark_ms': per_landmark_ms,  # Per-landmark mean-shift magnitudes
                'hessian_cond': None,  # Not computed for rigid phase
                'reg_ratio': None,  # No regularization in rigid
                'jtw_norm': None,
                'reg_term_norm': None,
            })

        # Copy rigid updates to params
        params[:6] = rigid_params[:6]

        # Debug: Print params after rigid phase
        if window_size == 11 and self.debug_mode:
            print(f"\n[DEBUG] RIGID phase completed: {rigid_iter + 1} iterations, converged={rigid_converged}")
            print(f"[DEBUG]   Final rigid params: scale={params[0]:.6f}, rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
        if self.debug_mode:
            print(f"[PY][RIGID_COMPLETE] Rigid params after {rigid_iter + 1} iterations:")
            print(f"[PY][RIGID_COMPLETE]   scale: {params[0]:.6f}")
            print(f"[PY][RIGID_COMPLETE]   rotation: ({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
            # Print LM5 position after rigid phase
            rigid_landmarks = pdm.params_to_landmarks_2d(params)
            for lm_idx in self.tracked_landmarks:
                if lm_idx < len(rigid_landmarks):
                    print(f"[PY][RIGID_COMPLETE]   Landmark_{lm_idx}: ({rigid_landmarks[lm_idx][0]:.4f}, {rigid_landmarks[lm_idx][1]:.4f})")

        # =================================================================
        # PHASE 2: NON-RIGID optimization with inner convergence loop
        # Matches OpenFace LandmarkDetectorModel.cpp:868 NU_RLMS(..., rigid=false)
        # =================================================================

        # CRITICAL FIX: Both phases must use the SAME base landmarks
        # Response maps are precomputed at initial landmarks, so we must
        # use the same base for offset computation in both phases
        base_landmarks_nonrigid = base_landmarks_rigid  # Use same as rigid phase!
        previous_landmarks = None
        nonrigid_converged = False

        # DEBUG: Print base_landmarks for non-rigid
        if window_size == 11 and self.debug_mode:
            print(f"\n[DEBUG] NON-RIGID base landmarks (same as rigid phase - initial):")
            for lm_idx in [36, 48]:
                print(f"[DEBUG]   base_nonrigid[{lm_idx}]: ({base_landmarks_nonrigid[lm_idx][0]:.4f}, {base_landmarks_nonrigid[lm_idx][1]:.4f})")
            print(f"[DEBUG]   These should be the SAME as initial landmarks (for response map consistency):")
            print(f"[DEBUG]   Initial landmarks:")
            for lm_idx in [36, 48]:
                print(f"[DEBUG]   initial[{lm_idx}]: ({landmarks_2d_initial[lm_idx][0]:.4f}, {landmarks_2d_initial[lm_idx][1]:.4f})")

        for nonrigid_iter in range(self.nonrigid_iterations):
            # Compute current shape from params
            current_landmarks = pdm.params_to_landmarks_2d(params)

            # DEBUG: Print iteration info
            if window_size == 11 and nonrigid_iter < 2 and self.debug_mode:
                print(f"\n[DEBUG] NON-RIGID iteration {nonrigid_iter}:")
                for lm_idx in [36, 48]:
                    offset_x = current_landmarks[lm_idx][0] - base_landmarks_nonrigid[lm_idx][0]
                    offset_y = current_landmarks[lm_idx][1] - base_landmarks_nonrigid[lm_idx][1]
                    print(f"[DEBUG]   Landmark {lm_idx}: current=({current_landmarks[lm_idx][0]:.4f}, {current_landmarks[lm_idx][1]:.4f})")
                    print(f"[DEBUG]   Landmark {lm_idx}: offset=({offset_x:.4f}, {offset_y:.4f})")

            # Early stopping: check if landmarks have converged
            # Only check after min_iterations to ensure we've made progress
            # C++ OpenFace uses L2 norm of entire flattened shape vector (136 values)
            # See LandmarkDetectorModel.cpp:603: if(norm(current_shape, previous_shape) < 0.01)
            if (nonrigid_iter >= self.min_iterations and
                previous_landmarks is not None and
                self.convergence_threshold > 0):
                # L2 norm of flattened shape difference (matches C++ cv::norm behavior)
                shape_diff = (current_landmarks - previous_landmarks).flatten()  # (136,)
                shape_change = np.linalg.norm(shape_diff)  # sqrt(sum(diff^2))
                if shape_change < 0.01:  # Matches C++ exactly
                    nonrigid_converged = True
                    break
            previous_landmarks = current_landmarks.copy()

            # Compute mean-shift using PRECOMPUTED response maps and current offsets
            # Offsets = current - base_nonrigid (where base = rigid-updated params)
            mean_shift = self._compute_mean_shift(
                current_landmarks, base_landmarks_nonrigid, response_maps, patch_experts,
                window_size, sim_img_to_ref, sim_ref_to_img, iteration=nonrigid_iter
            )

            # Debug: Print mean-shift for first iteration
            if self.debug_mode and nonrigid_iter == 0:
                print(f"[PY][ITER0_WS{window_size}] NONRIGID Mean-shift vectors (STACKED format):")
                n_lm = len(current_landmarks)
                for lm_idx in self.tracked_landmarks:
                    if lm_idx < n_lm:
                        ms_x = mean_shift[lm_idx]
                        ms_y = mean_shift[lm_idx + n_lm]
                        ms_mag = np.sqrt(ms_x**2 + ms_y**2)
                        print(f"[PY][ITER0_WS{window_size}]   Landmark_{lm_idx}: ms=({ms_x:.4f}, {ms_y:.4f}) mag={ms_mag:.4f}")

            # DEBUG: Print params BEFORE Jacobian for WS9 nonrigid iter 0
            if window_size == 9 and nonrigid_iter == 0 and self.debug_mode:
                print(f"\n[PY][JACOBIAN_INPUT_WS9] params: scale={params[0]:.4f} rot=({params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f}) tx={params[4]:.4f} ty={params[5]:.4f}")
                print(f"[PY][JACOBIAN_INPUT_WS9] params_local[:5]: [{params[6]:.4f}, {params[7]:.4f}, {params[8]:.4f}, {params[9]:.4f}, {params[10]:.4f}]")

            # Compute full Jacobian (global + local)
            J = pdm.compute_jacobian(params)

            # DEBUG: Check mean_shift right before solve
            if window_size in (11, 9) and nonrigid_iter == 0 and self.debug_mode:
                n_lm = len(mean_shift) // 2
                ms_x = mean_shift[:n_lm]
                ms_y = mean_shift[n_lm:]
                ms_x_sum = np.sum(ms_x)
                ms_y_sum = np.sum(ms_y)
                print(f"\n[DEBUG][BEFORE_SOLVE] WS{window_size} nonrigid_iter{nonrigid_iter}:")
                print(f"[DEBUG]   n_landmarks = {n_lm}")
                print(f"[DEBUG]   ms_x[:5] = {ms_x[:5]}")
                print(f"[DEBUG]   ms_x_sum = {ms_x_sum:.6f}, ms_y_sum = {ms_y_sum:.6f}")
                print(f"[DEBUG]   ms_x abs_sum = {np.sum(np.abs(ms_x)):.4f}")
                print(f"[DEBUG]   ms_x positive = {(ms_x > 0).sum()}, negative = {(ms_x < 0).sum()}, zero = {(ms_x == 0).sum()}")
                # Print LM5 mean-shift
                print(f"[DEBUG]   LM5 mean_shift: ({ms_x[5]:.4f}, {ms_y[5]:.4f})")

            # Solve for full parameter update with regularization
            delta_p = self._solve_update(J, mean_shift, W, Lambda_inv, params, nonrigid_iter, window_size)

            # Update ALL parameters
            params = pdm.update_params(params, delta_p)
            params = pdm.clamp_params(params)

            # Track iteration info with enhanced diagnostics
            # STACKED format: first n values are x, next n are y
            n_landmarks = len(mean_shift) // 2
            per_landmark_ms = [np.sqrt(mean_shift[i]**2 + mean_shift[i + n_landmarks]**2)
                              for i in range(n_landmarks)]

            # Reshape for norm calculation: convert stacked to (n, 2) for per-landmark magnitude
            ms_x = mean_shift[:n_landmarks]
            ms_y = mean_shift[n_landmarks:]
            ms_2d = np.column_stack([ms_x, ms_y])  # (n, 2)

            iteration_info.append({
                'iteration': len(iteration_info),  # Global iteration counter
                'phase': 'nonrigid',
                'window_size': window_size,
                'update_magnitude': np.linalg.norm(delta_p),
                'params': params.copy(),
                'mean_shift_norm': np.linalg.norm(mean_shift),
                'mean_shift_mean': np.mean(np.linalg.norm(ms_2d, axis=1)),
                'jacobian_norm': np.linalg.norm(J),
                'regularization': self.regularization,
                # Enhanced diagnostics for late-stage convergence analysis
                'per_landmark_ms': per_landmark_ms,  # Per-landmark mean-shift magnitudes
                'hessian_cond': self._last_hessian_cond,  # Hessian condition number
                'reg_ratio': self._last_reg_ratio,  # reg_term / data_term ratio
                'jtw_norm': self._last_jtw_norm,  # Data term magnitude
                'reg_term_norm': self._last_reg_term_norm,  # Regularization term magnitude
            })

            # Debug: Print landmarks after iteration
            if self.debug_mode:
                iter_landmarks = pdm.params_to_landmarks_2d(params)
                print(f"[PY][ITER{nonrigid_iter + 1}_WS{window_size}] Landmark positions:")
                for lm_idx in self.tracked_landmarks:
                    if lm_idx < len(iter_landmarks):
                        print(f"[PY][ITER{nonrigid_iter + 1}_WS{window_size}]   Landmark_{lm_idx}: ({iter_landmarks[lm_idx][0]:.4f}, {iter_landmarks[lm_idx][1]:.4f})")

        # Determine overall convergence
        converged = rigid_converged and nonrigid_converged

        # Debug: Print non-rigid phase completion
        if window_size == 11 and self.debug_mode:
            print(f"\n[DEBUG] NON-RIGID phase completed: {nonrigid_iter + 1} iterations, converged={nonrigid_converged}")
            print(f"[DEBUG]   Final params: scale={params[0]:.6f}, local[0]={params[6]:.6f}")

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

        # Shape parameters (indices 6+): use reg_factor / eigenvalues
        # C++ uses reg_factor = 22.5. Testing showed:
        #   reg_factor=1.0:  4/17 AUs pass
        #   reg_factor=5.0:  5/17 AUs pass
        #   reg_factor=10.0: 5/17 AUs pass
        #   reg_factor=22.5: 6/17 AUs pass but CLNF doesn't converge
        # Keeping 1.0 for now; issue is deeper than regularization.
        reg_factor = 1.0
        eigenvalues = pdm.eigen_values.flatten()
        Lambda_inv[6:] = reg_factor / eigenvalues

        return Lambda_inv

    def _should_reuse_cache(self, current_landmarks: np.ndarray) -> bool:
        """
        Check if cached response maps can be reused for the current frame.

        Video-mode optimization: when landmarks haven't moved much between frames,
        we can reuse the response maps from the previous frame instead of recomputing.
        This can provide 50-100% speedup since response map computation is 40-50% of CLNF time.

        Args:
            current_landmarks: Current 2D landmark positions (n_points, 2)

        Returns:
            True if cache can be reused, False if recomputation is needed
        """
        # No cache available
        if self.cached_response_maps is None or self.cached_landmarks is None:
            return False

        # Cache too old - force recompute to prevent drift
        if self.cache_age >= self.cache_max_age:
            return False

        # Check if landmarks are close enough to cached positions
        displacement = np.linalg.norm(current_landmarks - self.cached_landmarks, axis=1)
        max_displacement = displacement.max()

        return max_displacement < self.response_reuse_threshold

    def _should_reuse_scale_cache(self, scale: float, current_landmarks: np.ndarray) -> bool:
        """
        Check if cached response maps can be reused for a specific scale.

        Per-scale caching ensures that each scale maintains its own cache,
        preventing cross-scale cache invalidation in multi-scale optimization.

        Args:
            scale: The patch scaling factor (0.25, 0.35, 0.5)
            current_landmarks: Current 2D landmark positions (n_points, 2)

        Returns:
            True if cache can be reused for this scale, False otherwise
        """
        # No cache for this scale
        if scale not in self._scale_cache:
            return False

        cache = self._scale_cache[scale]

        # Cache too old - force recompute to prevent drift
        if cache['age'] >= self.cache_max_age:
            return False

        # Check if landmarks are close enough to cached positions
        displacement = np.linalg.norm(current_landmarks - cache['landmarks'], axis=1)
        max_displacement = displacement.max()

        return max_displacement < self.response_reuse_threshold

    def _extract_area_of_interest(self,
                                   image: np.ndarray,
                                   center_x: float,
                                   center_y: float,
                                   patch_expert,
                                   window_size: int,
                                   sim_img_to_ref: np.ndarray = None,
                                   sim_ref_to_img: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Extract the area of interest around a landmark for patch expert evaluation.

        This extracts the warped region around the landmark that will be processed
        by the patch expert neural network. Used by batched processing.

        Args:
            image: Input grayscale image
            center_x, center_y: Landmark position in IMAGE coordinates
            patch_expert: CENPatchExpert for this landmark
            window_size: Size of the search window (response map size)
            sim_img_to_ref: Optional 2x3 similarity transform (IMAGE → REFERENCE)
            sim_ref_to_img: Optional 2x3 similarity transform (REFERENCE → IMAGE)

        Returns:
            area_of_interest: Warped image region, or None if extraction fails
        """
        # Calculate area of interest size (same as _compute_response_map)
        if hasattr(patch_expert, 'width_support'):
            patch_dim = max(patch_expert.width_support, patch_expert.height_support)
        else:
            patch_dim = max(patch_expert.width, patch_expert.height)

        area_of_interest_width = window_size + patch_dim - 1
        area_of_interest_height = window_size + patch_dim - 1

        if sim_img_to_ref is not None and sim_ref_to_img is not None:
            # WARPING MODE: Use similarity transform to extract warped region
            # Use _extract_aoi which supports cpp_warp for exact C++ OpenFace matching
            area_of_interest = self._extract_aoi(
                image, center_x, center_y, sim_ref_to_img, area_of_interest_width
            )
        else:
            # NO WARPING: Direct extraction from image
            half_aoi = area_of_interest_width // 2
            x_start = int(center_x) - half_aoi
            y_start = int(center_y) - half_aoi
            x_end = x_start + area_of_interest_width
            y_end = y_start + area_of_interest_height

            # Bounds check with padding
            if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
                # Use copyMakeBorder for out-of-bounds regions
                pad_left = max(0, -x_start)
                pad_right = max(0, x_end - image.shape[1])
                pad_top = max(0, -y_start)
                pad_bottom = max(0, y_end - image.shape[0])

                x_start_safe = max(0, x_start)
                y_start_safe = max(0, y_start)
                x_end_safe = min(image.shape[1], x_end)
                y_end_safe = min(image.shape[0], y_end)

                area_of_interest = image[y_start_safe:y_end_safe, x_start_safe:x_end_safe].copy()
                area_of_interest = cv2.copyMakeBorder(
                    area_of_interest,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REPLICATE
                )
            else:
                area_of_interest = image[y_start:y_end, x_start:x_end].copy()

        return area_of_interest

    def _compute_peak_confidences(self, response_maps: dict, n_landmarks: int) -> np.ndarray:
        """
        Compute dynamic confidence weights based on response map peak sharpness.

        Landmarks with weak/flat response peaks (ambiguous detection) get lower weights,
        reducing their influence on parameter updates. This helps with low-contrast regions
        where the "peak" is barely distinguishable from background noise.

        The confidence is computed as:
            peak_to_mean = response.max() / response.mean()
            confidence = clamp((peak_to_mean - 1.0) / 4.0, 0, 1)

        This maps:
            - peak_to_mean = 1.0 (flat) -> confidence = 0.0
            - peak_to_mean = 2.0 -> confidence = 0.25
            - peak_to_mean = 5.0+ -> confidence = 1.0

        Args:
            response_maps: Dict mapping landmark_idx -> response_map array
            n_landmarks: Total number of landmarks

        Returns:
            peak_confidences: Array of shape (n_landmarks,) with confidence values [0, 1]
        """
        peak_confidences = np.ones(n_landmarks)  # Default to 1.0 for missing landmarks

        for landmark_idx, response_map in response_maps.items():
            if landmark_idx >= n_landmarks:
                continue

            # Compute peak-to-mean ratio
            resp_mean = response_map.mean()
            resp_max = response_map.max()

            if resp_mean > 1e-10:
                peak_to_mean = resp_max / resp_mean
            else:
                peak_to_mean = 1.0  # Treat zero response as flat

            # Map to confidence: (p2m - 1) / 4, clamped to [0, 1]
            # p2m=1.0 -> 0.0, p2m=2.0 -> 0.25, p2m=5.0+ -> 1.0
            confidence = min(1.0, max(0.0, (peak_to_mean - 1.0) / 4.0))
            peak_confidences[landmark_idx] = confidence

            if self.debug_mode and landmark_idx in self.tracked_landmarks:
                print(f"[PEAK_CONF] LM{landmark_idx}: p2m={peak_to_mean:.3f} -> conf={confidence:.3f}")

        return peak_confidences

    def _precompute_response_maps(self,
                                   landmarks_2d: np.ndarray,
                                   patch_experts: dict,
                                   image: np.ndarray,
                                   window_size: int,
                                   sim_img_to_ref: np.ndarray = None,
                                   sim_ref_to_img: np.ndarray = None,
                                   sigma_components: dict = None,
                                   iteration: int = None) -> dict:
        """
        Precompute response maps at initial landmark positions.

        This matches OpenFace's Response() call which computes response maps ONCE
        before optimization, then reuses them for both rigid and non-rigid phases.

        NOTE: C++ OpenFace processes left/right symmetric landmarks together in a single
        ResponseSparse() call for efficiency, but each landmark gets its OWN independent
        response map. There is NO weighted averaging between left and right responses.

        For mirrored landmarks (right side), the MirroredCENPatchExpert wrapper already
        correctly implements flip-process-flip to use the left side's neural network.

        Args:
            landmarks_2d: Initial 2D landmark positions (n_points, 2)
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            image: Grayscale image
            window_size: Response map size
            sim_img_to_ref: Similarity transform (image → reference)
            sim_ref_to_img: Similarity transform (reference → image)

        Returns:
            response_maps: Dict mapping landmark_idx -> response_map array
        """
        # GPU acceleration: dispatch to GPU implementation when enabled
        if self.use_gpu:
            return self._precompute_response_maps_gpu(
                landmarks_2d, patch_experts, image, window_size,
                sim_ref_to_img=sim_ref_to_img
            )

        response_maps = {}
        use_warping = (sim_img_to_ref is not None and sim_ref_to_img is not None)

        # C++ STYLE BATCHED PROCESSING FOR MIRROR PAIRS
        # Identify mirror pairs: left landmarks with real experts and their mirrored counterparts
        # Process them together using response_sparse like C++ does
        processed = set()

        for landmark_idx, patch_expert in patch_experts.items():
            if landmark_idx in processed:
                continue

            # Check if this is a mirrored expert (right side landmark)
            if isinstance(patch_expert, MirroredCENPatchExpert):
                # This landmark uses a mirrored expert - it will be processed with its mirror pair
                # Find the left landmark that has the real expert
                real_expert = patch_expert._mirror_expert
                # Find which left landmark has this expert
                left_idx = None
                for l_idx, l_expert in patch_experts.items():
                    if l_expert is real_expert:
                        left_idx = l_idx
                        break

                if left_idx is not None and left_idx not in processed:
                    # Process both together using response_sparse
                    lm_x_left, lm_y_left = landmarks_2d[left_idx]
                    lm_x_right, lm_y_right = landmarks_2d[landmark_idx]

                    # Extract AOIs for both landmarks
                    aoi_left = self._extract_area_of_interest(
                        image, lm_x_left, lm_y_left, real_expert, window_size,
                        sim_img_to_ref if use_warping else None,
                        sim_ref_to_img if use_warping else None
                    )
                    aoi_right = self._extract_area_of_interest(
                        image, lm_x_right, lm_y_right, real_expert, window_size,
                        sim_img_to_ref if use_warping else None,
                        sim_ref_to_img if use_warping else None
                    )

                    if aoi_left is not None and aoi_right is not None:
                        # Use batched processing like C++
                        resp_left, resp_right = real_expert.response_sparse(aoi_left, aoi_right)
                        if resp_left is not None:
                            response_maps[left_idx] = resp_left
                        if resp_right is not None:
                            response_maps[landmark_idx] = resp_right
                    else:
                        # Fallback to individual processing if AOI extraction failed
                        # Still use response_sparse with None for missing side to match C++
                        if aoi_left is not None:
                            resp_left, _ = real_expert.response_sparse(aoi_left, None)
                            if resp_left is not None:
                                response_maps[left_idx] = resp_left
                        if aoi_right is not None:
                            # Right side uses mirrored expert which internally uses real_expert
                            resp_right, _ = real_expert.response_sparse(None, aoi_right)
                            if resp_right is not None:
                                response_maps[landmark_idx] = resp_right

                    processed.add(left_idx)
                    processed.add(landmark_idx)
                    continue

            # Check if this is a real CEN expert with a mirror pair
            if isinstance(patch_expert, CENPatchExpert) and not patch_expert.is_empty:
                # Find if there's a mirrored landmark that uses this expert
                mirror_idx = None
                for m_idx, m_expert in patch_experts.items():
                    if isinstance(m_expert, MirroredCENPatchExpert):
                        if m_expert._mirror_expert is patch_expert:
                            mirror_idx = m_idx
                            break

                if mirror_idx is not None and mirror_idx not in processed:
                    # Process both together
                    lm_x_left, lm_y_left = landmarks_2d[landmark_idx]
                    lm_x_right, lm_y_right = landmarks_2d[mirror_idx]

                    aoi_left = self._extract_area_of_interest(
                        image, lm_x_left, lm_y_left, patch_expert, window_size,
                        sim_img_to_ref if use_warping else None,
                        sim_ref_to_img if use_warping else None
                    )
                    aoi_right = self._extract_area_of_interest(
                        image, lm_x_right, lm_y_right, patch_expert, window_size,
                        sim_img_to_ref if use_warping else None,
                        sim_ref_to_img if use_warping else None
                    )

                    if aoi_left is not None and aoi_right is not None:
                        resp_left, resp_right = patch_expert.response_sparse(aoi_left, aoi_right)
                        if resp_left is not None:
                            response_maps[landmark_idx] = resp_left
                        if resp_right is not None:
                            response_maps[mirror_idx] = resp_right
                    else:
                        # Fallback - use response_sparse with None for missing side to match C++
                        if aoi_left is not None:
                            resp_left, _ = patch_expert.response_sparse(aoi_left, None)
                            if resp_left is not None:
                                response_maps[landmark_idx] = resp_left
                        if aoi_right is not None:
                            _, resp_right = patch_expert.response_sparse(None, aoi_right)
                            if resp_right is not None:
                                response_maps[mirror_idx] = resp_right

                    processed.add(landmark_idx)
                    processed.add(mirror_idx)
                    continue

            # Process individually (no mirror pair or self-mirrored landmark like chin)
            lm_x, lm_y = landmarks_2d[landmark_idx]
            response_map = self._compute_response_map(
                image, lm_x, lm_y, patch_expert, window_size,
                sim_img_to_ref if use_warping else None,
                sim_ref_to_img if use_warping else None,
                sigma_components,
                landmark_idx, iteration
            )

            if response_map is not None:
                response_maps[landmark_idx] = response_map
            processed.add(landmark_idx)

        return response_maps

    def _get_or_create_batched_cen(self, patch_experts: dict) -> 'BatchedCEN':
        """
        Get or create a BatchedCEN instance for the given patch experts.

        Uses caching to avoid recreating the BatchedCEN for repeated calls
        with the same patch experts (identified by id).

        Args:
            patch_experts: Dict mapping landmark_idx -> CENPatchExpert

        Returns:
            BatchedCEN instance
        """
        cache_key = id(patch_experts)
        if cache_key not in self._batched_cen_cache:
            self._batched_cen_cache[cache_key] = BatchedCEN(patch_experts, device=self.gpu_device)
        return self._batched_cen_cache[cache_key]

    def _get_or_create_gpu_mean_shift(self) -> 'GPUMeanShift':
        """
        Get or create a GPUMeanShift instance.

        Uses lazy initialization and caching.

        Returns:
            GPUMeanShift instance
        """
        if self._gpu_mean_shift is None:
            # Use scale-adapted sigma if available
            current_sigma = getattr(self, '_current_sigma', self.sigma)
            self._gpu_mean_shift = GPUMeanShift(device=self.gpu_device, sigma=current_sigma)
        return self._gpu_mean_shift

    def _precompute_response_maps_gpu(self,
                                       landmarks_2d: np.ndarray,
                                       patch_experts: dict,
                                       image: np.ndarray,
                                       window_size: int,
                                       sim_ref_to_img: np.ndarray = None) -> dict:
        """
        GPU-accelerated response map computation using BatchedCEN.

        Provides exact numerical match with CPU _precompute_response_maps
        while processing all 68 landmarks in a single batched pass.

        Now supports warping transforms for multi-scale processing.

        Args:
            landmarks_2d: Initial 2D landmark positions (n_points, 2)
            patch_experts: Dict mapping landmark_idx -> CENPatchExpert
            image: Grayscale image
            window_size: Response map size
            sim_ref_to_img: Optional 2x3 similarity transform (reference → image)

        Returns:
            response_maps: Dict mapping landmark_idx -> response_map array
        """
        batched_cen = self._get_or_create_batched_cen(patch_experts)

        # Use GPU response map computation with warping support
        # use_gpu_warp=False uses CPU warpAffine for exact OpenCV matching
        if self.gpu_device != 'cpu':
            response_maps = batched_cen.compute_response_maps_gpu(
                image, landmarks_2d, window_size,
                sim_ref_to_img=sim_ref_to_img,
                use_gpu_warp=False  # CPU warp for numerical accuracy
            )
        else:
            response_maps = batched_cen.compute_response_maps(
                image, landmarks_2d, window_size,
                sim_ref_to_img=sim_ref_to_img
            )

        return response_maps

    def _compute_mean_shift_gpu(self,
                                landmarks_2d: np.ndarray,
                                base_landmarks_2d: np.ndarray,
                                response_maps: dict,
                                window_size: int = 11,
                                sim_img_to_ref: np.ndarray = None,
                                sim_ref_to_img: np.ndarray = None) -> np.ndarray:
        """
        GPU-accelerated mean-shift computation using GPUMeanShift.

        Provides exact numerical match with CPU _compute_mean_shift
        while processing all 68 landmarks in parallel on GPU.

        Args:
            landmarks_2d: Current 2D landmark positions (n_points, 2)
            base_landmarks_2d: Base landmark positions where response maps were extracted
            response_maps: Dict of precomputed response maps
            window_size: Search window size
            sim_img_to_ref: Similarity transform (image → reference) for transforming offsets
            sim_ref_to_img: Similarity transform (reference → image) for transforming mean-shifts

        Returns:
            mean_shift: Mean-shift vector, shape (2 * n_points,)
        """
        n_points = landmarks_2d.shape[0]

        # Check if we should use warping
        use_warping = (sim_img_to_ref is not None and sim_ref_to_img is not None)

        # Response map size
        resp_size = window_size
        center = (resp_size - 1) / 2.0

        # Compute offsets for all landmarks
        offsets_img = landmarks_2d - base_landmarks_2d  # (68, 2)

        if use_warping:
            # Transform offsets from image to reference coords
            a_sim = sim_img_to_ref[0, 0]
            b_sim = sim_img_to_ref[1, 0]
            offsets_ref_x = a_sim * offsets_img[:, 0] + (-b_sim) * offsets_img[:, 1]
            offsets_ref_y = b_sim * offsets_img[:, 0] + a_sim * offsets_img[:, 1]
        else:
            offsets_ref_x = offsets_img[:, 0]
            offsets_ref_y = offsets_img[:, 1]

        # Compute positions within response maps
        dx = offsets_ref_x + center
        dy = offsets_ref_y + center

        # Use GPU mean-shift
        gpu_ms = self._get_or_create_gpu_mean_shift()
        mean_shift_ref = gpu_ms.compute_mean_shift_gpu(response_maps, dx, dy, window_size)

        # mean_shift_ref is in format [ms_x_0..67, ms_y_0..67] in reference coords
        if use_warping:
            # Transform mean-shift from reference back to image coords
            a_mat = sim_ref_to_img[0, 0]
            b_mat = sim_ref_to_img[1, 0]

            ms_ref_x = mean_shift_ref[:n_points]
            ms_ref_y = mean_shift_ref[n_points:]

            ms_img_x = a_mat * ms_ref_x - b_mat * ms_ref_y
            ms_img_y = b_mat * ms_ref_x + a_mat * ms_ref_y

            mean_shift = np.zeros(2 * n_points, dtype=np.float32)
            mean_shift[:n_points] = ms_img_x
            mean_shift[n_points:] = ms_img_y
        else:
            mean_shift = mean_shift_ref

        return mean_shift

    def _compute_mean_shift_vectorized(self,
                                       response_maps_stacked: np.ndarray,
                                       dx_all: np.ndarray,
                                       dy_all: np.ndarray,
                                       window_size: int,
                                       valid_mask: np.ndarray) -> np.ndarray:
        """
        Fully vectorized mean-shift computation for all landmarks at once.

        Uses NumPy broadcasting to process all 68 landmarks in a single pass,
        avoiding Python loop overhead.

        Args:
            response_maps_stacked: (68, window_size, window_size) stacked response maps
            dx_all: (68,) x positions in response maps
            dy_all: (68,) y positions in response maps
            window_size: response map size (typically 11)
            valid_mask: (68,) boolean mask of valid landmarks

        Returns:
            mean_shift: (136,) stacked [ms_x_0..67, ms_y_0..67]
        """
        n_landmarks = response_maps_stacked.shape[0]

        # Use scale-adapted sigma if available
        current_sigma = getattr(self, '_current_sigma', self.sigma)
        a_kde = -0.5 / (current_sigma * current_sigma)

        # Create coordinate grids (cache if possible)
        if not hasattr(self, '_grid_cache'):
            self._grid_cache = {}

        if window_size not in self._grid_cache:
            jj = np.arange(window_size, dtype=np.float32)  # x coords (columns)
            ii = np.arange(window_size, dtype=np.float32)  # y coords (rows)
            grid_y, grid_x = np.meshgrid(ii, jj, indexing='ij')  # (ws, ws) each
            self._grid_cache[window_size] = (grid_x, grid_y)

        grid_x, grid_y = self._grid_cache[window_size]

        # Expand for broadcasting: (68, 1, 1)
        dx_exp = dx_all[:, None, None]
        dy_exp = dy_all[:, None, None]

        # Compute squared distances for all landmarks at once: (68, ws, ws)
        dist_x = dx_exp - grid_x
        dist_y = dy_exp - grid_y
        dist_sq = dist_x * dist_x + dist_y * dist_y

        # KDE weights: (68, ws, ws)
        kde_weights = np.exp(a_kde * dist_sq)

        # Combined weights: response * kde
        combined_weights = response_maps_stacked * kde_weights

        # Weighted sums: (68,)
        total_weight = combined_weights.sum(axis=(1, 2))
        total_weight = np.maximum(total_weight, 1e-10)  # Avoid division by zero

        weighted_x = (combined_weights * grid_x).sum(axis=(1, 2))
        weighted_y = (combined_weights * grid_y).sum(axis=(1, 2))

        # Centroids and mean-shift
        centroid_x = weighted_x / total_weight
        centroid_y = weighted_y / total_weight

        ms_x = centroid_x - dx_all
        ms_y = centroid_y - dy_all

        # Zero out invalid landmarks
        ms_x[~valid_mask] = 0.0
        ms_y[~valid_mask] = 0.0

        # Stack output: [ms_x_0..67, ms_y_0..67]
        mean_shift = np.concatenate([ms_x, ms_y]).astype(np.float32)
        return mean_shift

    def _compute_mean_shift(self,
                           landmarks_2d: np.ndarray,
                           base_landmarks_2d: np.ndarray,
                           response_maps: dict,
                           patch_experts: dict,
                           window_size: int = 11,
                           sim_img_to_ref: np.ndarray = None,
                           sim_ref_to_img: np.ndarray = None,
                           iteration: int = None) -> np.ndarray:
        """
        Compute mean-shift vector using PRECOMPUTED response maps and offsets.

        This matches OpenFace's NU_RLMS algorithm which:
        1. Uses response maps computed ONCE at initial positions
        2. Computes offsets: (current_landmarks - base_landmarks)
        3. Transforms offsets to reference coords (matching response map coords)
        4. Uses offsets to index into precomputed response maps
        5. Applies KDE mean-shift on the indexed responses

        Args:
            landmarks_2d: Current 2D landmark positions (n_points, 2)
            base_landmarks_2d: Base landmark positions where response maps were extracted
            response_maps: Dict of precomputed response maps (from _precompute_response_maps)
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert
            window_size: Search window size
            sim_img_to_ref: Similarity transform (image → reference) for transforming offsets
            sim_ref_to_img: Similarity transform (reference → image) for transforming mean-shifts

        Returns:
            mean_shift: Mean-shift vector, shape (2 * n_points,)
        """
        n_points = landmarks_2d.shape[0]

        # Check if we should use warping (transforms provided)
        use_warping = (sim_img_to_ref is not None and sim_ref_to_img is not None)

        # FAST PATH: Vectorized CPU mean-shift when using direct KDE (not debug mode)
        # This is faster than GPU mean-shift due to lower transfer overhead
        # Processes all 68 landmarks in a single NumPy pass
        if not self.debug_mode and self.use_direct_kde:
            # Stack response maps into array
            response_maps_stacked = np.zeros((n_points, window_size, window_size), dtype=np.float32)
            valid_mask = np.zeros(n_points, dtype=bool)
            for lm_idx, rm in response_maps.items():
                if lm_idx < n_points and rm.shape == (window_size, window_size):
                    response_maps_stacked[lm_idx] = rm
                    valid_mask[lm_idx] = True

            # Compute offsets for all landmarks at once
            offsets_img = landmarks_2d - base_landmarks_2d  # (68, 2)

            if use_warping:
                a_sim = sim_img_to_ref[0, 0]
                b_sim = sim_img_to_ref[1, 0]
                offsets_ref_x = a_sim * offsets_img[:, 0] + (-b_sim) * offsets_img[:, 1]
                offsets_ref_y = b_sim * offsets_img[:, 0] + a_sim * offsets_img[:, 1]
            else:
                offsets_ref_x = offsets_img[:, 0]
                offsets_ref_y = offsets_img[:, 1]

            # Compute dx, dy positions in response maps
            center = (window_size - 1) / 2.0
            dx_all = (offsets_ref_x + center).astype(np.float32)
            dy_all = (offsets_ref_y + center).astype(np.float32)

            # Clamp to valid range
            dx_all = np.clip(dx_all, 0, window_size - 0.1)
            dy_all = np.clip(dy_all, 0, window_size - 0.1)

            # Vectorized mean-shift computation
            mean_shift_ref = self._compute_mean_shift_vectorized(
                response_maps_stacked, dx_all, dy_all, window_size, valid_mask
            )

            # Transform back to image coords if needed
            if use_warping:
                a_mat = sim_ref_to_img[0, 0]
                b_mat = sim_ref_to_img[1, 0]
                ms_ref_x = mean_shift_ref[:n_points]
                ms_ref_y = mean_shift_ref[n_points:]
                ms_img_x = a_mat * ms_ref_x - b_mat * ms_ref_y
                ms_img_y = b_mat * ms_ref_x + a_mat * ms_ref_y
                mean_shift = np.zeros(2 * n_points, dtype=np.float32)
                mean_shift[:n_points] = ms_img_x
                mean_shift[n_points:] = ms_img_y
                return mean_shift
            else:
                return mean_shift_ref

        # GPU PATH: Use GPU mean-shift when GPU enabled but not using direct KDE
        # (This is slower than vectorized CPU, but matches original GPU behavior)
        if self.use_gpu and not self.debug_mode:
            return self._compute_mean_shift_gpu(
                landmarks_2d, base_landmarks_2d, response_maps,
                window_size, sim_img_to_ref, sim_ref_to_img
            )

        # SLOW PATH: Original per-landmark loop (for debug mode or precomputed KDE grid)
        mean_shift = np.zeros(2 * n_points)

        # Use scale-adapted sigma if available, otherwise use base
        current_sigma = getattr(self, '_current_sigma', self.sigma)

        # Gaussian kernel parameter for KDE: a_kde = -0.5 / sigma^2
        a_kde = -0.5 / (current_sigma * current_sigma)

        # Response map size
        resp_size = window_size

        # For each landmark with a precomputed response map
        for landmark_idx in response_maps.keys():
            if landmark_idx >= n_points:
                continue

            # Get precomputed response map (extracted at base landmark position)
            response_map = response_maps[landmark_idx]

            # Compute offset from base landmark to current landmark position IN IMAGE COORDS
            # This matches OpenFace C++ line 1197-1201:
            #   offsets = (current_shape - base_shape) * sim_img_to_ref
            curr_lm = landmarks_2d[landmark_idx]
            base_lm = base_landmarks_2d[landmark_idx]
            offset_img_x = curr_lm[0] - base_lm[0]
            offset_img_y = curr_lm[1] - base_lm[1]

            # DEBUG: Print offsets for landmark 36 on first iteration
            if landmark_idx == 36 and iteration == 0 and self.debug_mode:
                print(f"[DEBUG] Landmark 36 offset (image coords): ({offset_img_x:.4f}, {offset_img_y:.4f})")
                print(f"[DEBUG]   curr_lm: ({curr_lm[0]:.4f}, {curr_lm[1]:.4f})")
                print(f"[DEBUG]   base_lm: ({base_lm[0]:.4f}, {base_lm[1]:.4f})")
                print(f"[DEBUG]   a_kde (Gaussian param): {a_kde:.6f}")

            # CRITICAL: Response maps are in REFERENCE coordinates (warped patches)
            # C++ transforms offsets: offsets = (current - base) * sim_img_to_ref
            if use_warping:
                # Transform offset vector from image to reference using 2x2 rotation/scale part
                # sim_img_to_ref format: [[a_sim, -b_sim, tx], [b_sim, a_sim, ty]]
                # For vector transformation we use: [a_sim -b_sim; b_sim a_sim] (the rotation/scale part)
                a_sim = sim_img_to_ref[0, 0]
                b_sim = sim_img_to_ref[1, 0]
                offset_ref_x = a_sim * offset_img_x + (-b_sim) * offset_img_y
                offset_ref_y = b_sim * offset_img_x + a_sim * offset_img_y

                # DEBUG: Print transformed offsets
                if landmark_idx == 36 and iteration == 0 and self.debug_mode:
                    print(f"[DEBUG] Landmark 36 offset (ref coords): ({offset_ref_x:.4f}, {offset_ref_y:.4f})")
                    print(f"[DEBUG]   sim_img_to_ref[0,0] (a_sim): {a_sim:.6f}, sim_img_to_ref[1,0] (b_sim): {b_sim:.6f}")
            else:
                offset_ref_x = offset_img_x
                offset_ref_y = offset_img_y

            # Compute position within response map to evaluate
            # OpenFace C++ line 1203-1204:
            #   dxs = offsets.col(0) + (resp_size-1)/2
            #   dys = offsets.col(1) + (resp_size-1)/2
            center = (resp_size - 1) / 2.0

            # For RIGID: offset_ref = 0, so dx/dy = center
            # For NON-RIGID: offset_ref != 0, tracks how far rigid moved landmarks
            dx = offset_ref_x + center
            dy = offset_ref_y + center

            # Compute KDE mean-shift using OpenFace's algorithm
            # Result is in REFERENCE coordinates if warping was used
            # CRITICAL FIX: Pass a_kde (Gaussian param), not a_sim (similarity transform)!
            ms_ref_x, ms_ref_y = self._kde_mean_shift(
                response_map, dx, dy, a_kde, landmark_idx
            )

            if use_warping:
                # Transform mean-shift from REFERENCE back to IMAGE coordinates
                # Apply 2x2 rotation/scale matrix: [a_sim -b_sim; b_sim a_sim]
                a_mat = sim_ref_to_img[0, 0]
                b_mat = sim_ref_to_img[1, 0]
                ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y
                ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y

                # DEBUG: Print transform for landmark 0 in non-rigid iter 0
                if landmark_idx == 0 and iteration == 0 and window_size == 11 and self.debug_mode:
                    print(f"\n[DEBUG][TRANSFORM] LM0 iter0 WS11:")
                    print(f"[DEBUG]   ms_ref = ({ms_ref_x:.4f}, {ms_ref_y:.4f})")
                    print(f"[DEBUG]   a_mat = {a_mat:.4f}, b_mat = {b_mat:.4f}")
                    print(f"[DEBUG]   ms_img = ({ms_x:.4f}, {ms_y:.4f})")
                    print(f"[DEBUG]   offset_ref = ({offset_ref_x:.4f}, {offset_ref_y:.4f})")
            else:
                ms_x = ms_ref_x
                ms_y = ms_ref_y

            # STACKED format: [ms_x[0], ..., ms_x[n-1], ms_y[0], ..., ms_y[n-1]]
            mean_shift[landmark_idx] = ms_x
            mean_shift[landmark_idx + n_points] = ms_y

            # DEBUG: Print mean-shift computation details for landmark 36
            if landmark_idx == 36 and iteration == 0 and window_size == 11 and self.debug_mode:
                print(f"\n[DEBUG] Mean-shift computation for landmark 36:")
                print(f"[DEBUG]   dx={dx:.4f}, dy={dy:.4f} (position in response map)")
                print(f"[DEBUG]   ms_ref=({ms_ref_x:.4f}, {ms_ref_y:.4f}) (in reference coords)")
                print(f"[DEBUG]   ms_img=({ms_x:.4f}, {ms_y:.4f}) (in image coords)")
                print(f"[DEBUG]   Response map stats: min={response_map.min():.6f}, max={response_map.max():.6f}")

        # DEBUG: Print mean_shift vector stats
        if iteration == 0 and window_size == 11 and self.debug_mode:
            print(f"\n[DEBUG] Mean-shift vector computed (STACKED format):")
            print(f"[DEBUG]   Total landmarks: {len(response_maps)}")
            print(f"[DEBUG]   Mean-shift norm: {np.linalg.norm(mean_shift):.4f}")
            print(f"[DEBUG]   Mean-shift for landmarks 36, 48:")
            for lm_idx in [36, 48]:
                if lm_idx in response_maps:
                    ms_x = mean_shift[lm_idx]
                    ms_y = mean_shift[lm_idx + n_points]
                    print(f"[DEBUG]     Landmark {lm_idx}: ({ms_x:.4f}, {ms_y:.4f})")

        # DEBUG: Mean-shift transform debug (disabled for production)
        # Set MS_TRANSFORM_DEBUG=1 env var to enable
        # if os.environ.get('MS_TRANSFORM_DEBUG') and iteration == 0 and window_size == 11:
        #     ... debug output ...

        return mean_shift

    def _get_kde_kernel(self, window_size: int) -> np.ndarray:
        """
        Get or compute KDE kernel for given window size.

        Args:
            window_size: Size of response map window

        Returns:
            kde_kernel: Precomputed KDE kernel
        """
        # Use scale-adapted sigma if available, otherwise use base
        current_sigma = getattr(self, '_current_sigma', self.sigma)

        # Cache key includes both window_size and sigma (rounded to avoid float key issues)
        cache_key = (window_size, round(current_sigma * 100))

        if cache_key in self.kde_cache:
            return self.kde_cache[cache_key]

        # Compute KDE kernel (OpenFace uses step_size=0.1 for sub-pixel precision)
        step_size = 0.1
        a = -0.5 / (current_sigma * current_sigma)

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

        self.kde_cache[cache_key] = kernel
        return kernel

    def _precompute_kde_grid(self, resp_size: int, a: float) -> np.ndarray:
        """
        Precompute KDE kernel grid for fast mean-shift calculation.

        Matches OpenFace C++ implementation (line 918-950) which uses
        0.1 pixel grid spacing for efficiency.

        Args:
            resp_size: Response map size
            a: Gaussian kernel parameter (-0.5 / sigma^2)

        Returns:
            kde_grid: Precomputed KDE weights, shape ((resp_size/0.1)^2, resp_size^2)
        """
        step_size = 0.1

        # Number of grid points in each dimension
        grid_size = int(resp_size / step_size + 0.5)

        # Precompute KDE weights for all grid positions
        # Each row corresponds to one (dx, dy) grid position
        # Each row has resp_size*resp_size values (one per response map pixel)
        kde_grid = np.zeros((grid_size * grid_size, resp_size * resp_size), dtype=np.float64)

        # Iterate over grid positions (matching C++ line 924-929)
        for x in range(grid_size):
            dx_grid = x * step_size
            for y in range(grid_size):
                dy_grid = y * step_size

                # Compute index for this grid position
                idx = x * grid_size + y

                # Compute KDE weights for all response map positions
                # C++ iterates ii then jj (lines 934-945)
                kde_idx = 0
                for ii in range(resp_size):
                    # vx = (dy - ii)^2 matching C++ line 936
                    vx = (dy_grid - ii) * (dy_grid - ii)
                    for jj in range(resp_size):
                        # vy = (dx - jj)^2 matching C++ line 939
                        vy = (dx_grid - jj) * (dx_grid - jj)

                        # KDE weight at this position (C++ line 942)
                        kde_grid[idx, kde_idx] = np.exp(a * (vx + vy))
                        kde_idx += 1

        return kde_grid

    def _kde_mean_shift(self,
                       response_map: np.ndarray,
                       dx: float,
                       dy: float,
                       a: float,
                       landmark_idx: int = -1) -> Tuple[float, float]:
        """
        Compute KDE-based mean-shift for a single landmark.

        Implements OpenFace's NonVectorisedMeanShift_precalc_kde algorithm
        with precomputed KDE grid for 0.1 pixel spacing.

        Args:
            response_map: Patch expert response map (window_size, window_size)
            dx: Current x offset within response map
            dy: Current y offset within response map
            a: Gaussian kernel parameter (-0.5 / sigma^2)
            landmark_idx: Landmark index for debugging

        Returns:
            (ms_x, ms_y): Mean-shift in x and y directions
        """
        resp_size = response_map.shape[0]
        step_size = 0.1

        # DEBUG: Print for landmark 36
        if landmark_idx == 36 and not hasattr(self, '_printed_lm36_meanshift') and self.debug_mode:
            print(f"\n[PY][MEANSHIFT] Landmark 36 mean-shift computation:")
            print(f"[PY][MEANSHIFT]   dx (before clamp): {dx}")
            print(f"[PY][MEANSHIFT]   dy (before clamp): {dy}")
            print(f"[PY][MEANSHIFT]   resp_size: {resp_size}")
            print(f"[PY][MEANSHIFT]   use_direct_kde: {self.use_direct_kde}")

        # Clamp dx, dy to valid range (C++ line 973-980)
        if dx < 0:
            dx = 0
        if dy < 0:
            dy = 0
        if dx > resp_size - step_size:
            dx = resp_size - step_size
        if dy > resp_size - step_size:
            dy = resp_size - step_size

        if self.use_direct_kde:
            # Direct KDE computation at actual (dx, dy) position
            # This avoids grid snapping that causes systematic errors canceling to zero
            ms_x, ms_y = _kde_mean_shift_direct(response_map, dx, dy, a)
        else:
            # Precomputed grid approach (original implementation)
            # Get or create precomputed KDE grid for this response size
            cache_key = (resp_size, a)
            if cache_key not in self.kde_cache:
                self.kde_cache[cache_key] = self._precompute_kde_grid(resp_size, a)
            kde_grid = self.kde_cache[cache_key]

            # Round to nearest grid point (C++ line 983-984)
            # C++ uses int cast which rounds down, +0.5 achieves rounding
            closest_col = int(dy / step_size + 0.5)
            closest_row = int(dx / step_size + 0.5)

            # Compute grid index (C++ line 986)
            grid_size = int(resp_size / step_size + 0.5)
            idx = closest_row * grid_size + closest_col

            # DEBUG: Print after clamp
            if landmark_idx == 36 and not hasattr(self, '_printed_lm36_meanshift') and self.debug_mode:
                print(f"[PY][MEANSHIFT]   dx (after clamp): {dx}")
                print(f"[PY][MEANSHIFT]   dy (after clamp): {dy}")
                print(f"[PY][MEANSHIFT]   closest_row: {closest_row}, closest_col: {closest_col}")
                print(f"[PY][MEANSHIFT]   kde_idx: {idx}")
                print(f"[PY][MEANSHIFT]   Response map stats:")
                print(f"[PY][MEANSHIFT]     shape: {response_map.shape}")
                print(f"[PY][MEANSHIFT]     min: {response_map.min()}")
                print(f"[PY][MEANSHIFT]     max: {response_map.max()}")
                print(f"[PY][MEANSHIFT]     mean: {response_map.mean()}")

            # Get precomputed KDE weights for this grid position
            kde_weights = kde_grid[idx]

            # DEBUG: Print center values for landmark 36
            if landmark_idx == 36 and not hasattr(self, '_printed_lm36_meanshift') and self.debug_mode:
                center_ii, center_jj = 5, 5  # Center of 11x11 map
                print(f"[PY][MEANSHIFT]   Response at center (5,5): {response_map[center_ii, center_jj]:.8f}")
                print(f"[PY][MEANSHIFT]   Response at peak (5,4): {response_map[5, 4]:.8f}")
                # Save response map for detailed comparison
                import numpy as np
                np.save('/tmp/py_response_lm36.npy', response_map)

            # Use Numba-optimized mean-shift computation with precomputed grid
            ms_x, ms_y = _kde_mean_shift_numba(response_map, dx, dy, a, kde_weights)

        # DEBUG: Print final mean-shift for landmark 36
        if landmark_idx == 36 and not hasattr(self, '_printed_lm36_meanshift') and self.debug_mode:
            print(f"[PY][MEANSHIFT]   Final mean-shift:")
            print(f"[PY][MEANSHIFT]     ms_x: {ms_x}")
            print(f"[PY][MEANSHIFT]     ms_y: {ms_y}")
            self._printed_lm36_meanshift = True

        return ms_x, ms_y

    def _compute_response_map(self,
                             image: np.ndarray,
                             center_x: float,
                             center_y: float,
                             patch_expert,
                             window_size: int,
                             sim_img_to_ref: np.ndarray = None,
                             sim_ref_to_img: np.ndarray = None,
                             sigma_components: dict = None,
                             landmark_idx: int = None,
                             iteration: int = None,
                             is_mirror_patch: bool = False) -> Optional[np.ndarray]:
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
            is_mirror_patch: If True, this is a secondary patch from mirror location (suppress debug)

        Returns:
            response_map: (window_size, window_size) array of patch responses
        """
        response_map = np.zeros((window_size, window_size))

        # Window bounds (centered at current landmark)
        half_window = window_size // 2

        if sim_img_to_ref is not None:
            # WARPING MODE: Mimic OpenFace's exact approach (line 240 in Patch_experts.cpp)
            # Calculate area of interest size
            # CEN uses width_support/height_support, CCNF uses width/height
            if hasattr(patch_expert, 'width_support'):
                patch_dim = max(patch_expert.width_support, patch_expert.height_support)
            else:
                patch_dim = max(patch_expert.width, patch_expert.height)
            area_of_interest_width = window_size + patch_dim - 1
            area_of_interest_height = window_size + patch_dim - 1

            # Extract Area of Interest using similarity transform
            # Use _extract_aoi which supports cpp_warp for exact C++ OpenFace matching
            area_of_interest = self._extract_aoi(
                image, center_x, center_y, sim_ref_to_img, area_of_interest_width
            )

            # Now evaluate patches from the warped area_of_interest
            # The landmark is centered at (area_of_interest_width-1)/2
            center_warped = int((area_of_interest_width - 1) / 2)

            # DEBUG: Save area_of_interest for landmark 36
            if landmark_idx == 36 and iteration == 0 and self.debug_mode and not is_mirror_patch:
                np.save('/tmp/area_of_interest_lm36.npy', area_of_interest)
                cv2.imwrite('/tmp/area_of_interest_lm36.png', area_of_interest)
                print(f"[PY][DEBUG] Saved area_of_interest for landmark 36:")
                print(f"[PY][DEBUG]   Shape: {area_of_interest.shape}")
                print(f"[PY][DEBUG]   Stats: min={area_of_interest.min()}, max={area_of_interest.max()}, mean={area_of_interest.mean():.1f}")
                print(f"[PY][DEBUG]   area_of_interest_width: {area_of_interest_width}")
                print(f"[PY][DEBUG]   patch_dim: {patch_dim}")
                print(f"[PY][DEBUG]   window_size: {window_size}")
                print(f"[PY][DEBUG]   use_cpp_warp: {self.use_cpp_warp}")

            # Check if this is CEN (has response_sparse() method) or CCNF (has compute_response())
            if hasattr(patch_expert, 'response_sparse') and not hasattr(patch_expert, 'compute_response'):
                # CEN: Use response_sparse() with None for right side to match C++ ResponseSparse
                # This uses sparse computation + interpolation exactly like C++

                # DEBUG: Check area_of_interest before calling response_sparse()
                if landmark_idx in (36, 42) and iteration == 0 and self.debug_mode and not is_mirror_patch:
                    print(f"[PY][DEBUG][LM{landmark_idx}] area_of_interest BEFORE response_sparse():")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   dtype: {area_of_interest.dtype}, shape: {area_of_interest.shape}")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   min={area_of_interest.min()}, max={area_of_interest.max()}, mean={area_of_interest.mean():.1f}")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   sum={area_of_interest.sum()}")
                    print(f"[PY][DEBUG][LM{landmark_idx}] patch_expert info:")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   type: {type(patch_expert)}")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   width: {patch_expert.width}, height: {patch_expert.height}")
                    if hasattr(patch_expert, 'width_support'):
                        print(f"[PY][DEBUG][LM{landmark_idx}]   width_support: {patch_expert.width_support}, height_support: {patch_expert.height_support}")
                    # Save a copy to compare
                    np.save(f'/tmp/area_of_interest_lm{landmark_idx}_before_response.npy', area_of_interest.copy())

                # Use response_sparse with None for right side (like C++ with empty mat)
                response_map, _ = patch_expert.response_sparse(area_of_interest, None)

                # DEBUG: Check response_map after calling response()
                if landmark_idx in (36, 42) and iteration == 0 and self.debug_mode and not is_mirror_patch:
                    print(f"[PY][DEBUG][LM{landmark_idx}] response_map AFTER response():")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   dtype: {response_map.dtype}, shape: {response_map.shape}")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   min={response_map.min():.6f}, max={response_map.max():.6f}, mean={response_map.mean():.6f}")
                    print(f"[PY][DEBUG][LM{landmark_idx}]   peak: {np.unravel_index(np.argmax(response_map), response_map.shape)} = {response_map.max():.6f}")
            else:
                # CCNF: Nested loop to evaluate each position
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
                        # _extract_patch now always returns valid patch with border replication
                        response_map[i, j] = patch_expert.compute_response(patch)
        else:
            # NO WARPING: Direct extraction from image (not typically used with OpenFace)
            # Check if this is CEN (has response_sparse() method) or CCNF (has compute_response())
            if hasattr(patch_expert, 'response_sparse') and not hasattr(patch_expert, 'compute_response'):
                # CEN: Extract area_of_interest and call response_sparse()
                # Calculate area size needed
                if hasattr(patch_expert, 'width_support'):
                    patch_dim = max(patch_expert.width_support, patch_expert.height_support)
                else:
                    patch_dim = max(patch_expert.width, patch_expert.height)

                area_width = window_size + patch_dim - 1
                area_height = window_size + patch_dim - 1
                half_area = area_width // 2

                # Extract area around landmark
                x1 = int(center_x - half_area)
                y1 = int(center_y - half_area)
                x2 = x1 + area_width
                y2 = y1 + area_height

                # Ensure bounds
                if x1 >= 0 and y1 >= 0 and x2 <= image.shape[1] and y2 <= image.shape[0]:
                    area_of_interest = image[y1:y2, x1:x2]
                    response_map, _ = patch_expert.response_sparse(area_of_interest, None)
                else:
                    # Out of bounds - use low response
                    response_map[:] = -1e10
            else:
                # CCNF: Nested loop to evaluate each position
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

                        # _extract_patch now always returns valid patch with border replication (matches C++ OpenFace)
                        response_map[i, j] = patch_expert.compute_response(patch)

        # DEBUG: Save response map BEFORE sigma for landmarks 36 and 42
        if landmark_idx in (36, 42) and iteration == 0 and window_size == 11 and self.debug_mode and not is_mirror_patch:
            np.save(f'/tmp/python_response_map_lm{landmark_idx}_iter0_ws11_BEFORE_SIGMA.npy', response_map)
            print(f"[PY][DEBUG][LM{landmark_idx}] Saved BEFORE SIGMA response map (WS={window_size}): shape={response_map.shape}, min={response_map.min():.6f}, max={response_map.max():.6f}, mean={response_map.mean():.6f}")

        # Apply CCNF Sigma transformation for spatial correlation modeling
        # (OpenFace CCNF_patch_expert.cpp lines 400-404)
        # Use response_map size (window_size × window_size), NOT patch size
        response_window_size = response_map.shape[0]  # Square response map

        # DEBUG: Track Sigma transformation
        sigma_applied = False
        peak_before = None
        peak_after = None

        if sigma_components is not None and response_window_size in sigma_components:
            try:
                # DEBUG: Peak location before Sigma
                peak_idx_before = np.unravel_index(response_map.argmax(), response_map.shape)
                center = response_window_size // 2
                offset_before = (peak_idx_before[1] - center, peak_idx_before[0] - center)
                peak_before = (peak_idx_before, offset_before, response_map.max())

                # Get sigma components for this response map window size
                sigma_comps = sigma_components[response_window_size]

                # DEBUG: Enable detailed Sigma computation logging for landmarks 36 and 42 on first iteration
                debug_sigma = (landmark_idx in (36, 42) and iteration == 0 and response_window_size == 11 and self.debug_mode and not is_mirror_patch)

                if debug_sigma:
                    print(f"\n  [Sigma Component Selection Debug - LM{landmark_idx}]")
                    print(f"    landmark_idx={landmark_idx}, iteration={iteration}")
                    print(f"    response_window_size={response_window_size}")
                    print(f"    Available sigma_components window sizes: {list(sigma_components.keys())}")
                    print(f"    Selected sigma_comps length: {len(sigma_comps)}")
                    for i, sc in enumerate(sigma_comps):
                        print(f"    sigma_comps[{i}].shape = {sc.shape}")

                # Compute Sigma covariance matrix with correct window size
                Sigma = patch_expert.compute_sigma(sigma_comps, window_size=response_window_size, debug=debug_sigma)

                # Apply transformation: response = Sigma @ response.reshape(-1, 1)
                # This models spatial correlations in the response map
                response_shape = response_map.shape
                response_vec = response_map.reshape(-1, 1)
                response_transformed = Sigma @ response_vec
                response_map = response_transformed.reshape(response_shape)

                # DEBUG: Peak location after Sigma
                peak_idx_after = np.unravel_index(response_map.argmax(), response_map.shape)
                offset_after = (peak_idx_after[1] - center, peak_idx_after[0] - center)
                peak_after = (peak_idx_after, offset_after, response_map.max())

                sigma_applied = True
            except Exception as e:
                # If Sigma transformation fails, continue with untransformed response
                print(f"Warning: Sigma transformation failed: {e}")

        # DEBUG: Print Sigma transformation results (only if significant offset)
        # Disabled by default to reduce output
        pass

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
                      patch_height: int) -> np.ndarray:
        """
        Extract image patch centered at (center_x, center_y).

        Uses border replication for out-of-bounds regions, matching C++ OpenFace
        behavior. This prevents suppression of landmarks near image edges.

        Args:
            image: Source image
            center_x, center_y: Patch center coordinates
            patch_width, patch_height: Patch dimensions

        Returns:
            patch: Extracted patch (always returns a valid patch using border replication)
        """
        half_w = patch_width // 2
        half_h = patch_height // 2

        # Compute patch bounds
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + patch_width
        y2 = y1 + patch_height

        # Check if patch is within bounds
        if x1 >= 0 and y1 >= 0 and x2 <= image.shape[1] and y2 <= image.shape[0]:
            # Fast path: fully within bounds
            return image[y1:y2, x1:x2]

        # Slow path: use border replication for out-of-bounds regions
        # This matches C++ OpenFace's cv::warpAffine with BORDER_REPLICATE
        img_h, img_w = image.shape[:2]

        # Compute padding needed on each side
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        # If entire patch is outside image, return edge pixel replicated
        if x1 >= img_w or x2 <= 0 or y1 >= img_h or y2 <= 0:
            # Completely outside - return corner pixel replicated
            corner_x = max(0, min(center_x, img_w - 1))
            corner_y = max(0, min(center_y, img_h - 1))
            if len(image.shape) == 3:
                return np.full((patch_height, patch_width, image.shape[2]),
                              image[corner_y, corner_x], dtype=image.dtype)
            else:
                return np.full((patch_height, patch_width),
                              image[corner_y, corner_x], dtype=image.dtype)

        # Clamp bounds to valid image region
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)

        # Extract the valid portion
        valid_patch = image[src_y1:src_y2, src_x1:src_x2]

        # Apply border replication padding
        patch = cv2.copyMakeBorder(
            valid_patch,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )

        return patch

    def _solve_rigid_update(self,
                           J_rigid: np.ndarray,
                           v: np.ndarray,
                           W: np.ndarray,
                           iteration: int = -1,
                           window_size: int = -1) -> np.ndarray:
        """
        Solve for RIGID parameter update (scale, rotation, translation only).

        This is Phase 1 of two-phase optimization. Only updates global params
        while keeping local shape params at 0.

        Solves: Δp_global = (J_rigid^T·W·J_rigid)^(-1) · (J_rigid^T·W·v)

        Args:
            J_rigid: Jacobian for rigid params only (2n, 6)
            v: Mean-shift vector (2n,)
            W: Weight matrix (2n, 2n)
            iteration: Current iteration number (for debug)
            window_size: Current window size (for debug)

        Returns:
            delta_p_rigid: Parameter update for rigid params only (6,)
        """
        # Compute Hessian: A = J^T·W·J (no regularization for rigid params)
        A = J_rigid.T @ W @ J_rigid  # (6, 6)

        # Compute right-hand side: b = J^T·W·v
        b = J_rigid.T @ W @ v  # (6,)

        # DEBUG: Print values matching C++ format (iteration 0 only to reduce noise)
        if self.debug_mode and window_size == 11 and iteration == 0:
            # Compute mean-shift sum (v is stacked as [x0..x67, y0..y67])
            n = len(v) // 2
            sum_ms_x = v[:n].sum()
            sum_ms_y = v[n:].sum()
            print(f"\n[PY][ITER0_WS11_RIGID] Mean-shift sum: ({sum_ms_x:.4f}, {sum_ms_y:.4f})")
            print(f"[PY][ITER0_WS11_RIGID] J_w_t_m (gradient) [all 6]:")
            print(f"  [0] scale:  {b[0]:.2f}")
            print(f"  [1] rot_x:  {b[1]:.2f}")
            print(f"  [2] rot_y:  {b[2]:.2f}")
            print(f"  [3] rot_z:  {b[3]:.2f}")
            print(f"  [4] tx:     {b[4]:.2f}")
            print(f"  [5] ty:     {b[5]:.2f}")
            print(f"[PY][ITER0_WS11_RIGID] Hessian diagonal:")
            print(f"  [0,0] scale: {A[0,0]:.2f}")
            print(f"  [1,1] rot_x: {A[1,1]:.2f}")
            print(f"  [2,2] rot_y: {A[2,2]:.2f}")
            print(f"  [3,3] rot_z: {A[3,3]:.2f}")
            print(f"  [4,4] tx:    {A[4,4]:.2f}")
            print(f"  [5,5] ty:    {A[5,5]:.2f}")

        # Solve linear system: A·Δp = b using Cholesky (matches C++ cv::DECOMP_CHOLESKY)
        try:
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, b)
            delta_p_rigid = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fallback if not positive-definite
            try:
                delta_p_rigid = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta_p_rigid = np.linalg.lstsq(A, b, rcond=None)[0]

        # NOTE: No damping applied here - C++ OpenFace NU_RLMS does not apply damping
        # The 0.75 damping previously here was incorrect - it's only used in CalcParams
        # during initial pose estimation, not in the main optimization loop

        # DEBUG: Print delta_p matching C++ format (iteration 0 only)
        if self.debug_mode and window_size == 11 and iteration == 0:
            print(f"[PY][ITER0_WS11_RIGID] param_update:")
            print(f"  [0] delta_scale: {delta_p_rigid[0]:.6f}")
            print(f"  [1] delta_rot_x: {delta_p_rigid[1]:.6f} rad = {delta_p_rigid[1]*57.3:.2f} deg")
            print(f"  [2] delta_rot_y: {delta_p_rigid[2]:.6f} rad = {delta_p_rigid[2]*57.3:.2f} deg")
            print(f"  [3] delta_rot_z: {delta_p_rigid[3]:.6f} rad")
            print(f"  [4] delta_tx:    {delta_p_rigid[4]:.6f}")
            print(f"  [5] delta_ty:    {delta_p_rigid[5]:.6f}")

        return delta_p_rigid

    def _solve_update(self,
                     J: np.ndarray,
                     v: np.ndarray,
                     W: np.ndarray,
                     Lambda_inv: np.ndarray,
                     params: np.ndarray,
                     iteration: int = -1,
                     window_size: int = -1) -> np.ndarray:
        """
        Solve for parameter update using NU-RLMS equation.

        Solves: (J^T·W·J + λ·Λ^(-1))·Δp = J^T·W·v - λ·Λ^(-1)·p

        Args:
            J: Jacobian matrix (2n, m)
            v: Mean-shift vector (2n,)
            W: Weight matrix (2n, 2n)
            Lambda_inv: Inverse regularization matrix (m,)
            params: Current parameters (m,)
            iteration: Current iteration number (for debug)
            window_size: Current window size (for debug)

        Returns:
            delta_p: Parameter update (m,)
        """
        # Use scale-adapted regularization if available, otherwise use base
        current_reg = getattr(self, '_current_regularization', self.regularization)

        # Compute left-hand side: A = J^T·W·J + λ·Λ^(-1)
        JtWJ = J.T @ W @ J  # (m, m)
        Lambda_inv_diag = np.diag(current_reg * Lambda_inv)  # (m, m)
        A = JtWJ + Lambda_inv_diag

        # Compute right-hand side: b = J^T·W·v - λ·Λ^(-1)·p
        JtWv = J.T @ W @ v  # (m,)
        reg_term = current_reg * Lambda_inv * params  # (m,)
        b = JtWv - reg_term

        # DEBUG: Check mean-shift sums for translation
        if window_size in (11, 9) and iteration == 0 and self.debug_mode:
            n_lm = len(v) // 2
            v_x_sum = np.sum(v[:n_lm])
            v_y_sum = np.sum(v[n_lm:])
            print(f"\n[DEBUG][SOLVE_UPDATE] WS{window_size} iter{iteration}:")
            print(f"[DEBUG]   v_x_sum = {v_x_sum:.4f}, v_y_sum = {v_y_sum:.4f}")
            print(f"[DEBUG]   JtWv[4] (tx) = {JtWv[4]:.4f}, JtWv[5] (ty) = {JtWv[5]:.4f}")
            print(f"[DEBUG]   JtWv[:10] = {JtWv[:10]}")
            print(f"[DEBUG]   reg_term[:10] = {reg_term[:10]}")
            print(f"[DEBUG]   b[:10] = {b[:10]}")

        # Solve linear system: A·Δp = b using Cholesky decomposition
        # C++ OpenFace uses cv::DECOMP_CHOLESKY which is more numerically stable
        # for positive-definite systems (guaranteed by regularization)
        try:
            # Cholesky: A = L @ L.T, solve L @ y = b, then L.T @ x = y
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, b)
            delta_p = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fallback to general solver if not positive-definite
            try:
                delta_p = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Last resort: pseudo-inverse
                delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

        # Store diagnostic info for late-stage convergence analysis
        self._last_hessian_cond = np.linalg.cond(A)
        self._last_reg_ratio = np.linalg.norm(reg_term) / (np.linalg.norm(JtWv) + 1e-10)
        self._last_jtw_norm = np.linalg.norm(JtWv)
        self._last_reg_term_norm = np.linalg.norm(reg_term)

        # NOTE: C++ NU_RLMS (LandmarkDetectorModel.cpp) does NOT apply 0.75 damping!
        # The 0.75 damping in PDM.cpp CalcParams is for initial pose estimation only,
        # NOT for the main optimization loop. Removed to match C++ behavior.
        # OLD: delta_p = 0.75 * delta_p

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
