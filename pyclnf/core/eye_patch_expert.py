"""
Eye CCNF Patch Expert - Response map computation for eye landmark refinement

This implements the CCNF patch experts for 28-point eye models from OpenFace.
Eye models have scales 1.00 and 1.50 (different from face model scales).
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import cv2


def align_shapes_with_scale(source_shape: np.ndarray, destination_shape: np.ndarray) -> np.ndarray:
    """
    Compute optimal similarity transform to align source shape to destination.

    This matches C++ AlignShapesWithScale which uses Kabsch algorithm for rotation
    and computes scale separately. Returns 2x2 matrix A = s * R.

    Args:
        source_shape: Source shape as (n*2,) array [x0,y0,x1,y1,...] or (n, 2)
        destination_shape: Destination shape in same format

    Returns:
        2x2 similarity transform matrix from source to destination
    """
    # Reshape to (n, 2) if needed
    if source_shape.ndim == 1:
        n = len(source_shape) // 2
        src = source_shape.reshape(n, 2)
        dst = destination_shape.reshape(n, 2)
    else:
        src = source_shape.copy()
        dst = destination_shape.copy()
        n = len(src)

    # Mean normalize both shapes
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_mean_normed = src - src_mean
    dst_mean_normed = dst - dst_mean

    # Find scaling factor of each
    s_src = np.sqrt(np.sum(src_mean_normed**2) / n)
    s_dst = np.sqrt(np.sum(dst_mean_normed**2) / n)

    if s_src < 1e-10:
        return np.eye(2, dtype=np.float32)

    # Normalize to unit scale
    src_normed = src_mean_normed / s_src
    dst_normed = dst_mean_normed / s_dst

    # Scale ratio
    s = s_dst / s_src

    # Get rotation using Kabsch algorithm (SVD)
    H = src_normed.T @ dst_normed
    U, _, Vt = np.linalg.svd(H)

    # Ensure proper rotation (no reflection)
    d = np.linalg.det(Vt.T @ U.T)
    corr = np.eye(2)
    if d < 0:
        corr[1, 1] = -1

    R = Vt.T @ corr @ U.T

    # Final transform = scale * rotation
    A = s * R

    return A.astype(np.float32)


class EyeCCNFPatchExpert:
    """CCNF patch expert for a single eye landmark at a specific scale."""

    def __init__(self, patch_dir: str):
        """
        Load Eye CCNF patch expert from exported NumPy files.

        Args:
            patch_dir: Directory containing metadata.npz and neuron_*.npz files
        """
        self.patch_dir = Path(patch_dir)

        # Load patch metadata
        meta = np.load(self.patch_dir / 'metadata.npz', allow_pickle=True)
        self.width = int(meta['width'])
        self.height = int(meta['height'])
        self.betas = meta['betas']
        self.patch_confidence = float(meta['patch_confidence'])

        # Count and load neurons
        neuron_files = sorted(self.patch_dir.glob('neuron_*.npz'))
        self.num_neurons = len(neuron_files)

        self.neurons = []
        for neuron_file in neuron_files:
            neuron_data = np.load(neuron_file, allow_pickle=True)
            neuron = {
                'type': int(neuron_data['neuron_type']),
                'weights': neuron_data['weights'],
                'bias': float(neuron_data['bias']),
                'alpha': float(neuron_data['alpha']),
                'norm_weights': float(neuron_data['norm_weights'])
            }
            self.neurons.append(neuron)

    def compute_response(self, image_patch: np.ndarray, debug_file: str = None) -> float:
        """
        Compute response for this patch expert.

        Args:
            image_patch: Grayscale image patch, shape (height, width)
            debug_file: Optional file path to write debug output

        Returns:
            response: Scalar confidence value (normalized like C++)
        """
        if image_patch.shape != (self.height, self.width):
            image_patch = cv2.resize(image_patch, (self.width, self.height))

        # Extract features (raw intensity like C++ - no /255 normalization)
        features = image_patch.astype(np.float32)

        # Sum neuron responses and alphas
        total_response = 0.0
        sum_alphas = 0.0
        neuron_details = []

        for i, neuron in enumerate(self.neurons):
            alpha = neuron['alpha']
            if abs(alpha) < 1e-4:
                continue
            sum_alphas += alpha
            resp, sigmoid_in = self._compute_neuron_response(features, neuron, return_details=True)
            total_response += resp
            neuron_details.append((i, alpha, sigmoid_in, resp))

        # Apply Sigma normalization (when no sigma_components, Sigma = I / (2 * sum_alphas))
        # This scales response to [0, 1] range like C++
        normalized_response = total_response
        if sum_alphas > 1e-10:
            normalized_response = total_response / (2.0 * sum_alphas)

        # Debug output
        if debug_file:
            with open(debug_file, 'w') as f:
                f.write("=== Python CCNF NEURON DEBUG ===\n\n")
                f.write(f"patch_size: {self.height}x{self.width}\n")
                f.write(f"num_neurons: {self.num_neurons}\n\n")

                # Normalized input (ALL values to match C++)
                features_flat = features.flatten('F')  # Column-major like C++
                feature_mean = np.mean(features_flat)
                features_centered = features_flat - feature_mean
                feature_norm = np.linalg.norm(features_centered)
                normalized_features = features_centered / feature_norm if feature_norm > 1e-10 else features_centered

                # Raw patch values (first 5x5)
                f.write("Raw area_of_interest (first 5x5):\n")
                for row in range(min(5, features.shape[0])):
                    f.write("  ")
                    for col in range(min(5, features.shape[1])):
                        f.write(f"{features[row, col]:.1f} ")
                    f.write("\n")

                # ALL normalized input values (122 = 1 bias + 121 features)
                f.write(f"\nNormalized input for center pixel (all {len(normalized_features)+1} values):\n")
                f.write(f"  [0]: 1.00000000\n")  # Bias term
                for j in range(len(normalized_features)):
                    f.write(f"  [{j+1}]: {normalized_features[j]:.8f}\n")

                # Neuron responses with norm_weights
                f.write("\nNeuron responses for center pixel:\n")
                for i, alpha, sigmoid_in, resp in neuron_details:
                    neuron = self.neurons[i]
                    f.write(f"  Neuron {i}: alpha={alpha:.6f}, bias={neuron['bias']:.6f}, norm_w={neuron['norm_weights']:.6f}, sigmoid_input={sigmoid_in:.6f}, sigmoid_output={resp:.6f}\n")

                f.write(f"\nTotal response (before sigma): {total_response:.6f}\n")
                f.write(f"Sum of alphas: {sum_alphas:.6f}\n")
                f.write(f"Response / (2*sum_alphas): {normalized_response:.6f}\n")

                # Weight matrix first row (ALL values)
                if self.neurons:
                    neuron = self.neurons[0]
                    weights_flat = neuron['weights'].flatten('F') * neuron['norm_weights']
                    f.write(f"\nWeight matrix row 0 (all {len(weights_flat)+1} values):\n")
                    f.write(f"  [0]: {neuron['bias']:.8f}\n")  # Bias
                    for j in range(len(weights_flat)):
                        f.write(f"  [{j+1}]: {weights_flat[j]:.8f}\n")

                    # Raw weights before scaling
                    raw_weights = neuron['weights'].flatten('F')
                    f.write(f"\nNeuron 0 raw weights (first 10, before norm_weights scaling):\n")
                    for j in range(min(10, len(raw_weights))):
                        f.write(f"  [{j}]: {raw_weights[j]:.8f}\n")

        return float(normalized_response)

    def _compute_neuron_response(self, features: np.ndarray, neuron: dict, return_details: bool = False):
        """Compute response for a single neuron matching C++ ResponseOpenBlas."""
        weights = neuron['weights']

        if features.shape != weights.shape:
            features = cv2.resize(features, (weights.shape[1], weights.shape[0]))

        # C++ style contrast normalization (only center and normalize features, not weights)
        # C++ uses column-major ordering: colIdx = xx*height + yy
        features_flat = features.flatten('F')  # Column-major to match C++
        feature_mean = np.mean(features_flat)
        features_centered = features_flat - feature_mean
        feature_norm = np.linalg.norm(features_centered)

        if feature_norm > 1e-10:
            normalized_features = features_centered / feature_norm
        else:
            normalized_features = features_centered

        # C++ scales weights by norm_weights and includes bias
        # weight_matrix row = [bias, w0*norm_weights, w1*norm_weights, ...]
        # normalized_input col = [1, f0, f1, ...]
        weights_flat = weights.flatten('F') * neuron['norm_weights']  # Column-major

        # dot product: bias*1 + sum(scaled_weights * normalized_features)
        sigmoid_input = neuron['bias'] + np.dot(weights_flat, normalized_features)

        # Sigmoid: 2*alpha / (1 + exp(-x))
        response = (2.0 * neuron['alpha']) / (1.0 + np.exp(-np.clip(sigmoid_input, -500, 500)))

        if return_details:
            return response, sigmoid_input
        return response

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def get_info(self) -> dict:
        """Get patch expert information."""
        return {
            'width': self.width,
            'height': self.height,
            'num_neurons': self.num_neurons,
            'patch_confidence': self.patch_confidence,
        }


class EyeCCNFModel:
    """
    Complete CCNF model for 28-point eye landmarks.

    Manages patch experts for:
    - Two scales: 1.00 and 1.50
    - 28 landmarks per eye
    """

    def __init__(self, model_dir: str, side: str = 'left'):
        """
        Load Eye CCNF model from exported NumPy files.

        Args:
            model_dir: Base directory containing exported_eye_ccnf_{side}/ folders
            side: 'left' or 'right' eye
        """
        self.model_dir = Path(model_dir)
        self.side = side
        self.scales = [1.00, 1.50]  # Eye model scales
        self.n_landmarks = 28

        # Load scale models
        self.scale_models = {}
        eye_ccnf_dir = self.model_dir / f'exported_eye_ccnf_{side}'

        for scale in self.scales:
            scale_dir = eye_ccnf_dir / f'scale_{scale:.2f}'
            if scale_dir.exists():
                self.scale_models[scale] = self._load_scale_model(scale_dir)
            else:
                print(f"Warning: Eye CCNF scale {scale} not found at {scale_dir}")

    def _load_scale_model(self, scale_dir: Path) -> dict:
        """
        Load all patch experts for one scale.

        Args:
            scale_dir: Directory for one scale

        Returns:
            Dictionary mapping landmark_idx -> EyeCCNFPatchExpert
        """
        patches = {}

        # Eye CCNF exports use the same structure as main CCNF: view_00/patch_XX/
        # Check for patches in view_00 directory
        view_dir = scale_dir / 'view_00'
        if not view_dir.exists():
            # Try loading directly from scale dir (alternative structure)
            view_dir = scale_dir

        for landmark_idx in range(self.n_landmarks):
            patch_dir = view_dir / f'patch_{landmark_idx:02d}'
            if patch_dir.exists() and (patch_dir / 'metadata.npz').exists():
                try:
                    patches[landmark_idx] = EyeCCNFPatchExpert(str(patch_dir))
                except Exception as e:
                    print(f"Warning: Failed to load eye patch {landmark_idx}: {e}")

        return patches

    def get_patch_expert(self, scale: float, landmark_idx: int) -> Optional[EyeCCNFPatchExpert]:
        """
        Get patch expert for specific scale and landmark.

        Args:
            scale: Patch scale (1.00 or 1.50)
            landmark_idx: Eye landmark index (0-27)

        Returns:
            EyeCCNFPatchExpert or None
        """
        if scale not in self.scale_models:
            return None
        return self.scale_models[scale].get(landmark_idx)

    def get_all_patch_experts(self, scale: float) -> dict:
        """
        Get all patch experts for a specific scale.

        Args:
            scale: Patch scale (1.00 or 1.50)

        Returns:
            Dictionary mapping landmark_idx -> EyeCCNFPatchExpert
        """
        if scale not in self.scale_models:
            return {}
        return self.scale_models[scale]

    def get_info(self) -> dict:
        """Get Eye CCNF model information."""
        info = {
            'side': self.side,
            'n_landmarks': self.n_landmarks,
            'scales': list(self.scale_models.keys()),
            'patches_per_scale': {}
        }

        for scale, patches in self.scale_models.items():
            info['patches_per_scale'][scale] = len(patches)

        return info


# Mapping from main 68-landmark model to 28-point eye model
# These indices specify which main landmarks correspond to which eye model landmarks
# OpenFace: left eye uses landmarks 36-41, right eye uses 42-47
# In the 28-point eye model, eyelid landmarks are at different indices

# Left eye: main model indices -> eye model indices
LEFT_EYE_MAPPING = {
    36: 8,   # Outer corner
    37: 10,  # Upper outer
    38: 12,  # Upper inner
    39: 14,  # Inner corner
    40: 16,  # Lower inner
    41: 18,  # Lower outer
}

# Right eye: main model indices -> eye model indices
RIGHT_EYE_MAPPING = {
    42: 8,   # Inner corner
    43: 10,  # Upper inner
    44: 12,  # Upper outer
    45: 14,  # Outer corner
    46: 16,  # Lower outer
    47: 18,  # Lower inner
}

def map_main_to_eye_landmarks(main_landmarks: np.ndarray, side: str) -> np.ndarray:
    """
    Map main model landmarks (68) to eye model landmarks (28).

    This extracts the 6 eyelid landmarks from the main model and maps them
    to the corresponding indices in the 28-point eye model.

    Args:
        main_landmarks: Full 68 landmarks, shape (68, 2)
        side: 'left' or 'right'

    Returns:
        eye_landmarks: 28 landmarks for eye model (only 6 are filled in)
    """
    mapping = LEFT_EYE_MAPPING if side == 'left' else RIGHT_EYE_MAPPING

    # Create empty eye landmarks array
    eye_landmarks = np.zeros((28, 2), dtype=np.float32)

    # Fill in mapped landmarks
    for main_idx, eye_idx in mapping.items():
        eye_landmarks[eye_idx] = main_landmarks[main_idx]

    return eye_landmarks


def map_eye_to_main_landmarks(eye_landmarks: np.ndarray, main_landmarks: np.ndarray, side: str) -> np.ndarray:
    """
    Map refined eye model landmarks back to main model landmarks.

    This updates the 6 eyelid landmarks in the main model with refined
    positions from the eye model.

    Args:
        eye_landmarks: Refined 28 eye landmarks, shape (28, 2)
        main_landmarks: Current 68 landmarks, shape (68, 2)
        side: 'left' or 'right'

    Returns:
        updated_landmarks: 68 landmarks with refined eye positions
    """
    mapping = LEFT_EYE_MAPPING if side == 'left' else RIGHT_EYE_MAPPING

    # Copy main landmarks
    updated = main_landmarks.copy()

    # Update with refined eye landmarks
    for main_idx, eye_idx in mapping.items():
        updated[main_idx] = eye_landmarks[eye_idx]

    return updated


class HierarchicalEyeModel:
    """
    Hierarchical eye model for refining eye landmarks.

    Combines EyePDM (shape model) and EyeCCNF (patch experts) for
    accurate eye landmark detection using NU-RLMS optimization.
    """

    def __init__(self, model_dir: str, use_gpu: bool = True, gpu_device: str = 'auto'):
        """
        Load hierarchical eye models for both eyes.

        Args:
            model_dir: Base directory containing exported eye models
            use_gpu: Whether to use GPU for batched computation
            gpu_device: GPU device ('auto', 'mps', 'cuda', 'cpu')
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu

        # Auto-detect GPU device
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

        # Load PDMs for both eyes
        self.pdm = {
            'left': None,
            'right': None
        }

        # Import EyePDM
        from .eye_pdm import EyePDM

        left_pdm_dir = self.model_dir / 'exported_eye_pdm_left'
        right_pdm_dir = self.model_dir / 'exported_eye_pdm_right'

        if left_pdm_dir.exists():
            self.pdm['left'] = EyePDM(str(left_pdm_dir))
        else:
            print(f"Warning: Left eye PDM not found at {left_pdm_dir}")

        if right_pdm_dir.exists():
            self.pdm['right'] = EyePDM(str(right_pdm_dir))
        else:
            print(f"Warning: Right eye PDM not found at {right_pdm_dir}")

        # Load CCNF models for both eyes
        self.ccnf = {
            'left': EyeCCNFModel(str(self.model_dir), 'left'),
            'right': EyeCCNFModel(str(self.model_dir), 'right')
        }

        # Create batched CCNF for GPU acceleration
        self.batched_ccnf = {'left': {}, 'right': {}}
        if self.use_gpu:
            try:
                from .batched_eye_ccnf import BatchedEyeCCNF
                for side in ['left', 'right']:
                    self.batched_ccnf[side] = BatchedEyeCCNF(
                        self.ccnf[side], device=self.gpu_device
                    )
            except ImportError as e:
                print(f"Warning: Could not load BatchedEyeCCNF: {e}")
                self.use_gpu = False

        # Eye refinement parameters (OpenFace defaults for eye model)
        # C++ configures [3, 5, 9] but only uses [3, 5] at runtime because
        # OpenFace only ships scale 1.0 and 1.5 eye CCNF files (no scale 2.0 exists)
        # num_scales = patch_scaling.size() = 2, so loop only processes scales 0 and 1
        self.window_sizes = [3, 5]
        self.sigma = 1.0  # C++ uses sigma=1.0 for eye model (no refinement scaling)
        self.reg_factor = 0.5  # C++ uses reg_factor=0.5 for eye model
        self.max_iterations = 5  # C++ uses 5 iterations per phase (10 per window size)
        # Eye model scales - maps window_sizes[i] → patch_scaling[i]
        # Window 3 uses scale 1.0, Window 5 uses scale 1.5 (matches C++ runtime behavior)
        self.patch_scaling = [1.0, 1.5]

        # Validate configuration
        assert len(self.window_sizes) == len(self.patch_scaling), \
            f"window_sizes ({len(self.window_sizes)}) must match patch_scaling ({len(self.patch_scaling)})"

    def _compute_sim_transforms(self, eye_landmarks: np.ndarray, params: np.ndarray,
                                pdm, side: str, patch_scale: float = 1.0):
        """
        Compute similarity transform from current eye shape to reference shape.

        This matches C++ AlignShapesWithScale in Patch_experts.cpp.

        Args:
            eye_landmarks: Current 28 eye landmarks (28, 2)
            params: Current PDM parameters
            pdm: Eye PDM for this side
            side: 'left' or 'right'
            patch_scale: The patch scale for reference shape (1.0 or 1.5)

        Returns:
            sim_ref_to_img: 2x2 transform from reference to image space
            sim_img_to_ref: 2x2 transform from image to reference space
        """
        # Current shape in image coordinates
        image_shape = eye_landmarks.flatten()  # (56,)

        # Reference shape at patch_scaling
        ref_params = params.copy()
        ref_params[0] = patch_scale  # Use the scale for this window size
        ref_params[1:4] = 0  # no rotation
        ref_params[4:6] = 0  # no translation
        # Keep shape parameters (ref_params[6:]) same as current
        reference_shape = pdm.params_to_landmarks_2d(ref_params).flatten()

        # Compute similarity transform
        sim_img_to_ref = align_shapes_with_scale(image_shape, reference_shape)
        sim_ref_to_img = np.linalg.inv(sim_img_to_ref)

        return sim_ref_to_img, sim_img_to_ref

    def refine_eye_landmarks(self, image: np.ndarray, main_landmarks: np.ndarray,
                            side: str, main_rotation: np.ndarray = None,
                            main_scale: float = 1.0, track_iterations: bool = False) -> tuple:
        """
        Refine eye landmarks using hierarchical eye model with CCNF optimization.

        This runs NU-RLMS optimization on the 28-point eye model using
        eye-specific CCNF patch experts for accurate eye landmark detection.

        Args:
            image: Grayscale image
            main_landmarks: Current 68 landmarks
            side: 'left' or 'right'
            main_rotation: Rotation from main model [rx, ry, rz] for initialization
            main_scale: Detection scale from main model
            track_iterations: If True, return iteration history

        Returns:
            If track_iterations is False:
                refined_landmarks: Updated 68 landmarks with refined eye positions
            If track_iterations is True:
                (refined_landmarks, iteration_history): Tuple of landmarks and iteration data
        """
        # Initialize iteration tracking
        iteration_history = [] if track_iterations else None

        if self.pdm[side] is None:
            if track_iterations:
                return main_landmarks, iteration_history
            return main_landmarks

        pdm = self.pdm[side]
        ccnf = self.ccnf[side]

        # C++ uses original image coordinates - don't scale
        # The eye PDM scale will be larger to match (scale ~ main_scale * normalized_eye_scale)
        scale_factor = 1.0
        scaled_image = image
        scaled_landmarks = main_landmarks.copy()

        # Get eye landmark indices for this side
        if side == 'left':
            main_indices = [36, 37, 38, 39, 40, 41]
        else:
            main_indices = [42, 43, 44, 45, 46, 47]

        # Get the mapping for this side
        mapping = LEFT_EYE_MAPPING if side == 'left' else RIGHT_EYE_MAPPING

        # Extract current eye landmarks from main model (in scaled coordinates)
        current_eye = scaled_landmarks[main_indices]  # (6, 2)

        # Initialize eye PDM parameters using Procrustes alignment
        # Pass main model rotation for proper 3D pose initialization
        params = self._fit_eye_shape(current_eye, mapping, side, main_rotation)
        if params is None:
            if track_iterations:
                return main_landmarks, iteration_history
            return main_landmarks

        # Get all 28 eye landmarks for optimization
        eye_landmarks = pdm.params_to_landmarks_2d(params)

        # Hierarchical optimization over window sizes (C++ uses [3, 5, 9])
        for window_idx, window_size in enumerate(self.window_sizes):
            # Get the correct patch scale for this window size
            # window_sizes[i] -> patch_scaling[i]: ws=3 -> 1.0, ws=5 -> 1.5
            patch_scale = self.patch_scaling[window_idx]

            # Set current window size for response map computation
            self._current_window_size = window_size

            # NOTE: C++ sets refine_parameters=false for hierarchical models (eye, mouth, etc.)
            # This means sigma and reg_factor are NOT scaled for eye models - use fixed values
            self._current_sigma = self.sigma  # Fixed at 1.0 for eyes

            # Get patch experts for this scale
            patch_experts = ccnf.get_all_patch_experts(patch_scale)

            if not patch_experts:
                # No patch experts for this scale, skip to next window size
                continue

            # CRITICAL: Precompute response maps ONCE at initial positions for this window size
            # This matches the main model's approach - response maps are computed once
            # and then offsets track how landmarks move from initial positions
            initial_eye_landmarks = eye_landmarks.copy()

            # Compute similarity transform to align current shape to reference shape
            # This is critical for correct patch extraction and mean-shift transformation
            sim_ref_to_img, sim_img_to_ref = self._compute_sim_transforms(
                initial_eye_landmarks, params, pdm, side, patch_scale
            )

            response_maps = self._compute_eye_response_maps(
                scaled_image, initial_eye_landmarks, patch_experts, sim_ref_to_img,
                side=side, patch_scale=patch_scale
            )

            # Run RIGID phase first (like C++)
            for iteration in range(self.max_iterations):
                # Compute mean-shift using precomputed response maps and offset tracking
                # Pass sim_img_to_ref to transform offsets to reference space (like C++)
                mean_shift = self._compute_eye_mean_shift_with_offset(
                    eye_landmarks, initial_eye_landmarks, response_maps, patch_experts,
                    sim_img_to_ref, side=side
                )

                # Transform mean-shifts from reference space to image space (like C++)
                # C++: mean_shifts_2D = mean_shifts_2D * cv::Mat(sim_ref_to_img).t()
                # Convert stacked format to (n, 2) for transformation
                n_eye = 28
                ms_x = mean_shift[:n_eye]
                ms_y = mean_shift[n_eye:]
                mean_shift_2D = np.column_stack([ms_x, ms_y])  # (28, 2)
                mean_shift_2D = mean_shift_2D @ sim_ref_to_img.T
                # Convert back to stacked format
                mean_shift = np.concatenate([mean_shift_2D[:, 0], mean_shift_2D[:, 1]])

                # Solve for rigid-only parameter update
                delta_p = self._solve_eye_update_rigid(
                    pdm, params, mean_shift
                )

                # Apply update
                params = pdm.update_params(params, delta_p)
                params = pdm.clamp_params(params)

                # Update landmarks
                eye_landmarks = pdm.params_to_landmarks_2d(params)

                # Track iteration if requested
                if track_iterations:
                    iteration_data = {
                        'iteration': len(iteration_history),
                        'window_size': window_size,
                        'phase': 'rigid',
                        'eye_side': side,
                        'params': {
                            'global': params[:6].tolist(),  # scale, rot, trans
                            'local': params[6:].tolist() if len(params) > 6 else []
                        },
                        'landmarks': eye_landmarks.tolist(),
                        'update_magnitude': float(np.linalg.norm(delta_p))
                    }
                    iteration_history.append(iteration_data)

                # Check convergence
                if np.linalg.norm(delta_p) < 0.01:
                    break

            # CRITICAL FIX: C++ uses SAME response maps for both RIGID and NONRIGID phases
            # DO NOT recompute response maps here - use initial_eye_landmarks as base
            # Response maps were computed ONCE at start of this window_size
            # The offset tracking mechanism handles the landmark movement during optimization

            # Run NONRIGID phase (shape parameters)
            for iteration in range(self.max_iterations):
                # Compute mean-shift at current positions (offset from INITIAL base - same as RIGID)
                mean_shift = self._compute_eye_mean_shift_with_offset(
                    eye_landmarks, initial_eye_landmarks, response_maps, patch_experts,
                    sim_img_to_ref, side=side
                )

                # Transform mean-shifts from reference space to image space (like C++)
                # Use SAME transform as RIGID phase (sim_ref_to_img, not sim_ref_to_img_nr)
                # Convert stacked format to (n, 2) for transformation
                n_eye = 28
                ms_x = mean_shift[:n_eye]
                ms_y = mean_shift[n_eye:]
                mean_shift_2D = np.column_stack([ms_x, ms_y])  # (28, 2)
                mean_shift_2D = mean_shift_2D @ sim_ref_to_img.T
                # Convert back to stacked format
                mean_shift = np.concatenate([mean_shift_2D[:, 0], mean_shift_2D[:, 1]])

                # Solve for parameter update
                delta_p = self._solve_eye_update(
                    pdm, params, mean_shift, mapping
                )

                # Apply update
                params = pdm.update_params(params, delta_p)
                params = pdm.clamp_params(params)

                # Update landmarks
                eye_landmarks = pdm.params_to_landmarks_2d(params)

                # Track iteration if requested
                if track_iterations:
                    iteration_data = {
                        'iteration': len(iteration_history),
                        'window_size': window_size,
                        'phase': 'nonrigid',
                        'eye_side': side,
                        'params': {
                            'global': params[:6].tolist(),  # scale, rot, trans
                            'local': params[6:].tolist() if len(params) > 6 else []
                        },
                        'landmarks': eye_landmarks.tolist(),
                        'update_magnitude': float(np.linalg.norm(delta_p))
                    }
                    iteration_history.append(iteration_data)

                # Check convergence
                if np.linalg.norm(delta_p) < 0.01:
                    break

        # Update only the mapped landmarks in the main model
        # Scale landmarks back to original image coordinates
        refined_main = main_landmarks.copy()
        for main_idx, eye_idx in mapping.items():
            refined_main[main_idx] = eye_landmarks[eye_idx] / scale_factor

        if track_iterations:
            return refined_main, iteration_history
        return refined_main

    def _compute_sim_ref_to_img(self, eye_landmarks: np.ndarray, pdm) -> np.ndarray:
        """
        Compute transformation from reference space to image space.

        This aligns current eye shape to reference shape, like C++ AlignShapesWithScale.

        Returns:
            sim_ref_to_img: 2x2 transformation matrix [scale*cos, -scale*sin; scale*sin, scale*cos]
        """
        # Get reference shape (mean shape)
        # CRITICAL FIX: PDM stores as grouped [x0,x1,...,xn, y0,y1,...,yn], NOT interleaved
        x_coords = pdm.mean_shape[:pdm.n_points]
        y_coords = pdm.mean_shape[pdm.n_points:pdm.n_points * 2]
        ref_shape = np.column_stack([x_coords, y_coords])

        # Current shape
        curr_shape = eye_landmarks

        # Center both shapes
        ref_center = np.mean(ref_shape, axis=0)
        curr_center = np.mean(curr_shape, axis=0)

        ref_centered = ref_shape - ref_center
        curr_centered = curr_shape - curr_center

        # Compute scale
        ref_scale = np.sqrt(np.sum(ref_centered ** 2) / len(ref_shape))
        curr_scale = np.sqrt(np.sum(curr_centered ** 2) / len(curr_shape))

        if ref_scale < 1e-6:
            return np.eye(2, dtype=np.float32)

        scale = curr_scale / ref_scale

        # Compute rotation using Procrustes
        # For now, use identity rotation (eye model has minimal rotation)
        # Full Procrustes would compute optimal rotation angle

        sim_ref_to_img = np.array([[scale, 0],
                                   [0, scale]], dtype=np.float32)

        return sim_ref_to_img

    def _compute_eye_response_maps(self, image: np.ndarray,
                                   eye_landmarks: np.ndarray,
                                   patch_experts: dict,
                                   sim_ref_to_img: np.ndarray = None,
                                   side: str = None,
                                   patch_scale: float = None) -> dict:
        """
        Compute response maps for each eye landmark using CCNF patches.

        Uses C++ warpAffine approach: extract area_of_interest centered on landmark,
        then compute responses by sliding patch across entire area.

        Args:
            image: Grayscale image
            eye_landmarks: Current 28 eye landmarks
            patch_experts: Dict mapping landmark_idx -> EyeCCNFPatchExpert
            sim_ref_to_img: 2x2 similarity transform from reference to image space
            side: 'left' or 'right' eye (for GPU batched computation)
            patch_scale: Patch scale (1.0 or 1.5) for GPU batched computation

        Returns:
            response_maps: Dict mapping landmark_idx -> response_map
        """
        ws = getattr(self, '_current_window_size', self.window_sizes[0])

        # Use GPU batched computation if available
        if self.use_gpu and side is not None and patch_scale is not None:
            batched = self.batched_ccnf.get(side)
            if batched and hasattr(batched, 'compute_response_maps'):
                return batched.compute_response_maps(
                    image.astype(np.float32, copy=False),
                    eye_landmarks,
                    patch_scale,
                    ws,
                    sim_ref_to_img
                )

        response_maps = {}

        # Get a1, b1 from similarity transform (like C++)
        if sim_ref_to_img is not None:
            a1 = sim_ref_to_img[0, 0]
            b1 = -sim_ref_to_img[0, 1]  # C++ uses -sim_ref_to_img(0,1)
        else:
            # Fallback to identity
            a1 = 1.0
            b1 = 0.0

        for lm_idx, patch_expert in patch_experts.items():
            if lm_idx >= len(eye_landmarks):
                continue

            # Get landmark position
            x, y = eye_landmarks[lm_idx]

            # Compute area of interest size (like C++)
            # area_of_interest = window_size + patch_size - 1
            patch_size = patch_expert.height
            aoi_size = ws + patch_size - 1
            half_aoi = (aoi_size - 1) / 2.0

            # Create transformation matrix for warpAffine (like C++)
            # sim = [[a1, -b1, tx], [b1, a1, ty]]
            # tx = landmark_x - a1 * half_aoi + b1 * half_aoi
            # ty = landmark_y - a1 * half_aoi - b1 * half_aoi
            tx = x - a1 * half_aoi + b1 * half_aoi
            ty = y - a1 * half_aoi - b1 * half_aoi

            sim = np.array([[a1, -b1, tx],
                           [b1, a1, ty]], dtype=np.float32)

            # Extract area of interest using warpAffine with WARP_INVERSE_MAP
            area_of_interest = cv2.warpAffine(
                image.astype(np.float32, copy=False),
                sim,
                (aoi_size, aoi_size),
                flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
            )

            # Compute response map by sliding patch across area of interest
            # Response dimensions: aoi_size - patch_size + 1 = ws
            response_map = self._compute_ccnf_response_map(area_of_interest, patch_expert, landmark_idx=lm_idx)

            response_maps[lm_idx] = response_map

        return response_maps

    def _compute_ccnf_response_map(self, area_of_interest: np.ndarray,
                                   patch_expert, landmark_idx: int = -1) -> np.ndarray:
        """
        Compute CCNF response map over entire area of interest.

        This matches C++ CCNF_patch_expert::Response - slides patch across
        area and computes response at each position.

        Args:
            area_of_interest: Float image of size (aoi_height, aoi_width)
            patch_expert: EyeCCNFPatchExpert
            landmark_idx: Landmark index for debug purposes

        Returns:
            response_map: (response_height, response_width) where
                          response_height = aoi_height - patch_height + 1
        """
        aoi_h, aoi_w = area_of_interest.shape
        patch_h = patch_expert.height
        patch_w = patch_expert.width

        resp_h = aoi_h - patch_h + 1
        resp_w = aoi_w - patch_w + 1

        response_map = np.zeros((resp_h, resp_w), dtype=np.float32)

        # Debug for Eye landmarks 0 and 8 on 3x3 response maps (like C++)
        debug_enabled = (resp_h == 3 and resp_w == 3 and landmark_idx in [0, 8])
        debug_call = getattr(self, '_ccnf_debug_call', 0)

        # Slide patch across area of interest
        # CRITICAL: C++ im2col uses column-major order (iterate cols first, then rows)
        # to match, we need: outer loop j (cols), inner loop i (rows)
        for j in range(resp_w):
            for i in range(resp_h):
                # Extract patch at position (i, j) where i=row, j=col
                patch = area_of_interest[i:i+patch_h, j:j+patch_w]

                # Compute response - debug center pixel only
                debug_file = None
                if debug_enabled and i == 1 and j == 1:  # Center pixel of 3x3
                    debug_file = f'/tmp/python_ccnf_neuron_debug_call{debug_call}.txt'
                    self._ccnf_debug_call = debug_call + 1

                # Keep as float32 like C++ - don't convert to uint8 which loses fractional precision
                response = patch_expert.compute_response(patch.astype(np.float32), debug_file=debug_file)
                response_map[i, j] = response

        # Make sure response has no negative values (like C++)
        min_val = response_map.min()
        if min_val < 0:
            response_map = response_map - min_val

        return response_map

    def _compute_eye_mean_shift(self, eye_landmarks: np.ndarray,
                               response_maps: dict,
                               patch_experts: dict) -> np.ndarray:
        """
        Compute mean-shift vector from response maps.

        Args:
            eye_landmarks: Current 28 eye landmarks
            response_maps: Dict of response maps per landmark
            patch_experts: Dict of patch experts

        Returns:
            mean_shift: Mean-shift vector (2 * n_points,)
        """
        n_points = len(eye_landmarks)
        mean_shift = np.zeros(2 * n_points)

        # Gaussian kernel parameter (use scaled sigma if available)
        sigma = getattr(self, '_current_sigma', self.sigma)
        a_kde = -0.5 / (sigma * sigma)
        ws = getattr(self, '_current_window_size', self.window_sizes[0])
        center = (ws - 1) / 2.0

        for lm_idx, response_map in response_maps.items():
            if lm_idx >= n_points:
                continue

            # KDE mean-shift computation
            total_weight = 0.0
            mx = 0.0
            my = 0.0

            for ii in range(ws):
                for jj in range(ws):
                    # Distance from center
                    dist_sq = (ii - center)**2 + (jj - center)**2

                    # Gaussian weight
                    kde_weight = np.exp(a_kde * dist_sq)

                    # Combined weight
                    weight = response_map[ii, jj] * kde_weight

                    total_weight += weight
                    mx += weight * jj
                    my += weight * ii

            # Compute mean-shift
            if total_weight > 1e-10:
                ms_x = (mx / total_weight) - center
                ms_y = (my / total_weight) - center
            else:
                ms_x = 0.0
                ms_y = 0.0

            # STACKED format: [lm_idx] = x, [lm_idx + n_points] = y
            mean_shift[lm_idx] = ms_x
            mean_shift[lm_idx + n_points] = ms_y

        return mean_shift

    def _compute_eye_mean_shift_at_center(self, response_maps: dict,
                                         patch_experts: dict) -> np.ndarray:
        """
        Compute mean-shift vector from response maps, assuming we're at the center.

        This is simpler than offset tracking and avoids issues with small response
        maps being exceeded by large movements.

        Args:
            response_maps: Dict of response maps per landmark
            patch_experts: Dict of patch experts

        Returns:
            mean_shift: Mean-shift vector (2 * n_points,)
        """
        n_points = 28  # Eye model always has 28 points
        mean_shift = np.zeros(2 * n_points)

        # Gaussian kernel parameter (use scaled sigma if available)
        sigma = getattr(self, '_current_sigma', self.sigma)
        a_kde = -0.5 / (sigma * sigma)
        ws = getattr(self, '_current_window_size', self.window_sizes[0])
        center = (ws - 1) / 2.0

        for lm_idx, response_map in response_maps.items():
            if lm_idx >= n_points:
                continue

            # We're at the center of the response map
            dx = center
            dy = center

            # KDE mean-shift computation
            total_weight = 0.0
            mx = 0.0
            my = 0.0

            for ii in range(ws):
                for jj in range(ws):
                    # Distance from center
                    dist_sq = (ii - dy)**2 + (jj - dx)**2

                    # Gaussian weight
                    kde_weight = np.exp(a_kde * dist_sq)

                    # Combined weight
                    weight = response_map[ii, jj] * kde_weight

                    total_weight += weight
                    mx += weight * jj
                    my += weight * ii

            # Compute mean-shift relative to center
            if total_weight > 1e-10:
                ms_x = (mx / total_weight) - center
                ms_y = (my / total_weight) - center
            else:
                ms_x = 0.0
                ms_y = 0.0

            # STACKED format: [lm_idx] = x, [lm_idx + n_points] = y
            mean_shift[lm_idx] = ms_x
            mean_shift[lm_idx + n_points] = ms_y

        return mean_shift

    def _compute_eye_mean_shift_with_offset(self, eye_landmarks: np.ndarray,
                                           base_landmarks: np.ndarray,
                                           response_maps: dict,
                                           patch_experts: dict,
                                           sim_img_to_ref: np.ndarray = None,
                                           side: str = None) -> np.ndarray:
        """
        Compute mean-shift vector using precomputed response maps and offset tracking.

        This matches the C++ approach where:
        1. Offsets are computed in image space
        2. Offsets are transformed to reference space (divided by scale)
        3. Mean-shift is computed in reference space
        4. Mean-shift is later transformed back to image space

        Args:
            eye_landmarks: Current 28 eye landmarks
            base_landmarks: Base landmark positions where response maps were extracted
            response_maps: Dict of precomputed response maps per landmark
            patch_experts: Dict of patch experts
            sim_img_to_ref: 2x2 similarity transform from image to reference space
            side: 'left' or 'right' eye (for GPU batched computation)

        Returns:
            mean_shift: Mean-shift vector (2 * n_points,) in REFERENCE space
        """
        # Use GPU batched computation if available
        sigma = getattr(self, '_current_sigma', self.sigma)
        ws = getattr(self, '_current_window_size', self.window_sizes[0])

        if self.use_gpu and side is not None:
            batched = self.batched_ccnf.get(side)
            if batched and hasattr(batched, 'compute_mean_shift_batched'):
                return batched.compute_mean_shift_batched(
                    response_maps,
                    eye_landmarks,
                    base_landmarks,
                    ws,
                    sigma,
                    sim_img_to_ref
                )

        # Fallback to Python implementation
        n_points = len(eye_landmarks)
        mean_shift = np.zeros(2 * n_points)

        a_kde = -0.5 / (sigma * sigma)
        center = (ws - 1) / 2.0

        for lm_idx, response_map in response_maps.items():
            if lm_idx >= n_points:
                continue

            # Compute offset from base landmark to current landmark in IMAGE space
            offset_img_x = eye_landmarks[lm_idx, 0] - base_landmarks[lm_idx, 0]
            offset_img_y = eye_landmarks[lm_idx, 1] - base_landmarks[lm_idx, 1]

            # Transform offset to REFERENCE space (like C++)
            # C++: offsets = (current_shape - base_shape) * sim_img_to_ref.t()
            # NOTE: This multiplication is [x, y] @ sim_img_to_ref.T
            if sim_img_to_ref is not None:
                # Matrix multiply: [offset_x, offset_y] @ [[a, b], [c, d]]^T = [offset_x, offset_y] @ [[a, c], [b, d]]
                offset_ref_x = offset_img_x * sim_img_to_ref[0, 0] + offset_img_y * sim_img_to_ref[0, 1]
                offset_ref_y = offset_img_x * sim_img_to_ref[1, 0] + offset_img_y * sim_img_to_ref[1, 1]
            else:
                offset_ref_x = offset_img_x
                offset_ref_y = offset_img_y

            # Position within response map (center + offset in reference space)
            dx = offset_ref_x + center
            dy = offset_ref_y + center

            # Clamp to valid range
            if dx < 0:
                dx = 0
            if dy < 0:
                dy = 0
            if dx > ws - 1:
                dx = ws - 1
            if dy > ws - 1:
                dy = ws - 1

            # KDE mean-shift computation
            total_weight = 0.0
            mx = 0.0
            my = 0.0

            for ii in range(ws):
                for jj in range(ws):
                    # Distance from current position (dx, dy) in response map
                    dist_sq = (ii - dy)**2 + (jj - dx)**2

                    # Gaussian weight
                    kde_weight = np.exp(a_kde * dist_sq)

                    # Combined weight
                    weight = response_map[ii, jj] * kde_weight

                    total_weight += weight
                    mx += weight * jj
                    my += weight * ii

            # Compute mean-shift relative to current position in response map
            if total_weight > 1e-10:
                ms_x = (mx / total_weight) - dx
                ms_y = (my / total_weight) - dy
            else:
                ms_x = 0.0
                ms_y = 0.0

            # STACKED format: [lm_idx] = x, [lm_idx + n_points] = y
            mean_shift[lm_idx] = ms_x
            mean_shift[lm_idx + n_points] = ms_y

            # DEBUG: Detailed Eye_8 trace (left eye outer corner)
            if lm_idx == 8 and hasattr(self, '_eye8_trace_file'):
                with open(self._eye8_trace_file, 'a') as f:
                    f.write(f"\n--- Eye_8 Mean-Shift Computation ---\n")
                    f.write(f"Position: ({eye_landmarks[lm_idx, 0]:.6f}, {eye_landmarks[lm_idx, 1]:.6f})\n")
                    f.write(f"Base position: ({base_landmarks[lm_idx, 0]:.6f}, {base_landmarks[lm_idx, 1]:.6f})\n")
                    f.write(f"Offset (image): ({offset_img_x:.6f}, {offset_img_y:.6f})\n")
                    f.write(f"Offset (ref): ({offset_ref_x:.6f}, {offset_ref_y:.6f})\n")
                    f.write(f"dx, dy: ({dx:.6f}, {dy:.6f})\n")
                    f.write(f"Window size: {ws}, center: {center}\n")
                    f.write(f"Total weight: {total_weight:.8f}\n")
                    f.write(f"Weighted sum: mx={mx:.8f}, my={my:.8f}\n")
                    f.write(f"Raw mean-shift (ref): ({ms_x:.6f}, {ms_y:.6f})\n")

                    # Response map values
                    f.write(f"Response map ({ws}x{ws}):\n")
                    for ii in range(ws):
                        f.write("  ")
                        for jj in range(ws):
                            f.write(f"{response_map[ii, jj]:.4f} ")
                        f.write("\n")

        return mean_shift

    def _solve_eye_update_rigid(self, pdm, params: np.ndarray,
                                mean_shift: np.ndarray) -> np.ndarray:
        """
        Solve for rigid-only parameter update (no shape parameters).

        Args:
            pdm: Eye PDM
            params: Current parameters
            mean_shift: Mean-shift vector (2 * n_points,)

        Returns:
            delta_p: Parameter update (only first 6 elements non-zero)
        """
        # Compute Jacobian
        J = pdm.compute_jacobian(params)  # (2*28, 16)

        # Extract only rigid columns (first 6)
        J_rigid = J[:, :6]  # (2*28, 6)

        # Create weight matrix (identity)
        n_points = pdm.n_points
        W = np.eye(2 * n_points)

        # No regularization for rigid params
        regTerm = np.zeros((6, 6))

        # Solve: (J'WJ + reg)Δp = J'W·v
        A = J_rigid.T @ W @ J_rigid + regTerm
        b = J_rigid.T @ W @ mean_shift

        try:
            delta_p_rigid = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta_p_rigid = np.zeros(6)

        # Create full parameter update (zeros for shape params)
        delta_p = np.zeros(pdm.n_params)
        delta_p[:6] = delta_p_rigid

        # DEBUG: Eye_8 Jacobian and solver trace
        if hasattr(self, '_eye8_trace_file'):
            with open(self._eye8_trace_file, 'a') as f:
                f.write(f"\n--- Eye_8 Solver (Rigid) ---\n")
                # Jacobian rows for Eye_8 (STACKED format: rows 8 and 8+28=36)
                J8_x = J_rigid[8, :]  # X row
                J8_y = J_rigid[36, :]  # Y row (8 + 28)
                f.write(f"Jacobian row 8 (X): [{', '.join([f'{v:.6f}' for v in J8_x])}]\n")
                f.write(f"Jacobian row 36 (Y): [{', '.join([f'{v:.6f}' for v in J8_y])}]\n")
                # STACKED format: [lm_idx] = x, [lm_idx + 28] = y
                f.write(f"Mean-shift Eye_8: ({mean_shift[8]:.6f}, {mean_shift[8 + 28]:.6f})\n")
                f.write(f"b vector: [{', '.join([f'{v:.6f}' for v in b])}]\n")
                f.write(f"A matrix diagonal: [{', '.join([f'{A[i,i]:.6f}' for i in range(6)])}]\n")
                f.write(f"delta_p: [{', '.join([f'{v:.6f}' for v in delta_p_rigid])}]\n")

        # Note: C++ does NOT apply damping in eye model refinement
        # (Unlike main model which uses 0.75 damping in PDM.cpp CalcParams)
        # delta_p *= 0.5  # REMOVED - C++ doesn't dampen eye model updates

        return delta_p

    def _solve_eye_update(self, pdm, params: np.ndarray,
                         mean_shift: np.ndarray, mapping: dict) -> np.ndarray:
        """
        Solve for parameter update using mean-shift and regularization.

        Args:
            pdm: Eye PDM
            params: Current parameters
            mean_shift: Mean-shift vector (2 * n_points,)
            mapping: Main->Eye index mapping

        Returns:
            delta_p: Parameter update
        """
        # Compute Jacobian
        J = pdm.compute_jacobian(params)  # (2*28, 16)

        # Create weight matrix (identity for now)
        n_points = pdm.n_points
        W = np.eye(2 * n_points)

        # Create regularization matrix
        n_params = pdm.n_params
        Lambda_inv = np.zeros((n_params, n_params))

        # No regularization for rigid params
        Lambda_inv[0, 0] = 0  # scale
        Lambda_inv[1, 1] = 0  # rx
        Lambda_inv[2, 2] = 0  # ry
        Lambda_inv[3, 3] = 0  # rz
        Lambda_inv[4, 4] = 0  # tx
        Lambda_inv[5, 5] = 0  # ty

        # Regularize shape params by inverse eigenvalues
        for i in range(pdm.n_modes):
            if pdm.eigen_values.flatten()[i] > 1e-10:
                Lambda_inv[6+i, 6+i] = 1.0 / pdm.eigen_values.flatten()[i]
            else:
                Lambda_inv[6+i, 6+i] = 1e10

        # Solve: (J'WJ + λΛ⁻¹)Δp = J'W·v - λΛ⁻¹·p
        reg = self.reg_factor
        A = J.T @ W @ J + reg * Lambda_inv
        b = J.T @ W @ mean_shift - reg * Lambda_inv @ params

        try:
            delta_p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta_p = np.zeros(n_params)

        # Note: C++ does NOT apply damping in eye model refinement
        # delta_p *= 0.5  # REMOVED - C++ doesn't dampen eye model updates

        return delta_p

    def _fit_eye_shape(self, target_points: np.ndarray, mapping: dict, side: str,
                       main_rotation: np.ndarray = None):
        """
        Fit eye PDM to target points using least squares with shape regularization.

        This fits both rigid (scale, rotation, translation) and non-rigid (shape)
        parameters to best match the target points while staying close to mean shape.

        Args:
            target_points: Target 2D points (6 eyelid landmarks)
            mapping: Main->Eye index mapping
            side: 'left' or 'right'
            main_rotation: Rotation from main model [rx, ry, rz] for 3D pose

        Returns:
            params: Fitted eye PDM parameters
        """
        pdm = self.pdm[side]

        # Get eye indices
        eye_indices = list(mapping.values())  # [8, 10, 12, 14, 16, 18]

        # Step 1: Compute initial params using bounding box (like C++ CalcParams)
        # C++ CalcBoundingBox projects mean shape at scale=1, rot=0, trans=0
        # We need to do the same with ONLY the 6 visible landmarks

        # Get mean shape for the 6 visible eye landmarks only
        # C++ CalcParams subsamples to only visible landmarks (landmarks with non-zero values)
        mean_flat = pdm.mean_shape.flatten()
        n = pdm.n_points
        X_all = mean_flat[:n]
        Y_all = mean_flat[n:2*n]

        # Get just the 6 visible landmarks' 2D positions (at scale=1, rot=0, trans=0)
        mean_2d = np.column_stack([X_all[eye_indices], Y_all[eye_indices]])

        # Compute bounding boxes (C++ style)
        target_min_x, target_max_x = np.min(target_points[:, 0]), np.max(target_points[:, 0])
        target_min_y, target_max_y = np.min(target_points[:, 1]), np.max(target_points[:, 1])
        target_width = abs(target_max_x - target_min_x)
        target_height = abs(target_max_y - target_min_y)

        mean_min_x, mean_max_x = np.min(mean_2d[:, 0]), np.max(mean_2d[:, 0])
        mean_min_y, mean_max_y = np.min(mean_2d[:, 1]), np.max(mean_2d[:, 1])
        mean_width = abs(mean_max_x - mean_min_x)
        mean_height = abs(mean_max_y - mean_min_y)

        if mean_width < 1e-10 or mean_height < 1e-10:
            return None

        # Scale = average of width and height ratios (C++ style)
        scale = ((target_width / mean_width) + (target_height / mean_height)) / 2.0

        # Translation = center of target bounding box (uses visible landmarks)
        tx = (target_min_x + target_max_x) / 2.0
        ty = (target_min_y + target_max_y) / 2.0

        # C++ doesn't correct translation for mean center because the mean shape
        # is already centered (mean_center ≈ 0). The translation is just the target center.

        # Initialize parameters with rigid transform
        params = np.zeros(pdm.n_params)
        params[0] = scale
        params[4] = tx
        params[5] = ty

        # C++ CalcParams uses default rotation (0,0,0), not main model rotation
        # The optimization finds the rotation from the landmarks
        initial_rotation = np.array([0.0, 0.0, 0.0])
        # params[1:4] are already zero from np.zeros

        # Step 2: Fit all parameters using iterative optimization (like C++ CalcParams)
        # Create regularization matrix using inverse eigenvalues (like C++)
        reg_factor = 1.0  # C++ uses reg_factor=1 in CalcParams

        # Build regularization diagonal: [0,0,0,0,0,0, 1/eig[0], 1/eig[1], ...]
        # Also add small regularization for rotation to keep it close to main model
        reg_diag = np.zeros(pdm.n_params)

        # C++ CalcParams doesn't regularize rotation - leave reg_diag[1:4] at zero

        # Regularize shape parameters using eigenvalues
        for i in range(pdm.n_modes):
            if pdm.eigen_values.flatten()[i] > 1e-10:
                reg_diag[6+i] = reg_factor / pdm.eigen_values.flatten()[i]
            else:
                reg_diag[6+i] = 1e10
        regularisation = np.diag(reg_diag)

        # Weight matrix (identity for now)
        W = np.eye(12)

        prev_error = float('inf')
        not_improved = 0

        for iteration in range(1500):  # Up to 1500 iterations for better convergence
            # Get current model points
            current_2d = pdm.params_to_landmarks_2d(params)
            current_points = current_2d[eye_indices]

            # Compute error residual
            error_resid = (target_points - current_points).flatten()  # (12,)

            # Check convergence
            error = np.linalg.norm(error_resid)
            if error >= 0.999 * prev_error:
                not_improved += 1
                if not_improved >= 3:
                    break
            else:
                not_improved = 0
            prev_error = error

            # Get full Jacobian (STACKED format: rows 0:n = x, rows n:2n = y)
            J_full = pdm.compute_jacobian(params)  # (56, 16)
            n = pdm.n_points  # 28

            # Extract rows for our 6 landmarks
            # error_resid is interleaved [x0,y0,x1,y1,...], so J must match
            J = np.zeros((12, pdm.n_params))
            for i, eye_idx in enumerate(eye_indices):
                J[2*i] = J_full[eye_idx]        # x component from stacked row eye_idx
                J[2*i+1] = J_full[eye_idx + n]  # y component from stacked row eye_idx + n

            # Weighted Jacobian
            J_w_t = J.T @ W  # (16, 12)

            # Projection of residuals onto jacobians
            J_w_t_m = J_w_t @ error_resid  # (16,)

            # Add regularization term for shape params (pull towards zero)
            J_w_t_m[6:] = J_w_t_m[6:] - regularisation[6:, 6:] @ params[6:]

            # Compute Hessian
            Hessian = J_w_t @ J + regularisation  # (16, 16)

            # Solve for parameter update
            try:
                param_update = np.linalg.solve(Hessian, J_w_t_m)
            except np.linalg.LinAlgError:
                break

            # Damping (C++ uses 0.75)
            param_update *= 0.75

            # Update parameters
            params = pdm.update_params(params, param_update)
            params = pdm.clamp_params(params)

        return params

    def get_info(self) -> dict:
        """Get hierarchical eye model information."""
        info = {
            'window_sizes': self.window_sizes,
            'sigma': self.sigma,
            'reg_factor': self.reg_factor,
            'max_iterations': self.max_iterations,
            'pdm': {},
            'ccnf': {}
        }

        for side in ['left', 'right']:
            if self.pdm[side]:
                info['pdm'][side] = self.pdm[side].get_info()
            info['ccnf'][side] = self.ccnf[side].get_info()

        return info
