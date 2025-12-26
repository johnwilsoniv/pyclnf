#!/usr/bin/env python3
"""
CEN (Convolutional Expert Network) patch expert loader and inference.

Loads and runs patch expert models from OpenFace 2.2's .dat format.
"""

import numpy as np
import cv2
from pathlib import Path
import struct
from typing import List

# Try to import numba for JIT compilation (optional)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


class CENPatchExpert:
    """
    Single CEN patch expert for one landmark at one scale.

    A patch expert is a small neural network that evaluates how likely
    a landmark is at each position in a local patch.
    """

    def __init__(self):
        self.width_support = 0
        self.height_support = 0
        self.weights = []  # List of weight matrices for each layer
        self.biases = []   # List of bias vectors for each layer
        self.activation_function = []  # Activation type for each layer
        self.confidence = 0.0
        self.is_empty = False

    @property
    def width(self):
        """Compatibility property for CCNF interface"""
        return self.width_support

    @property
    def height(self):
        """Compatibility property for CCNF interface"""
        return self.height_support

    @property
    def patch_confidence(self):
        """Compatibility property for CCNF interface"""
        return self.confidence

    @classmethod
    def from_stream(cls, stream):
        """
        Load a CEN patch expert from binary stream.

        Args:
            stream: Binary file stream positioned at patch expert data

        Returns:
            CENPatchExpert instance
        """
        expert = cls()

        # Read header
        read_type = struct.unpack('i', stream.read(4))[0]
        if read_type != 6:
            raise ValueError(f"Invalid CEN patch expert type: {read_type}, expected 6")

        # Read dimensions and layer count
        expert.width_support = struct.unpack('i', stream.read(4))[0]
        expert.height_support = struct.unpack('i', stream.read(4))[0]
        num_layers = struct.unpack('i', stream.read(4))[0]

        # Empty patch (landmark invisible at this orientation)
        if num_layers == 0:
            expert.confidence = struct.unpack('d', stream.read(8))[0]
            expert.is_empty = True
            return expert

        # Read layers
        for i in range(num_layers):
            # Activation function type
            neuron_type = struct.unpack('i', stream.read(4))[0]
            expert.activation_function.append(neuron_type)

            # Read bias matrix
            bias = read_mat_bin(stream)
            expert.biases.append(bias)

            # Read weight matrix
            weight = read_mat_bin(stream)
            expert.weights.append(weight)

        # Read confidence
        expert.confidence = struct.unpack('d', stream.read(8))[0]

        return expert

    def compute_sigma(self, sigma_components, window_size=None, debug=False):
        """
        Compute Sigma covariance matrix for CEN patch experts.

        For CEN models without betas (quick/video mode), returns identity matrix.
        This matches OpenFace behavior when sigma is set to 0 for fast processing.

        Args:
            sigma_components: List of sigma component matrices (unused for CEN)
            window_size: Response map window size
            debug: Print debugging information

        Returns:
            Identity matrix for CEN (no spatial correlation modeling in quick mode)
        """
        if window_size is None:
            window_size = self.width_support
        matrix_size = window_size * window_size

        if debug:
            print(f"    [CEN Sigma] Returning identity matrix for quick mode (size={matrix_size}x{matrix_size})")

        # Return identity matrix - no spatial correlation modeling for CEN in quick mode
        return np.eye(matrix_size, dtype=np.float32)

    def response(self, area_of_interest):
        """
        Compute patch expert response map for an image patch.

        Matches C++ CEN_patch_expert::Response() exactly:
        1. im2col to extract patches
        2. contrast normalization on im2col output (row-wise on patches)
        3. neural network forward pass

        Args:
            area_of_interest: Grayscale image patch (H, W) as float32

        Returns:
            response: Response map (response_height, response_width) as float32
        """
        if self.is_empty:
            # Return zero response for empty patches
            response_height = max(1, area_of_interest.shape[0] - self.height_support + 1)
            response_width = max(1, area_of_interest.shape[1] - self.width_support + 1)
            return np.zeros((response_height, response_width), dtype=np.float32)

        response_height = area_of_interest.shape[0] - self.height_support + 1
        response_width = area_of_interest.shape[1] - self.width_support + 1

        # Use Numba-optimized version for 2-layer networks (most common case)
        if NUMBA_AVAILABLE and len(self.weights) == 2:
            # Ensure input is contiguous float32
            input_patch = np.ascontiguousarray(area_of_interest, dtype=np.float32)

            return _response_core_numba(
                input_patch,
                self.width_support,
                self.height_support,
                self.weights[0],
                self.biases[0],
                self.activation_function[0],
                self.weights[1],
                self.biases[1],
                self.activation_function[1],
                response_height,
                response_width
            )

        # Fallback for non-2-layer networks or when Numba unavailable
        # STEP 1: Convert to column format with bias (im2col)
        # Matches C++ im2colBias: extracts all patches
        input_col = im2col_bias(area_of_interest, self.width_support, self.height_support)

        # STEP 2: Apply contrast normalization on im2col output
        # Matches C++ contrastNorm: row-wise normalization on patches
        # Each row is one patch, skip first column (bias)
        normalized = contrast_norm(input_col)

        # STEP 3: Forward pass through neural network layers
        # Matches C++ Response/ResponseInternal
        layer_output = normalized
        for i in range(len(self.weights)):
            # Linear: output = input * weight^T + bias
            layer_output = layer_output @ self.weights[i].T + self.biases[i]

            # Apply activation function
            if self.activation_function[i] == 0:
                # Sigmoid with numerical stability clamping
                # Prevents overflow in exp(-x) for large positive x
                layer_output = np.clip(layer_output, -88, 88)
                layer_output = 1.0 / (1.0 + np.exp(-layer_output))
            elif self.activation_function[i] == 1:
                # Tanh
                layer_output = np.tanh(layer_output)
            elif self.activation_function[i] == 2:
                # ReLU
                layer_output = np.maximum(0, layer_output)
            # else: linear (no activation)

        # STEP 4: Reshape output to 2D response map
        # layer_output shape: (num_patches,) where num_patches = response_height * response_width
        # Flatten to 1D if needed
        layer_output_flat = layer_output.flatten()

        # Reshape in column-major (Fortran) order to match im2col_bias output
        # im2col_bias generates patches in column-major order (x outer loop, y inner loop)
        response = layer_output_flat.reshape(response_height, response_width, order='F')

        return response.astype(np.float32, copy=False)

    def response_sparse(self, area_of_interest_left, area_of_interest_right):
        """
        Compute response maps for both left and right landmarks together.

        Matches C++ CEN_patch_expert::ResponseSparse() exactly:
        1. Flip right AOI horizontally
        2. SPARSE im2col with contrast norm (skips every other patch)
        3. Concatenate and process through neural network together
        4. Apply interpolation matrix to fill in skipped positions
        5. Split results and flip right response back

        This is how C++ processes frontal faces - both symmetric landmarks
        are computed together in a single neural network pass. C++ uses
        sparse computation for speed, computing only ~50% of patches and
        interpolating the rest from neighbors.

        Args:
            area_of_interest_left: Left landmark's AOI (H, W) or None
            area_of_interest_right: Right landmark's AOI (H, W) or None

        Returns:
            response_left: Left response map (or None if left not provided)
            response_right: Right response map (or None if right not provided)
        """
        if self.is_empty:
            # Return zero responses for empty expert
            response_left = None
            response_right = None
            if area_of_interest_left is not None:
                h = area_of_interest_left.shape[0] - self.height_support + 1
                w = area_of_interest_left.shape[1] - self.width_support + 1
                response_left = np.zeros((h, w), dtype=np.float32)
            if area_of_interest_right is not None:
                h = area_of_interest_right.shape[0] - self.height_support + 1
                w = area_of_interest_right.shape[1] - self.width_support + 1
                response_right = np.zeros((h, w), dtype=np.float32)
            return response_left, response_right

        left_provided = area_of_interest_left is not None
        right_provided = area_of_interest_right is not None

        response_height = 0
        input_height = 0
        input_width = 0
        im2col_left = None
        im2col_right = None

        # Process right AOI: flip horizontally BEFORE sparse im2col (like C++ line 555)
        if right_provided:
            aoi_right_flipped = cv2.flip(area_of_interest_right.astype(np.float32, copy=False), 1)
            response_height = aoi_right_flipped.shape[0] - self.height_support + 1
            input_height = aoi_right_flipped.shape[0]
            input_width = aoi_right_flipped.shape[1]
            # Use sparse im2col with contrast norm (like C++ im2colBiasSparseContrastNorm)
            im2col_right = im2col_bias_sparse_contrast_norm(
                aoi_right_flipped, self.width_support, self.height_support
            )

        # Process left AOI (no flip)
        if left_provided:
            response_height = area_of_interest_left.shape[0] - self.height_support + 1
            input_height = area_of_interest_left.shape[0]
            input_width = area_of_interest_left.shape[1]
            # Use sparse im2col with contrast norm (like C++ im2colBiasSparseContrastNorm)
            im2col_left = im2col_bias_sparse_contrast_norm(
                area_of_interest_left.astype(np.float32, copy=False),
                self.width_support, self.height_support
            )

        # Concatenate and process together (like C++ line 570)
        if left_provided and right_provided:
            combined = np.vstack([im2col_left, im2col_right])
        elif left_provided:
            combined = im2col_left
        elif right_provided:
            combined = im2col_right
        else:
            return None, None

        # Forward pass through neural network (ResponseInternal)
        layer_output = combined
        for i in range(len(self.weights)):
            layer_output = layer_output @ self.weights[i].T + self.biases[i]

            if self.activation_function[i] == 0:
                layer_output = np.clip(layer_output, -88, 88)
                layer_output = 1.0 / (1.0 + np.exp(-layer_output))
            elif self.activation_function[i] == 1:
                layer_output = np.tanh(layer_output)
            elif self.activation_function[i] == 2:
                layer_output = np.maximum(0, layer_output)

        # Get interpolation matrix (cached)
        response_width = response_height  # Square response for CEN
        interp_matrix = get_interpolation_matrix(
            response_height, response_width, input_height, input_width
        )

        # Split and apply interpolation (like C++ lines 598-616)
        response_left = None
        response_right = None
        num_sparse = im2col_left.shape[0] if left_provided else im2col_right.shape[0]

        if left_provided and right_provided:
            # Split in half (first half = left, second half = right)
            resp_left_sparse = layer_output[:num_sparse].flatten()  # (num_sparse,)
            resp_right_sparse = layer_output[num_sparse:].flatten()

            # Apply interpolation: response = sparse @ interp_matrix
            # interp_matrix is (num_sparse, response_height * response_width)
            # resp_sparse is (num_sparse,) -> need (1, num_sparse) for matmul
            resp_left_full = resp_left_sparse @ interp_matrix  # (response_h * response_w,)
            resp_right_full = resp_right_sparse @ interp_matrix

            # Reshape with C++ ordering: t().reshape(1, response_height).t()
            # This is equivalent to reshape in column-major then transpose
            response_left = resp_left_full.reshape(response_height, response_width, order='F')
            response_right = resp_right_full.reshape(response_height, response_width, order='F')

        elif left_provided:
            resp_sparse = layer_output.flatten()
            resp_full = resp_sparse @ interp_matrix
            response_left = resp_full.reshape(response_height, response_width, order='F')

        elif right_provided:
            resp_sparse = layer_output.flatten()
            resp_full = resp_sparse @ interp_matrix
            response_right = resp_full.reshape(response_height, response_width, order='F')

        # Flip right response back (like C++ line 615)
        if response_right is not None:
            response_right = cv2.flip(response_right, 1)

        return response_left.astype(np.float32, copy=False) if response_left is not None else None, \
               response_right.astype(np.float32, copy=False) if response_right is not None else None


class MirroredCENPatchExpert:
    """
    Wrapper for CEN patch expert that applies horizontal flipping for mirrored landmarks.

    This implements OpenFace's mirroring logic where symmetric landmarks (e.g., left eye)
    use the same neural network as their mirror (e.g., right eye) but with flipped patches.

    The process is:
    1. Flip input patch horizontally
    2. Run through the mirror's neural network
    3. Flip output response horizontally
    """

    def __init__(self, mirror_expert: CENPatchExpert):
        """
        Create a mirrored patch expert wrapper.

        Args:
            mirror_expert: The actual CEN expert to use (from the mirror landmark)
        """
        self._mirror_expert = mirror_expert
        self.is_empty = False  # This wrapper is not empty

    @property
    def width_support(self):
        return self._mirror_expert.width_support

    @property
    def height_support(self):
        return self._mirror_expert.height_support

    @property
    def width(self):
        return self._mirror_expert.width

    @property
    def height(self):
        return self._mirror_expert.height

    @property
    def confidence(self):
        return self._mirror_expert.confidence

    @property
    def patch_confidence(self):
        return self._mirror_expert.patch_confidence

    def response(self, area_of_interest):
        """
        Compute response with horizontal flipping for mirrored landmarks.

        Args:
            area_of_interest: Input patch (H, W)

        Returns:
            response: Response map with flipping applied
        """
        # Step 1: Flip input horizontally (cv2.flip with axis=1)
        flipped_input = cv2.flip(area_of_interest.astype(np.float32, copy=False), 1)

        # Step 2: Run through mirror expert's neural network
        flipped_response = self._mirror_expert.response(flipped_input)

        # Step 3: Flip output horizontally
        response = cv2.flip(flipped_response, 1)

        return response

    def compute_sigma(self, sigma_components, window_size=None, debug=False):
        """Forward sigma computation to mirror expert."""
        return self._mirror_expert.compute_sigma(sigma_components, window_size, debug)


class CENModel:
    """
    CEN model wrapper that matches CCNFModel interface for pyclnf integration.

    Provides scale_models dict structure expected by clnf.py.
    """

    def __init__(self, model_base_dir: str, scales: List[float] = None,
                 use_shared_memory: bool = False, shared_memory_dir: str = None):
        """
        Load CEN patch experts and wrap in CCNFModel-compatible interface.

        Args:
            model_base_dir: Base directory containing patch_experts/ folder with .dat files
            scales: List of scales to load (default: [0.25, 0.35, 0.5])
            use_shared_memory: If True, use memory-mapped shared models for HPC
            shared_memory_dir: Directory for shared memory files (default: /dev/shm/pyclnf_models)
        """
        self.model_base_dir = Path(model_base_dir)
        self.scales = scales or [0.25, 0.35, 0.5]
        self.use_shared_memory = use_shared_memory

        # Load CEN patch experts (regular or shared memory)
        if use_shared_memory:
            from .shared_model_loader import get_cen_patch_experts
            self.cen_experts = get_cen_patch_experts(
                model_base_dir,
                shm_dir=shared_memory_dir,
                use_shared=True
            )
        else:
            self.cen_experts = CENPatchExperts(model_base_dir)

        # Load sigma components for response normalization
        # (Both CEN and CCNF use sigma components in OpenFace)
        from ..models.openface_loader import load_sigma_components
        self.sigma_components = load_sigma_components(str(model_base_dir))
        if self.sigma_components is not None:
            print(f"Loaded CEN sigma components for window sizes: {list(self.sigma_components.keys())}")
        else:
            print("Warning: No sigma components found for CEN model")
            self.sigma_components = {}

        # Build scale_models dict matching CCNFModel interface:
        # scale_models[scale]['views'][view_idx]['patches'][landmark_idx] -> patch_expert
        self.scale_models = {}

        # Get mirror indices for creating mirrored patch experts
        mirror_inds = self.cen_experts.mirror_inds

        for scale_idx, scale in enumerate(self.cen_experts.patch_scaling):
            if scale not in self.scales:
                continue

            experts_at_scale = self.cen_experts.patch_experts[scale_idx]

            # Build patches dict: landmark_idx -> patch_expert
            # For empty landmarks, create MirroredCENPatchExpert using their mirror's expert
            patches = {}
            n_empty = 0
            n_mirrored = 0

            for lm_idx, expert in enumerate(experts_at_scale):
                if not expert.is_empty:
                    # Normal expert with weights
                    patches[lm_idx] = expert
                else:
                    # Empty expert - use mirrored version
                    n_empty += 1
                    mirror_idx = mirror_inds[lm_idx]
                    mirror_expert = experts_at_scale[mirror_idx]

                    if not mirror_expert.is_empty:
                        # Create mirrored expert wrapper
                        patches[lm_idx] = MirroredCENPatchExpert(mirror_expert)
                        n_mirrored += 1
                    # else: Both empty (shouldn't happen for valid landmarks)

            if n_mirrored > 0:
                print(f"    Scale {scale}: {n_empty} empty landmarks, {n_mirrored} using mirrored experts")

            # Store in CCNFModel-compatible structure (frontal view = 0)
            self.scale_models[scale] = {
                'views': {
                    0: {  # Frontal view
                        'patches': patches
                    }
                }
            }


class CENPatchExperts:
    """
    Collection of CEN patch experts for all landmarks and scales.

    Manages multi-scale patch experts from OpenFace 2.2 .dat files.
    """

    def __init__(self, model_dir):
        """
        Load CEN patch experts from model directory.

        Args:
            model_dir: Path to directory containing cen_patches_*.dat files
        """
        self.model_dir = Path(model_dir)
        self.patch_scaling = [0.25, 0.35, 0.50, 1.00]  # Scales available
        self.num_landmarks = 68

        # patch_experts[scale][landmark]
        self.patch_experts = []
        self.mirror_inds = None  # Will store mirror indices (same for all scales)

        # Check if models exist, download if needed
        self._ensure_models_exist()

        # Load all scale levels
        print(f"Loading CEN patch experts (410 MB, ~5-10 seconds)...")
        for idx, scale in enumerate(self.patch_scaling):
            scale_file = self.model_dir / f"cen_patches_{scale:.2f}_of.dat"
            if not scale_file.exists():
                # Try patch_experts subdirectory
                scale_file = self.model_dir / "patch_experts" / f"cen_patches_{scale:.2f}_of.dat"
            if not scale_file.exists():
                raise FileNotFoundError(
                    f"CEN model not found: {scale_file}\n"
                    "Run: python -m pyclnf.model_downloader"
                )

            print(f"  [{idx+1}/4] Loading scale {scale}...")
            experts_at_scale, mirror_inds = self._load_scale(scale_file)
            self.patch_experts.append(experts_at_scale)

            # Store mirror_inds (same for all scales)
            if self.mirror_inds is None:
                self.mirror_inds = mirror_inds.flatten().astype(int)

            print(f"      ✓ {len(experts_at_scale)} patch experts loaded")

    def _ensure_models_exist(self):
        """Check if model files exist, attempt download if not."""
        # Check for any scale file
        for scale in self.patch_scaling:
            scale_file = self.model_dir / f"cen_patches_{scale:.2f}_of.dat"
            if scale_file.exists():
                return  # At least one file exists in model_dir
            scale_file = self.model_dir / "patch_experts" / f"cen_patches_{scale:.2f}_of.dat"
            if scale_file.exists():
                return  # At least one file exists in patch_experts subdir

        # No models found - try to download
        print("CEN patch expert models not found. Attempting download...")
        try:
            from ..model_downloader import download_models
            if not download_models():
                print("Model download failed. Please download manually from:")
                print("  https://github.com/johnwilsoniv/pyclnf/releases")
        except Exception as e:
            print(f"Could not download models: {e}")
            print("Please download manually from:")
            print("  https://github.com/johnwilsoniv/pyclnf/releases")

    def _load_scale(self, dat_file):
        """
        Load all patch experts at one scale level.

        Args:
            dat_file: Path to .dat file

        Returns:
            List of CENPatchExpert instances (frontal view only)
        """
        with open(dat_file, 'rb') as f:
            # File structure (confirmed from OpenFace C++ source):
            # 1. Header: patch_scale (double) + num_views (int)
            # 2. View centers: num_views × (x, y, z) as doubles
            # 3. Visibility matrices: num_views × cv::Mat<int>
            # 4. Patch experts: num_views × num_landmarks × CEN_patch_expert

            # Read file header
            patch_scale = struct.unpack('d', f.read(8))[0]
            num_views = struct.unpack('i', f.read(4))[0]

            # Read view centers (3D orientation for each view)
            # Each view center is immediately followed by an empty matrix
            view_centers = []
            for _ in range(num_views):
                x = struct.unpack('d', f.read(8))[0]
                y = struct.unpack('d', f.read(8))[0]
                z = struct.unpack('d', f.read(8))[0]
                view_centers.append((x, y, z))

                # Read empty matrix immediately after this view center
                empty_mat = read_mat_bin(f)  # Should be 0×0 matrix

            # Read visibility matrices (one per view)
            # These are cv::Mat<int> storing which landmarks are visible in each view
            visibilities = []
            for _ in range(num_views):
                vis_mat = read_mat_bin(f)
                visibilities.append(vis_mat)

            # Read mirror metadata (for facial symmetry)
            mirror_inds = read_mat_bin(f)  # 1×68 matrix
            mirror_views = read_mat_bin(f)  # 1×7 matrix

            # Now read patch experts for all views
            all_experts = []
            for view_idx in range(num_views):
                experts_for_view = []
                for lm_idx in range(self.num_landmarks):
                    try:
                        expert = CENPatchExpert.from_stream(f)
                        experts_for_view.append(expert)
                    except Exception as e:
                        print(f"Error loading expert (view={view_idx}, landmark={lm_idx}): {e}")
                        raise
                all_experts.append(experts_for_view)

        # Return only frontal view (view 0) for now
        # TODO: Add multi-view support for profile faces
        return all_experts[0], mirror_inds

    def response(self, image, landmarks, scale_idx):
        """
        Compute patch expert responses for all landmarks at given scale.

        Args:
            image: Grayscale image (H, W) as float32 [0, 255]
            landmarks: Current landmark positions (68, 2) as float32
            scale_idx: Scale index (0-3)

        Returns:
            responses: List of 68 response maps (one per landmark)
            extraction_bounds: List of 68 tuples (x1, y1, x2, y2) with actual extraction bounds
        """
        if scale_idx < 0 or scale_idx >= len(self.patch_experts):
            raise ValueError(f"Invalid scale index: {scale_idx}")

        experts_at_scale = self.patch_experts[scale_idx]
        responses = []
        extraction_bounds = []

        # For each landmark, extract patch and compute response
        for lm_idx in range(self.num_landmarks):
            expert = experts_at_scale[lm_idx]

            # Extract a SEARCH AREA around the landmark
            # The search area should be larger than the support window to allow
            # the patch expert to evaluate multiple positions
            # Search radius: 2.0x support (OpenFace default for robust refinement)
            search_radius = int(max(expert.width_support, expert.height_support) * 2.0)

            lm_x, lm_y = landmarks[lm_idx]
            x1 = max(0, int(lm_x - search_radius))
            y1 = max(0, int(lm_y - search_radius))
            x2 = min(image.shape[1], int(lm_x + search_radius))
            y2 = min(image.shape[0], int(lm_y + search_radius))

            # Extract and compute response
            patch = image[y1:y2, x1:x2]
            if patch.size > 0 and patch.shape[0] > expert.height_support and patch.shape[1] > expert.width_support:
                response = expert.response(patch)
            else:
                response = np.zeros((1, 1), dtype=np.float32)

            responses.append(response)
            extraction_bounds.append((x1, y1, x2, y2))

        return responses, extraction_bounds


def read_mat_bin(stream):
    """
    Read OpenCV matrix from binary stream (OpenFace format).

    Args:
        stream: Binary file stream

    Returns:
        numpy array with matrix data
    """
    # Read dimensions and type
    rows = struct.unpack('i', stream.read(4))[0]
    cols = struct.unpack('i', stream.read(4))[0]
    cv_type = struct.unpack('i', stream.read(4))[0]

    # Handle empty matrices (0×0 or 0×N or N×0)
    if rows == 0 or cols == 0:
        return np.array([], dtype=np.float32).reshape(rows, cols) if rows >= 0 and cols >= 0 else np.array([])

    # Map OpenCV type to numpy dtype
    # OpenCV type codes: CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3,
    #                    CV_32S=4, CV_32F=5, CV_64F=6
    if cv_type == 0:  # CV_8U
        dtype = np.uint8
    elif cv_type == 1:  # CV_8S
        dtype = np.int8
    elif cv_type == 2:  # CV_16U
        dtype = np.uint16
    elif cv_type == 3:  # CV_16S
        dtype = np.int16
    elif cv_type == 4:  # CV_32S
        dtype = np.int32
    elif cv_type == 5:  # CV_32F
        dtype = np.float32
    elif cv_type == 6:  # CV_64F
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported OpenCV matrix type: {cv_type} (rows={rows}, cols={cols})")

    # Read data
    size = rows * cols
    data = np.frombuffer(stream.read(size * np.dtype(dtype).itemsize), dtype=dtype)

    # Reshape to matrix (OpenCV uses row-major order like NumPy)
    matrix = data.reshape(rows, cols)

    # For weight/bias matrices, convert to float32; for visibility, keep as-is
    if cv_type in [5, 6]:  # Float types
        return matrix.astype(np.float32)
    else:  # Integer types
        return matrix


@njit(fastmath=True, cache=True)
def _contrast_norm_numba(input_patch, output):
    """
    Numba-optimized contrast normalization (5-10x faster).

    Matches C++ contrastNorm: norm = sqrt(sum((x - mean)²))
    """
    for y in range(input_patch.shape[0]):
        # Skip first column (bias), compute mean of rest
        row_sum = 0.0
        cols = input_patch.shape[1] - 1
        for x in range(1, input_patch.shape[1]):
            row_sum += input_patch[y, x]
        mean = row_sum / cols

        # Compute L2 norm: sqrt(sum((x - mean)²))
        # NOT standard deviation! C++ doesn't divide by n
        sum_sq = 0.0
        for x in range(1, input_patch.shape[1]):
            diff = input_patch[y, x] - mean
            sum_sq += diff * diff

        norm = np.sqrt(sum_sq)  # L2 norm, no division by n
        if norm == 0:  # C++ uses exact 0 comparison
            norm = 1.0

        # Normalize (skip first column which is bias)
        output[y, 0] = input_patch[y, 0]  # Keep first column (bias=1.0)
        for x in range(1, input_patch.shape[1]):
            output[y, x] = (input_patch[y, x] - mean) / norm

    return output


@njit(fastmath=True, cache=True)
def _im2col_bias_numba(input_patch, width, height, output):
    """
    Numba-optimized im2col with bias column.

    Extracts sliding window patches from input image in column-major order.
    This is a hot path called 16,320 times - optimized for nopython mode.
    """
    m = input_patch.shape[0]
    n = input_patch.shape[1]
    y_blocks = m - height + 1
    x_blocks = n - width + 1

    # Match C++ im2colBias loop structure EXACTLY:
    # Column-major patch ordering (x outer, y inner)
    for j in range(x_blocks):  # X positions (columns)
        for i in range(y_blocks):  # Y positions (rows)
            row_idx = i + j * y_blocks  # Column-major patch ordering

            # Extract patch at (i, j) position in column-major order
            # colIdx = xx*height + yy (column-major within patch)
            col_idx = 1  # Skip bias column
            for xx in range(width):
                for yy in range(height):
                    output[row_idx, col_idx] = input_patch[i + yy, j + xx]
                    col_idx += 1

    return output


@njit(fastmath=True, cache=True)
def _forward_pass_numba(layer_input, weights_T, bias, activation_type):
    """
    Numba-optimized single layer forward pass.

    Computes: output = input @ weights^T + bias, then applies activation.

    Args:
        layer_input: Input matrix (num_patches, input_dim)
        weights_T: Transposed weight matrix (input_dim, output_dim)
        bias: Bias vector (1, output_dim)
        activation_type: 0=sigmoid, 1=tanh, 2=relu, other=linear

    Returns:
        Output after activation (num_patches, output_dim)
    """
    num_patches = layer_input.shape[0]
    output_dim = weights_T.shape[1]

    # Allocate output
    output = np.empty((num_patches, output_dim), dtype=np.float32)

    # Matrix multiplication: output = input @ weights_T + bias
    for i in range(num_patches):
        for j in range(output_dim):
            acc = bias[0, j]  # Start with bias
            for k in range(layer_input.shape[1]):
                acc += layer_input[i, k] * weights_T[k, j]

            # Apply activation function
            if activation_type == 0:
                # Sigmoid with clamping
                if acc < -88.0:
                    acc = -88.0
                elif acc > 88.0:
                    acc = 88.0
                output[i, j] = 1.0 / (1.0 + np.exp(-acc))
            elif activation_type == 1:
                # Tanh
                output[i, j] = np.tanh(acc)
            elif activation_type == 2:
                # ReLU
                output[i, j] = max(0.0, acc)
            else:
                # Linear
                output[i, j] = acc

    return output


@njit(fastmath=True, cache=True, parallel=False)
def _response_core_numba(input_patch, width, height,
                         w0, b0, a0, w1, b1, a1,
                         response_height, response_width):
    """
    Numba-optimized complete response computation for 2-layer networks.

    Combines im2col, contrast normalization, and neural network forward pass
    into a single optimized function to minimize memory allocations.

    This is the hottest path - called 16,320 times per frame.
    """
    m = input_patch.shape[0]
    n = input_patch.shape[1]
    y_blocks = m - height + 1
    x_blocks = n - width + 1
    num_windows = y_blocks * x_blocks
    patch_size = height * width

    # STEP 1 & 2: im2col + contrast normalization combined
    # Allocate normalized output with bias column
    normalized = np.ones((num_windows, patch_size + 1), dtype=np.float32)

    for j in range(x_blocks):
        for i in range(y_blocks):
            row_idx = i + j * y_blocks

            # Extract patch and compute stats in one pass
            patch_sum = 0.0
            col_idx = 1

            # First pass: extract and compute mean
            for xx in range(width):
                for yy in range(height):
                    val = input_patch[i + yy, j + xx]
                    normalized[row_idx, col_idx] = val
                    patch_sum += val
                    col_idx += 1

            mean = patch_sum / patch_size

            # Second pass: compute norm
            sum_sq = 0.0
            for k in range(1, patch_size + 1):
                diff = normalized[row_idx, k] - mean
                sum_sq += diff * diff

            norm = np.sqrt(sum_sq)
            if norm == 0:  # C++ uses exact 0 comparison
                norm = 1.0

            # Third pass: normalize
            for k in range(1, patch_size + 1):
                normalized[row_idx, k] = (normalized[row_idx, k] - mean) / norm

    # STEP 3: Forward pass through layers
    # Layer 0
    layer0_out_dim = w0.shape[0]
    layer0_output = np.empty((num_windows, layer0_out_dim), dtype=np.float32)

    for i in range(num_windows):
        for j in range(layer0_out_dim):
            acc = b0[0, j]
            for k in range(patch_size + 1):
                acc += normalized[i, k] * w0[j, k]

            # Apply activation
            if a0 == 0:  # Sigmoid
                if acc < -88.0:
                    acc = -88.0
                elif acc > 88.0:
                    acc = 88.0
                layer0_output[i, j] = 1.0 / (1.0 + np.exp(-acc))
            elif a0 == 1:  # Tanh
                layer0_output[i, j] = np.tanh(acc)
            elif a0 == 2:  # ReLU
                layer0_output[i, j] = max(0.0, acc)
            else:
                layer0_output[i, j] = acc

    # Layer 1
    layer1_out_dim = w1.shape[0]
    layer1_output = np.empty((num_windows, layer1_out_dim), dtype=np.float32)

    for i in range(num_windows):
        for j in range(layer1_out_dim):
            acc = b1[0, j]
            for k in range(layer0_out_dim):
                acc += layer0_output[i, k] * w1[j, k]

            # Apply activation
            if a1 == 0:  # Sigmoid
                if acc < -88.0:
                    acc = -88.0
                elif acc > 88.0:
                    acc = 88.0
                layer1_output[i, j] = 1.0 / (1.0 + np.exp(-acc))
            elif a1 == 1:  # Tanh
                layer1_output[i, j] = np.tanh(acc)
            elif a1 == 2:  # ReLU
                layer1_output[i, j] = max(0.0, acc)
            else:
                layer1_output[i, j] = acc

    # STEP 4: Reshape to response map (column-major order)
    response = np.empty((response_height, response_width), dtype=np.float32)
    for j in range(response_width):
        for i in range(response_height):
            flat_idx = i + j * response_height
            response[i, j] = layer1_output[flat_idx, 0]

    return response


def contrast_norm(input_patch):
    """
    Apply row-wise contrast normalization.

    Matches C++ contrastNorm EXACTLY:
    - norm = sqrt(sum((x - mean)²)) [L2 norm, NOT standard deviation!]

    Uses Numba JIT if available for 5-10x speedup.

    Args:
        input_patch: Image patch (H, W) as float32

    Returns:
        normalized: Contrast-normalized patch
    """
    output = np.empty_like(input_patch, dtype=np.float32)

    if NUMBA_AVAILABLE:
        return _contrast_norm_numba(input_patch, output)
    else:
        # Fallback: NumPy vectorized version
        output = input_patch.copy()
        for y in range(input_patch.shape[0]):
            row = input_patch[y, 1:]  # Skip first column (bias)
            mean = np.mean(row)

            # C++ uses L2 norm: sqrt(sum((x - mean)²))
            # NOT standard deviation which divides by n!
            sum_sq = np.sum((row - mean) ** 2)
            norm = np.sqrt(sum_sq)

            if norm == 0:  # C++ uses exact 0 comparison
                norm = 1.0

            output[y, 1:] = (row - mean) / norm

        return output


def im2col_bias(input_patch, width, height):
    """
    Convert image to column format with bias for convolutional processing.

    Matches C++ im2colBias EXACTLY:
    - Column-major ordering of patches (x outer loop, y inner loop)
    - Column-major ordering within patches (xx*height + yy)

    Uses Numba JIT if available for significant speedup.

    Args:
        input_patch: Image patch (m, n) as float32
        width: Sliding window width
        height: Sliding window height

    Returns:
        output: Matrix (num_windows, width*height+1) with bias column
    """
    m, n = input_patch.shape
    y_blocks = m - height + 1  # yB in C++
    x_blocks = n - width + 1   # xB in C++
    num_windows = y_blocks * x_blocks

    # Allocate output with bias column
    output = np.ones((num_windows, height * width + 1), dtype=np.float32)

    if NUMBA_AVAILABLE:
        # Use Numba-optimized version
        input_contiguous = np.ascontiguousarray(input_patch, dtype=np.float32)
        return _im2col_bias_numba(input_contiguous, width, height, output)

    # Fallback: NumPy version
    # Match C++ im2colBias loop structure EXACTLY:
    # for (j = 0; j < xB; j++)                     // Outer loop: X
    #   for (i = 0; i < yB; i++)                   // Inner loop: Y
    #     rowIdx = i + j*yB                        // Column-major patch ordering
    #     for (yy = 0; yy < height; yy++)
    #       for (xx = 0; xx < width; xx++)
    #         colIdx = xx*height + yy              // Column-major within patch
    #         output[rowIdx, colIdx+1] = input[i+yy, j+xx]

    for j in range(x_blocks):  # X positions (columns)
        for i in range(y_blocks):  # Y positions (rows)
            row_idx = i + j * y_blocks  # Column-major patch ordering

            # Extract patch at (i, j) position
            patch = input_patch[i:i+height, j:j+width]

            # Flatten patch in column-major order (xx*height + yy)
            # This is Fortran order (column-major)
            patch_flat = patch.T.flatten()  # Transpose then flatten = column-major

            # Store in output (skip first column which is bias=1.0)
            output[row_idx, 1:] = patch_flat

    return output


# Global cache for interpolation matrices (same for all experts at same scale)
_INTERPOLATION_MATRIX_CACHE = {}


def get_interpolation_matrix(response_height, response_width, input_height, input_width):
    """
    Get or create interpolation matrix for sparse-to-full response mapping.

    Matches C++ interpolationMatrix() exactly.

    Caches matrices since they only depend on dimensions (not landmark or image).

    Args:
        response_height: Height of response map
        response_width: Width of response map
        input_height: Height of input area of interest
        input_width: Width of input area of interest

    Returns:
        map_matrix: (num_sparse, response_height * response_width) interpolation matrix
    """
    cache_key = (response_height, response_width, input_height, input_width)
    if cache_key in _INTERPOLATION_MATRIX_CACHE:
        return _INTERPOLATION_MATRIX_CACHE[cache_key]

    m = input_height
    n = input_width

    # C++ hardcodes 11x11 patch size (width_support/height_support)
    yB = m - 11 + 1
    xB = n - 11 + 1

    # Number of computed (non-skipped) sparse outputs
    out_size = (yB * xB - 1) // 2

    # Create map matrix: out_size x (response_height * response_width)
    map_matrix = np.zeros((out_size, response_height * response_width), dtype=np.float32)

    # Create value_id_matrix that maps sparse output indices
    # C++ assigns index to positions where k % 2 != 0
    value_id_matrix = np.zeros((response_width, response_height), dtype=np.int32)

    ind = 0
    for k in range(response_width * response_height):
        if k % 2 != 0:
            value_id_matrix.flat[k] = ind
            ind += 1

    # C++ transposes the value_id_matrix
    value_id_matrix = value_id_matrix.T

    # Fill the map_matrix with interpolation weights
    skip_counter = 0
    for x in range(response_width):
        for y in range(response_height):
            mapping_col = x * response_height + y
            skip_counter += 1

            if skip_counter % 2 == 0:
                # This position was computed directly (weight = 1)
                val_id = value_id_matrix[y, x]
                map_matrix[val_id, mapping_col] = 1.0
            else:
                # This position was skipped - average from neighbors
                num_neigh = 0.0
                val_ids = []

                if x - 1 >= 0:
                    num_neigh += 1
                    val_ids.append(value_id_matrix[y, x - 1])
                if y - 1 >= 0:
                    num_neigh += 1
                    val_ids.append(value_id_matrix[y - 1, x])
                if x + 1 < response_width:
                    num_neigh += 1
                    val_ids.append(value_id_matrix[y, x + 1])
                if y + 1 < response_height:
                    num_neigh += 1
                    val_ids.append(value_id_matrix[y + 1, x])

                for val_id in val_ids:
                    map_matrix[val_id, mapping_col] = 1.0 / num_neigh

    _INTERPOLATION_MATRIX_CACHE[cache_key] = map_matrix
    return map_matrix


def im2col_bias_sparse_contrast_norm(input_patch, width, height):
    """
    Sparse im2col with contrast normalization.

    Matches C++ im2colBiasSparseContrastNorm() exactly:
    - Skips every other patch in the iteration order
    - Applies contrast normalization inline

    Args:
        input_patch: Image patch (m, n) as float32
        width: Sliding window width (typically 11)
        height: Sliding window height (typically 11)

    Returns:
        output: Matrix (num_sparse, width*height+1) normalized patches with bias
    """
    m = input_patch.shape[0]
    n = input_patch.shape[1]

    yB = m - height + 1
    xB = n - width + 1

    out_size = (yB * xB - 1) // 2
    patch_size = height * width

    output = np.ones((out_size, patch_size + 1), dtype=np.float32)

    row_idx = 0
    skip_counter = 0

    for j in range(xB):
        for i in range(yB):
            skip_counter += 1
            if (skip_counter + 1) % 2 == 0:  # Skip when (skipCounter + 1) % 2 == 0
                continue

            # Extract patch values in column-major order (xx outer, yy inner)
            patch_sum = 0.0
            for yy in range(height):
                for xx in range(width):
                    col_idx = xx * height + yy + 1  # +1 for bias column
                    val = input_patch[i + yy, j + xx]
                    output[row_idx, col_idx] = val
                    patch_sum += val

            # Compute mean
            mean = patch_sum / patch_size

            # Compute L2 norm of (patch - mean)
            sum_sq = 0.0
            for k in range(1, patch_size + 1):
                diff = output[row_idx, k] - mean
                output[row_idx, k] = diff
                sum_sq += diff * diff

            norm = np.sqrt(sum_sq)
            if norm == 0:
                norm = 1.0

            # Normalize
            for k in range(1, patch_size + 1):
                output[row_idx, k] /= norm

            row_idx += 1

    return output
