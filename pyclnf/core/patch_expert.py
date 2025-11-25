"""
LNF Patch Expert - Response map computation for CLNF

Implements the Linear Neural Field patch expert from OpenFace CCNF model.

The patch expert computes a response map R(x, y) for each landmark that indicates
the likelihood of the landmark being at position (x, y) in the image patch.

Response computation:
    R(x, y) = Σ_k β_k * σ(α_k * ||w_k^T * f(x, y) + b_k||)

Where:
    - f(x, y): Image features at position (x, y)
    - w_k: Weight vector for neuron k
    - b_k: Bias for neuron k
    - α_k: Scaling factor for neuron k
    - β_k: Linear combination weight for neuron k
    - σ: Sigmoid activation function
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import cv2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.openface_loader import load_sigma_components


class CCNFPatchExpert:
    """CCNF patch expert for a single landmark at a specific scale."""

    def __init__(self, patch_dir: str):
        """
        Load CCNF patch expert from exported NumPy files.

        Args:
            patch_dir: Directory containing metadata.npz and neuron_*.npz files
        """
        self.patch_dir = Path(patch_dir)

        # Load patch metadata
        meta = np.load(self.patch_dir / 'metadata.npz')
        self.width = int(meta['width'])
        self.height = int(meta['height'])
        self.betas = meta['betas']
        self.patch_confidence = float(meta['patch_confidence'])

        # Count neuron files to determine num_neurons
        neuron_files = sorted(self.patch_dir.glob('neuron_*.npz'))
        self.num_neurons = len(neuron_files)

        # Load neurons
        self.neurons = []
        for neuron_file in neuron_files:
            neuron_data = np.load(neuron_file)

            neuron = {
                'type': int(neuron_data['neuron_type']),
                'weights': neuron_data['weights'],
                'bias': float(neuron_data['bias']),
                'alpha': float(neuron_data['alpha']),
                'norm_weights': float(neuron_data['norm_weights'])
            }
            self.neurons.append(neuron)

    def compute_response(self, image_patch: np.ndarray) -> float:
        """
        Compute response (confidence) for this patch expert.

        The response indicates how likely the landmark is at the CENTER
        of this image patch.

        Matches OpenFace C++ implementation:
        - Sum ALL neuron responses (skip neurons with alpha < 1e-4)
        - Do NOT group by sigma or weight by betas
        - Betas are used for edge features, not basic response computation

        Args:
            image_patch: Grayscale image patch, shape (height, width)

        Returns:
            response: Scalar confidence value
        """
        # Ensure patch is correct size
        if image_patch.shape != (self.height, self.width):
            image_patch = cv2.resize(image_patch, (self.width, self.height))

        # Extract features from patch (gradient magnitude)
        features = self._extract_features(image_patch)

        # Sum ALL neuron responses (OpenFace algorithm)
        total_response = 0.0

        for neuron in self.neurons:
            # Skip neurons with very small alpha (OpenFace does this for efficiency)
            if abs(neuron['alpha']) < 1e-4:
                continue

            neuron_response = self._compute_neuron_response(features, neuron)
            total_response += neuron_response

        return float(total_response)

    def _extract_features(self, image_patch: np.ndarray) -> np.ndarray:
        """
        Extract features from image patch.

        OpenFace CCNF uses RAW image intensity directly, NOT gradient features!
        See CCNF_patch_expert.cpp lines 247-256: if(neuron_type == 0) I = im;

        Args:
            image_patch: Grayscale image patch, shape (height, width), values 0-255

        Returns:
            features: Normalized image intensities, shape (height, width), float32
        """
        # Convert to float32 in range [0, 1]
        # This matches OpenFace's input to matchTemplate
        patch_float = image_patch.astype(np.float32) / 255.0

        return patch_float

    def _compute_neuron_response(self, features: np.ndarray, neuron: dict) -> np.ndarray:
        """
        Compute response for a single neuron using normalized cross-correlation.

        Matches OpenFace C++ implementation:
            response = (2 * alpha) * sigmoid(correlation * norm_weights + bias)

        Where correlation is computed using TM_CCOEFF_NORMED (normalized cross-correlation).

        Args:
            features: Feature map (gradient magnitude), shape (height, width)
            neuron: Neuron parameters (weights, bias, alpha, norm_weights)

        Returns:
            response: Single scalar response value
        """
        weights = neuron['weights']  # Shape: (height, width)
        bias = neuron['bias']
        alpha = neuron['alpha']
        norm_weights = neuron['norm_weights']

        # Ensure features and weights are same size
        if features.shape != weights.shape:
            features = cv2.resize(features, (weights.shape[1], weights.shape[0]))

        # Compute normalized cross-correlation using OpenCV's matchTemplate
        # TM_CCOEFF_NORMED: (T - mean(T)) · (I - mean(I)) / (||T - mean(T)|| * ||I - mean(I)||)
        # For a single patch, we can compute this directly

        # Compute means
        weight_mean = np.mean(weights)
        feature_mean = np.mean(features)

        # Center the data
        weights_centered = weights - weight_mean
        features_centered = features - feature_mean

        # Compute norms
        weight_norm = np.linalg.norm(weights_centered)
        feature_norm = np.linalg.norm(features_centered)

        # Compute normalized cross-correlation
        if weight_norm > 1e-10 and feature_norm > 1e-10:
            correlation = np.sum(weights_centered * features_centered) / (weight_norm * feature_norm)
        else:
            correlation = 0.0

        # Apply OpenFace formula: (2 * alpha) * sigmoid(correlation * norm_weights + bias)
        sigmoid_input = correlation * norm_weights + bias
        response = (2.0 * alpha) * self._sigmoid(sigmoid_input)

        return response

    def _sigmoid(self, x):
        """Numerically stable sigmoid function."""
        if isinstance(x, np.ndarray):
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )
        else:
            # Scalar version
            if x >= 0:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(x) / (1 + np.exp(x))

    def compute_sigma(self, sigma_components: List[np.ndarray], window_size: int = None, debug: bool = False) -> np.ndarray:
        """
        Compute Sigma covariance matrix for spatial correlation modeling.

        Based on OpenFace CCNF_patch_expert.cpp lines 81-117:
            sum_alphas = Σ(neuron.alpha)
            q1 = sum_alphas * Identity(window_size²)
            q2 = Σ(beta_i * sigma_component_i)
            SigmaInv = 2 * (q1 + q2)
            Sigma = inv(SigmaInv) using Cholesky decomposition

        The Sigma matrix models spatial correlations in the response map,
        transforming raw responses to account for local dependencies.

        Args:
            sigma_components: List of sigma component matrices for this window size
                             (loaded from exported CCNF model files)
            window_size: Response map window size (if None, uses patch width)
            debug: Print detailed debugging information

        Returns:
            Sigma matrix (window_size² × window_size²) for transforming response maps
        """
        # Calculate sum of alphas across all neurons
        sum_alphas = sum(neuron['alpha'] for neuron in self.neurons)

        # Get window size from parameter or patch dimensions
        if window_size is None:
            window_size = self.width
        matrix_size = window_size * window_size

        if debug:
            print(f"\n    [Sigma Debug] window_size={window_size}, matrix_size={matrix_size}")
            print(f"    [Sigma Debug] sum_alphas={sum_alphas:.6f}")
            print(f"    [Sigma Debug] num_neurons={len(self.neurons)}")
            print(f"    [Sigma Debug] num_betas={len(self.betas)}")
            print(f"    [Sigma Debug] num_sigma_components={len(sigma_components)}")

            # Verify sigma component shapes
            for i, sc in enumerate(sigma_components):
                expected_shape = (matrix_size, matrix_size)
                if sc.shape != expected_shape:
                    print(f"    [Sigma Debug] ⚠️  WARNING: sigma_component[{i}] shape {sc.shape} != expected {expected_shape}")
                else:
                    print(f"    [Sigma Debug] ✓ sigma_component[{i}] shape {sc.shape} correct")

        # q1 = sum_alphas * Identity
        q1 = sum_alphas * np.eye(matrix_size, dtype=np.float32)

        # q2 = Σ(beta_i * sigma_component_i)
        # Only use as many betas as we have sigma components
        q2 = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        num_components = min(len(self.betas), len(sigma_components))

        # Cast betas to float32 for consistent precision (matches C++ OpenFace)
        betas_f32 = np.asarray(self.betas, dtype=np.float32)

        if debug:
            print(f"    [Sigma Debug] Using {num_components} components (min of {len(self.betas)} betas, {len(sigma_components)} sigma_comps)")

        for i in range(num_components):
            if debug and i < 3:  # Print first 3 betas
                print(f"    [Sigma Debug] beta[{i}]={betas_f32[i]:.6f}")
            # Ensure sigma_components are also float32 to avoid mixed precision
            sigma_comp_f32 = np.asarray(sigma_components[i], dtype=np.float32)
            q2 += betas_f32[i] * sigma_comp_f32

        # SigmaInv = 2 * (q1 + q2)
        SigmaInv = 2.0 * (q1 + q2)

        if debug:
            # Check SigmaInv properties
            det_SigmaInv = np.linalg.det(SigmaInv)
            cond_SigmaInv = np.linalg.cond(SigmaInv)
            print(f"    [Sigma Debug] SigmaInv: det={det_SigmaInv:.6e}, cond={cond_SigmaInv:.2e}")
            print(f"    [Sigma Debug] SigmaInv: min={SigmaInv.min():.6f}, max={SigmaInv.max():.6f}")

        # Compute Sigma = inv(SigmaInv) using Cholesky decomposition (OpenFace uses DECOMP_CHOLESKY)
        try:
            # Try Cholesky decomposition first (OpenFace method)
            Sigma = np.linalg.inv(SigmaInv)
            if debug:
                print(f"    [Sigma Debug] ✓ Cholesky inversion succeeded")
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            print(f"Warning: Singular SigmaInv matrix for patch {self.width}×{self.height}, using pseudo-inverse")
            Sigma = np.linalg.pinv(SigmaInv)

        if debug:
            # Check Sigma properties
            det_Sigma = np.linalg.det(Sigma)
            cond_Sigma = np.linalg.cond(Sigma)
            print(f"    [Sigma Debug] Sigma: det={det_Sigma:.6e}, cond={cond_Sigma:.2e}")
            print(f"    [Sigma Debug] Sigma: min={Sigma.min():.6f}, max={Sigma.max():.6f}")

            # Verify Sigma * SigmaInv ≈ Identity
            product = Sigma @ SigmaInv
            identity_error = np.abs(product - np.eye(matrix_size)).max()
            print(f"    [Sigma Debug] Sigma*SigmaInv identity error: {identity_error:.6e}")

        return Sigma

    def get_info(self) -> dict:
        """Get patch expert information."""
        return {
            'width': self.width,
            'height': self.height,
            'num_neurons': self.num_neurons,
            'patch_confidence': self.patch_confidence,
            'num_betas': len(self.betas)
        }


class CCNFModel:
    """
    Complete CCNF model with multi-view, multi-scale patch experts.

    Manages loading patch experts for:
    - Multiple views (face orientations: frontal, profile, etc.)
    - Multiple scales (0.25, 0.35, 0.5 × interocular distance)
    - 68 landmarks per view
    """

    def __init__(self, model_base_dir: str, scales: Optional[List[float]] = None):
        """
        Load CCNF model from exported NumPy files.

        Args:
            model_base_dir: Base directory containing exported_ccnf_* folders
            scales: List of scales to load (default: [0.25, 0.35, 0.5])
        """
        self.model_base_dir = Path(model_base_dir)
        self.scales = scales or [0.25, 0.35, 0.5]

        # Load sigma components for CCNF spatial correlation modeling
        self.sigma_components = load_sigma_components(str(self.model_base_dir))
        if self.sigma_components is not None:
            print(f"Loaded sigma components for window sizes: {list(self.sigma_components.keys())}")
        else:
            print("Warning: Sigma components not found - CCNF will run without spatial correlation modeling")

        # Load multi-scale models
        self.scale_models = {}
        for scale in self.scales:
            scale_dir = self.model_base_dir / f'exported_ccnf_{scale}'
            if scale_dir.exists():
                self.scale_models[scale] = self._load_scale_model(scale_dir)
            else:
                print(f"Warning: Scale {scale} model not found at {scale_dir}")

    def _load_scale_model(self, scale_dir: Path) -> dict:
        """
        Load all patch experts for one scale.

        Args:
            scale_dir: Directory for one scale (e.g., exported_ccnf_0.25/)

        Returns:
            Dictionary with structure: {view_idx: {landmark_idx: CCNFPatchExpert}}
        """
        # Load global metadata
        global_meta = np.load(scale_dir / 'global_metadata.npz')
        num_views = int(global_meta['num_views'])
        num_landmarks = int(global_meta['num_landmarks'])

        scale_model = {
            'num_views': num_views,
            'num_landmarks': num_landmarks,
            'patch_scaling': float(global_meta['patch_scaling']),
            'views': {}
        }

        # Load each view
        for view_idx in range(num_views):
            view_dir = scale_dir / f'view_{view_idx:02d}'
            if not view_dir.exists():
                continue

            # Load view metadata
            view_meta_path = scale_dir / f'view_{view_idx:02d}_metadata.npz'
            view_meta = np.load(view_meta_path)

            view_data = {
                'center': view_meta['center'],
                'visibility': view_meta['visibility'],
                'patches': {}
            }

            # Load patch experts for this view
            for landmark_idx in range(num_landmarks):
                patch_dir = view_dir / f'patch_{landmark_idx:02d}'

                # Check if patch exists (some landmarks not visible in some views)
                if patch_dir.exists() and (patch_dir / 'metadata.npz').exists():
                    try:
                        patch_expert = CCNFPatchExpert(str(patch_dir))
                        view_data['patches'][landmark_idx] = patch_expert
                    except Exception as e:
                        print(f"Warning: Failed to load patch {landmark_idx} in view {view_idx}: {e}")

            scale_model['views'][view_idx] = view_data

        return scale_model

    def get_patch_expert(self, scale: float, view_idx: int, landmark_idx: int) -> Optional[CCNFPatchExpert]:
        """
        Get patch expert for specific scale, view, and landmark.

        Args:
            scale: Patch scale (0.25, 0.35, or 0.5)
            view_idx: View index (0-6)
            landmark_idx: Landmark index (0-67)

        Returns:
            CCNFPatchExpert or None if not available
        """
        if scale not in self.scale_models:
            return None

        scale_model = self.scale_models[scale]
        if view_idx not in scale_model['views']:
            return None

        view_data = scale_model['views'][view_idx]
        return view_data['patches'].get(landmark_idx)

    def get_best_view(self, pose: np.ndarray) -> int:
        """
        Select best view based on head pose.

        Args:
            pose: Head pose [pitch, yaw, roll] in degrees

        Returns:
            Best view index
        """
        # Use first scale to get view centers
        if not self.scale_models:
            return 0

        first_scale = list(self.scale_models.values())[0]

        # Find view with closest center to current pose
        best_view = 0
        min_distance = float('inf')

        for view_idx, view_data in first_scale['views'].items():
            center = view_data['center'].flatten()

            # Compute distance (primarily based on yaw)
            distance = np.linalg.norm(center - pose)

            if distance < min_distance:
                min_distance = distance
                best_view = view_idx

        return best_view

    def get_info(self) -> dict:
        """Get CCNF model information."""
        info = {
            'scales': list(self.scale_models.keys()),
            'scale_models': {}
        }

        for scale, model in self.scale_models.items():
            num_patches = sum(
                len(view_data['patches'])
                for view_data in model['views'].values()
            )
            info['scale_models'][scale] = {
                'num_views': model['num_views'],
                'num_landmarks': model['num_landmarks'],
                'patch_scaling': model['patch_scaling'],
                'total_patches': num_patches
            }

        return info


def test_patch_expert():
    """Test patch expert loading and response computation."""
    print("=" * 60)
    print("Testing LNF Patch Expert Implementation")
    print("=" * 60)

    # Test 1: Load CCNF model
    print("\nTest 1: Load CCNF model")
    model_dir = "pyclnf/models"
    ccnf = CCNFModel(model_dir)

    info = ccnf.get_info()
    print(f"  Loaded scales: {info['scales']}")
    for scale, scale_info in info['scale_models'].items():
        print(f"  Scale {scale}:")
        print(f"    Views: {scale_info['num_views']}")
        print(f"    Landmarks: {scale_info['num_landmarks']}")
        print(f"    Total patches: {scale_info['total_patches']}")

    # Test 2: Get specific patch expert
    print("\nTest 2: Get specific patch expert")
    scale = 0.25
    view_idx = 0
    landmark_idx = 30  # Nose tip (typically visible in frontal view)

    patch_expert = ccnf.get_patch_expert(scale, view_idx, landmark_idx)
    if patch_expert:
        patch_info = patch_expert.get_info()
        print(f"  Patch expert for landmark {landmark_idx}, view {view_idx}, scale {scale}:")
        print(f"    Size: {patch_info['width']}×{patch_info['height']}")
        print(f"    Neurons: {patch_info['num_neurons']}")
        print(f"    Confidence: {patch_info['patch_confidence']:.3f}")

        # Test 3: Compute response on random patch
        print("\nTest 3: Compute response")
        test_patch = np.random.randint(0, 256, (patch_info['height'], patch_info['width']), dtype=np.uint8)
        response = patch_expert.compute_response(test_patch)
        print(f"    Input patch shape: {test_patch.shape}")
        print(f"    Response value: {response:.6f}")
        print(f"    Response type: {type(response)}")
    else:
        print(f"  Patch expert not found for landmark {landmark_idx}, view {view_idx}")

    # Test 4: View selection
    print("\nTest 4: View selection based on pose")
    test_poses = [
        np.array([0, 0, 0]),      # Frontal
        np.array([0, 30, 0]),     # Right profile
        np.array([0, -30, 0]),    # Left profile
    ]

    for pose in test_poses:
        best_view = ccnf.get_best_view(pose)
        print(f"  Pose {pose} -> View {best_view}")

    print("\n" + "=" * 60)
    print("✓ LNF Patch Expert Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_patch_expert()
