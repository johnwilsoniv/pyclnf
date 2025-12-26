"""
CCNF (Constrained CNN Features) Patch Expert for inner face model.

This implements the CCNF response computation used by the hierarchical inner model.
CCNF is an older format than CEN, using template matching + sigmoid activation.

C++ Reference: OpenFace/lib/local/LandmarkDetector/src/CCNF_patch_expert.cpp
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class CCNFNeuron:
    """Single CCNF neuron - template matching + sigmoid activation."""

    def __init__(self, weights: np.ndarray, bias: float, alpha: float,
                 width: int, height: int, neuron_type: int = 0):
        """
        Initialize a CCNF neuron.

        Args:
            weights: Filter weights, shape (height * width,)
            bias: Bias term
            alpha: Neuron confidence/scaling factor
            width, height: Filter dimensions
            neuron_type: 0=raw intensity, 3=normalized
        """
        self.weights = weights.reshape(height, width).astype(np.float32)
        self.bias = float(bias)
        self.alpha = float(alpha)
        self.width = width
        self.height = height
        self.neuron_type = neuron_type

        # Precompute weight norm for normalized cross-correlation
        self.norm_weights = np.linalg.norm(self.weights)

    def response(self, area_of_interest: np.ndarray) -> np.ndarray:
        """
        Compute neuron response map.

        C++ algorithm (CCNF_patch_expert.cpp:213-285):
        1. Normalize input if neuron_type == 3
        2. Template matching with normalized cross-correlation
        3. Sigmoid activation: 2*alpha / (1 + exp(-(response * norm_weights + bias)))

        Args:
            area_of_interest: Input image patch (grayscale, float32)

        Returns:
            response: Response map
        """
        if area_of_interest.dtype != np.float32:
            aoi = area_of_interest.astype(np.float32)
        else:
            aoi = area_of_interest

        # Normalize if type 3 (per-area normalization)
        if self.neuron_type == 3:
            mean_val = aoi.mean()
            std_val = aoi.std()
            if std_val > 1e-6:
                aoi = (aoi - mean_val) / std_val
            else:
                aoi = aoi - mean_val

        # Template matching with normalized cross-correlation
        # TM_CCOEFF_NORMED: normalized cross-correlation coefficient
        result = cv2.matchTemplate(aoi, self.weights, cv2.TM_CCOEFF_NORMED)

        # Apply sigmoid activation with alpha scaling
        # C++ formula: 2 * alpha / (1 + exp(-(response * norm_weights + bias)))
        # Note: matchTemplate returns correlation [-1, 1], need to scale
        sigmoid_input = result * self.norm_weights + self.bias
        response = (2.0 * self.alpha) / (1.0 + np.exp(-sigmoid_input))

        return response


class CCNFPatchExpert:
    """CCNF patch expert - aggregates multiple neurons."""

    def __init__(self, patch_dir: str):
        """
        Load CCNF patch expert from exported files.

        Args:
            patch_dir: Directory containing info.npz, weight_matrix.npy, alphas.npy, betas.npy
        """
        patch_dir = Path(patch_dir)

        # Load metadata
        info = np.load(patch_dir / 'info.npz')
        self.width = int(info['width'])
        self.height = int(info['height'])
        self.confidence = float(info['confidence'])

        # Load weights and parameters
        self.weight_matrix = np.load(patch_dir / 'weight_matrix.npy')  # (n_neurons, patch_size+1)
        self.alphas = np.load(patch_dir / 'alphas.npy')  # (n_neurons,)
        self.betas = np.load(patch_dir / 'betas.npy')  # (n_betas,)

        self.n_neurons = len(self.alphas)
        self.n_betas = len(self.betas)
        self.patch_size = self.width * self.height

        # Create neurons
        self.neurons = self._create_neurons()

        # Compute sum of alphas for sigma computation
        self.sum_alphas = np.sum(np.abs(self.alphas))

    def _create_neurons(self):
        """Create neuron objects from weight matrix."""
        neurons = []
        for i in range(self.n_neurons):
            # weight_matrix row: [bias, w0, w1, ..., w_{patch_size-1}]
            row = self.weight_matrix[i]
            bias = row[0]
            weights = row[1:]  # (patch_size,)

            neuron = CCNFNeuron(
                weights=weights,
                bias=bias,
                alpha=self.alphas[i],
                width=self.width,
                height=self.height,
                neuron_type=0  # Default to raw intensity
            )
            neurons.append(neuron)
        return neurons

    def compute_response(self, area_of_interest: np.ndarray,
                          sigma_matrix: np.ndarray = None) -> np.ndarray:
        """
        Compute patch response map.

        C++ algorithm (CCNF_patch_expert.cpp:356-435):
        1. Aggregate all neuron responses
        2. Apply sigma smoothing: response = Sigma @ response_vec
        3. Ensure non-negative

        Args:
            area_of_interest: Input image patch
            sigma_matrix: Optional precomputed sigma matrix for smoothing

        Returns:
            response: Response map (window_size, window_size)
        """
        # Compute response map size
        resp_h = area_of_interest.shape[0] - self.height + 1
        resp_w = area_of_interest.shape[1] - self.width + 1

        if resp_h <= 0 or resp_w <= 0:
            return None

        # Initialize response
        response = np.zeros((resp_h, resp_w), dtype=np.float32)

        # Aggregate neuron responses
        for neuron in self.neurons:
            if abs(neuron.alpha) > 1e-4:
                neuron_resp = neuron.response(area_of_interest)
                response += neuron_resp

        # Apply sigma smoothing if provided
        if sigma_matrix is not None:
            resp_vec = response.flatten()
            smoothed = sigma_matrix @ resp_vec
            response = smoothed.reshape(resp_h, resp_w)

        # Ensure non-negative
        min_val = response.min()
        if min_val < 0:
            response = response - min_val

        return response.astype(np.float32, copy=False)


class CCNFSigmaComputation:
    """Compute sigma matrix from betas and sigma_components."""

    @staticmethod
    def compute_sigma(betas: np.ndarray, sigma_components: list,
                      sum_alphas: float, window_size: int) -> np.ndarray:
        """
        Compute sigma (covariance) matrix for response smoothing.

        C++ algorithm (CCNF_patch_expert.cpp:82-119):
            q1 = sum_alphas * I
            q2 = sum(beta[i] * sigma_components[i])
            SigmaInv = 2 * (q1 + q2)
            Sigma = inv(SigmaInv)

        Args:
            betas: Beta weights for sigma components
            sigma_components: List of pre-computed sigma component matrices
            sum_alphas: Sum of absolute neuron alphas
            window_size: Response window size

        Returns:
            sigma: Sigma matrix (ws^2, ws^2)
        """
        ws_sq = window_size * window_size

        # q1 = sum_alphas * Identity
        q1 = sum_alphas * np.eye(ws_sq, dtype=np.float32)

        # q2 = sum(beta_i * sigma_component_i)
        q2 = np.zeros((ws_sq, ws_sq), dtype=np.float32)
        for i, beta in enumerate(betas):
            if i < len(sigma_components):
                q2 += float(beta) * sigma_components[i].astype(np.float32)

        # SigmaInv = 2 * (q1 + q2)
        sigma_inv = 2.0 * (q1 + q2)

        # Sigma = inv(SigmaInv) using Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(sigma_inv)
            L_inv = np.linalg.inv(L)
            sigma = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # Fallback to standard inverse if Cholesky fails
            sigma = np.linalg.inv(sigma_inv + 1e-6 * np.eye(ws_sq))

        return sigma


class CCNFPatchExperts:
    """Collection of CCNF patch experts for all 51 inner landmarks."""

    def __init__(self, ccnf_dir: str):
        """
        Load all CCNF patch experts.

        Args:
            ccnf_dir: Directory containing exported_inner_ccnf/
        """
        ccnf_dir = Path(ccnf_dir)

        # Load metadata
        meta = np.load(ccnf_dir / 'metadata.npz')
        self.patch_scaling = float(meta['patch_scaling'])
        self.num_views = int(meta['num_views'])
        self.num_points = int(meta['num_points'])
        self.windows = meta['windows'].tolist()
        self.n_betas = int(meta['n_betas'])
        self.centers = meta['centers']
        self.visibilities = meta['visibilities']

        # Load sigma components for each window size
        self.sigma_components = {}
        for ws in self.windows:
            sigma_dir = ccnf_dir / f'sigma_ws{ws}'
            if sigma_dir.exists():
                components = []
                for i in range(self.n_betas):
                    sigma_file = sigma_dir / f'sigma_{i}.npy'
                    if sigma_file.exists():
                        components.append(np.load(sigma_file))
                self.sigma_components[ws] = components

        # Load patch experts for frontal view (view_id=0)
        self.experts = {}
        view_dir = ccnf_dir / 'view_0'
        for point_idx in range(self.num_points):
            patch_dir = view_dir / f'patch_{point_idx}'
            if patch_dir.exists():
                try:
                    self.experts[point_idx] = CCNFPatchExpert(str(patch_dir))
                except Exception as e:
                    print(f"Warning: Failed to load patch {point_idx}: {e}")

        # Precompute sigma matrices for each window size
        self.sigma_matrices = {}
        for ws in self.windows:
            if ws in self.sigma_components:
                # Use first expert's betas and sum_alphas (they're typically similar)
                if 0 in self.experts:
                    expert = self.experts[0]
                    sigma = CCNFSigmaComputation.compute_sigma(
                        expert.betas,
                        self.sigma_components[ws],
                        expert.sum_alphas,
                        ws
                    )
                    self.sigma_matrices[ws] = sigma

        print(f"Loaded {len(self.experts)}/{self.num_points} CCNF patch experts")
        print(f"Sigma matrices for window sizes: {list(self.sigma_matrices.keys())}")

    def compute_response(self, point_idx: int, area_of_interest: np.ndarray,
                          window_size: int = 9) -> Optional[np.ndarray]:
        """
        Compute response map for a landmark.

        Args:
            point_idx: Inner landmark index (0-50)
            area_of_interest: Image patch around landmark
            window_size: Response window size

        Returns:
            response: Response map or None if expert not available
        """
        if point_idx not in self.experts:
            return None

        expert = self.experts[point_idx]

        # Get sigma matrix for this window size
        sigma = self.sigma_matrices.get(window_size, None)

        # Compute response with sigma smoothing
        response = expert.compute_response(area_of_interest, sigma)

        return response

    def get_confidence(self, point_idx: int) -> float:
        """Get patch expert confidence for a landmark."""
        if point_idx not in self.experts:
            return 0.0
        return self.experts[point_idx].confidence


def test_ccnf_patch_expert():
    """Test CCNF patch expert implementation."""
    print("=" * 60)
    print("Testing CCNF Patch Expert")
    print("=" * 60)

    ccnf_dir = "pyclnf/pyclnf/models/exported_inner_ccnf"
    experts = CCNFPatchExperts(ccnf_dir)

    print(f"\nCCNF Experts Info:")
    print(f"  num_points: {experts.num_points}")
    print(f"  num_experts loaded: {len(experts.experts)}")
    print(f"  patch_scaling: {experts.patch_scaling}")
    print(f"  windows: {experts.windows}")
    print(f"  n_betas: {experts.n_betas}")

    # Test response computation
    print(f"\nTesting response computation:")

    # Create a synthetic test patch
    # Area of interest size = window_size + patch_size - 1
    window_size = 9
    expert = experts.experts[0]
    aoi_size = window_size + expert.width - 1  # 9 + 11 - 1 = 19

    # Random test image
    test_aoi = np.random.rand(aoi_size, aoi_size).astype(np.float32) * 255

    # Compute response
    response = experts.compute_response(0, test_aoi, window_size)
    print(f"  AOI shape: {test_aoi.shape}")
    print(f"  Response shape: {response.shape}")
    print(f"  Expected response shape: ({window_size}, {window_size})")
    print(f"  Response range: [{response.min():.4f}, {response.max():.4f}]")
    print(f"  Response is non-negative: {response.min() >= 0}")

    # Test with real-ish face patch (uniform with some structure)
    print(f"\nTesting with structured patch:")
    structured_aoi = np.ones((aoi_size, aoi_size), dtype=np.float32) * 128
    # Add some gradient
    for i in range(aoi_size):
        structured_aoi[i, :] += i * 2
    for j in range(aoi_size):
        structured_aoi[:, j] += j * 2

    response = experts.compute_response(0, structured_aoi, window_size)
    print(f"  Response range: [{response.min():.4f}, {response.max():.4f}]")
    print(f"  Peak location: {np.unravel_index(response.argmax(), response.shape)}")

    # Test sigma computation
    print(f"\nSigma matrices:")
    for ws, sigma in experts.sigma_matrices.items():
        print(f"  WS{ws}: shape={sigma.shape}, cond={np.linalg.cond(sigma):.2e}")

    print("\n" + "=" * 60)
    print("CCNF Patch Expert tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_ccnf_patch_expert()
