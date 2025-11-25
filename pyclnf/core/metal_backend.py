#!/usr/bin/env python3
"""
Metal GPU Backend for CLNF Response Map Computation

Uses PyTorch with MPS (Metal Performance Shaders) backend for GPU acceleration
on Apple Silicon. Computes all landmark response maps in parallel on GPU.

Target: 10-50x speedup over CPU by parallelizing:
- Image warping for all landmarks
- Patch extraction
- Normalized cross-correlation
- Response map computation
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time

# Try to import PyTorch with MPS support
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        USE_METAL = True
        print(f"Metal GPU backend available (MPS)")
    else:
        DEVICE = torch.device("cpu")
        USE_METAL = False
except ImportError:
    USE_METAL = False
    DEVICE = None


class MetalResponseMapComputer:
    """
    GPU-accelerated response map computation using Metal via PyTorch MPS.
    """

    def __init__(self, patch_experts: Dict, window_size: int = 11):
        """
        Initialize Metal backend with patch expert data.

        Args:
            patch_experts: Dictionary of CCNFPatchExpert objects
            window_size: Response map window size
        """
        if not USE_METAL:
            raise RuntimeError("Metal GPU backend not available")

        self.window_size = window_size
        self.n_landmarks = len(patch_experts)

        # Prepare GPU tensors for all patch experts
        self._prepare_gpu_data(patch_experts)

    def _prepare_gpu_data(self, patch_experts: Dict):
        """Upload all patch expert weights to GPU memory."""
        # Get dimensions from first expert
        sample_expert = list(patch_experts.values())[0]
        self.patch_height = sample_expert.height
        self.patch_width = sample_expert.width

        # Find max neurons across all experts
        max_neurons = max(exp.num_neurons for exp in patch_experts.values())

        # Prepare arrays for all landmarks
        n_landmarks = len(patch_experts)

        # Allocate arrays (landmarks × neurons × height × width)
        all_weights = np.zeros((n_landmarks, max_neurons, self.patch_height, self.patch_width), dtype=np.float32)
        all_biases = np.zeros((n_landmarks, max_neurons), dtype=np.float32)
        all_alphas = np.zeros((n_landmarks, max_neurons), dtype=np.float32)
        all_norm_weights = np.zeros((n_landmarks, max_neurons), dtype=np.float32)
        neuron_counts = np.zeros(n_landmarks, dtype=np.int32)

        # Fill arrays
        self.landmark_indices = []
        for i, (landmark_idx, expert) in enumerate(sorted(patch_experts.items())):
            self.landmark_indices.append(landmark_idx)
            neuron_counts[i] = expert.num_neurons

            for j, neuron in enumerate(expert.neurons):
                if j >= max_neurons:
                    break
                all_weights[i, j] = neuron['weights']
                all_biases[i, j] = neuron['bias']
                all_alphas[i, j] = neuron['alpha']
                all_norm_weights[i, j] = neuron['norm_weights']

        # Upload to GPU
        self.weights_gpu = torch.from_numpy(all_weights).to(DEVICE)
        self.biases_gpu = torch.from_numpy(all_biases).to(DEVICE)
        self.alphas_gpu = torch.from_numpy(all_alphas).to(DEVICE)
        self.norm_weights_gpu = torch.from_numpy(all_norm_weights).to(DEVICE)
        self.neuron_counts = torch.from_numpy(neuron_counts).to(DEVICE)

        # Precompute weight means and norms for NCC
        # Shape: (n_landmarks, max_neurons)
        weight_means = all_weights.mean(axis=(2, 3))
        weights_centered = all_weights - weight_means[:, :, None, None]
        weight_norms = np.sqrt((weights_centered ** 2).sum(axis=(2, 3)))

        self.weight_means_gpu = torch.from_numpy(weight_means).to(DEVICE)
        self.weights_centered_gpu = torch.from_numpy(weights_centered).to(DEVICE)
        self.weight_norms_gpu = torch.from_numpy(weight_norms).to(DEVICE)

        self.max_neurons = max_neurons
        print(f"Loaded {n_landmarks} landmarks with up to {max_neurons} neurons to GPU")

    def compute_all_response_maps(self,
                                   image: np.ndarray,
                                   landmarks_2d: np.ndarray,
                                   sim_matrices: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute response maps for all landmarks in parallel on GPU.

        Uses true batched GPU operations for all landmarks simultaneously.

        Args:
            image: Grayscale image (H, W)
            landmarks_2d: Landmark positions (n_landmarks, 2)
            sim_matrices: Similarity transform matrices for each landmark (n_landmarks, 2, 3)

        Returns:
            Dictionary mapping landmark_idx -> response_map (window_size, window_size)
        """
        n_landmarks = len(self.landmark_indices)
        h, w = image.shape

        # Upload image to GPU
        image_gpu = torch.from_numpy(image.astype(np.float32) / 255.0).to(DEVICE)
        image_4d = image_gpu.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Calculate area of interest size
        patch_dim = max(self.patch_width, self.patch_height)
        aoi_size = self.window_size + patch_dim - 1

        # Convert all similarity matrices to theta tensors for affine_grid
        # Need to normalize transforms for grid_sample's [-1, 1] coordinate system
        # Build on CPU first, then transfer to GPU
        thetas_np = np.zeros((n_landmarks, 2, 3), dtype=np.float32)

        for i, landmark_idx in enumerate(self.landmark_indices):
            sim = sim_matrices[i]
            # PyTorch grid_sample expects coordinates in [-1, 1] range
            # We need to convert the OpenCV-style transform
            thetas_np[i, 0, 0] = sim[0, 0]
            thetas_np[i, 0, 1] = sim[0, 1] * (h / w)  # Adjust for aspect ratio
            thetas_np[i, 0, 2] = 2 * sim[0, 2] / w - 1 + sim[0, 0] - 1
            thetas_np[i, 1, 0] = sim[1, 0] * (w / h)
            thetas_np[i, 1, 1] = sim[1, 1]
            thetas_np[i, 1, 2] = 2 * sim[1, 2] / h - 1 + sim[1, 1] - 1

        thetas = torch.from_numpy(thetas_np).to(DEVICE)

        # Generate grids for all landmarks at once
        grids = torch.nn.functional.affine_grid(
            thetas, [n_landmarks, 1, aoi_size, aoi_size], align_corners=False
        )

        # Expand image for batch sampling
        image_batch = image_4d.expand(n_landmarks, -1, -1, -1)

        # Warp all regions at once
        warped = torch.nn.functional.grid_sample(
            image_batch, grids, mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(1)  # (n_landmarks, aoi_size, aoi_size)

        # Extract patches for all landmarks using unfold
        # Result: (n_landmarks, window_size, window_size, patch_h, patch_w)
        patches = warped.unfold(1, self.patch_height, 1).unfold(2, self.patch_width, 1)

        # Reshape for batch NCC computation
        # (n_landmarks, ws*ws, ph*pw)
        ws = self.window_size
        patches_flat = patches.reshape(n_landmarks, ws * ws, -1)

        # Compute patch means and center
        patch_means = patches_flat.mean(dim=2, keepdim=True)  # (n_landmarks, ws*ws, 1)
        patches_centered = patches_flat - patch_means  # (n_landmarks, ws*ws, ph*pw)
        patch_norms = patches_centered.norm(dim=2)  # (n_landmarks, ws*ws)

        # Compute response maps for all landmarks
        response_maps_tensor = torch.zeros((n_landmarks, ws, ws), device=DEVICE)

        # Process each landmark's NCC computation
        # This part still needs per-landmark handling due to variable neuron counts
        for i in range(n_landmarks):
            n_neurons = int(self.neuron_counts[i].item())
            if n_neurons == 0:
                continue

            weights_centered = self.weights_centered_gpu[i, :n_neurons]  # (n_neurons, ph, pw)
            weight_norms = self.weight_norms_gpu[i, :n_neurons]  # (n_neurons,)
            biases = self.biases_gpu[i, :n_neurons]
            alphas = self.alphas_gpu[i, :n_neurons]
            norm_weights = self.norm_weights_gpu[i, :n_neurons]

            # Flatten weights
            weights_flat = weights_centered.reshape(n_neurons, -1)  # (n_neurons, ph*pw)

            # Compute correlations
            dot_products = patches_centered[i] @ weights_flat.T  # (ws*ws, n_neurons)
            safe_norms = patch_norms[i].unsqueeze(1) * weight_norms.unsqueeze(0) + 1e-10
            correlations = dot_products / safe_norms

            # Apply response function
            sigmoid_input = correlations * norm_weights.unsqueeze(0) + biases.unsqueeze(0)
            responses = (2.0 * alphas.unsqueeze(0)) * torch.sigmoid(sigmoid_input)

            # Sum across neurons
            response_maps_tensor[i] = responses.sum(dim=1).reshape(ws, ws)

        # Transfer back to CPU
        response_maps_np = response_maps_tensor.cpu().numpy()

        # Build result dictionary
        response_maps = {}
        for i, landmark_idx in enumerate(self.landmark_indices):
            response_maps[landmark_idx] = response_maps_np[i]

        return response_maps

    def _compute_single_response_map_gpu(self,
                                          image_gpu: torch.Tensor,
                                          center_x: float,
                                          center_y: float,
                                          sim_matrix: np.ndarray,
                                          expert_idx: int) -> torch.Tensor:
        """
        Compute response map for single landmark on GPU.
        """
        half_window = self.window_size // 2
        patch_dim = max(self.patch_width, self.patch_height)
        aoi_size = self.window_size + patch_dim - 1

        # Create sampling grid for affine warp
        # This is the GPU equivalent of cv2.warpAffine
        theta = torch.tensor([
            [sim_matrix[0, 0], sim_matrix[0, 1], sim_matrix[0, 2]],
            [sim_matrix[1, 0], sim_matrix[1, 1], sim_matrix[1, 2]]
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Normalize to [-1, 1] range for grid_sample
        h, w = image_gpu.shape
        theta[0, 0, 2] = 2 * theta[0, 0, 2] / w - 1
        theta[0, 1, 2] = 2 * theta[0, 1, 2] / h - 1

        grid = torch.nn.functional.affine_grid(
            theta, [1, 1, aoi_size, aoi_size], align_corners=False
        )

        # Warp image
        image_4d = image_gpu.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        warped = torch.nn.functional.grid_sample(
            image_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze()  # (aoi_size, aoi_size)

        # Extract all patches at once using unfold
        # After unfold: (aoi_size - patch_h + 1, aoi_size - patch_w + 1, patch_h, patch_w)
        # = (window_size, window_size, patch_h, patch_w)
        patches = warped.unfold(0, self.patch_height, 1).unfold(1, self.patch_width, 1)

        # Compute NCC for all patches with all neurons
        # patches: (ws, ws, ph, pw)
        # weights: (n_neurons, ph, pw)

        n_neurons = int(self.neuron_counts[expert_idx].item())
        weights_centered = self.weights_centered_gpu[expert_idx, :n_neurons]  # (n_neurons, ph, pw)
        weight_norms = self.weight_norms_gpu[expert_idx, :n_neurons]  # (n_neurons,)
        biases = self.biases_gpu[expert_idx, :n_neurons]  # (n_neurons,)
        alphas = self.alphas_gpu[expert_idx, :n_neurons]  # (n_neurons,)
        norm_weights = self.norm_weights_gpu[expert_idx, :n_neurons]  # (n_neurons,)

        # Reshape patches for batch computation
        # (ws*ws, ph*pw)
        patches_flat = patches.reshape(self.window_size * self.window_size, -1)

        # Compute patch means and center
        patch_means = patches_flat.mean(dim=1, keepdim=True)  # (ws*ws, 1)
        patches_centered = patches_flat - patch_means  # (ws*ws, ph*pw)
        patch_norms = patches_centered.norm(dim=1)  # (ws*ws,)

        # Compute correlations: (ws*ws, n_neurons)
        # correlation = (patches_centered @ weights_centered.T) / (patch_norms * weight_norms)
        weights_flat = weights_centered.reshape(n_neurons, -1)  # (n_neurons, ph*pw)
        dot_products = patches_centered @ weights_flat.T  # (ws*ws, n_neurons)

        # Avoid division by zero
        safe_norms = patch_norms.unsqueeze(1) * weight_norms.unsqueeze(0) + 1e-10
        correlations = dot_products / safe_norms  # (ws*ws, n_neurons)

        # Apply formula: (2 * alpha) * sigmoid(correlation * norm_weights + bias)
        sigmoid_input = correlations * norm_weights.unsqueeze(0) + biases.unsqueeze(0)
        responses = (2.0 * alphas.unsqueeze(0)) * torch.sigmoid(sigmoid_input)

        # Sum across neurons and reshape
        total_response = responses.sum(dim=1)  # (ws*ws,)
        response_map = total_response.reshape(self.window_size, self.window_size)

        return response_map


def test_metal_backend():
    """Test Metal backend."""
    if not USE_METAL:
        print("Metal backend not available")
        return

    import sys
    sys.path.insert(0, 'pyclnf')
    from pyclnf.core.patch_expert import CCNFModel

    print("\n=== Testing Metal Backend ===\n")

    # Load patch experts
    ccnf = CCNFModel('pyclnf/pyclnf/models')
    patch_experts = ccnf.scale_models[0.25]['views'][0]['patches']

    # Create Metal backend
    metal = MetalResponseMapComputer(patch_experts, window_size=11)

    # Create test image
    test_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    # Create dummy landmarks and transforms
    n_landmarks = len(patch_experts)
    landmarks = np.random.rand(68, 2) * [640, 480]
    sim_matrices = np.tile(np.eye(2, 3, dtype=np.float32), (n_landmarks, 1, 1))

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(10):
        response_maps = metal.compute_all_response_maps(test_image, landmarks, sim_matrices)
    elapsed = (time.perf_counter() - t0) / 10 * 1000

    print(f"Metal response maps: {elapsed:.1f}ms for {len(response_maps)} landmarks")
    print(f"Per landmark: {elapsed/len(response_maps):.2f}ms")


if __name__ == '__main__':
    test_metal_backend()
