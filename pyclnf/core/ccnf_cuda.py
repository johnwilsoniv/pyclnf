"""
CUDA-accelerated CCNF (Constrained Convolutional Neural Fields) inference.

This module provides GPU-accelerated batch inference for CCNF patch experts,
using batched normalized cross-correlation computation.

The implementation matches the CPU version in patch_expert.py exactly:
1. Normalize patch to [0, 1]
2. For each neuron:
   - Compute normalized cross-correlation (TM_CCOEFF_NORMED)
   - Apply: response = 2 * alpha * sigmoid(correlation * norm_weights + bias)
3. Sum all neuron responses
"""

import numpy as np
from typing import Dict, Optional

# Lazy import torch to avoid import errors when CUDA not available
_torch = None


def _ensure_torch():
    """Lazy import PyTorch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class CCNFInferenceCUDA:
    """
    GPU-accelerated CCNF inference for a single patch expert.

    Neuron weights are moved to GPU once at initialization.
    """

    def __init__(self, patch_expert, device: str = 'cuda'):
        """
        Initialize with weights on GPU.

        Args:
            patch_expert: CCNFPatchExpert instance from CPU
            device: 'cuda' or 'cpu'
        """
        torch = _ensure_torch()

        self.device = torch.device(device)
        self.width = patch_expert.width
        self.height = patch_expert.height

        # Pre-compute and store neuron parameters on GPU
        self.num_neurons = len(patch_expert.neurons)
        self.neurons_weights = []
        self.neurons_weights_mean = []
        self.neurons_weights_centered = []
        self.neurons_weights_norm = []
        self.neurons_bias = []
        self.neurons_alpha = []
        self.neurons_norm_weights = []
        self.neurons_active = []  # Track which neurons are active (alpha >= 1e-4)

        for neuron in patch_expert.neurons:
            alpha = float(neuron['alpha'])
            is_active = abs(alpha) >= 1e-4

            self.neurons_active.append(is_active)
            self.neurons_alpha.append(alpha)
            self.neurons_bias.append(float(neuron['bias']))
            self.neurons_norm_weights.append(float(neuron['norm_weights']))

            if is_active:
                # Pre-compute weight statistics for faster inference
                weights = torch.tensor(neuron['weights'], dtype=torch.float32, device=self.device)
                w_mean = weights.mean()
                w_centered = weights - w_mean
                w_norm = w_centered.norm()

                self.neurons_weights.append(weights)
                self.neurons_weights_mean.append(w_mean)
                self.neurons_weights_centered.append(w_centered)
                self.neurons_weights_norm.append(w_norm)
            else:
                # Placeholder for inactive neurons
                self.neurons_weights.append(None)
                self.neurons_weights_mean.append(None)
                self.neurons_weights_centered.append(None)
                self.neurons_weights_norm.append(None)

    def forward_batch(self, patches: 'torch.Tensor') -> 'torch.Tensor':
        """
        Batched forward pass on GPU.

        Args:
            patches: (batch_size, height, width) tensor, float32, values 0-255
                     Should already be on self.device

        Returns:
            responses: (batch_size,) tensor
        """
        torch = _ensure_torch()

        batch_size = patches.shape[0]

        # Normalize to [0, 1]
        features = patches / 255.0

        # Sum responses from all active neurons
        total = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for neuron_idx in range(self.num_neurons):
            if not self.neurons_active[neuron_idx]:
                continue

            # Get pre-computed weight statistics
            w_centered = self.neurons_weights_centered[neuron_idx]  # (H, W)
            w_norm = self.neurons_weights_norm[neuron_idx]  # scalar

            # Center features (batched)
            f_mean = features.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            f_centered = features - f_mean  # (B, H, W)

            # Compute feature norms (batched)
            f_norm = f_centered.pow(2).sum(dim=(1, 2)).sqrt()  # (B,)

            # Normalized cross-correlation
            # correlation = sum(w_centered * f_centered) / (w_norm * f_norm)
            correlation = (w_centered * f_centered).sum(dim=(1, 2)) / (w_norm * f_norm + 1e-10)

            # Apply neuron response formula:
            # response = 2 * alpha * sigmoid(correlation * norm_weights + bias)
            alpha = self.neurons_alpha[neuron_idx]
            norm_weights = self.neurons_norm_weights[neuron_idx]
            bias = self.neurons_bias[neuron_idx]

            sigmoid_input = correlation * norm_weights + bias
            response = 2.0 * alpha * torch.sigmoid(sigmoid_input)

            total = total + response

        return total


class CCNFBatchProcessor:
    """
    Optimized batch processor for multiple CCNF patch experts.

    Manages multiple CCNF experts on GPU for efficient per-frame processing.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize the batch processor.

        Args:
            device: 'cuda' or 'cpu'
        """
        torch = _ensure_torch()

        self.device = torch.device(device)
        self.experts: Dict[int, CCNFInferenceCUDA] = {}
        self._initialized = False

    def initialize_experts(self, patch_experts: dict) -> None:
        """
        Initialize CUDA experts from CPU patch experts.

        Args:
            patch_experts: Dict mapping landmark_idx -> CCNFPatchExpert (CPU)
        """
        self.experts.clear()

        for landmark_idx, cpu_expert in patch_experts.items():
            # Skip if no neurons
            if not hasattr(cpu_expert, 'neurons') or len(cpu_expert.neurons) == 0:
                continue

            self.experts[landmark_idx] = CCNFInferenceCUDA(cpu_expert, device=str(self.device))

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if experts have been initialized."""
        return self._initialized and len(self.experts) > 0

    def process_single(self, landmark_idx: int, patches: np.ndarray) -> np.ndarray:
        """
        Process patches for a single landmark on GPU.

        Args:
            landmark_idx: Index of the landmark
            patches: (batch_size, height, width) numpy array, uint8 or float32 values 0-255

        Returns:
            responses: (batch_size,) numpy array
        """
        torch = _ensure_torch()

        if landmark_idx not in self.experts:
            raise ValueError(f"No CUDA expert for landmark {landmark_idx}")

        expert = self.experts[landmark_idx]

        # Convert to float32 if needed and transfer to GPU
        if patches.dtype == np.uint8:
            patches = patches.astype(np.float32)
        patches_tensor = torch.tensor(patches, dtype=torch.float32, device=self.device)

        # Forward pass
        with torch.no_grad():
            responses = expert.forward_batch(patches_tensor)

        # Transfer back to CPU
        return responses.cpu().numpy()

    def process_batch(self, patches_by_landmark: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Process patches for multiple landmarks.

        Args:
            patches_by_landmark: Dict mapping landmark_idx -> (batch_size, H, W) patches

        Returns:
            responses_by_landmark: Dict mapping landmark_idx -> (batch_size,) responses
        """
        results = {}

        for landmark_idx, patches in patches_by_landmark.items():
            if landmark_idx not in self.experts:
                continue

            results[landmark_idx] = self.process_single(landmark_idx, patches)

        return results


# Global processor instance
_global_ccnf_processor: Optional[CCNFBatchProcessor] = None


def get_ccnf_cuda_processor(device: str = 'cuda') -> CCNFBatchProcessor:
    """Get or create the global CCNF CUDA batch processor."""
    global _global_ccnf_processor

    if _global_ccnf_processor is None:
        _global_ccnf_processor = CCNFBatchProcessor(device)

    return _global_ccnf_processor


def reset_ccnf_cuda_processor() -> None:
    """Reset the global CCNF processor."""
    global _global_ccnf_processor
    _global_ccnf_processor = None


def ccnf_compute_response_batch(patches: np.ndarray, patch_expert, device: str = 'cuda') -> np.ndarray:
    """
    Compute CCNF responses for a batch of patches.

    This is a convenience function for testing that creates a temporary CUDA expert.

    Args:
        patches: (batch_size, height, width) numpy array, values 0-255
        patch_expert: CCNFPatchExpert instance
        device: 'cuda' or 'cpu'

    Returns:
        responses: (batch_size,) numpy array
    """
    torch = _ensure_torch()

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Create temporary CUDA expert
    cuda_expert = CCNFInferenceCUDA(patch_expert, device=device)

    # Convert and transfer
    if patches.dtype == np.uint8:
        patches = patches.astype(np.float32)
    patches_tensor = torch.tensor(patches, dtype=torch.float32, device=device)

    # Forward pass
    with torch.no_grad():
        responses = cuda_expert.forward_batch(patches_tensor)

    return responses.cpu().numpy()
