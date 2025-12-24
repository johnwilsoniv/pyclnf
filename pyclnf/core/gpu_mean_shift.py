#!/usr/bin/env python3
"""
GPU-accelerated mean-shift computation for pyclnf.

Replaces the Numba JIT KDE mean-shift with batched GPU operations.
Provides exact numerical match with the CPU implementation.
"""

import numpy as np
from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMeanShift:
    """
    GPU-accelerated KDE mean-shift computation.

    Computes weighted centroid for all landmarks in parallel using
    batched tensor operations.
    """

    def __init__(self, device: str = 'cpu', sigma: float = 2.5):
        """
        Initialize GPU mean-shift computer.

        Args:
            device: 'cpu', 'cuda', or 'mps'
            sigma: Gaussian kernel sigma for KDE
        """
        self.device = device
        self.sigma = sigma
        self.a = -0.5 / (sigma * sigma)  # KDE parameter

        # Cache for coordinate grids (created per window size)
        self._grid_cache = {}

    def _get_grid(self, window_size: int) -> 'torch.Tensor':
        """Get or create coordinate grid for given window size."""
        if window_size not in self._grid_cache:
            # Create meshgrid of (y, x) coordinates
            y = torch.arange(window_size, dtype=torch.float32, device=self.device)
            x = torch.arange(window_size, dtype=torch.float32, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            self._grid_cache[window_size] = (grid_y, grid_x)

        return self._grid_cache[window_size]

    def compute_mean_shift_gpu(
        self,
        response_maps: Dict[int, np.ndarray],
        offsets_x: np.ndarray,
        offsets_y: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """
        Compute mean-shift for all landmarks using GPU.

        Args:
            response_maps: Dict mapping landmark_idx -> (h, w) response map
            offsets_x: Current x offsets in response map coords (68,)
            offsets_y: Current y offsets in response map coords (68,)
            window_size: Size of response map (e.g., 11)

        Returns:
            mean_shift: Stacked [ms_x_0..67, ms_y_0..67] shape (136,)
        """
        n_landmarks = 68

        # Stack response maps into tensor
        response_list = []
        valid_mask = np.zeros(n_landmarks, dtype=bool)

        for lm_idx in range(n_landmarks):
            if lm_idx in response_maps:
                response_list.append(response_maps[lm_idx])
                valid_mask[lm_idx] = True
            else:
                response_list.append(np.zeros((window_size, window_size), dtype=np.float32))

        responses = np.stack(response_list, axis=0)  # (68, h, w)
        responses_t = torch.from_numpy(responses).to(self.device)

        # Get coordinate grid
        grid_y, grid_x = self._get_grid(window_size)

        # Current positions in response map coords
        dx_t = torch.from_numpy(offsets_x.astype(np.float32)).to(self.device)
        dy_t = torch.from_numpy(offsets_y.astype(np.float32)).to(self.device)

        # Compute Gaussian KDE weights for all landmarks in parallel
        # dist_sq = (grid_x - dx)^2 + (grid_y - dy)^2
        # Shape: (68, h, w) after broadcasting

        # Expand dimensions for broadcasting
        dx_expanded = dx_t.view(-1, 1, 1)  # (68, 1, 1)
        dy_expanded = dy_t.view(-1, 1, 1)  # (68, 1, 1)

        dist_x = grid_x.unsqueeze(0) - dx_expanded  # (68, h, w)
        dist_y = grid_y.unsqueeze(0) - dy_expanded  # (68, h, w)
        dist_sq = dist_x ** 2 + dist_y ** 2

        # KDE weights: exp(a * dist_sq) where a = -0.5/sigma^2
        kde_weights = torch.exp(self.a * dist_sq)

        # Combined weights = response * kde_weight
        combined_weights = responses_t * kde_weights  # (68, h, w)

        # Compute weighted centroid
        total_weight = combined_weights.sum(dim=(1, 2))  # (68,)
        total_weight = torch.clamp(total_weight, min=1e-10)  # Avoid division by zero

        # Weighted sum of coordinates
        weighted_x = (combined_weights * grid_x.unsqueeze(0)).sum(dim=(1, 2))  # (68,)
        weighted_y = (combined_weights * grid_y.unsqueeze(0)).sum(dim=(1, 2))  # (68,)

        # Centroid
        centroid_x = weighted_x / total_weight
        centroid_y = weighted_y / total_weight

        # Mean-shift = centroid - current position
        ms_x = centroid_x - dx_t
        ms_y = centroid_y - dy_t

        # Stack into output format [ms_x_0..67, ms_y_0..67]
        mean_shift = torch.cat([ms_x, ms_y]).cpu().numpy()

        # Zero out invalid landmarks
        mean_shift[:n_landmarks][~valid_mask] = 0
        mean_shift[n_landmarks:][~valid_mask] = 0

        return mean_shift.astype(np.float32)


def create_gpu_mean_shift(device: str = 'cpu', sigma: float = 2.5) -> GPUMeanShift:
    """
    Create a GPU mean-shift computer.

    Args:
        device: 'cpu', 'cuda', or 'mps'
        sigma: Gaussian kernel sigma

    Returns:
        GPUMeanShift instance
    """
    return GPUMeanShift(device=device, sigma=sigma)
