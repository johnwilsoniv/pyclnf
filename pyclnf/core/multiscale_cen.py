#!/usr/bin/env python3
"""
Multi-scale batched CEN computation.

Processes all scales (0.25, 0.35, 0.5) in a single GPU call to minimize
transfer overhead and maximize GPU utilization.
"""

import numpy as np
from typing import Dict, List, Tuple
import cv2

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .batched_cen import BatchedCEN


class MultiScaleBatchedCEN:
    """
    Multi-scale batched CEN that processes all scales in one GPU call.

    Reduces GPU transfer overhead by batching 68 landmarks × 3 scales = 204 items.
    """

    def __init__(self, cen_model, scales: List[float] = [0.25, 0.35, 0.5], device: str = 'mps'):
        """
        Initialize multi-scale CEN from CENModel.

        Args:
            cen_model: CENModel instance with scale_models
            scales: List of scales to process
            device: GPU device ('mps', 'cuda', 'cpu')
        """
        self.scales = scales
        self.device = device
        self.num_scales = len(scales)

        # Create BatchedCEN for each scale
        self.batched_cens = {}
        for scale in scales:
            patch_experts = cen_model.scale_models[scale]['views'][0]['patches']
            self.batched_cens[scale] = BatchedCEN(patch_experts, device=device)

        # Stack weights from all scales for combined GPU processing
        self._stack_weights()

        # Cache
        self._grid_cache = {}

    def _stack_weights(self):
        """Stack weights from compatible scales for batched GPU processing."""
        if not TORCH_AVAILABLE:
            return

        # Group scales by architecture (layer dimensions)
        self.scale_groups = {}  # {arch_key: [scale1, scale2, ...]}

        for scale in self.scales:
            cen = self.batched_cens[scale]
            # Create key from layer dimensions
            arch_key = tuple(w.shape for w in cen.weights_t)
            if arch_key not in self.scale_groups:
                self.scale_groups[arch_key] = []
            self.scale_groups[arch_key].append(scale)

        # Stack weights per group
        self.group_weights = {}  # {arch_key: [stacked_weights_per_layer]}
        self.group_biases = {}

        for arch_key, group_scales in self.scale_groups.items():
            first_cen = self.batched_cens[group_scales[0]]
            num_layers = first_cen.num_layers

            weights_stacked = []
            biases_stacked = []

            for layer_idx in range(num_layers):
                layer_weights = []
                layer_biases = []

                for scale in group_scales:
                    cen = self.batched_cens[scale]
                    layer_weights.append(cen.weights_t[layer_idx])
                    layer_biases.append(cen.biases_t[layer_idx])

                stacked_w = torch.cat(layer_weights, dim=0)
                stacked_b = torch.cat(layer_biases, dim=0)

                weights_stacked.append(stacked_w)
                biases_stacked.append(stacked_b)

            self.group_weights[arch_key] = weights_stacked
            self.group_biases[arch_key] = biases_stacked

        # Get common properties
        first_cen = self.batched_cens[self.scales[0]]
        self.width_support = first_cen.width_support
        self.height_support = first_cen.height_support

    def compute_response_maps_multiscale(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        window_sizes: Dict[float, int],
        sim_ref_to_img_per_scale: Dict[float, np.ndarray]
    ) -> Dict[float, Dict[int, np.ndarray]]:
        """
        Compute response maps for all scales, batching compatible architectures.

        Scales with same architecture are processed together in one GPU call.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Current landmark positions (68, 2)
            window_sizes: Dict mapping scale -> window_size
            sim_ref_to_img_per_scale: Dict mapping scale -> similarity transform

        Returns:
            Dict mapping scale -> {landmark_idx: response_map}
        """
        result = {}

        # Process each architecture group
        for arch_key, group_scales in self.scale_groups.items():
            group_result = self._compute_group(
                image, landmarks, window_sizes, sim_ref_to_img_per_scale,
                group_scales, arch_key
            )
            result.update(group_result)

        return result

    def _compute_group(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        window_sizes: Dict[float, int],
        sim_ref_to_img_per_scale: Dict[float, np.ndarray],
        group_scales: List[float],
        arch_key: tuple
    ) -> Dict[float, Dict[int, np.ndarray]]:
        """Compute response maps for a group of scales with same architecture.

        When window sizes differ significantly, falls back to per-scale processing
        because using max window size adds more computation than batching saves.
        """
        group_ws = [window_sizes[s] for s in group_scales]
        first_cen = self.batched_cens[group_scales[0]]

        # Check if window sizes differ too much - if so, fall back to per-scale
        # Using max_ws adds (max_ws² - min_ws²) extra patches per landmark
        # Only batch if window sizes are the same (no extra computation)
        if len(set(group_ws)) > 1:
            result = {}
            for scale in group_scales:
                cen = self.batched_cens[scale]
                ws = window_sizes[scale]
                sim = sim_ref_to_img_per_scale.get(scale)
                result[scale] = cen.compute_response_maps_gpu(image, landmarks, ws, sim)
            return result

        # All same window size - batch efficiently
        max_window_size = group_ws[0]
        aoi_size = max_window_size + first_cen.width_support - 1

        # Step 1: Extract AOIs for all scales using max window size
        all_aois = []
        all_valid_masks = []

        for scale in group_scales:
            cen = self.batched_cens[scale]
            sim_ref_to_img = sim_ref_to_img_per_scale.get(scale)
            # Extract with max_window_size for all scales
            aois, valid_mask = cen.batch_extract_aoi(image, landmarks, max_window_size, sim_ref_to_img)

            for lm_idx in cen.mirror_indices:
                if valid_mask[lm_idx]:
                    aois[lm_idx] = cv2.flip(aois[lm_idx], 1)

            all_aois.append(aois)
            all_valid_masks.append(valid_mask)

        # Stack: (num_group_scales * 68, aoi_h, aoi_w)
        aois_stacked = np.concatenate(all_aois, axis=0)

        # Step 2: Single GPU transfer
        aois_t = torch.from_numpy(aois_stacked).to(self.device)

        # im2col + L2 norm (use first_cen for reorder indices)
        normalized = self._batch_im2col_l2norm_gpu(aois_t, aoi_size, first_cen)

        # Forward pass with group-stacked weights
        weights = self.group_weights[arch_key]
        biases = self.group_biases[arch_key]
        activations = first_cen.activations

        layer_output = normalized
        for layer_idx in range(len(weights)):
            layer_output = torch.bmm(layer_output, weights[layer_idx].transpose(1, 2))
            layer_output = layer_output + biases[layer_idx]
            layer_output = self._apply_activation_torch(layer_output, activations[layer_idx])

        # Single GPU->CPU transfer
        responses_flat = layer_output[:, :, 0].cpu().numpy()

        # Step 3: Reshape, crop to correct size, and split by scale
        max_response_height = max_window_size  # aoi_size - support + 1 = window_size

        result = {}
        for scale_idx, scale in enumerate(group_scales):
            offset = scale_idx * 68
            valid_mask = all_valid_masks[scale_idx]
            cen = self.batched_cens[scale]

            # Target response size for this scale
            target_ws = window_sizes[scale]

            # Crop offset (center crop from max_response_height to target_ws)
            crop_offset = (max_response_height - target_ws) // 2

            response_maps = {}
            for lm_idx in range(68):
                flat_idx = offset + lm_idx
                if valid_mask[lm_idx] and cen.landmark_valid[lm_idx]:
                    # Reshape to max size first
                    response = responses_flat[flat_idx].reshape(
                        max_response_height, max_response_height, order='F'
                    )
                    # Center crop to target size
                    if crop_offset > 0:
                        response = response[crop_offset:crop_offset+target_ws,
                                          crop_offset:crop_offset+target_ws]
                    if lm_idx in cen.mirror_indices:
                        response = cv2.flip(response, 1)
                    response_maps[lm_idx] = response.astype(np.float32, copy=False)
                else:
                    response_maps[lm_idx] = np.zeros(
                        (target_ws, target_ws), dtype=np.float32
                    )

            result[scale] = response_maps

        return result

    def _batch_im2col_l2norm_gpu(self, aois_t: 'torch.Tensor', aoi_size: int,
                                  cen: 'BatchedCEN') -> 'torch.Tensor':
        """GPU im2col + L2 contrast normalization."""
        n_batch = aois_t.shape[0]

        y_blocks = aoi_size - cen.height_support + 1
        x_blocks = y_blocks
        num_windows = y_blocks * x_blocks

        # Get reorder indices from the provided CEN
        patch_reorder, within_reorder = cen._get_reorder_indices(y_blocks, x_blocks)

        # GPU unfold for im2col
        aois_4d = aois_t.unsqueeze(1)  # (N, 1, h, w)
        patches = F.unfold(aois_4d, kernel_size=cen.height_support, stride=1)
        patches = patches.transpose(1, 2)  # (N, num_windows, patch_size)
        patches = patches[:, patch_reorder, :][:, :, within_reorder]

        # Add bias column
        ones = torch.ones(n_batch, num_windows, 1, device=self.device, dtype=torch.float32)
        patches = torch.cat([ones, patches], dim=-1)

        # L2 contrast norm (optimized)
        data = patches[:, :, 1:]
        mean = data.mean(dim=-1, keepdim=True)
        centered = data - mean
        sum_sq = (centered * centered).sum(dim=-1, keepdim=True)
        norm = sum_sq.sqrt()
        norm = torch.clamp(norm, min=1e-10)
        patches[:, :, 1:] = centered / norm

        return patches

    def _apply_activation_torch(self, x: 'torch.Tensor', activation_type: int) -> 'torch.Tensor':
        """Apply activation function."""
        if activation_type == 2:  # ReLU (type 2 in OpenFace CEN)
            return torch.relu(x)
        else:  # Sigmoid (type 0)
            return torch.sigmoid(x)

    def _compute_per_scale(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        window_sizes: Dict[float, int],
        sim_ref_to_img_per_scale: Dict[float, np.ndarray]
    ) -> Dict[float, Dict[int, np.ndarray]]:
        """Fallback: compute each scale separately."""
        result = {}
        for scale in self.scales:
            cen = self.batched_cens[scale]
            ws = window_sizes[scale]
            sim = sim_ref_to_img_per_scale.get(scale)
            result[scale] = cen.compute_response_maps_gpu(image, landmarks, ws, sim)
        return result


def create_multiscale_cen(cen_model, scales: List[float] = [0.25, 0.35, 0.5],
                          device: str = 'mps') -> MultiScaleBatchedCEN:
    """Factory function to create MultiScaleBatchedCEN."""
    return MultiScaleBatchedCEN(cen_model, scales, device)
