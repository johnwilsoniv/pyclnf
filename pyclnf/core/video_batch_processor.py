#!/usr/bin/env python3
"""
Video batch processor for pyclnf.

Processes multiple frames together to amortize GPU transfer overhead.
Provides significant speedup for video processing by batching across frames.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .batched_cen import BatchedCEN


class VideoBatchProcessor:
    """
    Batch processor for video frames.

    Processes multiple frames together to maximize GPU utilization
    and amortize transfer overhead.
    """

    def __init__(self, batched_cen: BatchedCEN, batch_size: int = 8, device: str = 'mps'):
        """
        Initialize video batch processor.

        Args:
            batched_cen: BatchedCEN instance for response map computation
            batch_size: Number of frames to process together
            device: GPU device ('mps', 'cuda', or 'cpu')
        """
        self.batched_cen = batched_cen
        self.batch_size = batch_size
        self.device = device

        # Frame buffer
        self._frame_buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # (image, landmarks, sim_ref_to_img)

    def add_frame(self, image: np.ndarray, landmarks: np.ndarray,
                  sim_ref_to_img: np.ndarray = None) -> Optional[List[Dict[int, np.ndarray]]]:
        """
        Add a frame to the buffer.

        When buffer reaches batch_size, processes all frames and returns results.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Landmark positions (68, 2)
            sim_ref_to_img: Optional similarity transform

        Returns:
            List of response maps dicts if batch was processed, None otherwise
        """
        self._frame_buffer.append((image.copy(), landmarks.copy(),
                                   sim_ref_to_img.copy() if sim_ref_to_img is not None else None))

        if len(self._frame_buffer) >= self.batch_size:
            return self.flush()

        return None

    def flush(self) -> List[Dict[int, np.ndarray]]:
        """
        Process all buffered frames and return results.

        Returns:
            List of response maps dicts, one per frame
        """
        if not self._frame_buffer:
            return []

        results = self._process_batch(self._frame_buffer)
        self._frame_buffer = []
        return results

    def _process_batch(self, frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[Dict[int, np.ndarray]]:
        """
        Process a batch of frames together.

        Args:
            frames: List of (image, landmarks, sim_ref_to_img) tuples

        Returns:
            List of response maps dicts
        """
        n_frames = len(frames)
        window_size = 11  # Default, could be parameterized
        aoi_size = window_size + self.batched_cen.width_support - 1

        # Step 1: Extract AOIs for all frames (CPU, uses OpenCV warpAffine)
        all_aois = []
        all_valid_masks = []

        for image, landmarks, sim_ref_to_img in frames:
            aois, valid_mask = self.batched_cen.batch_extract_aoi(
                image, landmarks, window_size, sim_ref_to_img
            )

            # Handle mirrored landmarks
            for lm_idx in self.batched_cen.mirror_indices:
                if valid_mask[lm_idx]:
                    aois[lm_idx] = cv2.flip(aois[lm_idx], 1)

            all_aois.append(aois)
            all_valid_masks.append(valid_mask)

        # Stack: (n_frames, 68, aoi_h, aoi_w)
        aois_stacked = np.stack(all_aois, axis=0)
        # Reshape for batch processing: (n_frames * 68, aoi_h, aoi_w)
        aois_flat = aois_stacked.reshape(-1, aoi_size, aoi_size)

        # Step 2: GPU batch processing
        if TORCH_AVAILABLE and self.device != 'cpu':
            results = self._batch_forward_gpu(aois_flat, n_frames, window_size)
        else:
            results = self._batch_forward_cpu(aois_flat, n_frames, window_size)

        # Step 3: Apply mirroring to results and build output dicts
        output = []
        response_height = window_size  # aoi - support + 1 = window_size

        for frame_idx in range(n_frames):
            response_maps = {}
            valid_mask = all_valid_masks[frame_idx]

            for lm_idx in range(68):
                flat_idx = frame_idx * 68 + lm_idx

                if valid_mask[lm_idx] and self.batched_cen.landmark_valid[lm_idx]:
                    response = results[flat_idx].reshape(
                        response_height, response_height, order='F'
                    )
                    if lm_idx in self.batched_cen.mirror_indices:
                        response = cv2.flip(response, 1)
                    response_maps[lm_idx] = response.astype(np.float32)
                else:
                    response_maps[lm_idx] = np.zeros(
                        (response_height, response_height), dtype=np.float32
                    )

            output.append(response_maps)

        return output

    def _batch_forward_gpu(self, aois_flat: np.ndarray, n_frames: int,
                           window_size: int) -> np.ndarray:
        """
        GPU batch forward pass for multiple frames.

        Args:
            aois_flat: Flattened AOIs (n_frames * 68, aoi_h, aoi_w)
            n_frames: Number of frames
            window_size: Response map size

        Returns:
            responses_flat: (n_frames * 68, num_windows)
        """
        total_batch = aois_flat.shape[0]
        aoi_h = aois_flat.shape[1]

        # Move to GPU
        aois_t = torch.from_numpy(aois_flat).to(self.device)

        # GPU im2col + L2 norm
        # We need to adapt batch_im2col_l2norm_gpu to handle arbitrary batch sizes
        # For now, use a loop over the weights (which are per-landmark)

        # Tile the weights for all frames
        # Original weights: (68, out_dim, in_dim)
        # Tiled weights: (n_frames * 68, out_dim, in_dim)

        y_blocks = aoi_h - self.batched_cen.height_support + 1
        x_blocks = y_blocks
        num_windows = y_blocks * x_blocks

        # Get reorder indices
        patch_reorder, within_reorder = self.batched_cen._get_reorder_indices(y_blocks, x_blocks)

        # GPU unfold for im2col
        aois_4d = aois_t.unsqueeze(1)  # (N, 1, h, w)
        patches = F.unfold(aois_4d, kernel_size=self.batched_cen.height_support, stride=1)
        patches = patches.transpose(1, 2)  # (N, num_windows, patch_size)
        patches = patches[:, patch_reorder, :][:, :, within_reorder]

        # Add bias column
        ones = torch.ones(total_batch, num_windows, 1, device=self.device, dtype=torch.float32)
        patches = torch.cat([ones, patches], dim=-1)

        # L2 contrast norm (skip bias column)
        data = patches[:, :, 1:]
        mean = data.mean(dim=-1, keepdim=True)
        centered = data - mean
        sum_sq = (centered ** 2).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(sum_sq)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        patches[:, :, 1:] = centered / norm

        # Forward pass: need to apply per-landmark weights
        # Reshape patches to (n_frames, 68, num_windows, features)
        patches = patches.view(n_frames, 68, num_windows, -1)

        # Apply MLP for each landmark
        layer_output = patches
        for layer_idx in range(self.batched_cen.num_layers):
            # weights: (68, out_dim, in_dim)
            # layer_output: (n_frames, 68, num_windows, in_dim)
            # Result: (n_frames, 68, num_windows, out_dim)

            # Use einsum for batched matmul with per-landmark weights
            w = self.batched_cen.weights_t[layer_idx]  # (68, out_dim, in_dim)
            b = self.batched_cen.biases_t[layer_idx]   # (68, 1, out_dim)

            layer_output = torch.einsum('fbni,boi->fbno', layer_output, w)
            layer_output = layer_output + b.unsqueeze(0)  # Broadcast over frames

            # Activation
            act = self.batched_cen.activations[layer_idx]
            if act == 1:  # ReLU
                layer_output = torch.relu(layer_output)
            else:  # Sigmoid (act == 0)
                layer_output = torch.sigmoid(layer_output)

        # Output: (n_frames, 68, num_windows, 1) -> (n_frames * 68, num_windows)
        responses = layer_output[:, :, :, 0].reshape(-1, num_windows)

        return responses.cpu().numpy()

    def _batch_forward_cpu(self, aois_flat: np.ndarray, n_frames: int,
                           window_size: int) -> np.ndarray:
        """
        CPU batch forward pass (fallback).

        Args:
            aois_flat: Flattened AOIs (n_frames * 68, aoi_h, aoi_w)
            n_frames: Number of frames
            window_size: Response map size

        Returns:
            responses_flat: (n_frames * 68, num_windows)
        """
        total_batch = n_frames * 68
        aoi_h = aois_flat.shape[1]
        num_windows = window_size * window_size

        responses = np.zeros((total_batch, num_windows), dtype=np.float32)

        for frame_idx in range(n_frames):
            frame_aois = aois_flat[frame_idx * 68:(frame_idx + 1) * 68]

            # Use batched CEN methods
            patches = self.batched_cen.batch_im2col(frame_aois)
            normalized = self.batched_cen.batch_l2_contrast_norm(patches)
            frame_responses = self.batched_cen.batch_forward_numpy(normalized)

            responses[frame_idx * 68:(frame_idx + 1) * 68] = frame_responses

        return responses


def create_video_batch_processor(cen_model, scale: float, batch_size: int = 8,
                                  device: str = 'mps') -> VideoBatchProcessor:
    """
    Create a video batch processor for a specific scale.

    Args:
        cen_model: CENModel instance
        scale: Scale to process (0.25, 0.35, 0.5)
        batch_size: Number of frames to batch together
        device: GPU device

    Returns:
        VideoBatchProcessor instance
    """
    from .batched_cen import BatchedCEN

    patches = cen_model.scale_models[scale]['views'][0]['patches']
    batched_cen = BatchedCEN(patches, device=device)

    return VideoBatchProcessor(batched_cen, batch_size=batch_size, device=device)
