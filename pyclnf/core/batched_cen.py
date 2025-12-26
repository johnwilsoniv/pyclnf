#!/usr/bin/env python3
"""
Batched CEN (Convolutional Expert Network) patch expert computation.

Provides GPU-accelerated batched inference for all 68 landmarks simultaneously,
while maintaining exact numerical equivalence with the sequential implementation.

Key optimizations:
- Batch all 68 AOI extractions
- Batch im2col + L2 contrast normalization
- Batch MLP forward passes (per-landmark weights via einsum/bmm)
- GPU acceleration via PyTorch when available
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


class BatchedCEN:
    """
    Batched CEN patch expert computation.

    Provides exact numerical match to sequential CEN while processing
    all 68 landmarks in a single forward pass.
    """

    def __init__(self, patch_experts: Dict, device: str = 'cpu'):
        """
        Initialize batched CEN from existing patch experts.

        Args:
            patch_experts: Dict mapping landmark_idx -> CENPatchExpert
            device: 'cpu', 'cuda', or 'mps' for Apple Silicon
        """
        self.device = device
        self.num_landmarks = len(patch_experts)

        # Extract weights from patch experts
        # Each landmark has its own 2-layer MLP
        self._extract_weights(patch_experts)

        # Cache for im2col indices (computed once per window size)
        self._im2col_cache = {}

    def _extract_weights(self, patch_experts: Dict):
        """Extract and stack weights from individual patch experts."""
        # Get dimensions from first non-empty expert
        sample_expert = None
        for expert in patch_experts.values():
            if not expert.is_empty:
                sample_expert = expert
                break

        if sample_expert is None:
            raise ValueError("No valid patch experts found")

        self.width_support = sample_expert.width_support
        self.height_support = sample_expert.height_support
        self.patch_size = self.width_support * self.height_support

        # Get number of layers and dimensions
        self.num_layers = len(sample_expert.weights)
        self.layer_dims = [w.shape for w in sample_expert.weights]

        # Activation types (assume same for all landmarks)
        self.activations = sample_expert.activation_function.copy()

        # Initialize lists for each layer
        weights_lists = [[] for _ in range(self.num_layers)]
        biases_lists = [[] for _ in range(self.num_layers)]

        self.landmark_valid = []  # Track which landmarks have valid experts
        self.mirror_indices = {}  # Map empty landmarks to their mirrors

        for lm_idx in range(68):
            expert = patch_experts.get(lm_idx)

            if expert is None or expert.is_empty:
                # Use zeros for empty experts (will be handled via mirroring)
                for layer_idx in range(self.num_layers):
                    out_dim, in_dim = self.layer_dims[layer_idx]
                    weights_lists[layer_idx].append(np.zeros((out_dim, in_dim), dtype=np.float32))
                    biases_lists[layer_idx].append(np.zeros((1, out_dim), dtype=np.float32))
                self.landmark_valid.append(False)
            else:
                # Check if this is a MirroredCENPatchExpert
                actual_expert = expert._mirror_expert if hasattr(expert, '_mirror_expert') else expert

                for layer_idx in range(self.num_layers):
                    weights_lists[layer_idx].append(actual_expert.weights[layer_idx].astype(np.float32))
                    biases_lists[layer_idx].append(actual_expert.biases[layer_idx].astype(np.float32))

                self.landmark_valid.append(True)
                if hasattr(expert, '_mirror_expert'):
                    self.mirror_indices[lm_idx] = True  # Mark as needing flip

        # Stack into batched tensors: list of (68, out_dim, in_dim) arrays
        self.weights = [np.stack(wl, axis=0) for wl in weights_lists]
        self.biases = [np.stack(bl, axis=0) for bl in biases_lists]

        # Convert to torch tensors if available
        if TORCH_AVAILABLE:
            self.weights_t = [torch.from_numpy(w).to(self.device) for w in self.weights]
            self.biases_t = [torch.from_numpy(b).to(self.device) for b in self.biases]

            # Precompute reorder indices for GPU unfold -> column-major conversion
            # Only needs to be done once per (patch_size, response_size) combination
            self._reorder_cache = {}

    def batch_extract_aoi(self, image: np.ndarray, landmarks: np.ndarray,
                          window_size: int,
                          sim_ref_to_img: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract areas of interest for all 68 landmarks.

        Supports optional warping via similarity transform for multi-scale processing.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Current landmark positions (68, 2) in IMAGE coordinates
            window_size: Response map window size (e.g., 11, 9, 7, 5)
            sim_ref_to_img: Optional 2x3 similarity transform (reference → image)
                           If provided, extracts warped patches matching C++ OpenFace

        Returns:
            aois: Batched AOI patches (68, aoi_h, aoi_w)
            valid_mask: Boolean mask for valid extractions
        """
        # AOI size = window_size + support_size - 1
        aoi_size = window_size + self.width_support - 1

        if sim_ref_to_img is not None:
            # WARPED extraction - matches C++ OpenFace behavior
            return self._batch_extract_aoi_warped(image, landmarks, aoi_size, sim_ref_to_img)
        else:
            # Direct extraction (no warping)
            return self._batch_extract_aoi_direct(image, landmarks, aoi_size)

    def _batch_extract_aoi_direct(self, image: np.ndarray, landmarks: np.ndarray,
                                   aoi_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Direct AOI extraction without warping."""
        half_aoi = aoi_size // 2
        h, w = image.shape
        aois = np.zeros((68, aoi_size, aoi_size), dtype=np.float32)
        valid_mask = np.ones(68, dtype=bool)

        for lm_idx in range(68):
            lm_x, lm_y = landmarks[lm_idx]

            # Compute extraction bounds
            x1 = int(lm_x - half_aoi)
            y1 = int(lm_y - half_aoi)
            x2 = x1 + aoi_size
            y2 = y1 + aoi_size

            # Handle boundary cases with padding
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - w)
            pad_bottom = max(0, y2 - h)

            # Clamp to image bounds
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(w, x2)
            y2_clamped = min(h, y2)

            # Extract valid region
            if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                patch = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                # Place in output with padding
                out_y1 = pad_top
                out_y2 = pad_top + patch.shape[0]
                out_x1 = pad_left
                out_x2 = pad_left + patch.shape[1]

                aois[lm_idx, out_y1:out_y2, out_x1:out_x2] = patch
            else:
                valid_mask[lm_idx] = False

        return aois, valid_mask

    def _batch_extract_aoi_warped(self, image: np.ndarray, landmarks: np.ndarray,
                                   aoi_size: int, sim_ref_to_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warped AOI extraction matching C++ OpenFace.

        Uses cv2.warpAffine with WARP_INVERSE_MAP to extract patches
        centered at each landmark in the warped reference frame.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Landmark positions (68, 2) in IMAGE coordinates
            aoi_size: Size of AOI patch to extract
            sim_ref_to_img: 2x3 similarity transform (reference → image)

        Returns:
            aois: Batched AOI patches (68, aoi_size, aoi_size)
            valid_mask: Boolean mask (all True for warped extraction)
        """
        aois = np.zeros((68, aoi_size, aoi_size), dtype=np.float32)
        valid_mask = np.ones(68, dtype=bool)

        # Extract similarity transform components
        a1 = sim_ref_to_img[0, 0]
        b1 = -sim_ref_to_img[0, 1]  # Note: NEGATIVE sign matches C++

        center_offset = (aoi_size - 1.0) / 2.0

        for lm_idx in range(68):
            center_x, center_y = landmarks[lm_idx]

            # Build warp matrix centered at this landmark
            # This matches _extract_aoi in optimizer.py
            tx = center_x - a1 * center_offset + b1 * center_offset
            ty = center_y - a1 * center_offset - b1 * center_offset

            sim_matrix = np.array([
                [a1, -b1, tx],
                [b1,  a1, ty]
            ], dtype=np.float32)

            # Extract warped patch
            aois[lm_idx] = cv2.warpAffine(
                image,
                sim_matrix,
                (aoi_size, aoi_size),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            )

        return aois, valid_mask

    def _batch_extract_aoi_warped_gpu(self, image: np.ndarray, landmarks: np.ndarray,
                                       aoi_size: int, sim_ref_to_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated warped AOI extraction using grid_sample.

        Extracts all 68 AOIs in a single GPU operation.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Landmark positions (68, 2) in IMAGE coordinates
            aoi_size: Size of AOI patch to extract
            sim_ref_to_img: 2x3 similarity transform (reference → image)

        Returns:
            aois: Batched AOI patches (68, aoi_size, aoi_size)
            valid_mask: Boolean mask (all True)
        """
        if not TORCH_AVAILABLE:
            return self._batch_extract_aoi_warped(image, landmarks, aoi_size, sim_ref_to_img)

        h, w = image.shape

        # Move image to GPU: (1, 1, H, W)
        image_t = torch.from_numpy(image).to(self.device).unsqueeze(0).unsqueeze(0)

        # Extract similarity transform components
        a1 = sim_ref_to_img[0, 0]
        b1 = -sim_ref_to_img[0, 1]  # Note: NEGATIVE sign

        center_offset = (aoi_size - 1.0) / 2.0

        # Create sampling grid for all 68 landmarks
        # Grid coordinates in [-1, 1] for grid_sample
        y_coords = torch.arange(aoi_size, dtype=torch.float32, device=self.device)
        x_coords = torch.arange(aoi_size, dtype=torch.float32, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Offset from center
        grid_x_centered = grid_x - center_offset  # (aoi_size, aoi_size)
        grid_y_centered = grid_y - center_offset

        # Transform grid points for each landmark
        landmarks_t = torch.from_numpy(landmarks.astype(np.float32)).to(self.device)

        # For each landmark, compute the sampling coordinates
        # sample_x = center_x + a1 * (grid_x - center) - b1 * (grid_y - center)
        # sample_y = center_y + b1 * (grid_x - center) + a1 * (grid_y - center)

        # Broadcast landmarks: (68, 1, 1)
        cx = landmarks_t[:, 0].view(68, 1, 1)
        cy = landmarks_t[:, 1].view(68, 1, 1)

        # Compute sample coordinates for all landmarks at once
        sample_x = cx + a1 * grid_x_centered - b1 * grid_y_centered  # (68, aoi_size, aoi_size)
        sample_y = cy + b1 * grid_x_centered + a1 * grid_y_centered

        # Normalize to [-1, 1] for grid_sample
        sample_x_norm = 2.0 * sample_x / (w - 1) - 1.0
        sample_y_norm = 2.0 * sample_y / (h - 1) - 1.0

        # Stack into grid: (68, aoi_size, aoi_size, 2)
        grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

        # Expand image for batch: (68, 1, H, W)
        image_batch = image_t.expand(68, -1, -1, -1)

        # Sample all AOIs at once
        aois_t = F.grid_sample(
            image_batch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (68, 1, aoi_size, aoi_size)

        # Remove channel dim and move to CPU
        aois = aois_t.squeeze(1).cpu().numpy()
        valid_mask = np.ones(68, dtype=bool)

        return aois, valid_mask

    def batch_im2col(self, aois: np.ndarray) -> np.ndarray:
        """
        Batched im2col with bias column.

        Extracts sliding window patches in column-major order to match C++.

        Args:
            aois: Batched AOI patches (batch, aoi_h, aoi_w)

        Returns:
            patches: (batch, num_windows, patch_size+1) with bias column
        """
        batch_size = aois.shape[0]
        aoi_h, aoi_w = aois.shape[1], aois.shape[2]

        y_blocks = aoi_h - self.height_support + 1
        x_blocks = aoi_w - self.width_support + 1
        num_windows = y_blocks * x_blocks

        # Output shape: (batch, num_windows, patch_size + 1)
        output = np.ones((batch_size, num_windows, self.patch_size + 1), dtype=np.float32)

        # Extract patches in column-major order (j outer, i inner)
        for j in range(x_blocks):
            for i in range(y_blocks):
                row_idx = i + j * y_blocks  # Column-major patch ordering

                # Extract patch for all batch items
                patch = aois[:, i:i+self.height_support, j:j+self.width_support]

                # Flatten in column-major order (transpose then flatten)
                # For each batch item: (height, width) -> (width, height) -> flatten
                patch_flat = patch.transpose(0, 2, 1).reshape(batch_size, -1)

                output[:, row_idx, 1:] = patch_flat

        return output

    def batch_l2_contrast_norm(self, patches: np.ndarray) -> np.ndarray:
        """
        Batched L2 contrast normalization.

        Matches C++ contrastNorm exactly:
        - norm = sqrt(sum((x - mean)²))  [L2 norm, NOT std deviation]

        Args:
            patches: (batch, num_windows, patch_size+1) with bias column

        Returns:
            normalized: Same shape, with columns 1: normalized
        """
        output = patches.copy()

        # Skip first column (bias), normalize rest
        data = output[:, :, 1:]  # (batch, num_windows, patch_size)

        # Compute mean per patch (axis=-1)
        mean = data.mean(axis=-1, keepdims=True)  # (batch, num_windows, 1)

        # Subtract mean
        centered = data - mean

        # Compute L2 norm (NOT std - no division by n)
        sum_sq = (centered ** 2).sum(axis=-1, keepdims=True)
        norm = np.sqrt(sum_sq)

        # Avoid division by zero (C++ uses exact 0 comparison)
        norm = np.where(norm == 0, 1.0, norm)

        # Normalize
        output[:, :, 1:] = centered / norm

        return output

    def _get_reorder_indices(self, y_blocks: int, x_blocks: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached reorder indices for unfold -> column-major conversion."""
        cache_key = (y_blocks, x_blocks, self.height_support)

        if cache_key not in self._reorder_cache:
            # Patch reorder: row-major iteration order -> column-major
            patch_reorder = []
            for j in range(x_blocks):
                for i in range(y_blocks):
                    row_major_idx = i * x_blocks + j
                    patch_reorder.append(row_major_idx)

            # Within-patch reorder: row-major flatten -> column-major flatten
            within_patch_reorder = []
            for xx in range(self.width_support):
                for yy in range(self.height_support):
                    row_major_idx = yy * self.width_support + xx
                    within_patch_reorder.append(row_major_idx)

            self._reorder_cache[cache_key] = (
                torch.tensor(patch_reorder, device=self.device, dtype=torch.long),
                torch.tensor(within_patch_reorder, device=self.device, dtype=torch.long)
            )

        return self._reorder_cache[cache_key]

    def batch_im2col_l2norm_gpu(self, aois_t: 'torch.Tensor') -> 'torch.Tensor':
        """
        GPU-accelerated im2col + L2 contrast normalization.

        Uses torch.nn.functional.unfold with reordering to match column-major im2col.

        Args:
            aois_t: (68, aoi_h, aoi_w) tensor on GPU

        Returns:
            normalized: (68, num_windows, patch_size+1) with bias column
        """
        batch_size = aois_t.shape[0]
        aoi_h, aoi_w = aois_t.shape[1], aois_t.shape[2]
        y_blocks = aoi_h - self.height_support + 1
        x_blocks = aoi_w - self.width_support + 1

        # Add channel dim for unfold: (68, 1, h, w)
        aois_4d = aois_t.unsqueeze(1)

        # Extract patches using unfold: (68, patch_size^2, num_windows)
        patches = F.unfold(aois_4d, kernel_size=self.height_support, stride=1)

        # Transpose to (68, num_windows, patch_size^2)
        patches = patches.transpose(1, 2)

        # Reorder to match column-major im2col
        patch_reorder, within_reorder = self._get_reorder_indices(y_blocks, x_blocks)
        patches = patches[:, patch_reorder, :][:, :, within_reorder]

        # Add bias column: (68, num_windows, patch_size^2+1)
        bias = torch.ones(batch_size, patches.shape[1], 1, device=self.device, dtype=patches.dtype)
        patches = torch.cat([bias, patches], dim=2)

        # L2 contrast normalization (skip bias column)
        # Optimized: use x*x instead of x**2 (2x faster on MPS)
        data = patches[:, :, 1:]
        mean = data.mean(dim=-1, keepdim=True)
        centered = data - mean
        sum_sq = (centered * centered).sum(dim=-1, keepdim=True)
        norm = sum_sq.sqrt()
        norm = torch.clamp(norm, min=1e-10)  # Faster than where()
        patches[:, :, 1:] = centered / norm

        return patches

    def _apply_activation(self, x: np.ndarray, activation_type: int) -> np.ndarray:
        """Apply activation function."""
        if activation_type == 0:  # Sigmoid
            x = np.clip(x, -88, 88)
            return 1.0 / (1.0 + np.exp(-x))
        elif activation_type == 1:  # Tanh
            return np.tanh(x)
        elif activation_type == 2:  # ReLU
            return np.maximum(0, x)
        else:  # Linear
            return x

    def batch_forward_numpy(self, patches: np.ndarray) -> np.ndarray:
        """
        Batched MLP forward pass using NumPy.

        Args:
            patches: (68, num_windows, patch_size+1) normalized patches

        Returns:
            responses: (68, num_windows) response values
        """
        layer_output = patches

        for layer_idx in range(self.num_layers):
            # weights[layer_idx]: (68, out_dim, in_dim)
            # biases[layer_idx]: (68, 1, out_dim)
            # layer_output: (68, num_windows, in_dim)

            # Batched matmul: output = input @ weights.T + bias
            # Using einsum: 'bni,boi->bno' where o=out_dim, i=in_dim
            layer_output = np.einsum('bni,boi->bno', layer_output, self.weights[layer_idx])
            layer_output = layer_output + self.biases[layer_idx]

            # Apply activation
            layer_output = self._apply_activation(layer_output, self.activations[layer_idx])

        return layer_output[:, :, 0]  # (68, num_windows)

    def _apply_activation_torch(self, x: 'torch.Tensor', activation_type: int) -> 'torch.Tensor':
        """Apply activation function (PyTorch version)."""
        if activation_type == 0:  # Sigmoid
            x = torch.clamp(x, -88, 88)
            return torch.sigmoid(x)
        elif activation_type == 1:  # Tanh
            return torch.tanh(x)
        elif activation_type == 2:  # ReLU
            return torch.relu(x)
        else:  # Linear
            return x

    def batch_forward_torch(self, patches: np.ndarray) -> np.ndarray:
        """
        Batched MLP forward pass using PyTorch for GPU acceleration.

        Args:
            patches: (68, num_windows, patch_size+1) normalized patches

        Returns:
            responses: (68, num_windows) response values
        """
        layer_output = torch.from_numpy(patches).to(self.device)

        for layer_idx in range(self.num_layers):
            # Batched matmul: output = input @ weights.T + bias
            # Using bmm (1.29x faster than einsum on MPS)
            layer_output = torch.bmm(layer_output, self.weights_t[layer_idx].transpose(1, 2))
            layer_output = layer_output + self.biases_t[layer_idx]

            # Apply activation
            layer_output = self._apply_activation_torch(layer_output, self.activations[layer_idx])

        return layer_output[:, :, 0].cpu().numpy()

    def compute_response_maps_gpu(self, image: np.ndarray, landmarks: np.ndarray,
                                   window_size: int,
                                   sim_ref_to_img: np.ndarray = None,
                                   use_gpu_warp: bool = False) -> Dict[int, np.ndarray]:
        """
        GPU-accelerated response map computation.

        Uses CPU for AOI extraction (to match OpenCV exactly) but GPU for
        im2col, L2 norm, and CEN forward pass.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Current landmark positions (68, 2)
            window_size: Response map window size
            sim_ref_to_img: Optional 2x3 similarity transform (reference → image)
            use_gpu_warp: If True, use GPU grid_sample for warping (faster but ~0.5px diff)
                         Default False for numerical accuracy

        Returns:
            response_maps: Dict mapping landmark_idx -> response map (h, w)
        """
        # Step 1: Extract AOIs with optional warping
        # Use CPU warpAffine by default to match OpenCV exactly
        # GPU warp is faster but has ~0.5px interpolation difference
        aoi_size = window_size + self.width_support - 1

        if sim_ref_to_img is not None and use_gpu_warp and TORCH_AVAILABLE and self.device != 'cpu':
            # GPU-accelerated warped extraction (faster, small interpolation diff)
            aois, valid_mask = self._batch_extract_aoi_warped_gpu(image, landmarks, aoi_size, sim_ref_to_img)
        else:
            # CPU extraction - matches OpenCV exactly
            aois, valid_mask = self.batch_extract_aoi(image, landmarks, window_size, sim_ref_to_img)

        # Handle mirrored landmarks
        for lm_idx in self.mirror_indices:
            if valid_mask[lm_idx]:
                aois[lm_idx] = cv2.flip(aois[lm_idx], 1)

        # Step 2: Move to GPU and run im2col + L2 norm + forward
        aois_t = torch.from_numpy(aois).to(self.device)

        # GPU im2col + L2 norm
        normalized = self.batch_im2col_l2norm_gpu(aois_t)

        # GPU forward pass (bmm is 1.29x faster than einsum on MPS)
        layer_output = normalized
        for layer_idx in range(self.num_layers):
            layer_output = torch.bmm(layer_output, self.weights_t[layer_idx].transpose(1, 2))
            layer_output = layer_output + self.biases_t[layer_idx]
            layer_output = self._apply_activation_torch(layer_output, self.activations[layer_idx])

        responses_flat = layer_output[:, :, 0].cpu().numpy()

        # Step 3: Reshape to response maps
        aoi_h = aois.shape[1]
        response_height = aoi_h - self.height_support + 1
        response_width = response_height  # Square

        response_maps = {}
        for lm_idx in range(68):
            if valid_mask[lm_idx] and self.landmark_valid[lm_idx]:
                response = responses_flat[lm_idx].reshape(
                    response_height, response_width, order='F'
                )
                if lm_idx in self.mirror_indices:
                    response = cv2.flip(response, 1)
                response_maps[lm_idx] = response.astype(np.float32, copy=False)
            else:
                response_maps[lm_idx] = np.zeros(
                    (response_height, response_width), dtype=np.float32
                )

        return response_maps

    def compute_response_maps(self, image: np.ndarray, landmarks: np.ndarray,
                               window_size: int,
                               sim_ref_to_img: np.ndarray = None) -> Dict[int, np.ndarray]:
        """
        Compute response maps for all 68 landmarks in a single batched pass.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Current landmark positions (68, 2)
            window_size: Response map window size (e.g., 11, 9, 7, 5)
            sim_ref_to_img: Optional 2x3 similarity transform (reference → image)

        Returns:
            response_maps: Dict mapping landmark_idx -> response map (h, w)
        """
        # Step 1: Extract AOIs for all landmarks (with optional warping)
        aois, valid_mask = self.batch_extract_aoi(image, landmarks, window_size, sim_ref_to_img)

        # Handle mirrored landmarks: flip their AOIs
        for lm_idx in self.mirror_indices:
            if valid_mask[lm_idx]:
                aois[lm_idx] = cv2.flip(aois[lm_idx], 1)  # Horizontal flip

        # Step 2: Batched im2col
        patches = self.batch_im2col(aois)  # (68, num_windows, patch_size+1)

        # Step 3: Batched L2 contrast normalization
        normalized = self.batch_l2_contrast_norm(patches)

        # Step 4: Batched MLP forward pass
        if TORCH_AVAILABLE and self.device != 'cpu':
            responses_flat = self.batch_forward_torch(normalized)
        else:
            responses_flat = self.batch_forward_numpy(normalized)

        # Step 5: Reshape to response maps (column-major order)
        aoi_h = aois.shape[1]
        aoi_w = aois.shape[2]
        response_height = aoi_h - self.height_support + 1
        response_width = aoi_w - self.width_support + 1

        response_maps = {}
        for lm_idx in range(68):
            if valid_mask[lm_idx] and self.landmark_valid[lm_idx]:
                # Reshape in column-major order to match original
                response = responses_flat[lm_idx].reshape(
                    response_height, response_width, order='F'
                )

                # Flip back mirrored landmarks
                if lm_idx in self.mirror_indices:
                    response = cv2.flip(response, 1)

                response_maps[lm_idx] = response.astype(np.float32, copy=False)
            else:
                # Return zero response for invalid landmarks
                response_maps[lm_idx] = np.zeros(
                    (response_height, response_width), dtype=np.float32
                )

        return response_maps


def create_batched_cen(scale_models: Dict, scale: float,
                       device: str = 'cpu') -> BatchedCEN:
    """
    Create a BatchedCEN instance from pyclnf's scale_models structure.

    Args:
        scale_models: Dict from CENModel.scale_models
        scale: Scale to create batched CEN for (e.g., 0.25, 0.35, 0.5)
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        BatchedCEN instance
    """
    if scale not in scale_models:
        raise ValueError(f"Scale {scale} not found in scale_models")

    # Get patch experts for frontal view (view 0)
    patches = scale_models[scale]['views'][0]['patches']

    return BatchedCEN(patches, device=device)
