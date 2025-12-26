#!/usr/bin/env python3
"""
CoreML backend for batched CEN computation.

Provides potentially faster inference on Apple Silicon by using
CoreML's optimized execution on GPU/ANE instead of PyTorch MPS.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import hashlib

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class BatchedCENTorch(nn.Module):
    """
    PyTorch module for batched CEN - used for CoreML conversion.

    Implements the exact same forward pass as BatchedCEN but as a
    proper nn.Module that can be traced and converted.
    """

    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray],
                 activations: List[int]):
        """
        Initialize from extracted weights.

        Args:
            weights: List of weight arrays per layer, each (68, out_dim, in_dim)
            biases: List of bias arrays per layer, each (68, 1, out_dim)
            activations: List of activation types per layer (0=sigmoid, 2=relu)
        """
        super().__init__()

        self.num_layers = len(weights)
        self.activations = activations

        # Register weights and biases as buffers (not parameters - we don't train)
        for i, (w, b) in enumerate(zip(weights, biases)):
            # Transpose weights for bmm: (68, out, in) -> (68, in, out)
            self.register_buffer(f'weight_{i}', torch.from_numpy(w.transpose(0, 2, 1).copy()))
            self.register_buffer(f'bias_{i}', torch.from_numpy(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (68, num_windows, input_dim)

        Returns:
            Output tensor (68, num_windows, 1)
        """
        for i in range(self.num_layers):
            weight = getattr(self, f'weight_{i}')
            bias = getattr(self, f'bias_{i}')

            # Batched matmul: (68, N, in) @ (68, in, out) -> (68, N, out)
            x = torch.bmm(x, weight) + bias

            # Apply activation
            if self.activations[i] == 0:  # Sigmoid
                x = torch.clamp(x, -88, 88)
                x = torch.sigmoid(x)
            elif self.activations[i] == 2:  # ReLU
                x = torch.relu(x)
            # else: linear (no activation)

        return x


class CoreMLCEN:
    """
    CoreML-accelerated CEN forward pass.

    Converts the batched CEN MLP to CoreML for potentially faster
    inference on Apple Silicon GPU/ANE.
    """

    # Class-level cache for compiled models
    _model_cache: Dict[str, 'ct.models.MLModel'] = {}

    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray],
                 activations: List[int], num_windows: int = 121,
                 compute_units: str = 'CPU_AND_GPU'):
        """
        Initialize CoreML CEN backend.

        Args:
            weights: List of weight arrays per layer, each (68, out_dim, in_dim)
            biases: List of bias arrays per layer, each (68, 1, out_dim)
            activations: List of activation types per layer
            num_windows: Number of windows (response map size squared)
            compute_units: 'CPU_ONLY', 'CPU_AND_GPU', or 'ALL' (includes ANE)
        """
        if not COREML_AVAILABLE:
            raise ImportError("coremltools not available")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CoreML conversion")

        self.num_windows = num_windows
        self.input_dim = weights[0].shape[2]  # (68, out, in) -> in
        self.compute_units = compute_units

        # Create cache key from weights hash
        self._cache_key = self._compute_cache_key(weights, biases, num_windows)

        # Get or create CoreML model
        if self._cache_key in CoreMLCEN._model_cache:
            self.model = CoreMLCEN._model_cache[self._cache_key]
        else:
            self.model = self._create_coreml_model(weights, biases, activations)
            CoreMLCEN._model_cache[self._cache_key] = self.model

    def _compute_cache_key(self, weights: List[np.ndarray], biases: List[np.ndarray],
                           num_windows: int) -> str:
        """Compute cache key from weights."""
        hasher = hashlib.md5()
        for w in weights:
            hasher.update(w.tobytes()[:1000])  # Sample first 1000 bytes
        hasher.update(str(num_windows).encode())
        return hasher.hexdigest()

    def _create_coreml_model(self, weights: List[np.ndarray], biases: List[np.ndarray],
                              activations: List[int]) -> 'ct.models.MLModel':
        """Create CoreML model from weights."""
        # Create PyTorch model
        torch_model = BatchedCENTorch(weights, biases, activations)
        torch_model.eval()

        # Create example input
        example_input = torch.randn(68, self.num_windows, self.input_dim)

        # Trace the model
        traced = torch.jit.trace(torch_model, example_input)

        # Convert to CoreML
        compute_unit_map = {
            'CPU_ONLY': ct.ComputeUnit.CPU_ONLY,
            'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
            'ALL': ct.ComputeUnit.ALL,
        }

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=(68, self.num_windows, self.input_dim))],
            outputs=[ct.TensorType(name="output")],
            compute_precision=ct.precision.FLOAT32,
            compute_units=compute_unit_map.get(self.compute_units, ct.ComputeUnit.CPU_AND_GPU),
            minimum_deployment_target=ct.target.macOS13,
        )

        return mlmodel

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run forward pass through CoreML model.

        Args:
            x: Input array (68, num_windows, input_dim)

        Returns:
            Output array (68, num_windows, 1)
        """
        # CoreML expects dict input
        result = self.model.predict({'input': x})
        return result['output']


class CoreMLBatchedCEN:
    """
    CoreML-accelerated BatchedCEN replacement.

    Drop-in replacement for BatchedCEN that uses CoreML for the
    forward pass while keeping AOI extraction on CPU.
    """

    def __init__(self, patch_experts: Dict, compute_units: str = 'CPU_AND_GPU'):
        """
        Initialize CoreML-accelerated batched CEN.

        Args:
            patch_experts: Dict mapping landmark_idx -> CENPatchExpert
            compute_units: 'CPU_ONLY', 'CPU_AND_GPU', or 'ALL'
        """
        self.compute_units = compute_units

        # Extract weights (same as BatchedCEN)
        self._extract_weights(patch_experts)

        # CoreML models for different window sizes
        self._coreml_models: Dict[int, CoreMLCEN] = {}

        # Reorder indices cache (same as BatchedCEN)
        self._reorder_cache = {}

    def _extract_weights(self, patch_experts: Dict):
        """Extract weights from patch experts (same as BatchedCEN)."""
        import cv2
        self.cv2 = cv2

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

        self.num_layers = len(sample_expert.weights)
        self.layer_dims = [w.shape for w in sample_expert.weights]
        self.activations = sample_expert.activation_function.copy()

        weights_lists = [[] for _ in range(self.num_layers)]
        biases_lists = [[] for _ in range(self.num_layers)]

        self.landmark_valid = []
        self.mirror_indices = {}

        for lm_idx in range(68):
            expert = patch_experts.get(lm_idx)

            if expert is None or expert.is_empty:
                for layer_idx in range(self.num_layers):
                    out_dim, in_dim = self.layer_dims[layer_idx]
                    weights_lists[layer_idx].append(np.zeros((out_dim, in_dim), dtype=np.float32))
                    biases_lists[layer_idx].append(np.zeros((1, out_dim), dtype=np.float32))
                self.landmark_valid.append(False)
            else:
                actual_expert = expert._mirror_expert if hasattr(expert, '_mirror_expert') else expert

                for layer_idx in range(self.num_layers):
                    weights_lists[layer_idx].append(actual_expert.weights[layer_idx].astype(np.float32))
                    biases_lists[layer_idx].append(actual_expert.biases[layer_idx].astype(np.float32))

                self.landmark_valid.append(True)
                if hasattr(expert, '_mirror_expert'):
                    self.mirror_indices[lm_idx] = True

        # Stack into batched arrays
        self.weights = [np.stack(wl, axis=0) for wl in weights_lists]
        self.biases = [np.stack(bl, axis=0) for bl in biases_lists]

    def _get_coreml_model(self, num_windows: int) -> CoreMLCEN:
        """Get or create CoreML model for given window count."""
        if num_windows not in self._coreml_models:
            self._coreml_models[num_windows] = CoreMLCEN(
                self.weights, self.biases, self.activations,
                num_windows=num_windows,
                compute_units=self.compute_units
            )
        return self._coreml_models[num_windows]

    def _batch_extract_aoi_warped(self, image: np.ndarray, landmarks: np.ndarray,
                                   aoi_size: int, sim_ref_to_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Warped AOI extraction (same as BatchedCEN)."""
        aois = np.zeros((68, aoi_size, aoi_size), dtype=np.float32)
        valid_mask = np.ones(68, dtype=bool)

        a1 = sim_ref_to_img[0, 0]
        b1 = -sim_ref_to_img[0, 1]
        center_offset = (aoi_size - 1.0) / 2.0

        for lm_idx in range(68):
            center_x, center_y = landmarks[lm_idx]
            tx = center_x - a1 * center_offset + b1 * center_offset
            ty = center_y - a1 * center_offset - b1 * center_offset

            sim_matrix = np.array([
                [a1, -b1, tx],
                [b1,  a1, ty]
            ], dtype=np.float32)

            aois[lm_idx] = self.cv2.warpAffine(
                image, sim_matrix, (aoi_size, aoi_size),
                flags=self.cv2.WARP_INVERSE_MAP | self.cv2.INTER_LINEAR
            )

        return aois, valid_mask

    def _batch_im2col_l2norm(self, aois: np.ndarray) -> np.ndarray:
        """Batched im2col + L2 contrast normalization."""
        batch_size = aois.shape[0]
        aoi_h, aoi_w = aois.shape[1], aois.shape[2]

        y_blocks = aoi_h - self.height_support + 1
        x_blocks = aoi_w - self.width_support + 1
        num_windows = y_blocks * x_blocks

        # Extract patches in column-major order
        output = np.ones((batch_size, num_windows, self.patch_size + 1), dtype=np.float32)

        for j in range(x_blocks):
            for i in range(y_blocks):
                row_idx = i + j * y_blocks
                patch = aois[:, i:i+self.height_support, j:j+self.width_support]
                patch_flat = patch.transpose(0, 2, 1).reshape(batch_size, -1)
                output[:, row_idx, 1:] = patch_flat

        # L2 contrast normalization
        data = output[:, :, 1:]
        mean = data.mean(axis=-1, keepdims=True)
        centered = data - mean
        sum_sq = (centered ** 2).sum(axis=-1, keepdims=True)
        norm = np.sqrt(sum_sq)
        norm = np.where(norm == 0, 1.0, norm)
        output[:, :, 1:] = centered / norm

        return output

    def compute_response_maps(self, image: np.ndarray, landmarks: np.ndarray,
                               window_size: int,
                               sim_ref_to_img: np.ndarray = None) -> Dict[int, np.ndarray]:
        """
        Compute response maps using CoreML backend.

        Args:
            image: Grayscale image (H, W) as float32
            landmarks: Current landmark positions (68, 2)
            window_size: Response map window size
            sim_ref_to_img: Optional 2x3 similarity transform

        Returns:
            response_maps: Dict mapping landmark_idx -> response map
        """
        aoi_size = window_size + self.width_support - 1

        # Extract AOIs
        if sim_ref_to_img is not None:
            aois, valid_mask = self._batch_extract_aoi_warped(image, landmarks, aoi_size, sim_ref_to_img)
        else:
            # Fallback to direct extraction
            aois, valid_mask = self._batch_extract_aoi_direct(image, landmarks, aoi_size)

        # Handle mirrored landmarks
        for lm_idx in self.mirror_indices:
            if valid_mask[lm_idx]:
                aois[lm_idx] = self.cv2.flip(aois[lm_idx], 1)

        # im2col + L2 normalization
        normalized = self._batch_im2col_l2norm(aois)

        # Get CoreML model for this window size
        num_windows = (aoi_size - self.height_support + 1) ** 2
        coreml_model = self._get_coreml_model(num_windows)

        # CoreML forward pass
        output = coreml_model.forward(normalized)  # (68, num_windows, 1)
        responses_flat = output[:, :, 0]  # (68, num_windows)

        # Reshape to response maps
        response_height = aoi_size - self.height_support + 1
        response_width = response_height

        response_maps = {}
        for lm_idx in range(68):
            if valid_mask[lm_idx] and self.landmark_valid[lm_idx]:
                response = responses_flat[lm_idx].reshape(
                    response_height, response_width, order='F'
                )
                if lm_idx in self.mirror_indices:
                    response = self.cv2.flip(response, 1)
                response_maps[lm_idx] = response.astype(np.float32, copy=False)
            else:
                response_maps[lm_idx] = np.zeros(
                    (response_height, response_width), dtype=np.float32
                )

        return response_maps

    def _batch_extract_aoi_direct(self, image: np.ndarray, landmarks: np.ndarray,
                                   aoi_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Direct AOI extraction without warping."""
        half_aoi = aoi_size // 2
        h, w = image.shape
        aois = np.zeros((68, aoi_size, aoi_size), dtype=np.float32)
        valid_mask = np.ones(68, dtype=bool)

        for lm_idx in range(68):
            lm_x, lm_y = landmarks[lm_idx]
            x1 = int(lm_x - half_aoi)
            y1 = int(lm_y - half_aoi)
            x2 = x1 + aoi_size
            y2 = y1 + aoi_size

            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - w)
            pad_bottom = max(0, y2 - h)

            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(w, x2)
            y2_clamped = min(h, y2)

            if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                patch = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                out_y1 = pad_top
                out_y2 = pad_top + patch.shape[0]
                out_x1 = pad_left
                out_x2 = pad_left + patch.shape[1]
                aois[lm_idx, out_y1:out_y2, out_x1:out_x2] = patch
            else:
                valid_mask[lm_idx] = False

        return aois, valid_mask


def create_coreml_cen(patch_experts: Dict, compute_units: str = 'CPU_AND_GPU') -> CoreMLBatchedCEN:
    """
    Create a CoreML-accelerated BatchedCEN.

    Args:
        patch_experts: Dict mapping landmark_idx -> CENPatchExpert
        compute_units: 'CPU_ONLY', 'CPU_AND_GPU', or 'ALL'

    Returns:
        CoreMLBatchedCEN instance
    """
    return CoreMLBatchedCEN(patch_experts, compute_units)
