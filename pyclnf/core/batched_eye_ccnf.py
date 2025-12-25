"""
Batched Eye CCNF - GPU-accelerated eye landmark response map computation.

This module provides GPU batching for eye CCNF patch experts, following the
same pattern as BatchedCEN for the main face model.

Eye CCNF structure:
- 28 landmarks per eye
- 11x11 patch size (121 features)
- 7 neurons per patch
- 2 scales: 1.0 and 1.5
"""

import numpy as np
import torch
from typing import Dict, Optional
import cv2


class EyeVectorizedMeanShift:
    """
    NumPy-vectorized KDE mean-shift for eye landmarks (28 points).

    Uses pure NumPy for vectorization - avoids GPU transfer overhead
    which dominates for small arrays like eye response maps.
    """

    def __init__(self, n_landmarks: int = 28):
        self.n_landmarks = n_landmarks
        # Cache for coordinate grids
        self._grid_cache = {}

    def _get_grid(self, window_size: int):
        """Get or create coordinate grid for given window size."""
        if window_size not in self._grid_cache:
            # Create meshgrid - jj varies along columns (x), ii along rows (y)
            jj, ii = np.meshgrid(
                np.arange(window_size, dtype=np.float32),
                np.arange(window_size, dtype=np.float32)
            )
            self._grid_cache[window_size] = (ii, jj)
        return self._grid_cache[window_size]

    def compute(self,
                response_maps: Dict[int, np.ndarray],
                eye_landmarks: np.ndarray,
                base_landmarks: np.ndarray,
                window_size: int,
                sigma: float,
                sim_img_to_ref: np.ndarray = None) -> np.ndarray:
        """
        Compute mean-shift for all 28 eye landmarks using vectorized NumPy.

        Args:
            response_maps: Dict mapping landmark_idx -> response_map (ws, ws)
            eye_landmarks: Current landmarks (28, 2)
            base_landmarks: Base landmarks where response maps were extracted (28, 2)
            window_size: Response window size (3 or 5)
            sigma: Gaussian kernel sigma
            sim_img_to_ref: 2x2 transform from image to reference space

        Returns:
            mean_shift: (56,) in stacked format [x0..x27, y0..y27]
        """
        n = self.n_landmarks
        ws = window_size
        center = (ws - 1) / 2.0
        a_kde = -0.5 / (sigma * sigma)

        # Get cached grid
        ii, jj = self._get_grid(ws)  # ii = y coords (rows), jj = x coords (cols)

        # Stack response maps into array (28, ws, ws)
        responses = np.zeros((n, ws, ws), dtype=np.float32)
        valid_mask = np.zeros(n, dtype=bool)

        for lm_idx, resp_map in response_maps.items():
            if lm_idx < n:
                responses[lm_idx] = resp_map
                valid_mask[lm_idx] = True

        # Compute offsets from base to current
        offsets_img = eye_landmarks - base_landmarks  # (28, 2)

        # Transform to reference space if needed
        if sim_img_to_ref is not None:
            offsets_ref = offsets_img @ sim_img_to_ref.T
        else:
            offsets_ref = offsets_img

        # dx, dy = center + offset, clamped to [0, ws-1]
        dx = np.clip(offsets_ref[:, 0] + center, 0, ws - 1)  # (28,)
        dy = np.clip(offsets_ref[:, 1] + center, 0, ws - 1)  # (28,)

        # Expand for broadcasting: (28, 1, 1)
        dx_exp = dx[:, np.newaxis, np.newaxis]
        dy_exp = dy[:, np.newaxis, np.newaxis]

        # Compute squared distances: (28, ws, ws)
        dist_sq = (ii - dy_exp) ** 2 + (jj - dx_exp) ** 2

        # KDE weights
        kde_weights = np.exp(a_kde * dist_sq)

        # Combined weights = response * kde
        weights = responses * kde_weights  # (28, ws, ws)

        # Weighted sums
        total_weight = np.sum(weights, axis=(1, 2))  # (28,)
        total_weight = np.maximum(total_weight, 1e-10)  # Avoid division by zero
        mx = np.sum(weights * jj, axis=(1, 2))  # (28,)
        my = np.sum(weights * ii, axis=(1, 2))  # (28,)

        # Mean-shift = centroid - current position
        ms_x = (mx / total_weight) - dx
        ms_y = (my / total_weight) - dy

        # Zero out invalid landmarks
        ms_x[~valid_mask] = 0
        ms_y[~valid_mask] = 0

        # Stack output: [x0..x27, y0..y27]
        return np.concatenate([ms_x, ms_y]).astype(np.float32)


class BatchedEyeCCNF:
    """
    GPU-batched Eye CCNF patch expert computation.

    Processes all 28 eye landmarks in a single batched forward pass,
    providing 10-20x speedup over sequential Python computation.
    """

    def __init__(self, eye_ccnf_model, device: str = 'mps'):
        """
        Initialize batched eye CCNF from existing EyeCCNFModel.

        Args:
            eye_ccnf_model: EyeCCNFModel instance with loaded patch experts
            device: PyTorch device ('mps', 'cuda', 'cpu')
        """
        self.device = torch.device(device)
        self.n_landmarks = 28
        self.patch_size = 11  # Eye patches are 11x11
        self.n_features = self.patch_size * self.patch_size  # 121

        # Create cached vectorized mean-shift computer (NumPy, not GPU - faster for small arrays)
        self._mean_shift = EyeVectorizedMeanShift(n_landmarks=28)

        # Extract and stack weights for each scale
        self.scale_weights = {}

        for scale, patches in eye_ccnf_model.scale_models.items():
            if not patches:
                continue

            # Get number of neurons (should be 7 for all eye patches)
            sample_patch = list(patches.values())[0]
            n_neurons = sample_patch.num_neurons

            # Stack weights for all 28 landmarks
            # Shape: (28, n_neurons, 121) for weights
            # Shape: (28, n_neurons) for bias, alpha, norm_weights
            all_weights = []
            all_bias = []
            all_alpha = []
            all_norm_weights = []

            for lm_idx in range(self.n_landmarks):
                if lm_idx in patches:
                    patch = patches[lm_idx]
                    lm_weights = []
                    lm_bias = []
                    lm_alpha = []
                    lm_norm = []

                    for neuron in patch.neurons:
                        # Flatten weights in column-major order (like C++)
                        w = neuron['weights'].flatten('F')
                        lm_weights.append(w)
                        lm_bias.append(neuron['bias'])
                        lm_alpha.append(neuron['alpha'])
                        lm_norm.append(neuron['norm_weights'])

                    all_weights.append(np.array(lm_weights))
                    all_bias.append(np.array(lm_bias))
                    all_alpha.append(np.array(lm_alpha))
                    all_norm_weights.append(np.array(lm_norm))
                else:
                    # Missing landmark - use zeros
                    all_weights.append(np.zeros((n_neurons, self.n_features)))
                    all_bias.append(np.zeros(n_neurons))
                    all_alpha.append(np.zeros(n_neurons))
                    all_norm_weights.append(np.zeros(n_neurons))

            # Convert to tensors
            weights = torch.tensor(np.array(all_weights), dtype=torch.float32, device=self.device)
            bias = torch.tensor(np.array(all_bias), dtype=torch.float32, device=self.device)
            alpha = torch.tensor(np.array(all_alpha), dtype=torch.float32, device=self.device)
            norm_weights = torch.tensor(np.array(all_norm_weights), dtype=torch.float32, device=self.device)

            # Precompute scaled weights: weights * norm_weights
            # Shape: (28, n_neurons, 121)
            scaled_weights = weights * norm_weights.unsqueeze(-1)

            # Precompute sum of alphas for normalization
            sum_alphas = torch.sum(alpha, dim=1, keepdim=True)  # (28, 1)

            self.scale_weights[scale] = {
                'weights': scaled_weights,  # (28, n_neurons, 121)
                'bias': bias,               # (28, n_neurons)
                'alpha': alpha,             # (28, n_neurons)
                'sum_alphas': sum_alphas,   # (28, 1)
                'n_neurons': n_neurons,
            }

    def compute_response_maps(self,
                              image: np.ndarray,
                              eye_landmarks: np.ndarray,
                              scale: float,
                              window_size: int,
                              sim_ref_to_img: np.ndarray = None) -> Dict[int, np.ndarray]:
        """
        Compute response maps for all 28 eye landmarks in a batched pass.

        Args:
            image: Grayscale image (H, W) as float32
            eye_landmarks: Current 28 eye landmarks (28, 2)
            scale: Patch scale (1.0 or 1.5)
            window_size: Response window size (3 or 5)
            sim_ref_to_img: 2x2 similarity transform from reference to image

        Returns:
            response_maps: Dict mapping landmark_idx -> response_map (window_size, window_size)
        """
        if scale not in self.scale_weights:
            return {}

        weights_data = self.scale_weights[scale]

        # Get transform parameters
        if sim_ref_to_img is not None:
            a1 = sim_ref_to_img[0, 0]
            b1 = -sim_ref_to_img[0, 1]
        else:
            a1 = 1.0
            b1 = 0.0

        # Compute area of interest size
        aoi_size = window_size + self.patch_size - 1  # ws + 11 - 1 = ws + 10
        half_aoi = (aoi_size - 1) / 2.0

        # Extract all AOIs using warpAffine (still sequential, but fast)
        # Shape: (28, aoi_size, aoi_size)
        all_aois = []
        valid_landmarks = []

        for lm_idx in range(self.n_landmarks):
            x, y = eye_landmarks[lm_idx]

            # Create transform matrix
            tx = x - a1 * half_aoi + b1 * half_aoi
            ty = y - a1 * half_aoi - b1 * half_aoi

            sim = np.array([[a1, -b1, tx],
                           [b1, a1, ty]], dtype=np.float32)

            aoi = cv2.warpAffine(
                image,
                sim,
                (aoi_size, aoi_size),
                flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
            )

            all_aois.append(aoi)
            valid_landmarks.append(lm_idx)

        # Stack AOIs: (28, aoi_size, aoi_size)
        all_aois = np.array(all_aois, dtype=np.float32)

        # Compute response maps using batched CCNF
        response_maps = self._compute_batched_responses(
            all_aois, weights_data, window_size
        )

        # Convert to dict format
        result = {}
        for i, lm_idx in enumerate(valid_landmarks):
            result[lm_idx] = response_maps[i]

        return result

    def _compute_batched_responses(self,
                                   aois: np.ndarray,
                                   weights_data: dict,
                                   window_size: int) -> np.ndarray:
        """
        Compute CCNF responses for all landmarks in batched mode.

        Args:
            aois: Areas of interest (28, aoi_h, aoi_w)
            weights_data: Dict with scaled_weights, bias, alpha, sum_alphas
            window_size: Response window size

        Returns:
            response_maps: (28, window_size, window_size)
        """
        n_landmarks = aois.shape[0]
        aoi_size = aois.shape[1]

        # Response map dimensions
        resp_h = resp_w = window_size

        # Extract all patches using im2col-style extraction
        # For each position (i, j) in response map, extract 11x11 patch
        # Total patches per landmark: window_size^2
        # Total patches: 28 * window_size^2

        n_patches_per_lm = resp_h * resp_w
        total_patches = n_landmarks * n_patches_per_lm

        # Extract patches: (28, ws*ws, 121)
        all_patches = np.zeros((n_landmarks, n_patches_per_lm, self.n_features), dtype=np.float32)

        patch_idx = 0
        # CRITICAL FIX: Use row-major order (i outer, j inner) to match .view() reshape
        # Previously used column-major order (j outer, i inner) which caused response map
        # to be transposed, making mean-shift move landmarks in wrong Y direction.
        for i in range(resp_h):
            for j in range(resp_w):
                # Extract patch at (i, j) for all landmarks
                patches = aois[:, i:i+self.patch_size, j:j+self.patch_size]  # (28, 11, 11)
                # Flatten in column-major order (matches C++ im2col within each patch)
                all_patches[:, patch_idx, :] = patches.reshape(n_landmarks, -1, order='F')
                patch_idx += 1

        # Convert to tensor: (28, ws*ws, 121)
        patches_tensor = torch.tensor(all_patches, dtype=torch.float32, device=self.device)

        # Normalize patches (center and L2 normalize)
        # Mean per patch
        patch_mean = patches_tensor.mean(dim=-1, keepdim=True)  # (28, ws*ws, 1)
        patches_centered = patches_tensor - patch_mean

        # L2 norm per patch
        patch_norm = torch.norm(patches_centered, dim=-1, keepdim=True)  # (28, ws*ws, 1)
        patch_norm = torch.clamp(patch_norm, min=1e-10)
        patches_normalized = patches_centered / patch_norm  # (28, ws*ws, 121)

        # Compute neuron responses for all patches
        # weights_data['weights']: (28, n_neurons, 121)
        # patches_normalized: (28, ws*ws, 121)
        # Result: (28, ws*ws, n_neurons)

        scaled_weights = weights_data['weights']  # (28, n_neurons, 121)
        bias = weights_data['bias']               # (28, n_neurons)
        alpha = weights_data['alpha']             # (28, n_neurons)
        sum_alphas = weights_data['sum_alphas']   # (28, 1)

        # Einsum: for each landmark, compute dot product of each patch with each neuron's weights
        # sigmoid_input[l, p, n] = bias[l, n] + sum_k(weights[l, n, k] * patches[l, p, k])
        sigmoid_input = torch.einsum('lnk,lpk->lpn', scaled_weights, patches_normalized)  # (28, ws*ws, n_neurons)
        sigmoid_input = sigmoid_input + bias.unsqueeze(1)  # Add bias

        # Sigmoid: 2 * alpha / (1 + exp(-x))
        sigmoid_output = 2.0 * alpha.unsqueeze(1) / (1.0 + torch.exp(-sigmoid_input.clamp(-500, 500)))

        # Sum over neurons to get total response
        total_response = sigmoid_output.sum(dim=-1)  # (28, ws*ws)

        # Normalize by 2 * sum_alphas
        sum_alphas_expanded = sum_alphas.expand(-1, n_patches_per_lm)  # (28, ws*ws)
        normalized_response = total_response / (2.0 * sum_alphas_expanded.clamp(min=1e-10))

        # Reshape to (28, ws, ws)
        response_maps = normalized_response.view(n_landmarks, resp_h, resp_w)

        # Ensure non-negative (like C++)
        min_vals = response_maps.amin(dim=(1, 2), keepdim=True)
        response_maps = response_maps - min_vals.clamp(max=0)

        # Convert back to numpy
        return response_maps.cpu().numpy()

    def compute_mean_shift_batched(self,
                                   response_maps: Dict[int, np.ndarray],
                                   eye_landmarks: np.ndarray,
                                   base_landmarks: np.ndarray,
                                   window_size: int,
                                   sigma: float,
                                   sim_img_to_ref: np.ndarray = None) -> np.ndarray:
        """
        Compute mean-shift for all landmarks using cached GPU computer.

        Args:
            response_maps: Dict mapping landmark_idx -> response_map (ws, ws)
            eye_landmarks: Current 28 eye landmarks (28, 2)
            base_landmarks: Base landmark positions (28, 2)
            window_size: Response window size (3 or 5)
            sigma: Gaussian kernel sigma
            sim_img_to_ref: 2x2 similarity transform from image to reference space

        Returns:
            mean_shift: Mean-shift vector (56,) in stacked format [x0..x27, y0..y27]
        """
        return self._mean_shift.compute(
            response_maps,
            eye_landmarks,
            base_landmarks,
            window_size,
            sigma,
            sim_img_to_ref
        )


def create_batched_eye_ccnf(eye_ccnf_model, device: str = 'mps') -> BatchedEyeCCNF:
    """
    Factory function to create BatchedEyeCCNF from EyeCCNFModel.

    Args:
        eye_ccnf_model: EyeCCNFModel instance
        device: PyTorch device

    Returns:
        BatchedEyeCCNF instance
    """
    return BatchedEyeCCNF(eye_ccnf_model, device)
