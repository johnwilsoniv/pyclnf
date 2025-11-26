"""
CEN Core Inference - CPU Baseline

This is the hot path extracted from pyclnf. It's called 16,320 times per frame.
Your goal: make a CUDA version that produces identical output but batched.

The math:
1. im2col: Extract sliding window patches from input
2. contrast_norm: Normalize each patch (subtract mean, divide by L2 norm)  
3. Layer 0: linear + sigmoid
4. Layer 1: linear + sigmoid
5. Reshape to response map
"""

import numpy as np
from typing import Tuple


def cen_forward_single(input_patch: np.ndarray,
                       width: int,
                       height: int,
                       w0: np.ndarray,
                       b0: np.ndarray,
                       a0: int,
                       w1: np.ndarray,
                       b1: np.ndarray,
                       a1: int) -> np.ndarray:
    """
    Single patch CEN forward pass (CPU baseline).
    
    This is what currently runs 16,320 times per frame.
    
    Args:
        input_patch: (H, W) grayscale image region, float32, range [0, 255]
        width, height: Patch dimensions (typically 11x11)
        w0, b0, a0: Layer 0 weights, biases, activation type
        w1, b1, a1: Layer 1 weights, biases, activation type
        
    Returns:
        response_map: (response_h, response_w) float32 array
    """
    m, n = input_patch.shape
    response_height = m - height + 1
    response_width = n - width + 1
    
    # STEP 1 & 2: im2col + contrast normalization
    y_blocks = m - height + 1
    x_blocks = n - width + 1
    num_windows = y_blocks * x_blocks
    patch_size = height * width
    
    # Allocate normalized output with bias column
    normalized = np.ones((num_windows, patch_size + 1), dtype=np.float32)
    
    for j in range(x_blocks):
        for i in range(y_blocks):
            row_idx = i + j * y_blocks
            
            # Extract patch
            patch = input_patch[i:i+height, j:j+width]
            patch_flat = patch.T.flatten()  # Column-major order
            
            # Contrast normalize
            mean = np.mean(patch_flat)
            centered = patch_flat - mean
            norm = np.sqrt(np.sum(centered ** 2))
            if norm < 1e-10:
                norm = 1.0
            
            normalized[row_idx, 1:] = centered / norm
    
    # STEP 3: Layer 0 forward
    layer0_out = normalized @ w0.T + b0
    if a0 == 0:  # Sigmoid
        layer0_out = np.clip(layer0_out, -88, 88)
        layer0_out = 1.0 / (1.0 + np.exp(-layer0_out))
    elif a0 == 2:  # ReLU
        layer0_out = np.maximum(0, layer0_out)
    
    # STEP 4: Layer 1 forward
    layer1_out = layer0_out @ w1.T + b1
    if a1 == 0:  # Sigmoid
        layer1_out = np.clip(layer1_out, -88, 88)
        layer1_out = 1.0 / (1.0 + np.exp(-layer1_out))
    elif a1 == 2:  # ReLU
        layer1_out = np.maximum(0, layer1_out)
    
    # STEP 5: Reshape to response map (column-major order)
    response = layer1_out.flatten().reshape(response_height, response_width, order='F')
    
    return response.astype(np.float32)


def cen_forward_batch_cpu(patches: np.ndarray,
                          width: int,
                          height: int,
                          w0: np.ndarray,
                          b0: np.ndarray,
                          a0: int,
                          w1: np.ndarray,
                          b1: np.ndarray,
                          a1: int) -> np.ndarray:
    """
    Batched CEN forward pass (CPU reference implementation).
    
    This processes multiple ALREADY-EXTRACTED patches at once.
    (The im2col step is skipped - patches are pre-extracted)
    
    Args:
        patches: (batch_size, height, width) float32, already normalized to [0,1]
        width, height: Patch dimensions
        w0, b0, a0: Layer 0 params
        w1, b1, a1: Layer 1 params
        
    Returns:
        responses: (batch_size,) float32 array
    """
    batch_size = patches.shape[0]
    patch_size = height * width
    
    # Flatten and contrast normalize
    flat = patches.reshape(batch_size, -1)
    
    # Contrast norm: center and normalize each row
    mean = flat.mean(axis=1, keepdims=True)
    centered = flat - mean
    norm = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
    norm = np.maximum(norm, 1e-10)
    normalized = centered / norm
    
    # Add bias column
    bias_col = np.ones((batch_size, 1), dtype=np.float32)
    layer_input = np.hstack([bias_col, normalized])
    
    # Layer 0
    layer0_out = layer_input @ w0.T + b0
    if a0 == 0:  # Sigmoid
        layer0_out = np.clip(layer0_out, -88, 88)
        layer0_out = 1.0 / (1.0 + np.exp(-layer0_out))
    elif a0 == 2:  # ReLU
        layer0_out = np.maximum(0, layer0_out)
    
    # Layer 1
    layer1_out = layer0_out @ w1.T + b1
    if a1 == 0:  # Sigmoid
        layer1_out = np.clip(layer1_out, -88, 88)
        layer1_out = 1.0 / (1.0 + np.exp(-layer1_out))
    elif a1 == 2:  # ReLU
        layer1_out = np.maximum(0, layer1_out)
    
    return layer1_out.squeeze(-1).astype(np.float32)


def generate_synthetic_weights(patch_size: int = 11,
                               hidden_dim: int = 32,
                               seed: int = 42) -> dict:
    """
    Generate synthetic CEN weights for testing.
    
    These aren't real trained weights, but they have the right shapes
    and reasonable values for testing the forward pass.
    
    Args:
        patch_size: Width/height of input patches
        hidden_dim: Hidden layer dimension
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with w0, b0, a0, w1, b1, a1
    """
    np.random.seed(seed)
    
    input_dim = patch_size * patch_size + 1  # +1 for bias column
    
    # Layer 0: input_dim -> hidden_dim
    w0 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
    b0 = np.zeros((1, hidden_dim), dtype=np.float32)
    
    # Layer 1: hidden_dim -> 1
    w1 = np.random.randn(1, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros((1, 1), dtype=np.float32)
    
    return {
        'width': patch_size,
        'height': patch_size,
        'w0': w0,
        'b0': b0,
        'a0': 0,  # Sigmoid
        'w1': w1,
        'b1': b1,
        'a1': 0,  # Sigmoid
    }


def generate_test_patches(batch_size: int = 1000,
                          patch_size: int = 11,
                          seed: int = 123) -> np.ndarray:
    """
    Generate random test patches.
    
    Args:
        batch_size: Number of patches
        patch_size: Width/height
        seed: Random seed
        
    Returns:
        patches: (batch_size, patch_size, patch_size) float32 in [0, 1]
    """
    np.random.seed(seed)
    return np.random.rand(batch_size, patch_size, patch_size).astype(np.float32)


if __name__ == "__main__":
    # Quick sanity test
    print("Testing CEN CPU baseline...")
    
    weights = generate_synthetic_weights()
    patches = generate_test_patches(batch_size=100)
    
    # Test single forward
    single_result = cen_forward_single(
        patches[0] * 255,  # Scale to [0, 255] for single version
        **{k: v for k, v in weights.items() if k not in ['width', 'height']},
        width=weights['width'],
        height=weights['height']
    )
    print(f"Single forward output shape: {single_result.shape}")
    
    # Test batch forward
    batch_result = cen_forward_batch_cpu(patches, **weights)
    print(f"Batch forward output shape: {batch_result.shape}")
    print(f"Output range: [{batch_result.min():.4f}, {batch_result.max():.4f}]")
    
    print("[OK] CPU baseline working!")
