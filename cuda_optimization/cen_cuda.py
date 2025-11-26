"""
CEN CUDA Implementation

Your task: Fill in cen_forward_batch_cuda() to match the CPU version's output.

The CPU version (in cen_core.py) does:
1. Flatten patches
2. Contrast normalize (center, divide by L2 norm)
3. Add bias column
4. Layer 0: matmul + sigmoid
5. Layer 1: matmul + sigmoid

All of these operations have direct PyTorch equivalents.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CENInferenceCUDA:
    """
    GPU-accelerated CEN inference.
    
    Usage:
        cen = CENInferenceCUDA(weights_dict, device='cuda')
        responses = cen.forward(patches_tensor)
    """
    
    def __init__(self, weights: dict, device: str = 'cuda'):
        """
        Initialize with weights on GPU.
        
        Args:
            weights: Dict with w0, b0, a0, w1, b1, a1, width, height
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.width = weights['width']
        self.height = weights['height']
        
        # Move weights to GPU (do this ONCE at init, not per-call)
        self.w0 = torch.tensor(weights['w0'], dtype=torch.float32, device=self.device)
        self.b0 = torch.tensor(weights['b0'], dtype=torch.float32, device=self.device)
        self.a0 = weights['a0']
        
        self.w1 = torch.tensor(weights['w1'], dtype=torch.float32, device=self.device)
        self.b1 = torch.tensor(weights['b1'], dtype=torch.float32, device=self.device)
        self.a1 = weights['a1']
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass on GPU.
        
        Args:
            patches: (batch_size, height, width) tensor, float32, range [0, 1]
                     Should already be on self.device
                     
        Returns:
            responses: (batch_size,) tensor
        """
        batch_size = patches.shape[0]
        
        # =====================================================================
        # TODO: Implement the forward pass here
        # 
        # Hint: The operations you need are:
        #   - patches.view(batch_size, -1)           # Flatten
        #   - tensor.mean(dim=1, keepdim=True)       # Mean per row
        #   - torch.norm(tensor, dim=1, keepdim=True) # L2 norm per row
        #   - torch.cat([a, b], dim=1)               # Concatenate
        #   - tensor @ matrix.T                       # Matrix multiply
        #   - torch.sigmoid(tensor)                   # Sigmoid activation
        #   - F.relu(tensor)                          # ReLU activation
        #
        # The CPU reference is in cen_core.py - match its math exactly!
        # =====================================================================
        
        # Step 1: Flatten patches
        flat = patches.view(batch_size, -1)
        
        # Step 2: Contrast normalization
        mean = flat.mean(dim=1, keepdim=True)
        centered = flat - mean
        norm = torch.norm(centered, dim=1, keepdim=True).clamp(min=1e-10)
        normalized = centered / norm
        
        # Step 3: Add bias column
        bias_col = torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)
        layer_input = torch.cat([bias_col, normalized], dim=1)
        
        # Step 4: Layer 0
        layer0_out = layer_input @ self.w0.T + self.b0
        if self.a0 == 0:  # Sigmoid
            layer0_out = torch.sigmoid(layer0_out)
        elif self.a0 == 2:  # ReLU
            layer0_out = F.relu(layer0_out)
        
        # Step 5: Layer 1
        layer1_out = layer0_out @ self.w1.T + self.b1
        if self.a1 == 0:  # Sigmoid
            layer1_out = torch.sigmoid(layer1_out)
        elif self.a1 == 2:  # ReLU
            layer1_out = F.relu(layer1_out)
        
        return layer1_out.squeeze(-1)


def cen_forward_batch_cuda(patches: np.ndarray,
                           width: int,
                           height: int,
                           w0: np.ndarray,
                           b0: np.ndarray,
                           a0: int,
                           w1: np.ndarray,
                           b1: np.ndarray,
                           a1: int,
                           device: str = 'cuda') -> np.ndarray:
    """
    Convenience function matching cen_forward_batch_cpu signature.
    
    This handles numpy<->torch conversion so you can drop it in as a replacement.
    
    Args:
        patches: (batch_size, height, width) numpy array
        ... (same as CPU version)
        
    Returns:
        responses: (batch_size,) numpy array
    """
    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA support.")
    
    dev = torch.device(device)
    
    # Build weights dict
    weights = {
        'width': width,
        'height': height,
        'w0': w0,
        'b0': b0,
        'a0': a0,
        'w1': w1,
        'b1': b1,
        'a1': a1,
    }
    
    # Create inference object (in real usage, you'd cache this)
    cen = CENInferenceCUDA(weights, device=device)
    
    # Convert input to tensor and move to GPU
    patches_tensor = torch.tensor(patches, dtype=torch.float32, device=dev)
    
    # Forward pass
    with torch.no_grad():  # No need for gradients
        responses = cen.forward(patches_tensor)
    
    # Move back to CPU and convert to numpy
    return responses.cpu().numpy()


# =============================================================================
# BONUS: Optimized version with persistent state
# =============================================================================

class CENBatchProcessor:
    """
    Optimized processor for repeated inference.
    
    Key optimizations:
    1. Weights stay on GPU (no repeated transfers)
    2. Reusable CUDA streams
    3. Optional: torch.compile for kernel fusion (PyTorch 2.0+)
    """
    
    def __init__(self, weights: dict, device: str = 'cuda', compile: bool = False):
        self.device = torch.device(device)
        self.cen = CENInferenceCUDA(weights, device)
        
        # Optional: Use torch.compile for additional speedup (PyTorch 2.0+)
        if compile and hasattr(torch, 'compile'):
            self.cen.forward = torch.compile(self.cen.forward)
            print("Using torch.compile for kernel fusion")
    
    def process(self, patches: np.ndarray) -> np.ndarray:
        """Process a batch of patches."""
        patches_t = torch.tensor(patches, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            result = self.cen.forward(patches_t)
        
        return result.cpu().numpy()
    
    def process_tensor(self, patches: torch.Tensor) -> torch.Tensor:
        """Process patches that are already tensors (avoids numpy conversion)."""
        with torch.no_grad():
            return self.cen.forward(patches)


if __name__ == "__main__":
    print("Testing CUDA CEN implementation...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        print("Install PyTorch with CUDA:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        exit(1)

    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Import CPU baseline
    from cen_core import generate_synthetic_weights, generate_test_patches, cen_forward_batch_cpu
    
    # Generate test data
    weights = generate_synthetic_weights()
    patches = generate_test_patches(batch_size=1000)
    
    # Run CPU
    cpu_result = cen_forward_batch_cpu(patches, **weights)
    
    # Run CUDA
    cuda_result = cen_forward_batch_cuda(patches, **weights, device='cuda')
    
    # Compare
    max_diff = np.abs(cpu_result - cuda_result).max()
    print(f"Max difference CPU vs CUDA: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("[OK] CUDA implementation matches CPU!")
    else:
        print("[ERROR] Results differ - check your implementation")
