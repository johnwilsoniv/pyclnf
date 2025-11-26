# pyclnf CUDA Acceleration - Implementation Guide

## Overview

**Goal:** Accelerate pyclnf facial landmark detection from ~3 FPS to 15-30 FPS using CUDA.

**Approach:** Drop-in GPU acceleration with automatic backend selection. CPU code remains untouched as fallback.

**Constraint:** All changes must be testable in isolation without running the full pipeline.

---

## Current State

### Performance Bottleneck

The CEN (Convolutional Expert Network) forward pass in `pyclnf/core/cen_patch_expert.py` is called **16,320 times per frame**. It currently runs sequentially on CPU via Numba JIT.

### Existing Files

```
pyclnf/
├── pyclnf/
│   ├── clnf.py                      # Main entry point
│   ├── core/
│   │   ├── cen_patch_expert.py      # CEN inference (TARGET FOR OPTIMIZATION)
│   │   ├── optimizer.py             # NU-RLMS optimization loop
│   │   ├── pdm.py                   # Point Distribution Model
│   │   ├── numba_accelerator.py     # Existing Numba JIT functions
│   │   ├── coreml_backend.py        # Mac ANE backend (reference)
│   │   └── metal_backend.py         # Mac Metal backend (reference)
│   └── cuda_optimization/           # Standalone CUDA dev/test environment
│       ├── cen_core.py              # Extracted CPU baseline
│       ├── cen_cuda.py              # CUDA implementation (COMPLETE)
│       ├── test_accuracy.py         # Accuracy tests
│       └── benchmark.py             # Speed benchmarks
```

### Key Function to Accelerate

Location: `pyclnf/core/cen_patch_expert.py`

Function: `_response_core_numba()` (lines ~688-799)

This function:
1. Extracts patches via im2col
2. Applies contrast normalization
3. Runs 2-layer neural network (matmul + sigmoid)
4. Returns response map

---

## Implementation Plan

### Phase 1: Standalone CUDA Module (COMPLETE)

Files in `pyclnf/cuda_optimization/` contain a working CUDA implementation:

- `cen_core.py` - CPU reference implementation
- `cen_cuda.py` - PyTorch CUDA implementation with `CENInferenceCUDA` class
- `test_accuracy.py` - Verifies CUDA matches CPU output
- `benchmark.py` - Measures speedup

**Status:** Code complete, needs testing on GPU machine.

### Phase 2: Integrate into Main Codebase

Create a backend selection system and integrate CUDA into the main pipeline.

#### Task 2.1: Create Backend Selector

Create new file: `pyclnf/core/cuda_backend.py`

```python
"""
CUDA Backend Selection

Automatically detects CUDA availability and provides unified interface.
"""

import numpy as np
from typing import Optional

class ComputeBackend:
    CUDA = "cuda"
    CPU = "cpu"
    
    _active_backend: Optional[str] = None
    _cuda_available: Optional[bool] = None
    
    @classmethod
    def get(cls) -> str:
        """Get the active compute backend."""
        if cls._active_backend is not None:
            return cls._active_backend
        return cls.CUDA if cls.is_cuda_available() else cls.CPU
    
    @classmethod
    def set(cls, backend: str):
        """Force a specific backend."""
        if backend not in [cls.CUDA, cls.CPU]:
            raise ValueError(f"Unknown backend: {backend}")
        cls._active_backend = backend
    
    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available."""
        if cls._cuda_available is None:
            try:
                import torch
                cls._cuda_available = torch.cuda.is_available()
            except ImportError:
                cls._cuda_available = False
        return cls._cuda_available
    
    @classmethod
    def reset(cls):
        """Reset to auto-detection."""
        cls._active_backend = None
```

#### Task 2.2: Create CUDA CEN Module

Create new file: `pyclnf/core/cen_cuda.py`

Copy the `CENInferenceCUDA` and `CENBatchProcessor` classes from `cuda_optimization/cen_cuda.py`.

Add a factory function:

```python
def create_cen_processor(weights: dict, device: str = 'auto'):
    """
    Create CEN processor with automatic backend selection.
    
    Args:
        weights: Dict with w0, b0, a0, w1, b1, a1, width, height
        device: 'auto', 'cuda', or 'cpu'
    
    Returns:
        Processor with .forward(patches) method
    """
    if device == 'auto':
        from .cuda_backend import ComputeBackend
        device = ComputeBackend.get()
    
    if device == 'cuda':
        return CENBatchProcessor(weights, device='cuda')
    else:
        return CENBatchProcessorCPU(weights)
```

#### Task 2.3: Modify CEN Patch Expert

File: `pyclnf/core/cen_patch_expert.py`

Add CUDA path to the `CENPatchExpert.response()` method:

```python
def response(self, area_of_interest):
    """Compute patch expert response map."""
    if self.is_empty:
        # ... existing empty handling ...
        
    # Check for CUDA backend
    from .cuda_backend import ComputeBackend
    if ComputeBackend.get() == ComputeBackend.CUDA:
        return self._response_cuda(area_of_interest)
    
    # Existing CPU/Numba path
    if NUMBA_AVAILABLE and len(self.weights) == 2:
        return _response_core_numba(...)
    
    # ... rest of existing code ...

def _response_cuda(self, area_of_interest):
    """CUDA-accelerated response computation."""
    from .cen_cuda import cen_forward_cuda
    # ... implementation ...
```

#### Task 2.4: Batch Response Map Computation

File: `pyclnf/core/optimizer.py`

Modify `_precompute_response_maps()` to process all landmarks in parallel on GPU:

```python
def _precompute_response_maps(self, landmarks_2d, patch_experts, image, ...):
    from .cuda_backend import ComputeBackend
    
    if ComputeBackend.get() == ComputeBackend.CUDA:
        return self._precompute_response_maps_cuda(landmarks_2d, patch_experts, image, ...)
    else:
        return self._precompute_response_maps_cpu(landmarks_2d, patch_experts, image, ...)

def _precompute_response_maps_cpu(self, ...):
    """Original implementation - renamed."""
    # ... existing code unchanged ...

def _precompute_response_maps_cuda(self, ...):
    """
    GPU-accelerated response map computation.
    
    Instead of processing landmarks sequentially, batch all operations:
    1. Batch warpAffine for all landmarks
    2. Batch CEN forward pass
    3. Batch sigma transforms
    """
    # ... new implementation ...
```

---

## Testing Strategy

### Unit Tests

Each component must pass accuracy tests before integration:

```bash
# Test standalone CUDA implementation
cd pyclnf/cuda_optimization
python test_accuracy.py

# Test backend selection
python -c "from pyclnf.core.cuda_backend import ComputeBackend; print(ComputeBackend.get())"

# Test integrated CEN
python -c "
from pyclnf.core.cen_patch_expert import CENPatchExpert
# ... load expert and test ...
"
```

### Accuracy Criteria

- CUDA output must match CPU output within tolerance: `max_diff < 1e-5`
- No NaN or Inf values in output
- Works for batch sizes: 1, 100, 1000, 8228, 16320

### Speed Criteria

- Minimum 5x speedup over CPU baseline
- Target: 15-30 FPS end-to-end (up from ~3 FPS)

---

## Dependencies

```
torch >= 2.0.0  # With CUDA support matching system CUDA version
numpy >= 1.20.0
```

Install PyTorch with CUDA (adjust version to match `nvidia-smi` output):

```bash
# For CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `pyclnf/core/cuda_backend.py` | CREATE | Backend detection and selection |
| `pyclnf/core/cen_cuda.py` | CREATE | CUDA CEN implementation |
| `pyclnf/core/cen_patch_expert.py` | MODIFY | Add CUDA dispatch in `response()` |
| `pyclnf/core/optimizer.py` | MODIFY | Add batched GPU path in `_precompute_response_maps()` |

---

## Implementation Order

1. **Verify CUDA works** - Run `python cuda_optimization/cen_cuda.py`
2. **Create `cuda_backend.py`** - Backend selection infrastructure  
3. **Create `cen_cuda.py`** - Copy from cuda_optimization, add factory
4. **Modify `cen_patch_expert.py`** - Add CUDA dispatch
5. **Test single-landmark CUDA** - Verify accuracy
6. **Modify `optimizer.py`** - Batch processing
7. **End-to-end benchmark** - Measure FPS improvement

---

## Notes for Coding Agent

- **Do not modify CPU code paths** - They must remain as fallback
- **Use `ComputeBackend.get()` for all dispatch decisions** - Enables easy A/B testing
- **Keep GPU tensors on GPU** - Avoid unnecessary `.cpu().numpy()` conversions in hot paths
- **Pre-load weights to GPU once** - Not per-frame
- **Test each change in isolation** - Use files in `cuda_optimization/` as reference
- **The cuda_optimization/ folder contains working, tested code** - Use it as ground truth
