# CCNF CUDA Acceleration Plan

## Overview

Add CUDA support to CCNFModel while maintaining backward compatibility with CPU-only environments.

## Current Architecture

### CCNFPatchExpert
- Located in `pyclnf/core/patch_expert.py`
- Each patch expert has:
  - `width`, `height` (typically 11x11)
  - `betas` array (3 values for sigma components)
  - `patch_confidence` scalar
  - `neurons` list (~7 neurons per patch)
    - Each neuron: `type`, `weights` (11x11), `bias`, `alpha`, `norm_weights`

### Response Computation Algorithm
```python
def compute_response(image_patch):
    # 1. Normalize to [0,1]
    features = image_patch.astype(float32) / 255.0

    # 2. Sum neuron responses
    total = 0.0
    for neuron in neurons:
        if abs(neuron['alpha']) < 1e-4:
            continue

        # Normalized cross-correlation (TM_CCOEFF_NORMED)
        weights_centered = weights - mean(weights)
        features_centered = features - mean(features)
        correlation = sum(weights_centered * features_centered) / (norm(w) * norm(f))

        # Response formula
        response = 2 * alpha * sigmoid(correlation * norm_weights + bias)
        total += response

    return total
```

### Current Call Flow
```
optimizer._compute_response_map()
  -> for each (i,j) in window_size x window_size:
       patch = extract_patch(area_of_interest, i, j)
       response_map[i,j] = patch_expert.compute_response(patch)
```

This results in ~121 calls to `compute_response()` per landmark (11x11 window).

## CUDA Optimization Strategy

### Key Insight
The main bottleneck is computing 121 normalized cross-correlations per landmark, times ~7 neurons each = ~847 correlation operations per landmark, times 68 landmarks = ~57,000 correlation operations per frame.

### Batched GPU Implementation

1. **Batch all patches across all landmarks**
   - Extract 121 patches per landmark on CPU (im2col)
   - Stack all patches: (68 landmarks × 121 patches) = 8,228 patches per frame

2. **GPU-accelerated correlation computation**
   - Move all neuron weights to GPU once at model load
   - For each unique (width, height) configuration:
     - Batch compute: mean, centered values, norms, correlations
     - Apply sigmoid and alpha scaling
     - Sum neuron responses

### Implementation Files

1. **`pyclnf/core/ccnf_cuda.py`** (NEW)
   - `CCNFInferenceCUDA` - Single expert GPU inference
   - `CCNFBatchProcessor` - Multi-expert batch processor
   - `ccnf_forward_batch_cuda()` - Drop-in NumPy replacement

2. **Modify `pyclnf/core/patch_expert.py`**
   - Add `device` parameter to `CCNFModel.__init__()`
   - Add `cuda_processors` dict (scale -> processor)
   - Add `get_cuda_processor(scale)` method
   - Keep all existing interfaces unchanged

3. **Modify `pyclnf/core/optimizer.py`**
   - Add CCNF CUDA path in `_compute_response_map()`
   - Add `_compute_response_ccnf_cuda()` helper

4. **Revert `pyclnf/clnf.py`**
   - Change import from `CENModel` back to `CCNFModel`
   - Keep `device` parameter (already added)

## Interface Boundaries (Must Preserve)

### CCNFPatchExpert
```python
class CCNFPatchExpert:
    def __init__(self, patch_dir: str)
    def compute_response(self, image_patch: np.ndarray) -> float
    def compute_sigma(self, sigma_components, window_size=None, debug=False) -> np.ndarray
    def get_info(self) -> dict

    # Properties
    .width, .height, .betas, .patch_confidence, .num_neurons, .neurons
```

### CCNFModel
```python
class CCNFModel:
    def __init__(self, model_base_dir: str, scales: List[float] = None)
    def get_patch_expert(self, scale, view_idx, landmark_idx) -> CCNFPatchExpert
    def get_best_view(self, pose: np.ndarray) -> int
    def get_info(self) -> dict

    # Properties
    .scale_models, .sigma_components
```

### Optimizer Integration
```python
# Must accept these patch_expert types:
# - CCNFPatchExpert (has .compute_response())
# - CENPatchExpert (has .response())
# Detection via: hasattr(patch_expert, 'response') and not hasattr(patch_expert, 'compute_response')
```

## CUDA Implementation Details

### CCNFInferenceCUDA Class
```python
class CCNFInferenceCUDA:
    def __init__(self, patch_expert: CCNFPatchExpert, device='cuda'):
        # Pre-load all neuron weights to GPU
        self.neurons_weights = []  # List of (11,11) tensors
        self.neurons_bias = []
        self.neurons_alpha = []
        self.neurons_norm_weights = []

    def forward_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass.

        Args:
            patches: (batch_size, height, width) float32 tensor, values 0-255

        Returns:
            responses: (batch_size,) float32 tensor
        """
        # Normalize to [0,1]
        features = patches / 255.0

        # Batch compute all neuron responses
        total = torch.zeros(batch_size, device=self.device)

        for neuron_idx in range(self.num_neurons):
            if self.neurons_alpha[neuron_idx] < 1e-4:
                continue

            # Batched normalized cross-correlation
            weights = self.neurons_weights[neuron_idx]  # (H, W)

            # Center features and weights
            w_mean = weights.mean()
            w_centered = weights - w_mean  # (H, W)

            f_mean = features.mean(dim=(1,2), keepdim=True)  # (B, 1, 1)
            f_centered = features - f_mean  # (B, H, W)

            # Compute norms
            w_norm = w_centered.norm()
            f_norm = f_centered.norm(dim=(1,2))  # (B,)

            # Correlation: sum of element-wise product
            correlation = (w_centered * f_centered).sum(dim=(1,2)) / (w_norm * f_norm + 1e-10)

            # Apply neuron response formula
            sigmoid_input = correlation * self.neurons_norm_weights[neuron_idx] + self.neurons_bias[neuron_idx]
            response = 2.0 * self.neurons_alpha[neuron_idx] * torch.sigmoid(sigmoid_input)

            total += response

        return total
```

### CCNFBatchProcessor Class
```python
class CCNFBatchProcessor:
    def __init__(self, device='cuda'):
        self.experts = {}  # landmark_idx -> CCNFInferenceCUDA

    def initialize_experts(self, patch_experts: dict):
        """Initialize CUDA experts from CPU patch experts."""
        for landmark_idx, cpu_expert in patch_experts.items():
            self.experts[landmark_idx] = CCNFInferenceCUDA(cpu_expert, self.device)

    def process_single(self, landmark_idx, patches):
        """Process patches for single landmark on GPU."""
        # patches: (batch_size, H, W) numpy array
        # returns: (batch_size,) numpy array
```

## Testing Plan

1. **Unit Tests** (`test_ccnf_cuda.py`)
   - Test CCNFInferenceCUDA matches CPU CCNFPatchExpert.compute_response()
   - Test CCNFBatchProcessor initialization
   - Test batch processing accuracy

2. **Integration Tests** (`test_integration.py`)
   - Test CCNFModel with device='cuda' vs device='cpu'
   - Test optimizer produces same results
   - Test full CLNF pipeline

3. **Benchmarks** (`benchmark_ccnf.py`)
   - Measure per-landmark speedup
   - Measure full-frame speedup
   - Compare CPU vs CUDA end-to-end

## Implementation Order

1. Create `ccnf_cuda.py` with core CUDA classes
2. Add tests comparing CPU vs CUDA accuracy
3. Modify `patch_expert.py` to add device support to CCNFModel
4. Modify `optimizer.py` to use CUDA processor for CCNF
5. Revert `clnf.py` to use CCNFModel
6. Run full integration tests
7. Run benchmarks

## Backward Compatibility Guarantees

- `CCNFModel()` without device parameter works exactly as before (CPU)
- `CCNFPatchExpert` interface unchanged
- All existing code paths continue to work
- CUDA is optional - graceful fallback to CPU if PyTorch/CUDA unavailable
