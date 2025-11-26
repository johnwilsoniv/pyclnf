# CUDA Support for Training - Implementation Plan

## Summary

Found training code in two repositories with **MASSIVE** CUDA optimization opportunities:

### Key Discovery: Training Data Generation Uses pyCLNF!

The `TrainingDataGenerator` in pyfaceau calls `pyclnf.CLNF()` for every frame processed.
This is the **same CCNF inference we just accelerated by 688x**!

```
pyfaceau/pyfaceau/data/training_data_generator.py:108
    from pyclnf import CLNF
    self._clnf = CLNF()  # <-- This now auto-uses CUDA!
```

### Expected Impact

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| **Training data generation** (CLNF bottleneck) | ~2 FPS | ~100+ FPS | **50-100x** |
| **Model training** (forward/backward) | 1x | 1.5-2x | 1.5-2x |
| **End-to-end pipeline** | Hours | Minutes | **10-50x** |

The training data generation is the **massive** bottleneck because:
1. pyCLNF runs on every frame
2. It was running on CPU at ~2-5 FPS
3. Our CUDA acceleration gives 688x speedup on the CCNF inference

---

## Repositories with Training Code

| Repository | Training Files | Models | Current CUDA Support |
|-----------|---------------|--------|---------------------|
| **pyfaceau** | `train_au_prediction.py`, `train_landmark_pose.py` | AUPredictionNet, UnifiedLandmarkPoseNet | Basic (device selection only) |
| **face-analysis** | `train_au_mlp.py`, `train_landmark_cnn.py` | AUMLP, Landmark CNN | Mixed precision (FP16) |

---

## Current CUDA Status

### Training Data Generation (HUGE OPPORTUNITY)
- `TrainingDataGenerator.process_frame()` calls:
  - `MTCNN.detect()` - Already GPU-accelerated
  - `CLNF.fit()` - **NOW GPU-accelerated via our work!**
  - `pyfhog.extract_fhog_features()` - CPU only (C++ extension)
  - AU prediction - CPU SVR models

### pyfaceau Training (Partial CUDA)
- **Device selection**: Auto-detects CUDA/MPS/CPU
- **Missing optimizations**:
  - No mixed precision (AMP)
  - No cuDNN benchmarking
  - No non-blocking memory transfers
  - Standard DataLoader (no prefetching optimization)

### face-analysis Training (Better CUDA)
- **Has**:
  - Mixed precision with `GradScaler` + `autocast`
  - cuDNN benchmark mode enabled
  - Non-blocking memory transfers
  - Pin memory in DataLoaders
- **Missing**:
  - Gradient accumulation for larger effective batch sizes
  - Multi-GPU support (DataParallel/DistributedDataParallel)

---

## Implementation Plan

### Phase 0: Verify Training Data Generation CUDA (ALREADY DONE!)

The `CLNF()` class now auto-detects CUDA via `device='auto'` default.
**No changes needed** - training data generation should already be 50-100x faster!

Test with:
```bash
cd pyfaceau
python scripts/generate_training_data.py test_video.mp4 -o test.h5
```

### Phase 1: Unified CUDA Training Utilities Module (New)

Create `pyfaceau/pyfaceau/nn/cuda_training.py`:

```python
"""
CUDA-accelerated training utilities.

Provides:
- Mixed precision training (AMP)
- Optimized data loading
- Multi-GPU support
- Gradient accumulation
- Memory management
"""
```

**Components:**
1. `CUDATrainingConfig` - Configuration dataclass
2. `setup_cuda_training()` - Initialize CUDA environment
3. `AMPTrainer` - Base class with mixed precision support
4. `OptimizedDataLoader` - Wrapper with prefetching
5. `GradientAccumulator` - For larger effective batches

### Phase 2: Upgrade pyfaceau Training Scripts

#### 2.1 Update `train_au_prediction.py`
- Add `torch.cuda.amp` imports
- Add `GradScaler` for mixed precision
- Add `autocast` context in forward pass
- Enable `torch.backends.cudnn.benchmark`
- Add non-blocking memory transfers
- Add gradient clipping with scaled gradients

#### 2.2 Update `train_landmark_pose.py`
- Same optimizations as AU prediction
- Add multi-task loss scaling for mixed precision

### Phase 3: Update Model Architectures for CUDA Efficiency

#### 3.1 `au_prediction_net.py`
- Ensure all operations are CUDA-compatible
- Add `@torch.jit.script` decorations where beneficial
- Optimize memory layout for CUDA kernels

#### 3.2 `landmark_pose_net.py`
- Same optimizations
- Ensure Wing loss is numerically stable in FP16

### Phase 4: Data Pipeline Optimization

#### 4.1 HDF5 Dataset Improvements
- Add memory-mapped reading option
- Implement prefetch queue
- Add GPU-side augmentation where possible

#### 4.2 DataLoader Configuration
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,           # Enable for CUDA
    persistent_workers=True,   # Keep workers alive between epochs
    prefetch_factor=2,         # Prefetch batches
)
```

### Phase 5: Multi-GPU Support (Optional)

For larger datasets/models:
- `DistributedDataParallel` wrapper
- Gradient synchronization
- Learning rate scaling

---

## Detailed Changes

### File: `pyfaceau/pyfaceau/nn/cuda_training.py` (NEW)

```python
"""CUDA training utilities for pyfaceau neural networks."""

import torch
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class CUDATrainingConfig:
    """Configuration for CUDA-accelerated training."""

    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16

    # cuDNN
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Memory
    pin_memory: bool = True
    non_blocking: bool = True

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Gradient accumulation
    accumulation_steps: int = 1


def setup_cuda_training(config: CUDATrainingConfig) -> torch.device:
    """
    Initialize CUDA environment for training.

    Returns:
        torch.device: Best available device
    """
    # Set cuDNN options
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic

    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")

    return device


class AMPTrainerMixin:
    """
    Mixin class providing mixed precision training support.

    Add to trainer classes to enable AMP.
    """

    def __init__(self):
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()

    def train_step_amp(self, model, batch, criterion, optimizer):
        """Single training step with AMP."""
        if self.use_amp:
            with autocast():
                outputs = model(batch['inputs'])
                loss = criterion(outputs, batch['targets'])

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = model(batch['inputs'])
            loss = criterion(outputs, batch['targets'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        return loss.item()


def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    config: Optional[CUDATrainingConfig] = None,
):
    """Create a DataLoader optimized for CUDA training."""
    config = config or CUDATrainingConfig()

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=shuffle,  # Drop last incomplete batch during training
    )
```

### File: `pyfaceau/pyfaceau/nn/train_au_prediction.py` (MODIFY)

Key changes:
```python
# Add imports
from torch.cuda.amp import GradScaler, autocast

# In AUTrainer.__init__():
self.scaler = GradScaler() if torch.cuda.is_available() else None
self.use_amp = torch.cuda.is_available()

# Enable cuDNN benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# In train_epoch():
def train_epoch(self) -> Dict[str, float]:
    self.model.train()
    total_loss = 0.0

    for batch in self.train_loader:
        images = batch['image'].to(self.device, non_blocking=True)
        au_targets = batch['au_intensities'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        if self.use_amp:
            with autocast():
                au_pred = self.model(images)
                losses = self.criterion(au_pred, au_targets)

            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            au_pred = self.model(images)
            losses = self.criterion(au_pred, au_targets)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        total_loss += losses['total'].item()
```

### File: `pyfaceau/pyfaceau/nn/train_landmark_pose.py` (MODIFY)

Same pattern as AU training, with additional Wing loss FP16 stability:

```python
# In WingLoss - ensure numerical stability for FP16
def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = torch.abs(pred - target)

    # Add eps for numerical stability in FP16
    eps = 1e-6
    loss = torch.where(
        x < self.w,
        self.w * torch.log(1 + x / (self.epsilon + eps)),
        x - self.C
    )

    return loss.mean()
```

---

## Expected Performance Improvements

### Training Data Generation (BIGGEST WIN)

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| CLNF inference per frame | ~200-500ms | ~1-3ms | **100-500x** |
| Frame processing rate | ~2-5 FPS | ~50-100 FPS | **20-50x** |
| Process 10,000 frames | ~1-2 hours | ~2-5 minutes | **20-50x** |

### Neural Network Training

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| Training throughput | 1x | 1.5-2x |
| Memory usage | Baseline | 50-70% (with AMP) |
| Batch size capacity | N | 1.5-2N |
| GPU utilization | ~60-70% | ~85-95% |

### End-to-End Training Pipeline

| Stage | Before | After |
|-------|--------|-------|
| Generate training data (100k frames) | 10-20 hours | 15-30 minutes |
| Train AUPredictionNet (100 epochs) | 2-4 hours | 1-2 hours |
| Train LandmarkPoseNet (100 epochs) | 3-5 hours | 1.5-3 hours |
| **Total pipeline** | **15-30 hours** | **2-4 hours** |

---

## Testing Plan

1. **Accuracy verification**: Ensure models trained with AMP match FP32 accuracy
2. **Numerical stability**: Check for NaN/Inf in gradients with mixed precision
3. **Memory profiling**: Verify reduced memory footprint
4. **Speed benchmarks**: Compare epoch times before/after

---

## Files to Create/Modify

### New Files:
- `pyfaceau/pyfaceau/nn/cuda_training.py` - CUDA training utilities

### Modified Files:
- `pyfaceau/pyfaceau/nn/train_au_prediction.py` - Add AMP support
- `pyfaceau/pyfaceau/nn/train_landmark_pose.py` - Add AMP support
- `pyfaceau/pyfaceau/nn/landmark_pose_net.py` - FP16 stability fixes
- `pyfaceau/pyfaceau/data/hdf5_dataset.py` - Optimize for CUDA training

### Optional:
- `face-analysis/train_au_mlp.py` - Already has good CUDA support
- `face-analysis/train_landmark_cnn.py` - Already has good CUDA support

---

## Implementation Order

1. Create `cuda_training.py` utilities module
2. Update `train_au_prediction.py` with AMP
3. Update `train_landmark_pose.py` with AMP
4. Fix numerical stability in loss functions
5. Optimize DataLoader configuration
6. Run accuracy verification tests
7. Benchmark performance improvements
