# Minimal CUDA Optimization Setup

This folder contains everything you need to develop and test CUDA acceleration for pyclnf's CEN inference - **without building the full project**.

## What You Need to Install

```bash
# 1. Python 3.10+ (you already have this via winget)

# 2. PyTorch with CUDA (pick your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# OR
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

# 3. NumPy (for CPU baseline)
pip install numpy

# 4. Optional: for benchmarking
pip install tqdm
```

## Verify CUDA Works

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Files in This Folder

```
cuda_optimization/
├── README.md                 # This file
├── cen_core.py              # Extracted CEN inference (CPU baseline)
├── cen_cuda.py              # Your CUDA implementation goes here
├── test_accuracy.py         # Verify CUDA matches CPU
├── benchmark.py             # Speed comparison
└── sample_data.npz          # Synthetic test data (no model files needed)
```

## Quick Test

```bash
cd cuda_optimization
python test_accuracy.py      # Should show "PASS" for all tests
python benchmark.py          # Shows CPU vs CUDA speed
```

## The Goal

The CEN forward pass does this (simplified):
```
Input: patches (batch_size, 11, 11)
  ↓
Contrast normalize each patch
  ↓  
Layer 0: matmul + sigmoid
  ↓
Layer 1: matmul + sigmoid
  ↓
Output: responses (batch_size,)
```

Currently this runs 16,320 times per frame sequentially.
Your job: make it run once as a batched GPU operation.
