"""
Test CUDA integration with pyclnf.

This script tests that the CUDA acceleration is properly integrated
and produces results matching the CPU implementation.
"""

import sys
import os
import time

# Add pyclnf to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_cuda_backend():
    """Test CUDA backend detection."""
    print("=" * 60)
    print("Test 1: CUDA Backend Detection")
    print("=" * 60)

    from pyclnf.core.cuda_backend import ComputeBackend, is_cuda_available

    cuda_available = is_cuda_available()
    print(f"  CUDA available: {cuda_available}")

    if cuda_available:
        device_name = ComputeBackend.get_device_name()
        print(f"  Device: {device_name}")

    backend = ComputeBackend.get()
    print(f"  Selected backend: {backend}")

    return cuda_available


def test_cen_cuda_module():
    """Test CEN CUDA module standalone."""
    print("\n" + "=" * 60)
    print("Test 2: CEN CUDA Module")
    print("=" * 60)

    from pyclnf.core.cen_cuda import cen_forward_batch_cuda, CENBatchProcessor
    import torch

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return True

    # Create test data
    np.random.seed(42)
    batch_size = 121  # typical response map size (11x11)
    patches = np.random.rand(batch_size, 11, 11).astype(np.float32)

    # Create synthetic weights
    input_dim = 11 * 11 + 1
    hidden_dim = 32
    w0 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
    b0 = np.zeros((1, hidden_dim), dtype=np.float32)
    w1 = np.random.randn(1, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros((1, 1), dtype=np.float32)

    # Run CUDA forward
    responses = cen_forward_batch_cuda(
        patches, width=11, height=11,
        w0=w0, b0=b0, a0=0,
        w1=w1, b1=b1, a1=0
    )

    print(f"  Input shape: {patches.shape}")
    print(f"  Output shape: {responses.shape}")
    print(f"  Output range: [{responses.min():.4f}, {responses.max():.4f}]")

    return True


def test_cen_model_cuda_init():
    """Test CENModel initialization with CUDA."""
    print("\n" + "=" * 60)
    print("Test 3: CENModel CUDA Initialization")
    print("=" * 60)

    try:
        from pyclnf.core.cen_patch_expert import CENModel, is_cuda_available

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyclnf", "models")

        if not os.path.exists(model_dir):
            print(f"  SKIP: Model directory not found: {model_dir}")
            return True

        # Check if model files exist
        dat_file = os.path.join(model_dir, "patch_experts", "cen_patches_0.25_of.dat")
        if not os.path.exists(dat_file):
            print(f"  SKIP: Model files not found (expected at {dat_file})")
            print("  Note: This is expected in test environments without the full model")
            return True

        cuda_available = is_cuda_available()
        print(f"  CUDA available for CEN: {cuda_available}")

        # Initialize with auto device selection
        print("  Loading CEN model...")
        model = CENModel(model_dir, device='auto')

        print(f"  Device: {model.device}")
        print(f"  CUDA processors: {len(model.cuda_processors)}")

        if model.cuda_processors:
            for scale, processor in model.cuda_processors.items():
                print(f"    Scale {scale}: {len(processor.experts)} experts initialized")

        return True

    except FileNotFoundError as e:
        print(f"  SKIP: Model files not found: {e}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_accuracy_cpu_vs_cuda():
    """Test that CUDA produces same results as CPU."""
    print("\n" + "=" * 60)
    print("Test 4: CPU vs CUDA Accuracy")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        # Use the standalone test
        sys.path.insert(0, os.path.dirname(__file__))
        from cen_core import generate_synthetic_weights, generate_test_patches, cen_forward_batch_cpu
        from cen_cuda import cen_forward_batch_cuda

        weights = generate_synthetic_weights()
        patches = generate_test_patches(batch_size=1000)

        # CPU
        cpu_result = cen_forward_batch_cpu(patches, **weights)

        # CUDA
        cuda_result = cen_forward_batch_cuda(patches, **weights)

        # Compare
        max_diff = np.abs(cpu_result - cuda_result).max()
        mean_diff = np.abs(cpu_result - cuda_result).mean()

        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        passed = max_diff < 1e-5
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return passed

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """Quick benchmark comparison."""
    print("\n" + "=" * 60)
    print("Test 5: Quick Benchmark")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        sys.path.insert(0, os.path.dirname(__file__))
        from cen_core import generate_synthetic_weights, generate_test_patches, cen_forward_batch_cpu
        from cen_cuda import cen_forward_batch_cuda

        weights = generate_synthetic_weights()
        batch_size = 8228  # typical per-frame batch
        patches = generate_test_patches(batch_size=batch_size)

        # Warmup
        _ = cen_forward_batch_cuda(patches, **weights)
        torch.cuda.synchronize()

        # Benchmark CPU
        n_runs = 5
        cpu_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = cen_forward_batch_cpu(patches, **weights)
            cpu_times.append(time.perf_counter() - t0)
        cpu_avg = np.mean(cpu_times) * 1000

        # Benchmark CUDA
        cuda_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = cen_forward_batch_cuda(patches, **weights)
            torch.cuda.synchronize()
            cuda_times.append(time.perf_counter() - t0)
        cuda_avg = np.mean(cuda_times) * 1000

        speedup = cpu_avg / cuda_avg

        print(f"  Batch size: {batch_size}")
        print(f"  CPU: {cpu_avg:.2f} ms")
        print(f"  CUDA: {cuda_avg:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        return speedup > 1.0  # Should be faster

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CUDA Integration Tests for pyclnf")
    print("=" * 60)

    results = []

    results.append(("CUDA Backend", test_cuda_backend()))
    results.append(("CEN CUDA Module", test_cen_cuda_module()))
    results.append(("CENModel CUDA Init", test_cen_model_cuda_init()))
    results.append(("CPU vs CUDA Accuracy", test_accuracy_cpu_vs_cuda()))
    results.append(("Benchmark", test_benchmark()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("=" * 60)
    sys.exit(0 if all_passed else 1)
