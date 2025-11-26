"""
Test CCNF CUDA implementation.

Tests that CUDA produces same results as CPU and measures speedup.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_cuda_available():
    """Test CUDA availability."""
    print("=" * 60)
    print("Test 1: CUDA Availability")
    print("=" * 60)

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        return cuda_available
    except ImportError:
        print("  PyTorch not installed")
        return False


def test_ccnf_inference_cuda():
    """Test CCNFInferenceCUDA matches CPU."""
    print("\n" + "=" * 60)
    print("Test 2: CCNF Inference CUDA vs CPU")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        from pyclnf.core.patch_expert import CCNFPatchExpert
        from pyclnf.core.ccnf_cuda import CCNFInferenceCUDA

        # Load a real patch expert
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyclnf", "models")
        patch_dir = os.path.join(model_dir, "exported_ccnf_0.25", "view_00", "patch_30")

        if not os.path.exists(patch_dir):
            print(f"  SKIP: Patch dir not found: {patch_dir}")
            return True

        print(f"  Loading patch expert from {patch_dir}")
        cpu_expert = CCNFPatchExpert(patch_dir)
        cuda_expert = CCNFInferenceCUDA(cpu_expert, device='cuda')

        print(f"  Patch size: {cpu_expert.width}x{cpu_expert.height}")
        print(f"  Num neurons: {cpu_expert.num_neurons}")

        # Test with random patches
        np.random.seed(42)
        batch_size = 121  # 11x11 window
        patches = np.random.randint(0, 256, (batch_size, cpu_expert.height, cpu_expert.width)).astype(np.float32)

        # CPU results
        cpu_results = np.array([cpu_expert.compute_response(p.astype(np.uint8)) for p in patches])

        # CUDA results
        patches_tensor = torch.tensor(patches, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            cuda_results = cuda_expert.forward_batch(patches_tensor).cpu().numpy()

        # Compare
        max_diff = np.abs(cpu_results - cuda_results).max()
        mean_diff = np.abs(cpu_results - cuda_results).mean()

        print(f"  CPU results range: [{cpu_results.min():.4f}, {cpu_results.max():.4f}]")
        print(f"  CUDA results range: [{cuda_results.min():.4f}, {cuda_results.max():.4f}]")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        # Allow some tolerance for floating point differences
        passed = max_diff < 1e-4
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return passed

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ccnf_batch_processor():
    """Test CCNFBatchProcessor."""
    print("\n" + "=" * 60)
    print("Test 3: CCNF Batch Processor")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        from pyclnf.core.patch_expert import CCNFModel

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyclnf", "models")

        if not os.path.exists(os.path.join(model_dir, "exported_ccnf_0.25")):
            print(f"  SKIP: Model dir not found")
            return True

        print("  Loading CCNFModel with device='cuda'...")
        model = CCNFModel(model_dir, scales=[0.25], device='cuda')

        print(f"  Device: {model.device}")
        print(f"  CUDA processors: {len(model.cuda_processors)}")

        if model.cuda_processors:
            processor = model.get_cuda_processor(0.25)
            if processor:
                print(f"  Scale 0.25 processor: {len(processor.experts)} experts")
                return True
            else:
                print("  ERROR: No processor for scale 0.25")
                return False
        else:
            print("  WARNING: No CUDA processors initialized")
            return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_accuracy_single_landmark():
    """Test accuracy for a single landmark's response map."""
    print("\n" + "=" * 60)
    print("Test 4: Single Landmark Accuracy")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        from pyclnf.core.patch_expert import CCNFPatchExpert
        from pyclnf.core.ccnf_cuda import CCNFBatchProcessor

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyclnf", "models")
        patch_dir = os.path.join(model_dir, "exported_ccnf_0.25", "view_00", "patch_36")

        if not os.path.exists(patch_dir):
            print(f"  SKIP: Patch dir not found")
            return True

        cpu_expert = CCNFPatchExpert(patch_dir)

        # Initialize CUDA processor
        processor = CCNFBatchProcessor(device='cuda')
        processor.initialize_experts({36: cpu_expert})

        # Create test area_of_interest (21x21 for window_size=11 with 11x11 patch)
        np.random.seed(42)
        window_size = 11
        area_size = window_size + cpu_expert.width - 1
        area_of_interest = np.random.randint(0, 256, (area_size, area_size)).astype(np.uint8)

        # CPU: compute response map
        cpu_response_map = np.zeros((window_size, window_size))
        half_window = window_size // 2
        center = (area_size - 1) // 2

        for i in range(window_size):
            for j in range(window_size):
                py = center - half_window + i
                px = center - half_window + j
                y1 = py - cpu_expert.height // 2
                x1 = px - cpu_expert.width // 2
                patch = area_of_interest[y1:y1+cpu_expert.height, x1:x1+cpu_expert.width]
                cpu_response_map[i, j] = cpu_expert.compute_response(patch)

        # CUDA: batch compute
        patches = []
        for i in range(window_size):
            for j in range(window_size):
                py = center - half_window + i
                px = center - half_window + j
                y1 = py - cpu_expert.height // 2
                x1 = px - cpu_expert.width // 2
                patches.append(area_of_interest[y1:y1+cpu_expert.height, x1:x1+cpu_expert.width])
        patches = np.array(patches, dtype=np.float32)

        cuda_responses = processor.process_single(36, patches)
        cuda_response_map = cuda_responses.reshape(window_size, window_size)

        # Compare
        max_diff = np.abs(cpu_response_map - cuda_response_map).max()
        mean_diff = np.abs(cpu_response_map - cuda_response_map).mean()

        print(f"  Window size: {window_size}x{window_size}")
        print(f"  CPU response range: [{cpu_response_map.min():.4f}, {cpu_response_map.max():.4f}]")
        print(f"  CUDA response range: [{cuda_response_map.min():.4f}, {cuda_response_map.max():.4f}]")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        passed = max_diff < 1e-4
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return passed

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """Benchmark CPU vs CUDA."""
    print("\n" + "=" * 60)
    print("Test 5: Benchmark")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: CUDA not available")
            return True

        from pyclnf.core.patch_expert import CCNFPatchExpert
        from pyclnf.core.ccnf_cuda import CCNFInferenceCUDA

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyclnf", "models")
        patch_dir = os.path.join(model_dir, "exported_ccnf_0.25", "view_00", "patch_30")

        if not os.path.exists(patch_dir):
            print(f"  SKIP: Patch dir not found")
            return True

        cpu_expert = CCNFPatchExpert(patch_dir)
        cuda_expert = CCNFInferenceCUDA(cpu_expert, device='cuda')

        # Test different batch sizes
        batch_sizes = [121, 1000, 8228]

        for batch_size in batch_sizes:
            patches = np.random.randint(0, 256, (batch_size, cpu_expert.height, cpu_expert.width)).astype(np.float32)

            # Warmup CUDA
            patches_tensor = torch.tensor(patches, dtype=torch.float32, device='cuda')
            with torch.no_grad():
                _ = cuda_expert.forward_batch(patches_tensor)
            torch.cuda.synchronize()

            # Benchmark CPU
            n_runs = 3
            cpu_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                for p in patches:
                    _ = cpu_expert.compute_response(p.astype(np.uint8))
                cpu_times.append(time.perf_counter() - t0)
            cpu_avg = np.mean(cpu_times) * 1000

            # Benchmark CUDA
            cuda_times = []
            for _ in range(n_runs):
                patches_tensor = torch.tensor(patches, dtype=torch.float32, device='cuda')
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = cuda_expert.forward_batch(patches_tensor)
                torch.cuda.synchronize()
                cuda_times.append(time.perf_counter() - t0)
            cuda_avg = np.mean(cuda_times) * 1000

            speedup = cpu_avg / cuda_avg

            print(f"  Batch {batch_size}: CPU={cpu_avg:.1f}ms, CUDA={cuda_avg:.2f}ms, Speedup={speedup:.1f}x")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CCNF CUDA Tests")
    print("=" * 60)

    results = []

    results.append(("CUDA Available", test_cuda_available()))
    results.append(("CCNF Inference CUDA", test_ccnf_inference_cuda()))
    results.append(("CCNF Batch Processor", test_ccnf_batch_processor()))
    results.append(("Single Landmark Accuracy", test_accuracy_single_landmark()))
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
