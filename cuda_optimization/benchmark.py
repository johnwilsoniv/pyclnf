"""
Benchmark: CPU vs CUDA CEN Inference

Run with: python benchmark.py
"""

import numpy as np
import time
import sys

# Benchmark configuration
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
BATCH_SIZES = [121, 1000, 8228, 16320]  # 121 = single response map, 8228 = typical frame


def benchmark_cpu(patches, weights, iterations):
    """Benchmark CPU implementation."""
    from cen_core import cen_forward_batch_cpu
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        cen_forward_batch_cpu(patches, **weights)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        cen_forward_batch_cpu(patches, **weights)
        times.append(time.perf_counter() - t0)
    
    return times


def benchmark_cuda(patches, weights, iterations):
    """Benchmark CUDA implementation."""
    import torch
    from cen_cuda import CENBatchProcessor
    
    # Create processor (weights stay on GPU)
    processor = CENBatchProcessor(weights, device='cuda')
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        processor.process(patches)
    
    # Sync to ensure warmup is complete
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        processor.process(patches)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return times


def benchmark_cuda_tensor_only(patches, weights, iterations):
    """
    Benchmark CUDA with data already on GPU.
    
    This shows the theoretical maximum speedup when you avoid
    CPU<->GPU data transfers (which you should in a real pipeline).
    """
    import torch
    from cen_cuda import CENBatchProcessor
    
    # Create processor
    processor = CENBatchProcessor(weights, device='cuda')
    
    # Pre-move data to GPU
    patches_gpu = torch.tensor(patches, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        processor.process_tensor(patches_gpu)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        processor.process_tensor(patches_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return times


def print_stats(name, times, batch_size):
    """Print timing statistics."""
    times_ms = np.array(times) * 1000
    patches_per_sec = batch_size / np.mean(times)
    
    print(f"  {name:20s}: {np.mean(times_ms):7.2f}ms ± {np.std(times_ms):5.2f}ms  "
          f"({patches_per_sec/1000:.1f}K patches/sec)")


def run_benchmarks():
    """Run all benchmarks."""
    from cen_core import generate_synthetic_weights, generate_test_patches
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        cuda_available = False
        print("PyTorch not installed - CUDA benchmarks will be skipped")
    
    weights = generate_synthetic_weights()
    
    print("\n" + "=" * 70)
    print("CEN INFERENCE BENCHMARK")
    print("=" * 70)
    
    for batch_size in BATCH_SIZES:
        patches = generate_test_patches(batch_size=batch_size)
        
        print(f"\nBatch size: {batch_size}")
        print("-" * 50)
        
        # CPU benchmark
        cpu_times = benchmark_cpu(patches, weights, BENCHMARK_ITERATIONS)
        print_stats("CPU (NumPy)", cpu_times, batch_size)
        
        if cuda_available:
            # CUDA with data transfer
            cuda_times = benchmark_cuda(patches, weights, BENCHMARK_ITERATIONS)
            print_stats("CUDA (with transfer)", cuda_times, batch_size)
            
            # CUDA without data transfer
            cuda_notrans_times = benchmark_cuda_tensor_only(patches, weights, BENCHMARK_ITERATIONS)
            print_stats("CUDA (GPU only)", cuda_notrans_times, batch_size)
            
            # Speedup
            speedup_with_transfer = np.mean(cpu_times) / np.mean(cuda_times)
            speedup_gpu_only = np.mean(cpu_times) / np.mean(cuda_notrans_times)
            print(f"  {'Speedup':20s}: {speedup_with_transfer:.1f}x (with transfer), "
                  f"{speedup_gpu_only:.1f}x (GPU only)")
    
    # Frame-level estimate
    print("\n" + "=" * 70)
    print("FRAME-LEVEL PERFORMANCE ESTIMATE")
    print("=" * 70)
    
    # Typical frame: 16,320 patches
    frame_patches = generate_test_patches(batch_size=16320)
    
    cpu_frame_times = benchmark_cpu(frame_patches, weights, 50)
    cpu_fps = 1.0 / np.mean(cpu_frame_times)
    print(f"\nCPU: {np.mean(cpu_frame_times)*1000:.1f}ms/frame = {cpu_fps:.1f} FPS")
    
    if cuda_available:
        cuda_frame_times = benchmark_cuda_tensor_only(frame_patches, weights, 50)
        cuda_fps = 1.0 / np.mean(cuda_frame_times)
        speedup = cpu_fps / cuda_fps if cuda_fps > 0 else 0
        print(f"CUDA: {np.mean(cuda_frame_times)*1000:.1f}ms/frame = {cuda_fps:.1f} FPS")
        print(f"Speedup: {np.mean(cpu_frame_times)/np.mean(cuda_frame_times):.1f}x")
        
        print(f"\nNote: This is just the CEN inference portion.")
        print(f"Full pipeline has other costs (image loading, warping, etc.)")


if __name__ == "__main__":
    run_benchmarks()
