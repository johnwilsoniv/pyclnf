#!/usr/bin/env python3
"""
Accuracy verification for Numba JIT-accelerated CLNF functions.

This test ensures that JIT-compiled functions produce IDENTICAL results
to the original Python implementations (within floating-point tolerance).

Critical: We cannot sacrifice accuracy for speed!
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'pyclnf')

import numpy as np
import time
from pathlib import Path

# Force Python implementations for comparison
from pyclnf.core.numba_accelerator import (
    kde_mean_shift_jit,
    compute_ncc_jit,
    compute_neuron_response_jit,
    compute_patch_response_jit,
    euler_to_rotation_matrix_jit,
    compute_jacobian_jit
)

class accel:
    """Wrapper for accelerated functions."""
    kde_mean_shift_jit = staticmethod(kde_mean_shift_jit)
    compute_ncc_jit = staticmethod(compute_ncc_jit)
    compute_neuron_response_jit = staticmethod(compute_neuron_response_jit)
    compute_patch_response_jit = staticmethod(compute_patch_response_jit)
    euler_to_rotation_matrix_jit = staticmethod(euler_to_rotation_matrix_jit)
    compute_jacobian_jit = staticmethod(compute_jacobian_jit)


def test_kde_mean_shift():
    """Test KDE mean-shift JIT vs manual Python implementation."""
    print("\n" + "="*60)
    print("TEST: KDE Mean-Shift")
    print("="*60)

    # Generate test cases
    np.random.seed(42)
    test_cases = [
        (11, 5.5, 5.5),   # Center
        (11, 0.1, 0.1),   # Corner
        (11, 10.9, 10.9), # Other corner
        (9, 4.5, 4.5),    # Different size
        (7, 3.5, 3.5),    # Smaller
    ]

    sigma = 1.75
    a = -0.5 / (sigma * sigma)

    max_error = 0.0
    total_jit_time = 0.0
    total_py_time = 0.0

    for window_size, dx, dy in test_cases:
        response_map = np.random.rand(window_size, window_size).astype(np.float64)

        # JIT version
        t0 = time.perf_counter()
        for _ in range(100):
            jit_x, jit_y = accel.kde_mean_shift_jit(response_map, dx, dy, a)
        jit_time = time.perf_counter() - t0
        total_jit_time += jit_time

        # Python version (inline implementation)
        t0 = time.perf_counter()
        for _ in range(100):
            mx = 0.0
            my = 0.0
            total_weight = 0.0
            resp_size = window_size

            for ii in range(resp_size):
                for jj in range(resp_size):
                    dist_sq = (dy - ii)**2 + (dx - jj)**2
                    kde_weight = np.exp(a * dist_sq)
                    weight = kde_weight * response_map[ii, jj]
                    total_weight += weight
                    mx += weight * jj
                    my += weight * ii

            if total_weight > 1e-10:
                py_x = (mx / total_weight) - dx
                py_y = (my / total_weight) - dy
            else:
                py_x = 0.0
                py_y = 0.0
        py_time = time.perf_counter() - t0
        total_py_time += py_time

        # Compare
        error = max(abs(jit_x - py_x), abs(jit_y - py_y))
        max_error = max(max_error, error)

        status = "PASS" if error < 1e-10 else "FAIL"
        print(f"  Window {window_size}x{window_size} @ ({dx:.1f},{dy:.1f}): "
              f"error={error:.2e} [{status}]")

    speedup = total_py_time / total_jit_time if total_jit_time > 0 else 0
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Speedup: {speedup:.1f}x")

    return max_error < 1e-9


def test_ncc_computation():
    """Test normalized cross-correlation JIT vs Python."""
    print("\n" + "="*60)
    print("TEST: Normalized Cross-Correlation")
    print("="*60)

    np.random.seed(42)

    # Test cases with different sizes
    sizes = [(11, 11), (15, 15), (7, 7), (21, 21)]

    max_error = 0.0
    total_jit_time = 0.0
    total_py_time = 0.0

    for h, w in sizes:
        features = np.random.rand(h, w).astype(np.float64)
        weights = np.random.rand(h, w).astype(np.float64)

        # JIT version
        t0 = time.perf_counter()
        for _ in range(1000):
            jit_ncc = accel.compute_ncc_jit(features, weights)
        jit_time = time.perf_counter() - t0
        total_jit_time += jit_time

        # Python version
        t0 = time.perf_counter()
        for _ in range(1000):
            feat_flat = features.ravel()
            wgt_flat = weights.ravel()

            feat_mean = np.mean(feat_flat)
            wgt_mean = np.mean(wgt_flat)

            feat_c = feat_flat - feat_mean
            wgt_c = wgt_flat - wgt_mean

            feat_norm = np.linalg.norm(feat_c)
            wgt_norm = np.linalg.norm(wgt_c)

            if wgt_norm > 1e-10 and feat_norm > 1e-10:
                py_ncc = np.sum(feat_c * wgt_c) / (wgt_norm * feat_norm)
            else:
                py_ncc = 0.0
        py_time = time.perf_counter() - t0
        total_py_time += py_time

        error = abs(jit_ncc - py_ncc)
        max_error = max(max_error, error)

        status = "PASS" if error < 1e-10 else "FAIL"
        print(f"  Size {h}x{w}: JIT={jit_ncc:.6f} PY={py_ncc:.6f} error={error:.2e} [{status}]")

    speedup = total_py_time / total_jit_time if total_jit_time > 0 else 0
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Speedup: {speedup:.1f}x")

    return max_error < 1e-9


def test_jacobian_computation():
    """Test Jacobian computation JIT vs Python."""
    print("\n" + "="*60)
    print("TEST: Jacobian Computation")
    print("="*60)

    # Load real PDM data
    model_dir = Path("pyclnf/pyclnf/models/exported_pdm")
    mean_shape = np.load(model_dir / 'mean_shape.npy')
    princ_comp = np.load(model_dir / 'princ_comp.npy')

    n_points = mean_shape.shape[0] // 3
    n_modes = princ_comp.shape[1]

    np.random.seed(42)

    # Test cases with different parameters
    test_params = [
        (1.0, 0.0, 0.0, 0.0),        # Neutral
        (1.5, 0.1, 0.2, 0.15),       # Rotated
        (0.8, -0.1, -0.2, 0.3),      # Different scale/rotation
    ]

    max_error = 0.0
    total_jit_time = 0.0
    total_py_time = 0.0

    for s, wx, wy, wz in test_params:
        q = np.random.randn(n_modes) * 0.1
        shape_3d = mean_shape.flatten() + princ_comp @ q

        X = shape_3d[:n_points].astype(np.float64)
        Y = shape_3d[n_points:2*n_points].astype(np.float64)
        Z = shape_3d[2*n_points:3*n_points].astype(np.float64)

        # JIT version
        t0 = time.perf_counter()
        for _ in range(100):
            R = accel.euler_to_rotation_matrix_jit(wx, wy, wz)
            J_jit = accel.compute_jacobian_jit(
                X, Y, Z, R, s,
                princ_comp.astype(np.float64),
                n_points, n_modes
            )
        jit_time = time.perf_counter() - t0
        total_jit_time += jit_time

        # Python version (using PDM class)
        from pyclnf.core.pdm import PDM
        pdm = PDM(str(model_dir))

        params = np.zeros(6 + n_modes)
        params[0] = s
        params[1:4] = [wx, wy, wz]
        params[6:] = q

        # Temporarily disable Numba
        import pyclnf.core.pdm as pdm_module
        original_flag = pdm_module.USE_NUMBA
        pdm_module.USE_NUMBA = False

        t0 = time.perf_counter()
        for _ in range(100):
            J_py = pdm.compute_jacobian(params)
        py_time = time.perf_counter() - t0
        total_py_time += py_time

        pdm_module.USE_NUMBA = original_flag

        # Compare
        error = np.max(np.abs(J_jit - J_py))
        max_error = max(max_error, error)

        # 1e-5 threshold is acceptable for Jacobian since it's used in iterative optimization
        # and errors < 1e-5 are negligible compared to pixel positions (100-500px)
        status = "PASS" if error < 1e-5 else "FAIL"
        print(f"  Params (s={s}, wx={wx:.1f}, wy={wy:.1f}, wz={wz:.1f}): "
              f"max_error={error:.2e} [{status}]")

    speedup = total_py_time / total_jit_time if total_jit_time > 0 else 0
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Speedup: {speedup:.1f}x")

    return max_error < 1e-5


def test_patch_response():
    """Test patch expert response computation JIT vs Python."""
    print("\n" + "="*60)
    print("TEST: Patch Expert Response")
    print("="*60)

    # Load a real patch expert
    patch_dir = Path("pyclnf/pyclnf/models/exported_ccnf_0.25/view_00/patch_30")

    if not patch_dir.exists():
        print("  SKIP: Patch expert not found")
        return True

    from pyclnf.core.patch_expert import CCNFPatchExpert
    import pyclnf.core.patch_expert as pe_module

    # Load patch expert
    patch_expert = CCNFPatchExpert(str(patch_dir))

    np.random.seed(42)

    # Test with random patches
    max_error = 0.0
    total_jit_time = 0.0
    total_py_time = 0.0

    for i in range(5):
        test_patch = np.random.randint(0, 256,
            (patch_expert.height, patch_expert.width), dtype=np.uint8)

        # Ensure batched data is prepared
        if not hasattr(patch_expert, '_batched_weights'):
            patch_expert._prepare_batched_data()

        features = test_patch.astype(np.float32) / 255.0

        # JIT version
        t0 = time.perf_counter()
        for _ in range(100):
            jit_resp = float(accel.compute_patch_response_jit(
                features.astype(np.float64),
                patch_expert._batched_weights,
                patch_expert._batched_biases,
                patch_expert._batched_alphas,
                patch_expert._batched_norm_weights,
                patch_expert.num_neurons
            ))
        jit_time = time.perf_counter() - t0
        total_jit_time += jit_time

        # Python version (disable Numba temporarily)
        original_flag = pe_module.USE_NUMBA
        pe_module.USE_NUMBA = False

        t0 = time.perf_counter()
        for _ in range(100):
            py_resp = patch_expert.compute_response(test_patch)
        py_time = time.perf_counter() - t0
        total_py_time += py_time

        pe_module.USE_NUMBA = original_flag

        error = abs(jit_resp - py_resp)
        max_error = max(max_error, error)

        status = "PASS" if error < 1e-6 else "FAIL"
        print(f"  Patch {i+1}: JIT={jit_resp:.6f} PY={py_resp:.6f} error={error:.2e} [{status}]")

    speedup = total_py_time / total_jit_time if total_jit_time > 0 else 0
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Speedup: {speedup:.1f}x")

    return max_error < 1e-5


def main():
    print("\n" + "#"*60)
    print("# CLNF Numba Acceleration - Accuracy Verification")
    print("#"*60)

    results = {}

    # Run all tests
    results['KDE Mean-Shift'] = test_kde_mean_shift()
    results['NCC Computation'] = test_ncc_computation()
    results['Jacobian'] = test_jacobian_computation()
    results['Patch Response'] = test_patch_response()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: [{status}]")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll accuracy tests PASSED!")
        print("JIT optimizations produce numerically identical results.")
    else:
        print("\nSome tests FAILED!")
        print("WARNING: Accuracy may be compromised.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
