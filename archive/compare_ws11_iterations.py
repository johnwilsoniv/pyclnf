#!/usr/bin/env python3
"""
Compare WS11 RIGID iterations: Python vs C++

C++ data from: /tmp/cpp_ws11_rigid_iterations.txt
"""

import sys
import numpy as np
import cv2

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

# C++ WS11 RIGID iteration data from dump
CPP_WS11_RIGID = [
    {  # Iteration 0
        'gradient': np.array([-1869.3754, 51210.8711, 29970.7871, -28833.3535, -519.7841, 1564.0923]),
        'hessian_diag': np.array([201627.1406, 420364.0, 420364.0, 2443865.75, 68.0, 68.0]),
        'delta_p': np.array([-0.006725, 0.122627, 0.071190, -0.013097, -7.643884, 23.001356]),
        'mean_shift_sum': (-519.7840, 1564.0923),
        'mean_shift_lm36': (-3.3696, 26.1750),
    },
    {  # Iteration 1
        'gradient': np.array([11223.6094, 14511.3076, 11944.0576, -10864.9053, -179.8759, 652.8416]),
        'hessian_diag': np.array([199910.0, 439266.4062, 441470.0, 2411911.75, 68.0, 68.0]),
        'delta_p': np.array([0.065444, 0.037068, 0.031675, -0.005131, -2.645233, 9.600612]),
    },
    {  # Iteration 2
        'gradient': np.array([6491.1562, -4644.1665, 3608.1135, -4130.1323, -33.2616, 221.7483]),
        'hessian_diag': np.array([198688.1875, 473550.5312, 474929.0, 2484678.5, 68.0, 68.0]),
        'delta_p': np.array([0.032485, -0.007503, 0.010395, -0.002728, -0.489141, 3.261004]),
    },
    {  # Iteration 3
        'gradient': np.array([2910.0442, -6349.9092, 1248.2825, -1958.0750, -4.6210, 88.8553]),
        'hessian_diag': np.array([198698.9688, 483265.5938, 483180.4062, 2529868.5, 68.0, 68.0]),
        'delta_p': np.array([0.012647, -0.012508, 0.003845, -0.001681, -0.067957, 1.306695]),
    },
    {  # Iteration 4
        'gradient': np.array([1146.7036, -4726.9282, 512.1938, -1744.6879, 2.2786, 45.4091]),
        'hessian_diag': np.array([198908.75, 484477.8125, 483381.6875, 2550553.5, 68.0, 68.0]),
        'delta_p': np.array([0.004167, -0.009803, 0.001673, -0.001320, 0.033508, 0.667782]),
    },
]


def main():
    from pyclnf import CLNF
    from pymtcnn import MTCNN

    print("=" * 80)
    print("WS11 RIGID ITERATION COMPARISON: Python vs C++")
    print("=" * 80)

    # Run Python
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    img = cv2.imread(img_path)

    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None

    clnf = CLNF(detector=None)
    clnf.optimizer.debug_mode = True  # Enable debug output

    # Run fit
    print("\nRunning Python CLNF with debug output...")
    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5_arr)

    # Get Python WS11 iterations
    iteration_history = info.get('iteration_history', [])
    ws11_rigid = [h for h in iteration_history if h['window_size'] == 11 and h['phase'] == 'rigid']

    print("\n" + "=" * 80)
    print("ITERATION-BY-ITERATION COMPARISON")
    print("=" * 80)

    param_names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']

    # Compare each iteration
    for iter_idx in range(min(5, len(ws11_rigid), len(CPP_WS11_RIGID))):
        print(f"\n{'='*80}")
        print(f"ITERATION {iter_idx}")
        print(f"{'='*80}")

        cpp = CPP_WS11_RIGID[iter_idx]
        py = ws11_rigid[iter_idx] if iter_idx < len(ws11_rigid) else None

        if py is None:
            print("  No Python data for this iteration")
            continue

        # C++ delta_p
        print("\nC++ delta_p:")
        for i, name in enumerate(param_names):
            print(f"  {name:8}: {cpp['delta_p'][i]:+.8f}")

        # Python params after this iteration - compute delta from previous
        py_params = py['params']
        if iter_idx > 0:
            prev_params = ws11_rigid[iter_idx - 1]['params']
            py_delta = py_params[:6] - prev_params[:6]
        else:
            # For iter 0, we need initial params
            pdm = clnf.pdm
            init_params = pdm.init_params(bbox)
            py_delta = py_params[:6] - init_params[:6]

        print("\nPython delta_p (computed from param diff):")
        for i, name in enumerate(param_names):
            print(f"  {name:8}: {py_delta[i]:+.8f}")

        # Compare
        print("\nComparison:")
        print(f"  {'Param':8} {'C++ delta':>14} {'Py delta':>14} {'Diff':>12} {'%Err':>8}")
        print(f"  {'-'*60}")

        for i, name in enumerate(param_names):
            cpp_val = cpp['delta_p'][i]
            py_val = py_delta[i]
            diff = py_val - cpp_val
            pct = abs(diff / cpp_val * 100) if abs(cpp_val) > 1e-6 else 0
            marker = " ⚠" if pct > 10 else ""
            print(f"  {name:8} {cpp_val:>+14.6f} {py_val:>+14.6f} {diff:>+12.6f} {pct:>7.1f}%{marker}")

        # Compare gradients if we have them
        if 'gradient' in cpp:
            print("\n  Gradient comparison (tx, ty only - directly from mean-shift sum):")
            cpp_tx = cpp['gradient'][4]
            cpp_ty = cpp['gradient'][5]
            # Python mean-shift sum should match tx/ty gradient
            py_ms_norm = py['mean_shift_norm']
            print(f"    C++ tx gradient: {cpp_tx:.4f}")
            print(f"    C++ ty gradient: {cpp_ty:.4f}")
            print(f"    Python mean-shift norm: {py_ms_norm:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Where does divergence start?")
    print("=" * 80)

    # Compute cumulative rotation error
    print("\nCumulative rotation after each iteration:")
    print(f"  {'Iter':4} {'C++ rot_x':>12} {'Py rot_x':>12} {'Diff':>10}")
    print(f"  {'-'*45}")

    cpp_rot_x = 0.0
    py_prev_params = None

    for iter_idx in range(min(5, len(ws11_rigid), len(CPP_WS11_RIGID))):
        cpp_rot_x += CPP_WS11_RIGID[iter_idx]['delta_p'][1]  # rot_x is index 1

        py = ws11_rigid[iter_idx]
        py_rot_x = py['params'][1]  # rot_x in Python params

        diff = py_rot_x - cpp_rot_x
        print(f"  {iter_idx:4} {cpp_rot_x:>+12.6f} {py_rot_x:>+12.6f} {diff:>+10.6f}")


if __name__ == '__main__':
    main()
