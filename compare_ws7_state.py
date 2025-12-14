#!/usr/bin/env python3
"""
Compare Python state at WS7 entry with C++ WS7 iter 0 dump.

C++ dump from: /tmp/cpp_ws7_rigid_iter0_dump.txt
"""

import sys
import numpy as np

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

# C++ WS7 iter 0 state from dump
CPP_GLOBAL_PARAMS = np.array([
    2.7569327354,   # scale
    -0.0625211820,  # rot_x
    0.1299121678,   # rot_y
    -0.0631994084,  # rot_z
    1597.0220947266,  # tx
    919.0394897461    # ty
])

CPP_LOCAL_PARAMS = np.array([
    -1.9738134146, 0.7379326224, -17.6084537506, 10.0534906387, -1.7963975668,
    1.7258944511, -5.4227809906, 17.0406837463, -12.6726417542, 11.6865844727
])

# C++ landmarks at WS7 entry (from dump)
CPP_LM36 = (1470.2796630859, 827.8734130859)
CPP_LM48 = (1525.7143554688, 1007.2557373047)
CPP_LM8 = (1610.1534423828, 1113.4343261719)


def main():
    import cv2
    from pyclnf import CLNF
    from pymtcnn import MTCNN

    print("=" * 80)
    print("WS7 STATE COMPARISON: Python vs C++ Dump")
    print("=" * 80)

    # Run Python to get state at WS7 entry
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    img = cv2.imread(img_path)

    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None

    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    # Run fit
    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5_arr)

    # Get Python state after WS9 (= WS7 entry)
    iteration_history = info.get('iteration_history', [])

    # Find last WS9 iteration
    ws9_iters = [h for h in iteration_history if h['window_size'] == 9]
    if not ws9_iters:
        print("ERROR: No WS9 iterations found")
        return

    py_ws9_final = ws9_iters[-1]
    py_params = py_ws9_final['params']

    print("\n" + "=" * 80)
    print("GLOBAL PARAMETERS (WS7 Entry State)")
    print("=" * 80)

    param_names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']

    print(f"\n{'Param':8} {'C++':>14} {'Python':>14} {'Diff':>12} {'%Err':>8}")
    print("-" * 60)

    for i, name in enumerate(param_names):
        cpp_val = CPP_GLOBAL_PARAMS[i]
        py_val = py_params[i]
        diff = py_val - cpp_val
        pct = abs(diff / cpp_val * 100) if abs(cpp_val) > 1e-6 else 0
        print(f"{name:8} {cpp_val:>14.8f} {py_val:>14.8f} {diff:>+12.8f} {pct:>7.2f}%")

    print("\n" + "=" * 80)
    print("LOCAL PARAMETERS (First 10)")
    print("=" * 80)

    print(f"\n{'Index':6} {'C++':>14} {'Python':>14} {'Diff':>12}")
    print("-" * 50)

    for i in range(min(10, len(CPP_LOCAL_PARAMS))):
        cpp_val = CPP_LOCAL_PARAMS[i]
        py_val = py_params[6 + i] if (6 + i) < len(py_params) else 0
        diff = py_val - cpp_val
        print(f"{i:6} {cpp_val:>14.4f} {py_val:>14.4f} {diff:>+12.4f}")

    print("\n" + "=" * 80)
    print("LANDMARK POSITIONS (WS7 Entry)")
    print("=" * 80)

    # Compute Python landmarks at WS9-end params
    py_landmarks = pdm.params_to_landmarks_2d(py_params)

    cpp_landmarks = {
        36: CPP_LM36,
        48: CPP_LM48,
        8: CPP_LM8
    }

    print(f"\n{'LM':4} {'C++ X':>12} {'C++ Y':>12} {'Py X':>12} {'Py Y':>12} {'Error':>10}")
    print("-" * 65)

    for lm_idx, (cpp_x, cpp_y) in cpp_landmarks.items():
        py_x, py_y = py_landmarks[lm_idx]
        err = np.sqrt((py_x - cpp_x)**2 + (py_y - cpp_y)**2)
        print(f"{lm_idx:4} {cpp_x:>12.4f} {cpp_y:>12.4f} {py_x:>12.4f} {py_y:>12.4f} {err:>10.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Global param differences
    global_diff = np.abs(py_params[:6] - CPP_GLOBAL_PARAMS)
    print(f"\nGlobal param total abs diff: {global_diff.sum():.8f}")
    print(f"  Scale diff: {global_diff[0]:.8f}")
    print(f"  Rotation diff sum: {global_diff[1:4].sum():.8f}")
    print(f"  Translation diff sum: {global_diff[4:6].sum():.8f}")

    # Check if states match closely enough
    if global_diff.sum() < 0.01:
        print("\n✓ States match closely - divergence likely in WS7 optimization itself")
    else:
        print("\n✗ States already diverged BEFORE WS7 entry")
        print("  → Root cause is in WS11 or WS9")


if __name__ == '__main__':
    main()
