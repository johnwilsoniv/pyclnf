#!/usr/bin/env python3
"""
Test pyCLNF error progression at each window size.

This tracks the error compared to C++ at each WS transition to understand
where error accumulates (WS11 → WS9 → WS7 → WS5).
"""

import sys
import numpy as np
import cv2
import subprocess
import tempfile
import os

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

OPENFACE_BIN = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'


def run_cpp_openface(image_path: str) -> np.ndarray:
    """Run C++ OpenFace and return 68 landmarks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [OPENFACE_BIN, '-f', image_path, '-out_dir', tmpdir, '-2Dfp']
        subprocess.run(cmd, capture_output=True, check=True)

        import pandas as pd
        csv_path = os.path.join(tmpdir, os.path.basename(image_path).replace('.jpg', '.csv').replace('.png', '.csv'))
        df = pd.read_csv(csv_path)

        landmarks = np.zeros((68, 2))
        for i in range(68):
            landmarks[i, 0] = df[f'x_{i}'].values[0]
            landmarks[i, 1] = df[f'y_{i}'].values[0]

        return landmarks


def main():
    from pyclnf import CLNF
    from pymtcnn import MTCNN

    print("=" * 80)
    print("PYCLNF ERROR PROGRESSION BY WINDOW SIZE")
    print("=" * 80)

    # Use test image
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load {img_path}")
        return

    print(f"\nTest image: {img_path}")
    print(f"Image size: {img.shape}")

    # Get C++ reference landmarks
    print("\n[1] Running C++ OpenFace for reference...")
    cpp_landmarks = run_cpp_openface(img_path)
    print(f"    C++ LM36: ({cpp_landmarks[36,0]:.2f}, {cpp_landmarks[36,1]:.2f})")
    print(f"    C++ LM8:  ({cpp_landmarks[8,0]:.2f}, {cpp_landmarks[8,1]:.2f})")

    # Detect face with MTCNN
    print("\n[2] Detecting face with MTCNN...")
    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None
    print(f"    MTCNN bbox: {bbox}")

    # Initialize CLNF
    print("\n[3] Running Python CLNF with window size tracking...")
    clnf = CLNF(detector=None)
    pdm = clnf.pdm
    optimizer = clnf.optimizer

    # Get initial params
    if lm5_arr is not None:
        init_params = pdm.init_params_from_5pt(bbox, lm5_arr)
    else:
        init_params = pdm.init_params(bbox)

    init_landmarks = pdm.params_to_landmarks_2d(init_params)

    # Compute init error vs C++
    init_errors = np.linalg.norm(init_landmarks - cpp_landmarks, axis=1)
    print(f"\n    Init error vs C++: mean={init_errors.mean():.4f} px, max={init_errors.max():.4f} px")

    # Run full fit which handles patch experts internally
    final_landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5_arr, return_params=True)

    # Extract landmarks at each WS transition
    print("\n" + "=" * 80)
    print("ERROR PROGRESSION BY WINDOW SIZE")
    print("=" * 80)

    ws_errors = {}

    if 'iteration_history' in info:
        # Find last iteration for each window size
        ws_final_params = {}
        for iter_info in info['iteration_history']:
            ws = iter_info['window_size']
            phase = iter_info['phase']
            ws_final_params[ws] = iter_info['params'].copy()

        for ws in [11, 9, 7, 5]:
            if ws not in ws_final_params:
                continue

            params = ws_final_params[ws]
            py_landmarks = pdm.params_to_landmarks_2d(params)
            errors = np.linalg.norm(py_landmarks - cpp_landmarks, axis=1)

            ws_errors[ws] = (errors.mean(), errors.max())

    # Summary table
    print("\n| Stage                | Mean Error  | Max Error   |")
    print("|----------------------|-------------|-------------|")
    print(f"| Init (bbox, frontal) | {init_errors.mean():>10.6f} px | {init_errors.max():>10.6f} px |")

    for ws in [11, 9, 7, 5]:
        if ws in ws_errors:
            mean_err, max_err = ws_errors[ws]
            print(f"| After WS{ws:<13}| {mean_err:>10.6f} px | {max_err:>10.6f} px |")

    # Final result (final_landmarks already returned from fit)
    final_errors = np.linalg.norm(final_landmarks - cpp_landmarks, axis=1)
    print(f"| Final                | {final_errors.mean():>10.6f} px | {final_errors.max():>10.6f} px |")

    # Show worst landmarks
    print("\n" + "=" * 80)
    print("WORST LANDMARKS")
    print("=" * 80)
    worst_idx = np.argsort(final_errors)[-5:][::-1]
    for idx in worst_idx:
        py_lm = final_landmarks[idx]
        cpp_lm = cpp_landmarks[idx]
        print(f"  LM{idx:2d}: error={final_errors[idx]:.4f} px "
              f"| py=({py_lm[0]:.1f}, {py_lm[1]:.1f}) "
              f"| cpp=({cpp_lm[0]:.1f}, {cpp_lm[1]:.1f})")


if __name__ == '__main__':
    main()
