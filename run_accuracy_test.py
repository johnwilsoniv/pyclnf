#!/usr/bin/env python3
"""
Accuracy test comparing pyCLNF vs C++ OpenFace.

This script:
1. Runs C++ OpenFace to get ground truth landmarks
2. Runs Python CLNF with per-window-size tracking
3. Compares error at each optimization stage
"""

import sys
import os

# Add paths for local modules
BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

import numpy as np
import cv2
import subprocess
import pandas as pd
from pathlib import Path


def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Run C++ OpenFace and extract landmarks."""
    out_dir = '/tmp/cpp_accuracy_test'
    os.makedirs(out_dir, exist_ok=True)

    # Run OpenFace
    result = subprocess.run([
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir,
        '-2Dfp'
    ], capture_output=True, timeout=60)

    # Load landmarks from CSV
    csv_name = Path(image_path).stem + '.csv'
    csv_path = os.path.join(out_dir, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"C++ output not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check column format (with or without space prefix)
    if 'x_0' in df.columns:
        x_col = 'x_{}'
        y_col = 'y_{}'
    elif ' x_0' in df.columns:
        x_col = ' x_{}'
        y_col = ' y_{}'
    else:
        raise ValueError(f"Unknown CSV format. Columns: {list(df.columns[:10])}")

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[x_col.format(i)].values[0]
        landmarks[i, 1] = df[y_col.format(i)].values[0]

    return landmarks


def run_accuracy_test(image_path: str):
    """Run accuracy comparison between Python and C++."""
    print("=" * 70)
    print("PYCLNF vs C++ OPENFACE ACCURACY TEST")
    print("=" * 70)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return
    print(f"\nImage: {image_path}")
    print(f"Shape: {img.shape}")

    # Get C++ reference landmarks
    print("\n[1] Running C++ OpenFace...")
    try:
        cpp_landmarks = get_cpp_landmarks(image_path)
        print(f"    ✓ C++ landmarks loaded")
        print(f"    LM0: ({cpp_landmarks[0,0]:.2f}, {cpp_landmarks[0,1]:.2f})")
        print(f"    LM36: ({cpp_landmarks[36,0]:.2f}, {cpp_landmarks[36,1]:.2f})")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # Initialize Python CLNF
    print("\n[2] Initializing Python CLNF...")
    from pyclnf import CLNF
    from pyclnf.core.pdm import PDM

    # Try to use MTCNN for detection
    try:
        from pymtcnn import MTCNN
        mtcnn = MTCNN(backend='coreml')
        print("    ✓ MTCNN detector loaded")

        # Detect face
        bboxes, landmarks_5pt = mtcnn.detect(img)
        if len(bboxes) == 0:
            print("    ERROR: No face detected by MTCNN")
            return

        bbox = tuple(bboxes[0][:4])
        lm5 = landmarks_5pt[0] if landmarks_5pt is not None else None
        print(f"    ✓ Face detected: bbox={bbox}")
    except Exception as e:
        print(f"    MTCNN failed: {e}")
        print("    Falling back to bbox from C++ landmarks...")

        # Compute bbox from C++ landmarks
        x_min, y_min = cpp_landmarks.min(axis=0)
        x_max, y_max = cpp_landmarks.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        pad = 0.2
        bbox = (
            x_min - width * pad,
            y_min - height * pad,
            x_max + width * pad,
            y_max + height * pad
        )
        lm5 = None
        print(f"    ✓ Computed bbox: {bbox}")

    # Initialize CLNF (without detector since we have bbox)
    clnf = CLNF(detector=None)

    # Get PDM for initialization comparison
    pdm = clnf.pdm

    # Initialize from bbox
    print("\n[3] Comparing initialization...")
    # Note: PDM.init_params takes (x, y, w, h) format
    # MTCNN returns (x, y, w, h) format already
    init_params = pdm.init_params(bbox)
    init_landmarks = pdm.params_to_landmarks_2d(init_params)
    init_error = np.linalg.norm(init_landmarks - cpp_landmarks, axis=1)
    print(f"    Init error: mean={init_error.mean():.6f} px, max={init_error.max():.6f} px")

    # Run optimization with per-stage tracking
    print("\n[4] Running optimization with stage tracking...")

    # We need to modify the fit to get per-window-size results
    # For now, run full fit and get final
    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5)

    if landmarks is None:
        print("    ERROR: Fit failed")
        return

    # Final error
    final_error = np.linalg.norm(landmarks - cpp_landmarks, axis=1)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Analyze per-window-size error from iteration history
    ws_errors = {}
    if 'iteration_history' in info:
        for iter_info in info['iteration_history']:
            ws = iter_info['window_size']
            params = iter_info['params']

            # Compute landmarks at this iteration
            iter_landmarks = pdm.params_to_landmarks_2d(params)
            iter_error = np.linalg.norm(iter_landmarks - cpp_landmarks, axis=1)

            # Keep the last iteration for each window size
            ws_errors[ws] = {
                'mean': iter_error.mean(),
                'max': iter_error.max(),
                'phase': iter_info['phase']
            }

    print(f"\n| Stage                | Mean Error  | Max Error   |")
    print(f"|----------------------|-------------|-------------|")
    print(f"| Init (bbox)          | {init_error.mean():.6f} px | {init_error.max():.6f} px |")

    # Print per-window-size errors in order
    for ws in [11, 9, 7, 5]:
        if ws in ws_errors:
            err = ws_errors[ws]
            print(f"| After WS{ws}           | {err['mean']:.6f} px | {err['max']:.6f} px |")

    print(f"| Final (all WS)       | {final_error.mean():.6f} px | {final_error.max():.6f} px |")

    # Per-landmark error analysis
    print(f"\nWorst 5 landmarks:")
    worst_indices = np.argsort(final_error)[-5:]
    for idx in reversed(worst_indices):
        cpp_pos = cpp_landmarks[idx]
        py_pos = landmarks[idx]
        print(f"  LM{idx}: error={final_error[idx]:.4f} px")
        print(f"         C++: ({cpp_pos[0]:.2f}, {cpp_pos[1]:.2f})")
        print(f"         Py:  ({py_pos[0]:.2f}, {py_pos[1]:.2f})")

    # Check if iteration history is available
    if 'iteration_history' in info:
        print(f"\nIteration history: {len(info['iteration_history'])} iterations recorded")

    # Show eye refinement status
    if 'eye_refined' in info:
        print(f"Eye refinement applied: {info['eye_refined']}")

    # Error progression analysis
    print(f"\nError progression analysis:")
    ws_list = [11, 9, 7, 5]
    prev_ws = None
    for ws in ws_list:
        if ws in ws_errors:
            if prev_ws and prev_ws in ws_errors:
                change = ws_errors[ws]['mean'] - ws_errors[prev_ws]['mean']
                pct = (change / ws_errors[prev_ws]['mean']) * 100
                direction = "↑" if change > 0 else "↓"
                print(f"  WS{prev_ws} → WS{ws}: {direction} {abs(change):.4f} px ({pct:+.1f}%)")
            prev_ws = ws

    return {
        'init_error': init_error,
        'final_error': final_error,
        'cpp_landmarks': cpp_landmarks,
        'py_landmarks': landmarks,
        'info': info
    }


def find_test_image():
    """Find a suitable test image."""
    candidates = [
        '/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_frame_0030.jpg',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/comparison_frame_0150.jpg',
        '/Users/johnwilsoniv/Documents/SplitFace Open3/debug_face_positions.jpg',
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Search for any jpg
    import glob
    jpgs = glob.glob('/Users/johnwilsoniv/Documents/SplitFace Open3/**/*.jpg', recursive=True)
    for jpg in jpgs:
        if 'archive' not in jpg and os.path.getsize(jpg) > 50000:
            return jpg

    return None


if __name__ == '__main__':
    # Find test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_test_image()

    if image_path is None:
        print("ERROR: No test image found")
        sys.exit(1)

    print(f"Using test image: {image_path}")
    run_accuracy_test(image_path)
