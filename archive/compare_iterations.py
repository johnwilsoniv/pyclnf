#!/usr/bin/env python3
"""
Compare pyCLNF vs C++ OpenFace iteration-by-iteration.

This script:
1. Runs Python CLNF with detailed iteration logging
2. Parses C++ debug output
3. Compares parameters and landmarks at each iteration
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf import CLNF
from pyclnf.core.pdm import PDM
from pymtcnn import MTCNN


def run_python_clnf(image_path):
    """Run Python CLNF with detailed iteration tracking."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return None, None, None

    # Detect face with MTCNN
    mtcnn = MTCNN(backend='coreml')
    bboxes, landmarks_5pt = mtcnn.detect(img)

    if len(bboxes) == 0:
        print("No faces detected")
        return None, None, None

    bbox = tuple(bboxes[0][:4])
    lm5 = landmarks_5pt[0] if landmarks_5pt is not None else None

    print("=" * 70)
    print("PYTHON CLNF ITERATION TRACKING")
    print("=" * 70)
    print(f"\nInput: {image_path}")
    print(f"MTCNN bbox: x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}")

    # Run CLNF with debug mode for iteration tracking
    clnf = CLNF(debug_mode=True, tracked_landmarks=[36, 42, 30, 8, 2, 6])

    # Run fit with detailed return
    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5, return_params=True)

    print(f"\n{'='*70}")
    print("PYTHON FINAL RESULTS")
    print("=" * 70)
    print(f"Converged: {info['converged']}")
    print(f"Total iterations: {info['iterations']}")

    final_params = info['params']
    print(f"\nFinal global parameters:")
    print(f"  scale:  {final_params[0]:.6f}")
    print(f"  rot_x:  {final_params[1]:.6f}")
    print(f"  rot_y:  {final_params[2]:.6f}")
    print(f"  rot_z:  {final_params[3]:.6f}")
    print(f"  tx:     {final_params[4]:.6f}")
    print(f"  ty:     {final_params[5]:.6f}")

    print(f"\nFinal local parameters (first 10):")
    print(f"  {final_params[6:16]}")

    # Print iteration history
    if 'iteration_history' in info:
        print(f"\n{'='*70}")
        print("ITERATION HISTORY")
        print("=" * 70)

        for iter_info in info['iteration_history']:
            phase = iter_info['phase']
            ws = iter_info['window_size']
            it = iter_info['iteration']
            update_mag = iter_info['update_magnitude']
            ms_norm = iter_info['mean_shift_norm']
            ms_mean = iter_info['mean_shift_mean']

            params = iter_info['params']

            print(f"\n[{phase.upper()}] Window {ws}, Iteration {it}:")
            print(f"  Update magnitude: {update_mag:.6f}")
            print(f"  Mean-shift norm: {ms_norm:.4f}, mean: {ms_mean:.4f}")
            print(f"  Global params: scale={params[0]:.6f}, rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
            print(f"  Translation: tx={params[4]:.6f}, ty={params[5]:.6f}")
            print(f"  Local params (first 5): {params[6:11]}")

    return landmarks, info, final_params


def parse_cpp_debug_output(debug_file_path):
    """Parse C++ OpenFace debug output for iteration comparison."""
    if not os.path.exists(debug_file_path):
        print(f"C++ debug file not found: {debug_file_path}")
        return None

    iterations = []
    current_iter = {}

    with open(debug_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse iteration markers
            if 'RIGID iter' in line or 'NONRIGID iter' in line:
                if current_iter:
                    iterations.append(current_iter)
                current_iter = {'phase': 'rigid' if 'RIGID' in line else 'nonrigid'}

                # Extract iteration number
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if 'iter' in p.lower() and i + 1 < len(parts):
                            current_iter['iteration'] = int(parts[i + 1].replace(':', ''))
                except (ValueError, IndexError):
                    pass

            # Parse parameters
            if 'scale=' in line:
                try:
                    current_iter['scale'] = float(line.split('scale=')[1].split()[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass

            if 'rot_x=' in line or 'wx=' in line:
                try:
                    if 'rot_x=' in line:
                        current_iter['rot_x'] = float(line.split('rot_x=')[1].split()[0].replace(',', ''))
                    elif 'wx=' in line:
                        current_iter['rot_x'] = float(line.split('wx=')[1].split()[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass

    if current_iter:
        iterations.append(current_iter)

    return iterations


def run_cpp_openface(image_path):
    """Run C++ OpenFace and capture output."""
    import subprocess
    import tempfile

    cpp_exe = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [cpp_exe, '-f', image_path, '-out_dir', tmpdir, '-2Dfp']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Get landmarks from CSV
        csv_files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(tmpdir, csv_files[0])
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            if len(lines) > 1:
                header = lines[0].strip().split(',')
                values = lines[1].strip().split(',')

                x_cols = [i for i, h in enumerate(header) if h.strip().startswith('x_')]
                y_cols = [i for i, h in enumerate(header) if h.strip().startswith('y_')]

                landmarks = np.zeros((len(x_cols), 2))
                for i, (xi, yi) in enumerate(zip(x_cols, y_cols)):
                    landmarks[i, 0] = float(values[xi])
                    landmarks[i, 1] = float(values[yi])

                return landmarks, result.stderr
    return None, None


def compare_results(py_landmarks, cpp_landmarks, py_params):
    """Compare Python and C++ results."""
    print("\n" + "=" * 70)
    print("COMPARISON: PYTHON vs C++")
    print("=" * 70)

    if cpp_landmarks is None:
        print("C++ landmarks not available for comparison")
        return

    # Per-landmark errors
    errors = np.linalg.norm(py_landmarks - cpp_landmarks, axis=1)

    print(f"\nLandmark Error Statistics:")
    print(f"  Mean error: {np.mean(errors):.4f} px")
    print(f"  Max error:  {np.max(errors):.4f} px")
    print(f"  Min error:  {np.min(errors):.4f} px")
    print(f"  Std error:  {np.std(errors):.4f} px")

    # Per-region analysis
    regions = {
        'Jawline (0-16)': range(0, 17),
        'Left eyebrow (17-21)': range(17, 22),
        'Right eyebrow (22-26)': range(22, 27),
        'Nose (27-35)': range(27, 36),
        'Left eye (36-41)': range(36, 42),
        'Right eye (42-47)': range(42, 48),
        'Outer mouth (48-59)': range(48, 60),
        'Inner mouth (60-67)': range(60, 68),
    }

    print(f"\nPer-Region Error Analysis:")
    for region_name, indices in regions.items():
        region_errors = errors[list(indices)]
        print(f"  {region_name}: mean={np.mean(region_errors):.4f}, max={np.max(region_errors):.4f}")

    # Show worst landmarks
    print(f"\nTop 10 Worst Landmarks:")
    worst_indices = np.argsort(errors)[-10:][::-1]
    for idx in worst_indices:
        py_pos = py_landmarks[idx]
        cpp_pos = cpp_landmarks[idx]
        print(f"  LM{idx:2d}: error={errors[idx]:.2f}px | "
              f"PY=({py_pos[0]:.1f}, {py_pos[1]:.1f}) | "
              f"C++=({cpp_pos[0]:.1f}, {cpp_pos[1]:.1f}) | "
              f"diff=({py_pos[0]-cpp_pos[0]:+.2f}, {py_pos[1]-cpp_pos[1]:+.2f})")

    # Y-axis analysis (common source of error)
    y_errors = py_landmarks[:, 1] - cpp_landmarks[:, 1]
    print(f"\nY-Axis Error Analysis (Python - C++):")
    print(f"  Mean Y error: {np.mean(y_errors):+.4f} px (positive = Python below C++)")
    print(f"  Max Y error:  {np.max(y_errors):+.4f} px")
    print(f"  Min Y error:  {np.min(y_errors):+.4f} px")

    # Check if jawline has systematic bias
    jawline_y_errors = y_errors[:17]
    print(f"\n  Jawline Y errors: {jawline_y_errors}")
    print(f"  Jawline mean Y error: {np.mean(jawline_y_errors):+.4f} px")


def visualize_comparison(image_path, py_landmarks, cpp_landmarks, output_path=None):
    """Visualize Python vs C++ landmarks side by side."""
    img = cv2.imread(image_path)
    if img is None:
        return

    vis = img.copy()

    # Draw C++ landmarks in blue
    if cpp_landmarks is not None:
        for i, (x, y) in enumerate(cpp_landmarks):
            cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)
            if i in [0, 8, 16, 36, 42, 30, 48]:  # Key landmarks
                cv2.putText(vis, f"{i}", (int(x)+5, int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # Draw Python landmarks in green
    if py_landmarks is not None:
        for i, (x, y) in enumerate(py_landmarks):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Draw lines between corresponding landmarks for error visualization
    if py_landmarks is not None and cpp_landmarks is not None:
        for i in range(len(py_landmarks)):
            py_pt = (int(py_landmarks[i, 0]), int(py_landmarks[i, 1]))
            cpp_pt = (int(cpp_landmarks[i, 0]), int(cpp_landmarks[i, 1]))
            cv2.line(vis, py_pt, cpp_pt, (0, 0, 255), 1)

    # Add legend
    cv2.putText(vis, "Blue: C++", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(vis, "Green: Python", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "Red lines: Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"\nVisualization saved to: {output_path}")
    else:
        cv2.imwrite('/tmp/py_cpp_comparison.png', vis)
        print(f"\nVisualization saved to: /tmp/py_cpp_comparison.png")

    return vis


def main():
    """Main comparison routine."""
    image_path = '/tmp/test_face_clean.png'

    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        print("Creating a test image from the face directory...")

        # Try to find a test image
        test_dirs = [
            '/Users/johnwilsoniv/Documents/SplitFace Open3/test_images',
            '/Users/johnwilsoniv/repo/fea_tool/test_images',
        ]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    image_path = os.path.join(test_dir, images[0])
                    print(f"Using: {image_path}")
                    break
        else:
            print("No test images found. Please provide an image path.")
            return

    # Run Python CLNF
    py_landmarks, py_info, py_params = run_python_clnf(image_path)

    if py_landmarks is None:
        print("Python CLNF failed")
        return

    # Run C++ OpenFace
    print("\n" + "=" * 70)
    print("RUNNING C++ OPENFACE")
    print("=" * 70)
    cpp_landmarks, cpp_stderr = run_cpp_openface(image_path)

    if cpp_landmarks is not None:
        print(f"C++ landmarks extracted: {cpp_landmarks.shape}")
    else:
        print("C++ extraction failed")

    # Compare results
    compare_results(py_landmarks, cpp_landmarks, py_params)

    # Visualize
    visualize_comparison(image_path, py_landmarks, cpp_landmarks)


if __name__ == '__main__':
    main()
