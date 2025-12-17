#!/usr/bin/env python3
"""
Compare pyCLNF initialization parameters with C++ OpenFace.

This script:
1. Runs MTCNN detection (should match C++ exactly)
2. Computes init params with bbox correction
3. Compares with C++ debug output
"""

import cv2
import numpy as np
import sys
import subprocess
import tempfile
import os

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf import CLNF
from pyclnf.core.pdm import PDM
from pymtcnn import MTCNN


def get_cpp_init_params(image_path):
    """
    Run C++ OpenFace and extract initialization parameters.

    Need to add debug output to C++ to get these values.
    For now, parse from existing debug files.
    """
    # Check if debug file exists from previous run
    debug_file = '/tmp/cpp_init_landmarks_68.txt'
    if os.path.exists(debug_file):
        params = {}
        with open(debug_file, 'r') as f:
            content = f.read()

        # Parse params
        for line in content.split('\n'):
            if 'params[0] (scale):' in line:
                params['scale'] = float(line.split(':')[1].strip())
            elif 'params[1] (rot_x):' in line:
                params['rot_x'] = float(line.split(':')[1].strip())
            elif 'params[2] (rot_y):' in line:
                params['rot_y'] = float(line.split(':')[1].strip())
            elif 'params[3] (rot_z):' in line:
                params['rot_z'] = float(line.split(':')[1].strip())
            elif 'params[4] (trans_x):' in line:
                params['tx'] = float(line.split(':')[1].strip())
            elif 'params[5] (trans_y):' in line:
                params['ty'] = float(line.split(':')[1].strip())

        return params
    return None


def run_cpp_feature_extraction(image_path):
    """Run C++ OpenFace and get final landmarks."""
    cpp_exe = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [cpp_exe, '-f', image_path, '-out_dir', tmpdir, '-2Dfp']
        result = subprocess.run(cmd, capture_output=True, text=True)

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

                return landmarks
    return None


def compare_init():
    """Compare initialization parameters."""
    image_path = '/tmp/test_face_clean.png'
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Cannot load {image_path}")
        return

    print("=" * 60)
    print("INITIALIZATION PARAMETER COMPARISON")
    print("=" * 60)

    # Step 1: MTCNN Detection
    print("\n1. MTCNN Detection")
    print("-" * 40)

    mtcnn = MTCNN(backend='coreml')
    bboxes, landmarks_5pt = mtcnn.detect(img)

    if len(bboxes) == 0:
        print("No faces detected")
        return

    bbox = tuple(bboxes[0][:4])
    lm5 = landmarks_5pt[0] if landmarks_5pt is not None else None

    print(f"Raw MTCNN bbox: x={bbox[0]:.4f}, y={bbox[1]:.4f}, w={bbox[2]:.4f}, h={bbox[3]:.4f}")

    # Step 2: Apply bbox correction (like C++)
    print("\n2. Bbox Correction")
    print("-" * 40)

    pdm = PDM('/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/pyclnf/models/exported_pdm')
    corrected_bbox = pdm._apply_mtcnn_bbox_preprocessing(bbox)

    print(f"Corrected bbox: x={corrected_bbox[0]:.4f}, y={corrected_bbox[1]:.4f}, w={corrected_bbox[2]:.4f}, h={corrected_bbox[3]:.4f}")
    print(f"  y offset applied: {corrected_bbox[1] - bbox[1]:.2f} px")
    print(f"  h change: {corrected_bbox[3] - bbox[3]:.2f} px")

    # Step 3: Initialize params
    print("\n3. Parameter Initialization")
    print("-" * 40)

    # Using just bbox (no 5pt landmarks)
    params_bbox = pdm.init_params(bbox)
    print(f"\nPython init_params (with bbox correction):")
    print(f"  scale:  {params_bbox[0]:.6f}")
    print(f"  rot_x:  {params_bbox[1]:.6f}")
    print(f"  rot_y:  {params_bbox[2]:.6f}")
    print(f"  rot_z:  {params_bbox[3]:.6f}")
    print(f"  tx:     {params_bbox[4]:.6f}")
    print(f"  ty:     {params_bbox[5]:.6f}")
    print(f"  local params: all zeros (mean shape)")

    # Using 5pt landmarks
    if lm5 is not None:
        params_5pt = pdm.init_params_from_5pt(bbox, lm5)
        print(f"\nPython init_params_from_5pt:")
        print(f"  scale:  {params_5pt[0]:.6f}")
        print(f"  rot_x:  {params_5pt[1]:.6f}")
        print(f"  rot_y:  {params_5pt[2]:.6f}")
        print(f"  rot_z:  {params_5pt[3]:.6f}")
        print(f"  tx:     {params_5pt[4]:.6f}")
        print(f"  ty:     {params_5pt[5]:.6f}")

    # Step 4: Get C++ init params (if available)
    print("\n4. C++ Comparison")
    print("-" * 40)

    cpp_params = get_cpp_init_params(image_path)
    if cpp_params:
        print(f"C++ init params (from debug file):")
        print(f"  scale:  {cpp_params.get('scale', 'N/A')}")
        print(f"  rot_x:  {cpp_params.get('rot_x', 'N/A')}")
        print(f"  rot_y:  {cpp_params.get('rot_y', 'N/A')}")
        print(f"  rot_z:  {cpp_params.get('rot_z', 'N/A')}")
        print(f"  tx:     {cpp_params.get('tx', 'N/A')}")
        print(f"  ty:     {cpp_params.get('ty', 'N/A')}")

        # Compare
        print(f"\nDifferences (Python - C++):")
        if 'scale' in cpp_params:
            print(f"  scale:  {params_bbox[0] - cpp_params['scale']:+.6f}")
        if 'rot_x' in cpp_params:
            print(f"  rot_x:  {params_bbox[1] - cpp_params['rot_x']:+.6f}")
        if 'rot_y' in cpp_params:
            print(f"  rot_y:  {params_bbox[2] - cpp_params['rot_y']:+.6f}")
        if 'rot_z' in cpp_params:
            print(f"  rot_z:  {params_bbox[3] - cpp_params['rot_z']:+.6f}")
        if 'tx' in cpp_params:
            print(f"  tx:     {params_bbox[4] - cpp_params['tx']:+.6f}")
        if 'ty' in cpp_params:
            print(f"  ty:     {params_bbox[5] - cpp_params['ty']:+.6f}")
    else:
        print("C++ debug file not found. Run C++ with debug output first.")

    # Step 5: Get initial landmarks
    print("\n5. Initial Landmarks")
    print("-" * 40)

    init_landmarks = pdm.params_to_landmarks_2d(params_bbox)
    print(f"Initial landmark positions (first 10):")
    for i in range(10):
        print(f"  LM{i}: ({init_landmarks[i, 0]:.2f}, {init_landmarks[i, 1]:.2f})")

    # Step 6: Run full CLNF and compare final result
    print("\n6. Final Landmarks Comparison")
    print("-" * 40)

    clnf = CLNF(debug_mode=False)
    py_landmarks, info = clnf.fit(img, bbox, return_params=True)

    print(f"Python CLNF converged: {info['converged']}")
    print(f"Python iterations: {info['iterations']}")

    cpp_landmarks = run_cpp_feature_extraction(image_path)

    if cpp_landmarks is not None:
        errors = np.linalg.norm(py_landmarks - cpp_landmarks, axis=1)
        print(f"\nFinal error vs C++:")
        print(f"  Mean: {np.mean(errors):.4f} px")
        print(f"  Max:  {np.max(errors):.4f} px")
        print(f"  Min:  {np.min(errors):.4f} px")

        # Show worst landmarks
        print(f"\nTop 5 worst landmarks:")
        worst = np.argsort(errors)[-5:][::-1]
        for idx in worst:
            print(f"  LM{idx}: error={errors[idx]:.2f}px, py=({py_landmarks[idx, 0]:.1f}, {py_landmarks[idx, 1]:.1f}), cpp=({cpp_landmarks[idx, 0]:.1f}, {cpp_landmarks[idx, 1]:.1f})")


if __name__ == '__main__':
    compare_init()
