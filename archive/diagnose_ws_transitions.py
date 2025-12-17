#!/usr/bin/env python3
"""
Diagnose state at window size transitions.

Compares Python state after each window size (WS11, WS9, WS7, WS5)
to find where divergence from C++ begins.
"""

import sys
import os
import numpy as np
import cv2
import subprocess
import pandas as pd

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')


def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Run C++ OpenFace and extract landmarks."""
    out_dir = '/tmp/cpp_ws_diag'
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run([
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir,
        '-2Dfp'
    ], capture_output=True, timeout=60)

    from pathlib import Path
    csv_name = Path(image_path).stem + '.csv'
    csv_path = os.path.join(out_dir, csv_name)
    df = pd.read_csv(csv_path)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].values[0]
        landmarks[i, 1] = df[f'y_{i}'].values[0]

    return landmarks


def run_python_with_state_tracking(image_path: str, cpp_landmarks: np.ndarray):
    """Run Python CLNF and track state at each WS transition."""
    from pyclnf import CLNF
    from pymtcnn import MTCNN

    # Load image and detect face
    img = cv2.imread(image_path)
    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None

    # Initialize CLNF
    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    # Run fit
    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5_arr)

    # Analyze state at each WS transition from iteration history
    print("\n" + "=" * 80)
    print("STATE AT WINDOW SIZE TRANSITIONS")
    print("=" * 80)

    # Track last iteration of each window size
    ws_states = {}
    for iter_info in info.get('iteration_history', []):
        ws = iter_info['window_size']
        ws_states[ws] = {
            'params': iter_info['params'].copy(),
            'phase': iter_info['phase'],
            'iteration': iter_info['iteration'],
        }

    # Print state after each WS
    prev_params = None
    for ws in [11, 9, 7, 5]:
        if ws not in ws_states:
            continue

        state = ws_states[ws]
        params = state['params']

        # Compute landmarks
        landmarks_at_ws = pdm.params_to_landmarks_2d(params)
        error = np.linalg.norm(landmarks_at_ws - cpp_landmarks, axis=1)

        print(f"\n{'='*80}")
        print(f"AFTER WS{ws} ({state['phase'].upper()} phase)")
        print(f"{'='*80}")

        print(f"\nGlobal params:")
        print(f"  scale:    {params[0]:.8f}")
        print(f"  rot_x:    {params[1]:.8f} rad ({np.degrees(params[1]):.4f}°)")
        print(f"  rot_y:    {params[2]:.8f} rad ({np.degrees(params[2]):.4f}°)")
        print(f"  rot_z:    {params[3]:.8f} rad ({np.degrees(params[3]):.4f}°)")
        print(f"  tx:       {params[4]:.4f}")
        print(f"  ty:       {params[5]:.4f}")

        print(f"\nLocal params (first 5): {params[6:11]}")

        print(f"\nSample landmarks vs C++:")
        for lm_idx in [36, 48, 8, 30]:
            py_lm = landmarks_at_ws[lm_idx]
            cpp_lm = cpp_landmarks[lm_idx]
            lm_err = np.linalg.norm(py_lm - cpp_lm)
            print(f"  LM{lm_idx}: Py=({py_lm[0]:.2f}, {py_lm[1]:.2f}) "
                  f"C++=({cpp_lm[0]:.2f}, {cpp_lm[1]:.2f}) err={lm_err:.4f} px")

        print(f"\nError stats: mean={error.mean():.4f} px, max={error.max():.4f} px")

        # Param changes from previous WS
        if prev_params is not None:
            delta = params[:6] - prev_params[:6]
            print(f"\nParam changes from previous WS:")
            print(f"  Δscale: {delta[0]:+.8f}")
            print(f"  Δrot_x: {delta[1]:+.8f} rad ({np.degrees(delta[1]):+.4f}°)")
            print(f"  Δrot_y: {delta[2]:+.8f} rad ({np.degrees(delta[2]):+.4f}°)")
            print(f"  Δrot_z: {delta[3]:+.8f} rad ({np.degrees(delta[3]):+.4f}°)")
            print(f"  Δtx:    {delta[4]:+.4f}")
            print(f"  Δty:    {delta[5]:+.4f}")

        prev_params = params.copy()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Error progression")
    print("=" * 80)
    print("\n| WS   | Mean Error | Max Error  | Δ Mean from prev |")
    print("|------|------------|------------|------------------|")

    prev_mean_err = None
    for ws in [11, 9, 7, 5]:
        if ws not in ws_states:
            continue
        landmarks_at_ws = pdm.params_to_landmarks_2d(ws_states[ws]['params'])
        error = np.linalg.norm(landmarks_at_ws - cpp_landmarks, axis=1)
        mean_err = error.mean()

        if prev_mean_err is not None:
            delta_str = f"{mean_err - prev_mean_err:+.4f}"
        else:
            delta_str = "-"

        print(f"| WS{ws:<2} | {mean_err:>10.4f} | {error.max():>10.4f} | {delta_str:>16} |")
        prev_mean_err = mean_err

    return ws_states, landmarks, info


def compute_jacobian_comparison(params, pdm, cpp_landmarks):
    """Compute and analyze Jacobian at given state."""
    J = pdm.compute_jacobian_rigid(params)

    print(f"\nJacobian analysis (shape {J.shape}):")
    print(f"  Column norms: scale={np.linalg.norm(J[:,0]):.2f}, "
          f"rot_x={np.linalg.norm(J[:,1]):.2f}, "
          f"rot_y={np.linalg.norm(J[:,2]):.2f}, "
          f"rot_z={np.linalg.norm(J[:,3]):.2f}, "
          f"tx={np.linalg.norm(J[:,4]):.2f}, "
          f"ty={np.linalg.norm(J[:,5]):.2f}")

    # Sample rows
    print(f"\n  J[36, :] (LM36 x): {J[36, :]}")
    print(f"  J[36+68, :] (LM36 y): {J[36+68, :]}")

    return J


def main():
    print("=" * 80)
    print("WINDOW SIZE TRANSITION STATE DIAGNOSTIC")
    print("=" * 80)

    # Find test image
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    if not os.path.exists(img_path):
        print(f"Error: Test image not found: {img_path}")
        return

    print(f"\nTest image: {img_path}")

    # Get C++ reference
    print("\n[1] Running C++ OpenFace for reference...")
    cpp_landmarks = get_cpp_landmarks(img_path)
    print(f"    C++ landmarks loaded. LM36: ({cpp_landmarks[36,0]:.2f}, {cpp_landmarks[36,1]:.2f})")

    # Run Python with tracking
    print("\n[2] Running Python CLNF with state tracking...")
    ws_states, landmarks, info = run_python_with_state_tracking(img_path, cpp_landmarks)

    # Jacobian analysis at WS7 entry
    print("\n" + "=" * 80)
    print("JACOBIAN ANALYSIS AT WS7 ENTRY")
    print("=" * 80)

    from pyclnf import CLNF
    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    if 9 in ws_states:
        # Use state after WS9 (which is WS7 entry)
        params_at_ws7_entry = ws_states[9]['params']
        print(f"\nUsing state after WS9 (= WS7 entry state)")
        J = compute_jacobian_comparison(params_at_ws7_entry, pdm, cpp_landmarks)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compare these params with C++ dumps at same points")
    print("2. Look for where state first diverges significantly")
    print("3. Check if Jacobian values match C++ at WS7 entry")


if __name__ == '__main__':
    main()
