#!/usr/bin/env python3
"""
Trace the origin of state divergence between Python and C++.

This script systematically compares Python vs C++ state at:
1. After initialization (before any optimization)
2. After each window size (WS11, WS9, WS7, WS5)
3. After each phase (RIGID, NONRIGID) within each window size
4. Per-iteration within WS11 to find exact divergence point

Uses C++ reference landmarks from CSV output.
"""

import sys
import os
import numpy as np
import cv2
import subprocess
import pandas as pd
from pathlib import Path

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')


def get_cpp_landmarks(image_path: str) -> np.ndarray:
    """Run C++ OpenFace and extract final landmarks."""
    out_dir = '/tmp/cpp_divergence_trace'
    os.makedirs(out_dir, exist_ok=True)

    # Clean old files
    for f in Path(out_dir).glob('*'):
        f.unlink()

    result = subprocess.run([
        '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction',
        '-f', image_path,
        '-out_dir', out_dir,
        '-2Dfp'
    ], capture_output=True, timeout=60)

    csv_name = Path(image_path).stem + '.csv'
    csv_path = os.path.join(out_dir, csv_name)
    df = pd.read_csv(csv_path)

    landmarks = np.zeros((68, 2))
    for i in range(68):
        landmarks[i, 0] = df[f'x_{i}'].values[0]
        landmarks[i, 1] = df[f'y_{i}'].values[0]

    return landmarks


def run_python_with_full_trace(image_path: str, cpp_landmarks: np.ndarray):
    """Run Python CLNF with comprehensive state tracing."""
    from pyclnf import CLNF
    from pymtcnn import MTCNN

    # Load image and detect face
    img = cv2.imread(image_path)
    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None

    print("=" * 80)
    print("DIVERGENCE TRACE: Python vs C++")
    print("=" * 80)

    # Initialize CLNF
    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    # ===== CHECKPOINT 0: Initial state from bbox =====
    print("\n" + "=" * 80)
    print("CHECKPOINT 0: Initial State (from bbox)")
    print("=" * 80)

    initial_params = pdm.init_params(bbox)
    initial_landmarks = pdm.params_to_landmarks_2d(initial_params)

    print(f"\nInitial params:")
    print(f"  scale:  {initial_params[0]:.8f}")
    print(f"  rot_x:  {initial_params[1]:.8f}")
    print(f"  rot_y:  {initial_params[2]:.8f}")
    print(f"  rot_z:  {initial_params[3]:.8f}")
    print(f"  tx:     {initial_params[4]:.8f}")
    print(f"  ty:     {initial_params[5]:.8f}")

    init_error = np.linalg.norm(initial_landmarks - cpp_landmarks, axis=1)
    print(f"\nInitial vs C++ final: mean={init_error.mean():.4f} px, max={init_error.max():.4f} px")
    print(f"  LM36: Py=({initial_landmarks[36,0]:.2f}, {initial_landmarks[36,1]:.2f}) "
          f"C++_final=({cpp_landmarks[36,0]:.2f}, {cpp_landmarks[36,1]:.2f})")

    # ===== Run fit with iteration history =====
    print("\n" + "=" * 80)
    print("Running optimization with full iteration history...")
    print("=" * 80)

    landmarks, info = clnf.fit(img, bbox, landmarks_5pt=lm5_arr)

    # ===== Analyze iteration history =====
    iteration_history = info.get('iteration_history', [])

    # Group by window size and phase
    ws_phase_states = {}
    for iter_info in iteration_history:
        ws = iter_info['window_size']
        phase = iter_info['phase']
        key = (ws, phase)
        ws_phase_states[key] = {
            'params': iter_info['params'].copy(),
            'iteration': iter_info['iteration'],
        }

    # ===== CHECKPOINT 1-4: After each window size =====
    prev_error = init_error.mean()

    for ws in [11, 9, 7, 5]:
        print(f"\n{'='*80}")
        print(f"CHECKPOINT: After WS{ws}")
        print("=" * 80)

        # Get state after RIGID phase
        rigid_key = (ws, 'rigid')
        if rigid_key in ws_phase_states:
            rigid_state = ws_phase_states[rigid_key]
            rigid_params = rigid_state['params']
            rigid_landmarks = pdm.params_to_landmarks_2d(rigid_params)
            rigid_error = np.linalg.norm(rigid_landmarks - cpp_landmarks, axis=1)

            print(f"\n  After RIGID phase:")
            print(f"    scale:  {rigid_params[0]:.8f}")
            print(f"    rot:    ({rigid_params[1]:.8f}, {rigid_params[2]:.8f}, {rigid_params[3]:.8f})")
            print(f"    tx,ty:  ({rigid_params[4]:.4f}, {rigid_params[5]:.4f})")
            print(f"    Error vs C++: mean={rigid_error.mean():.6f} px")

        # Get state after NONRIGID phase
        nonrigid_key = (ws, 'nonrigid')
        if nonrigid_key in ws_phase_states:
            nonrigid_state = ws_phase_states[nonrigid_key]
            nonrigid_params = nonrigid_state['params']
            nonrigid_landmarks = pdm.params_to_landmarks_2d(nonrigid_params)
            nonrigid_error = np.linalg.norm(nonrigid_landmarks - cpp_landmarks, axis=1)

            print(f"\n  After NONRIGID phase:")
            print(f"    scale:  {nonrigid_params[0]:.8f}")
            print(f"    rot:    ({nonrigid_params[1]:.8f}, {nonrigid_params[2]:.8f}, {nonrigid_params[3]:.8f})")
            print(f"    tx,ty:  ({nonrigid_params[4]:.4f}, {nonrigid_params[5]:.4f})")
            print(f"    Local params[0:5]: {nonrigid_params[6:11]}")
            print(f"    Error vs C++: mean={nonrigid_error.mean():.6f} px")

            delta_error = nonrigid_error.mean() - prev_error
            print(f"    Change from prev WS: {delta_error:+.6f} px")
            prev_error = nonrigid_error.mean()

    # ===== DETAILED WS11 ITERATION TRACE =====
    print("\n" + "=" * 80)
    print("DETAILED TRACE: WS11 Per-Iteration")
    print("=" * 80)

    ws11_iters = [h for h in iteration_history if h['window_size'] == 11]

    print(f"\nWS11 has {len(ws11_iters)} iterations")
    print("\n  Iter | Phase    | Scale     | Rot_X      | Rot_Y      | Rot_Z      | TX       | TY")
    print("  " + "-" * 95)

    for iter_info in ws11_iters:
        params = iter_info['params']
        phase = iter_info['phase']
        iter_num = iter_info['iteration']

        print(f"  {iter_num:4d} | {phase:8s} | {params[0]:.6f} | {params[1]:+.8f} | "
              f"{params[2]:+.8f} | {params[3]:+.8f} | {params[4]:.2f} | {params[5]:.2f}")

    # ===== Compare update magnitudes =====
    print("\n" + "=" * 80)
    print("UPDATE MAGNITUDE ANALYSIS")
    print("=" * 80)

    print("\n  WS   | Phase    | Avg Update Mag | Mean-Shift Norm")
    print("  " + "-" * 55)

    for ws in [11, 9, 7, 5]:
        for phase in ['rigid', 'nonrigid']:
            ws_phase_iters = [h for h in iteration_history
                            if h['window_size'] == ws and h['phase'] == phase]
            if ws_phase_iters:
                avg_update = np.mean([h['update_magnitude'] for h in ws_phase_iters])
                avg_ms = np.mean([h['mean_shift_norm'] for h in ws_phase_iters])
                print(f"  WS{ws:<2} | {phase:8s} | {avg_update:14.6f} | {avg_ms:.6f}")

    # ===== Final comparison =====
    print("\n" + "=" * 80)
    print("FINAL STATE COMPARISON")
    print("=" * 80)

    final_error = np.linalg.norm(landmarks - cpp_landmarks, axis=1)
    print(f"\nFinal error vs C++: mean={final_error.mean():.6f} px, max={final_error.max():.6f} px")

    # Per-landmark error distribution
    print("\nPer-landmark error distribution:")
    print("  Region      | Landmarks        | Mean Error")
    print("  " + "-" * 45)

    regions = {
        'Jaw':        range(0, 17),
        'L.Eyebrow':  range(17, 22),
        'R.Eyebrow':  range(22, 27),
        'Nose':       range(27, 36),
        'L.Eye':      range(36, 42),
        'R.Eye':      range(42, 48),
        'Mouth':      range(48, 68),
    }

    for region, indices in regions.items():
        region_error = final_error[list(indices)].mean()
        print(f"  {region:12s} | {indices.start:2d}-{indices.stop-1:2d}             | {region_error:.6f} px")

    # ===== Key landmarks comparison =====
    print("\n" + "=" * 80)
    print("KEY LANDMARKS: Python vs C++ Final")
    print("=" * 80)

    key_landmarks = [36, 39, 42, 45, 30, 48, 54, 8]  # Eyes, nose tip, mouth corners, chin

    print("\n  LM  | Python X  | Python Y  | C++ X     | C++ Y     | Error")
    print("  " + "-" * 65)

    for lm_idx in key_landmarks:
        py_x, py_y = landmarks[lm_idx]
        cpp_x, cpp_y = cpp_landmarks[lm_idx]
        err = np.sqrt((py_x - cpp_x)**2 + (py_y - cpp_y)**2)
        print(f"  {lm_idx:3d} | {py_x:9.4f} | {py_y:9.4f} | {cpp_x:9.4f} | {cpp_y:9.4f} | {err:.6f}")

    return iteration_history, landmarks, info


def analyze_first_divergence(iteration_history, cpp_landmarks, pdm):
    """Find the first iteration where significant divergence occurs."""
    print("\n" + "=" * 80)
    print("FIRST DIVERGENCE ANALYSIS")
    print("=" * 80)

    prev_error = None
    first_jump = None

    for iter_info in iteration_history:
        params = iter_info['params']
        landmarks = pdm.params_to_landmarks_2d(params)
        error = np.linalg.norm(landmarks - cpp_landmarks, axis=1).mean()

        if prev_error is not None:
            delta = error - prev_error
            if abs(delta) > 0.01 and first_jump is None:  # >0.01 px jump
                first_jump = iter_info
                print(f"\nFIRST SIGNIFICANT JUMP at iteration {iter_info['iteration']}:")
                print(f"  Window Size: {iter_info['window_size']}")
                print(f"  Phase: {iter_info['phase']}")
                print(f"  Error before: {prev_error:.6f} px")
                print(f"  Error after:  {error:.6f} px")
                print(f"  Delta: {delta:+.6f} px")

        prev_error = error

    if first_jump is None:
        print("\nNo significant single-iteration jump found (all deltas < 0.01 px)")
        print("Divergence is gradual accumulation across iterations.")


def main():
    print("=" * 80)
    print("STATE DIVERGENCE TRACER")
    print("=" * 80)

    # Find test image
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    if not os.path.exists(img_path):
        # Try alternate locations
        for alt in ['comparison_frame_0030.png', 'test_frame.jpg']:
            alt_path = f'{BASE_DIR}/{alt}'
            if os.path.exists(alt_path):
                img_path = alt_path
                break

    if not os.path.exists(img_path):
        print(f"Error: Test image not found")
        return

    print(f"\nTest image: {img_path}")

    # Get C++ reference (final landmarks)
    print("\n[1] Running C++ OpenFace for reference...")
    cpp_landmarks = get_cpp_landmarks(img_path)
    print(f"    C++ final landmarks loaded.")

    # Run Python with full trace
    print("\n[2] Running Python CLNF with full trace...")
    iteration_history, landmarks, info = run_python_with_full_trace(img_path, cpp_landmarks)

    # Analyze where divergence first appears
    from pyclnf import CLNF
    clnf = CLNF(detector=None)
    analyze_first_divergence(iteration_history, cpp_landmarks, clnf.pdm)

    print("\n" + "=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compare these values with C++ debug dumps at same checkpoints")
    print("2. Look for first significant parameter divergence")
    print("3. Focus debugging on that specific iteration/component")


if __name__ == '__main__':
    main()
