#!/usr/bin/env python3
"""
Compare C++ WS7 RIGID iter 0 dump with Python implementation.
Finds the exact point of divergence.
"""

import numpy as np
import cv2
import sys
import re

sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn')

from pyclnf import CLNF
from pyclnf.core.utils import align_shapes_with_scale, invert_similarity_transform


def parse_cpp_dump(filepath: str) -> dict:
    """Parse the C++ WS7 RIGID iter 0 dump file."""
    data = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse current_global
    match = re.search(r'current_global: \[([\d\.\-e,\s]+)\]', content)
    if match:
        data['current_global'] = np.array([float(x.strip()) for x in match.group(1).split(',')], dtype=np.float32)

    # Parse sim_img_to_ref
    match = re.search(r'sim_img_to_ref: \[\[([\d\.\-e,\s]+)\], \[([\d\.\-e,\s]+)\]\]', content)
    if match:
        row1 = [float(x.strip()) for x in match.group(1).split(',')]
        row2 = [float(x.strip()) for x in match.group(2).split(',')]
        data['sim_img_to_ref'] = np.array([row1, row2], dtype=np.float32)

    # Parse sim_ref_to_img
    match = re.search(r'sim_ref_to_img: \[\[([\d\.\-e,\s]+)\], \[([\d\.\-e,\s]+)\]\]', content)
    if match:
        row1 = [float(x.strip()) for x in match.group(1).split(',')]
        row2 = [float(x.strip()) for x in match.group(2).split(',')]
        data['sim_ref_to_img'] = np.array([row1, row2], dtype=np.float32)

    # Parse current_shape (68 landmarks)
    data['current_shape'] = np.zeros((68, 2), dtype=np.float32)
    for match in re.finditer(r'current_shape.*\n(?:.*\n)*?.*LM(\d+): \(([\d\.\-e]+), ([\d\.\-e]+)\)', content):
        lm = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        if lm < 68:
            data['current_shape'][lm] = [x, y]

    # Parse base_shape (68 landmarks)
    data['base_shape'] = np.zeros((68, 2), dtype=np.float32)
    base_section = re.search(r'base_shape \(68 landmarks\):\n((?:.*LM\d+:.*\n)+)', content)
    if base_section:
        for match in re.finditer(r'LM(\d+): \(([\d\.\-e]+), ([\d\.\-e]+)\)', base_section.group(1)):
            lm = int(match.group(1))
            x = float(match.group(2))
            y = float(match.group(3))
            if lm < 68:
                data['base_shape'][lm] = [x, y]

    # Parse offsets, dxs, dys
    data['offsets'] = np.zeros((68, 2), dtype=np.float32)
    data['dxs'] = np.zeros(68, dtype=np.float32)
    data['dys'] = np.zeros(68, dtype=np.float32)
    for match in re.finditer(r'LM(\d+): offset=\(([\d\.\-e]+), ([\d\.\-e]+)\) dx=([\d\.\-e]+) dy=([\d\.\-e]+)', content):
        lm = int(match.group(1))
        if lm < 68:
            data['offsets'][lm] = [float(match.group(2)), float(match.group(3))]
            data['dxs'][lm] = float(match.group(4))
            data['dys'][lm] = float(match.group(5))

    # Parse response maps for LM 4, 36, 48
    data['response_maps'] = {}
    for lm in [4, 36, 48]:
        match = re.search(rf'Response LM{lm} \((\d+)x(\d+)\):\n((?:\s+[\d\.\-e\s]+\n)+)', content)
        if match:
            rows = int(match.group(1))
            resp_text = match.group(3)
            resp_lines = [line.strip().split() for line in resp_text.strip().split('\n')]
            resp = np.array([[float(x) for x in line] for line in resp_lines], dtype=np.float32)
            data['response_maps'][lm] = resp

    # Parse sigma
    match = re.search(r'sigma=([\d\.\-e]+)', content)
    if match:
        data['sigma'] = float(match.group(1))

    # Parse mean_shifts (68 landmarks)
    data['mean_shifts'] = np.zeros((68, 2), dtype=np.float32)
    ms_section = re.search(r'=== MEAN-SHIFTS ===\nmean_shifts.*:\n((?:.*LM\d+:.*\n)+)', content)
    if ms_section:
        for match in re.finditer(r'LM(\d+): \(([\d\.\-e]+), ([\d\.\-e]+)\)', ms_section.group(1)):
            lm = int(match.group(1))
            if lm < 68:
                data['mean_shifts'][lm] = [float(match.group(2)), float(match.group(3))]

    # Parse J_w_t_m (gradient)
    match = re.search(r'J_w_t_m \(6\): \[([\d\.\-e,\s]+)\]', content)
    if match:
        data['J_w_t_m'] = np.array([float(x.strip()) for x in match.group(1).split(',')], dtype=np.float32)

    # Parse Hessian (6x6)
    hessian_match = re.search(r'Hessian \(6x6\):\n((?:\s+\[[\d\.\-e,\s]+\]\n)+)', content)
    if hessian_match:
        hess_lines = hessian_match.group(1).strip().split('\n')
        hess_rows = []
        for line in hess_lines:
            line = line.strip()
            if line.startswith('['):
                line = line[1:-1]  # Remove [ ]
                hess_rows.append([float(x.strip()) for x in line.split(',')])
        data['Hessian'] = np.array(hess_rows, dtype=np.float32)

    # Parse param_update (delta_p)
    match = re.search(r'param_update \(6\): \[([\d\.\-e,\s]+)\]', content)
    if match:
        data['param_update'] = np.array([float(x.strip()) for x in match.group(1).split(',')], dtype=np.float32)

    # Parse weight matrix diagonal
    data['weight_diag'] = np.zeros(68, dtype=np.float32)
    for match in re.finditer(r'W\[(\d+),\1\]=([\d\.\-e]+)', content):
        idx = int(match.group(1))
        if idx < 68:
            data['weight_diag'][idx] = float(match.group(2))

    return data


def compare_values(name: str, cpp_val, py_val, tol: float = 1e-5) -> tuple:
    """Compare two values and return (matches, max_diff, message)."""
    if cpp_val is None or py_val is None:
        return False, float('inf'), f"{name}: Missing value (cpp={cpp_val is not None}, py={py_val is not None})"

    cpp_arr = np.asarray(cpp_val, dtype=np.float64)
    py_arr = np.asarray(py_val, dtype=np.float64)

    if cpp_arr.shape != py_arr.shape:
        return False, float('inf'), f"{name}: Shape mismatch (cpp={cpp_arr.shape}, py={py_arr.shape})"

    diff = np.abs(cpp_arr - py_arr)
    max_diff = np.max(diff)

    if max_diff < tol:
        return True, max_diff, f"{name}: MATCH (max_diff={max_diff:.2e})"
    else:
        # Find where max diff occurs
        if cpp_arr.ndim == 1:
            idx = np.argmax(diff)
            return False, max_diff, f"{name}: DIVERGE at [{idx}] cpp={cpp_arr[idx]:.10f} py={py_arr[idx]:.10f} diff={max_diff:.2e}"
        elif cpp_arr.ndim == 2:
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            return False, max_diff, f"{name}: DIVERGE at {idx} cpp={cpp_arr[idx]:.10f} py={py_arr[idx]:.10f} diff={max_diff:.2e}"
        else:
            return False, max_diff, f"{name}: DIVERGE (max_diff={max_diff:.2e})"


def main():
    print("=" * 80)
    print("WS7 RIGID ITER 0 COMPARISON: C++ vs PYTHON")
    print("=" * 80)

    # Load C++ dump
    cpp_dump_path = '/tmp/cpp_ws7_rigid_iter0_dump.txt'
    try:
        cpp = parse_cpp_dump(cpp_dump_path)
        print(f"\nLoaded C++ dump from {cpp_dump_path}")
    except FileNotFoundError:
        print(f"\nERROR: C++ dump not found at {cpp_dump_path}")
        print("Run OpenFace first to generate the dump.")
        return

    # Print what we parsed
    print(f"  current_global: {cpp.get('current_global')}")
    print(f"  sigma: {cpp.get('sigma')}")
    print(f"  param_update: {cpp.get('param_update')}")

    # =========================================================================
    # Run Python to same state
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING PYTHON TO WS7 RIGID ITER 0")
    print("=" * 80)

    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    clnf = CLNF(
        convergence_profile='cpp_match',
        detector='pymtcnn',
        use_validator=False,
        use_eye_refinement=False,
        debug_mode=False
    )

    cpp_bbox = (199.912231, 786.471008, 524.311890, 520.743896)
    params = clnf.pdm.init_params(bbox=cpp_bbox)

    # Run WS11 and WS9
    for ws, scale in [(11, 0.25), (9, 0.35)]:
        patch_experts = clnf._get_patch_experts(0, scale)
        weights = np.ones(clnf.pdm.n_points)
        for lm_idx, pe in patch_experts.items():
            if hasattr(pe, 'patch_confidence'):
                weights[lm_idx] = pe.patch_confidence
        params, _ = clnf.optimizer.optimize(
            clnf.pdm, params, patch_experts, gray,
            weights=weights, window_size=ws,
            patch_scaling=scale,
            sigma_components=clnf.ccnf.sigma_components
        )

    # Now run WS7 RIGID iter 0 step by step
    ws = 7
    patch_scale = 0.5
    patch_experts = clnf._get_patch_experts(0, patch_scale)

    # Get current state
    landmarks_2d = clnf.pdm.params_to_landmarks_2d(params)
    reference_shape = clnf.pdm.get_reference_shape(patch_scale, params[6:])

    sim_img_to_ref = align_shapes_with_scale(landmarks_2d, reference_shape)
    sim_ref_to_img = invert_similarity_transform(sim_img_to_ref)

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    results = []
    first_divergence = None

    # 1. Compare current_global (params)
    py_global = params[:6].astype(np.float32)
    match, diff, msg = compare_values("current_global", cpp.get('current_global'), py_global)
    results.append((match, diff, msg))
    print(f"\n{msg}")
    if not match and first_divergence is None:
        first_divergence = ("current_global", diff)

    # 2. Compare current_shape (landmarks)
    py_shape = landmarks_2d.astype(np.float32)
    match, diff, msg = compare_values("current_shape", cpp.get('current_shape'), py_shape)
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("current_shape", diff)

    # 3. Compare base_shape
    py_base = reference_shape.astype(np.float32)
    match, diff, msg = compare_values("base_shape", cpp.get('base_shape'), py_base)
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("base_shape", diff)

    # 4. Compare similarity transforms
    match, diff, msg = compare_values("sim_img_to_ref", cpp.get('sim_img_to_ref'), sim_img_to_ref.astype(np.float32))
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("sim_img_to_ref", diff)

    match, diff, msg = compare_values("sim_ref_to_img", cpp.get('sim_ref_to_img'), sim_ref_to_img.astype(np.float32))
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("sim_ref_to_img", diff)

    # 5. Compute and compare offsets
    print("\n--- OFFSET COMPUTATION ---")
    current_shape_2D = landmarks_2d  # (68, 2)
    base_shape_2D = reference_shape  # (68, 2)

    py_offsets = (current_shape_2D - base_shape_2D) @ sim_img_to_ref.T
    py_dxs = py_offsets[:, 0] + (ws - 1) / 2.0
    py_dys = py_offsets[:, 1] + (ws - 1) / 2.0

    match, diff, msg = compare_values("offsets", cpp.get('offsets'), py_offsets.astype(np.float32))
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("offsets", diff)

    match, diff, msg = compare_values("dxs", cpp.get('dxs'), py_dxs.astype(np.float32))
    results.append((match, diff, msg))
    print(f"{msg}")

    match, diff, msg = compare_values("dys", cpp.get('dys'), py_dys.astype(np.float32))
    results.append((match, diff, msg))
    print(f"{msg}")

    # 6. Compare response maps (for LM 4, 36, 48)
    print("\n--- RESPONSE MAPS ---")

    # Precompute response maps
    sigma_components = clnf.ccnf.sigma_components
    if isinstance(sigma_components, list):
        sigma = sigma_components[0] if len(sigma_components) > 0 else 1.0
    else:
        sigma = sigma_components

    response_maps = clnf.optimizer._precompute_response_maps(
        landmarks_2d, patch_experts, gray, ws,
        sim_img_to_ref, sim_ref_to_img,
        clnf.ccnf.sigma_components, iteration=0
    )

    for lm in [4, 36, 48]:
        if lm in cpp.get('response_maps', {}) and lm in response_maps:
            cpp_resp = cpp['response_maps'][lm]
            py_resp = response_maps[lm].astype(np.float32)
            match, diff, msg = compare_values(f"response_map_LM{lm}", cpp_resp, py_resp)
            results.append((match, diff, msg))
            print(f"{msg}")
            if not match and first_divergence is None:
                first_divergence = (f"response_map_LM{lm}", diff)

    # 7. Compare mean-shifts (computed from KDE)
    print("\n--- MEAN-SHIFTS ---")

    # Compute mean-shifts using the optimizer's method
    # Compute mean-shifts using the optimizer's internal method
    mean_shifts_ref = {}
    for lm_idx in response_maps:
        ms = clnf.optimizer._compute_mean_shift(response_maps[lm_idx], ws, sigma)
        mean_shifts_ref[lm_idx] = ms

    # Convert to (68, 2) format
    py_mean_shifts = np.zeros((68, 2), dtype=np.float32)
    for lm, ms in mean_shifts_ref.items():
        if lm < 68:
            py_mean_shifts[lm] = ms

    match, diff, msg = compare_values("mean_shifts", cpp.get('mean_shifts'), py_mean_shifts)
    results.append((match, diff, msg))
    print(f"{msg}")
    if not match and first_divergence is None:
        first_divergence = ("mean_shifts", diff)

    # Print mean-shifts for selected landmarks
    print("\n  Detailed mean-shifts for LM 4, 36, 48:")
    for lm in [4, 36, 48]:
        cpp_ms = cpp.get('mean_shifts', np.zeros((68,2)))[lm]
        py_ms = py_mean_shifts[lm]
        diff_ms = np.abs(cpp_ms - py_ms)
        print(f"    LM{lm}: cpp=({cpp_ms[0]:.6f}, {cpp_ms[1]:.6f}) py=({py_ms[0]:.6f}, {py_ms[1]:.6f}) diff=({diff_ms[0]:.6f}, {diff_ms[1]:.6f})")

    # 8. Compare gradient (J_w_t_m)
    print("\n--- GRADIENT & SOLVE ---")
    if cpp.get('J_w_t_m') is not None:
        # We would need to compute the full Jacobian and gradient in Python
        # This is complex - for now just show C++ values
        print(f"  C++ J_w_t_m: {cpp['J_w_t_m']}")
        print(f"  C++ param_update: {cpp.get('param_update')}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    matches = sum(1 for r in results if r[0])
    total = len(results)
    print(f"\nMatched: {matches}/{total}")

    if first_divergence:
        print(f"\n>>> FIRST DIVERGENCE: {first_divergence[0]} (diff={first_divergence[1]:.2e})")
    else:
        print("\n>>> ALL VALUES MATCH!")


if __name__ == '__main__':
    main()
