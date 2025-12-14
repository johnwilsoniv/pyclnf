#!/usr/bin/env python3
"""
Debug WS11 iteration 0 mean-shift values.

Compare Python mean-shifts with C++ to find why they differ by 10x.
"""

import sys
import numpy as np
import cv2

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

# C++ WS11 RIGID iteration 0 data
CPP_MEAN_SHIFT_SUM = (-519.7840, 1564.0923)
CPP_MEAN_SHIFT_LM36 = (-3.3696, 26.1750)
CPP_MEAN_SHIFT_LM5 = (-16.7918, 8.8628)


def main():
    from pyclnf import CLNF
    from pyclnf.core.optimizer import NuRLMSOptimizer
    from pymtcnn import MTCNN

    print("=" * 80)
    print("WS11 ITER 0 MEAN-SHIFT DEBUG")
    print("=" * 80)

    # Load image and detect face
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    img = cv2.imread(img_path)

    mtcnn = MTCNN(backend='coreml')
    bboxes, lm5 = mtcnn.detect(img)
    bbox = tuple(bboxes[0][:4])
    lm5_arr = lm5[0] if lm5 is not None else None

    # Initialize CLNF
    clnf = CLNF(detector=None)
    pdm = clnf.pdm
    optimizer = clnf.optimizer

    # Get initial params (same as what fit() uses)
    initial_params = pdm.init_params(bbox)
    initial_landmarks = pdm.params_to_landmarks_2d(initial_params)

    print(f"\nInitial state:")
    print(f"  scale: {initial_params[0]:.6f}")
    print(f"  rotation: ({initial_params[1]:.6f}, {initial_params[2]:.6f}, {initial_params[3]:.6f})")
    print(f"  translation: ({initial_params[4]:.2f}, {initial_params[5]:.2f})")

    # Compute similarity transform (same as optimizer.optimize)
    window_size = 11
    patch_scale = optimizer.window_size_to_patch_scale.get(window_size, 0.35)

    # Reference shape at scale 1.0
    ref_params = np.zeros(6 + pdm.n_modes)
    ref_params[0] = 1.0  # Unit scale
    ref_shape = pdm.params_to_landmarks_2d(ref_params)

    # Compute similarity transform (initial_landmarks → ref_shape)
    from pyclnf.core.utils import compute_similarity_transform
    sim_img_to_ref, sim_ref_to_img = compute_similarity_transform(initial_landmarks, ref_shape)

    print(f"\nSimilarity transforms:")
    print(f"  sim_img_to_ref:\n{sim_img_to_ref}")
    print(f"  sim_ref_to_img:\n{sim_ref_to_img}")

    scale_factor = np.sqrt(sim_ref_to_img[0,0]**2 + sim_ref_to_img[1,0]**2)
    print(f"  Scale factor (ref→img): {scale_factor:.4f}")

    # Compute response maps (same as optimizer)
    patch_experts = optimizer.patch_experts.get(patch_scale, {})
    sigma_components = optimizer.sigma_components.get(patch_scale, {}).get(window_size, {})

    print(f"\nComputing response maps...")
    response_maps = optimizer._precompute_response_maps(
        img, initial_landmarks, patch_experts, window_size,
        sim_img_to_ref, sim_ref_to_img, sigma_components, iteration=0
    )
    print(f"  {len(response_maps)} response maps computed")

    # Compute mean-shift
    print(f"\nComputing mean-shift vectors...")
    mean_shift = optimizer._compute_mean_shift(
        initial_landmarks, initial_landmarks, response_maps, patch_experts,
        window_size, sim_img_to_ref, sim_ref_to_img, iteration=0
    )

    # Analyze mean-shift
    n_lm = len(mean_shift) // 2
    ms_x = mean_shift[:n_lm]
    ms_y = mean_shift[n_lm:]

    ms_sum_x = np.sum(ms_x)
    ms_sum_y = np.sum(ms_y)

    print("\n" + "=" * 80)
    print("MEAN-SHIFT COMPARISON: Python vs C++")
    print("=" * 80)

    print(f"\nMean-shift SUM (all landmarks):")
    print(f"  {'':10} {'X':>14} {'Y':>14} {'Norm':>14}")
    print(f"  {'-'*55}")
    print(f"  {'C++':10} {CPP_MEAN_SHIFT_SUM[0]:>14.4f} {CPP_MEAN_SHIFT_SUM[1]:>14.4f} {np.linalg.norm(CPP_MEAN_SHIFT_SUM):>14.4f}")
    print(f"  {'Python':10} {ms_sum_x:>14.4f} {ms_sum_y:>14.4f} {np.sqrt(ms_sum_x**2 + ms_sum_y**2):>14.4f}")

    ratio_x = CPP_MEAN_SHIFT_SUM[0] / ms_sum_x if abs(ms_sum_x) > 0.01 else 0
    ratio_y = CPP_MEAN_SHIFT_SUM[1] / ms_sum_y if abs(ms_sum_y) > 0.01 else 0
    print(f"\n  C++/Python ratio: X={ratio_x:.2f}x, Y={ratio_y:.2f}x")
    print(f"  Expected ratio (scale factor): {scale_factor:.2f}x")

    print(f"\nIndividual landmark mean-shifts:")
    print(f"  {'LM':4} {'C++ X':>12} {'C++ Y':>12} {'Py X':>12} {'Py Y':>12} {'Ratio X':>10} {'Ratio Y':>10}")
    print(f"  {'-'*80}")

    cpp_samples = {
        36: CPP_MEAN_SHIFT_LM36,
        5: CPP_MEAN_SHIFT_LM5,
    }

    for lm_idx, (cpp_x, cpp_y) in cpp_samples.items():
        py_x = ms_x[lm_idx]
        py_y = ms_y[lm_idx]
        ratio_x = cpp_x / py_x if abs(py_x) > 0.001 else 0
        ratio_y = cpp_y / py_y if abs(py_y) > 0.001 else 0
        print(f"  {lm_idx:4} {cpp_x:>12.4f} {cpp_y:>12.4f} {py_x:>12.4f} {py_y:>12.4f} {ratio_x:>10.2f} {ratio_y:>10.2f}")

    # Show first few mean-shifts
    print(f"\nFirst 10 landmark mean-shifts (Python):")
    for i in range(min(10, n_lm)):
        print(f"  LM{i}: ({ms_x[i]:.4f}, {ms_y[i]:.4f})")

    # Check if mean-shifts are being computed in reference or image coords
    print("\n" + "=" * 80)
    print("COORDINATE SYSTEM ANALYSIS")
    print("=" * 80)

    print(f"\nPython mean-shift vector norm: {np.linalg.norm(mean_shift):.4f}")
    print(f"C++ mean-shift sum norm: {np.linalg.norm(CPP_MEAN_SHIFT_SUM):.4f}")

    # If Python is in ref coords, transforming should match C++
    ms_transformed = np.zeros_like(mean_shift)
    a_mat = sim_ref_to_img[0, 0]
    b_mat = sim_ref_to_img[1, 0]

    for i in range(n_lm):
        ms_ref_x = ms_x[i]
        ms_ref_y = ms_y[i]
        # Transform ref → img
        ms_img_x = a_mat * ms_ref_x - b_mat * ms_ref_y
        ms_img_y = b_mat * ms_ref_x + a_mat * ms_ref_y
        ms_transformed[i] = ms_img_x
        ms_transformed[i + n_lm] = ms_img_y

    ms_trans_sum_x = np.sum(ms_transformed[:n_lm])
    ms_trans_sum_y = np.sum(ms_transformed[n_lm:])

    print(f"\nIf Python mean-shifts were in REF coords, after transforming to IMG:")
    print(f"  Sum X: {ms_trans_sum_x:.4f} (C++: {CPP_MEAN_SHIFT_SUM[0]:.4f})")
    print(f"  Sum Y: {ms_trans_sum_y:.4f} (C++: {CPP_MEAN_SHIFT_SUM[1]:.4f})")

    # Check gradient computation
    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)

    # Compute Jacobian
    J = pdm.compute_jacobian_rigid(initial_params)

    # Gradient with Python mean-shifts (as-is)
    gradient_py = J.T @ mean_shift

    # Gradient with transformed mean-shifts
    gradient_trans = J.T @ ms_transformed

    cpp_gradient = np.array([-1869.3754, 51210.8711, 29970.7871, -28833.3535, -519.7841, 1564.0923])

    param_names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']

    print(f"\n{'Param':8} {'C++':>14} {'Py (as-is)':>14} {'Py (trans)':>14}")
    print(f"{'-'*55}")

    for i, name in enumerate(param_names):
        print(f"{name:8} {cpp_gradient[i]:>14.2f} {gradient_py[i]:>14.2f} {gradient_trans[i]:>14.2f}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Check which version matches better
    err_asis = np.linalg.norm(gradient_py - cpp_gradient) / np.linalg.norm(cpp_gradient) * 100
    err_trans = np.linalg.norm(gradient_trans - cpp_gradient) / np.linalg.norm(cpp_gradient) * 100

    print(f"\nGradient error (relative to C++):")
    print(f"  Using Python mean-shifts as-is: {err_asis:.1f}%")
    print(f"  After transforming to img coords: {err_trans:.1f}%")

    if err_trans < err_asis:
        print(f"\n→ Python mean-shifts are in REFERENCE coords, need to transform to IMAGE!")
        print(f"  This is the ROOT CAUSE of the divergence!")
    else:
        print(f"\n→ Coordinate transform is not the issue - need to investigate further")


if __name__ == '__main__':
    main()
