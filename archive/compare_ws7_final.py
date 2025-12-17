#!/usr/bin/env python3
"""Compare Python WS7 RIGID iter 0 with C++ dump - focused on key values."""

import numpy as np
import sys
sys.path.insert(0, '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf')

# C++ values from dump
CPP_GLOBAL = np.array([3.7053751945, 0.1247465163, 0.1459977180, -0.0650392026, 454.3930358887, 1013.7211303711], dtype=np.float32)
CPP_SIM_IMG_TO_REF = np.array([[0.1355176121, -0.0074605849], [0.0074605858, 0.1355176270]], dtype=np.float32)
CPP_SIM_REF_TO_IMG = np.array([[7.3568177223, 0.4050112665], [-0.4050112963, 7.3568172455]], dtype=np.float32)
CPP_SIGMA = 2.5

# C++ mean-shifts (68 landmarks, in reference coords)
CPP_MEAN_SHIFTS = np.array([
    [-0.4016962051, 0.5293622017], [-0.1068627834, -0.1101186275], [-0.3711023331, -0.1579661369],
    [-0.3938851357, -0.1370797157], [0.1594028473, -0.3014392853], [-0.2388868332, 1.0006389618],
    [0.2368395329, 0.0088577271], [0.3704259396, -0.5921900272], [-0.1561720371, -0.3117122650],
    [-0.1604306698, -0.1396424770], [0.3240282536, 0.5774393082], [0.3935127258, -0.5810060501],
    [0.2642405033, 0.1888706684], [0.4645836353, 0.1094574928], [0.4523882866, -0.0772569180],
    [0.2188594341, -0.2412779331], [-0.2508368492, 0.4389889240], [-0.1807813644, 0.3485145569],
    [0.1302344799, 0.1924495697], [0.0500631332, 0.2714793682], [-0.2518930435, 0.0679805279],
    [0.0685081482, 0.1157610416], [0.0246226788, -0.2781500816], [-0.2424659729, 0.1776938438],
    [0.0625801086, 0.0985398293], [-0.1250267029, 0.0466961861], [-0.0236959457, -0.1467947960],
    [0.2873535156, -0.4368009567], [0.3643028736, -0.2326610088], [0.3791558743, -0.3327829838],
    [0.4846088886, 0.0507555008], [-0.2490787506, 0.1248807907], [-0.4826819897, 0.4000725746],
    [-0.5468919277, 0.5147850513], [0.3039484024, 0.0640814304], [0.4588344097, -0.4989936352],
    [-0.1274178028, -0.4419162273], [0.3475983143, -0.0710823536], [0.5275504589, 0.0021920204],
    [0.3964543343, 0.2532346249], [0.3233265877, 0.2637672424], [0.3560788631, 0.1558225155],
    [-0.3009743690, 0.1753869057], [-0.0602719784, -0.2276537418], [0.0253477097, -0.0719428062],
    [0.4265260696, -0.0138719082], [-0.1548404694, 0.4234611988], [-0.3762631416, 0.3579363823],
    [0.5891211033, 0.4286904335], [-0.2924177647, -0.2891168594], [0.9437184334, 0.5592706203],
    [-0.1565642357, 0.0256528854], [0.2899699211, 0.8046841621], [-0.0656266212, -0.6879742146],
    [-0.9278130531, -0.2995545864], [-0.8812701702, 0.1202919483], [-0.3538205624, 0.1902966499],
    [0.1177933216, 0.3130590916], [0.6540658474, 0.8222360611], [1.8403348923, 0.1180589199],
    [-0.7808692455, 0.1620092392], [0.4300866127, 0.1150197983], [0.5331230164, 0.9498014450],
    [0.1690456867, 0.8762636185], [-0.0803442001, -0.3307173252], [-0.4088127613, 0.8806819916],
    [0.2125616074, 0.9670722485], [0.5247833729, 0.8593134880],
], dtype=np.float32)

# C++ gradient and param_update
CPP_GRADIENT = np.array([1097.0189208984, 3550.7390136719, -2447.6115722656, -4865.8476562500, 40.4956893921, 57.8499298096], dtype=np.float32)
CPP_DELTA_P = np.array([0.0058227107, 0.0053943912, -0.0032489037, -0.0012575650, 0.5955247879, 0.8507342935], dtype=np.float32)

# C++ Hessian
CPP_HESSIAN = np.array([
    [193279.8593750000, -23699.7753906250, -30572.9492187500, -98.1437683105, 0.0000867844, 0.0000400543],
    [-23699.7753906250, 631602.5625000000, -9657.6494140625, -198994.1718750000, -0.0000019073, -0.0003433228],
    [-30572.9492187500, -9657.6494140625, 625331.8125000000, 147789.0468750000, -0.0000228882, 0.0000022650],
    [-98.1437683105, -198994.1718750000, 147789.0468750000, 2633399.7500000000, -0.0005264282, 0.0000019073],
    [0.0000867844, -0.0000019073, -0.0000228882, -0.0005264282, 68.0000000000, 0.0000000000],
    [0.0000400543, -0.0003433228, 0.0000022650, 0.0000019073, 0.0000000000, 68.0000000000],
], dtype=np.float32)

# C++ Jacobian selected rows
CPP_J_ROWS = {
    4: np.array([-54.1049385071, 9.2667207718, 195.3800659180, -149.2447052002, 1.0, 0.0], dtype=np.float32),
    72: np.array([33.8680534363, -178.4304351807, -36.6184730530, -227.5918579102, 0.0, 1.0], dtype=np.float32),
    36: np.array([-42.8145942688, -21.2002086639, 48.9841842651, 122.2201614380, 1.0, 0.0], dtype=np.float32),
    104: np.array([-34.7678108215, -9.9235897064, -20.4271736145, -160.7553710938, 0.0, 1.0], dtype=np.float32),
    48: np.array([-25.0964012146, 16.4502849579, -2.9558007717, -110.3302383423, 1.0, 0.0], dtype=np.float32),
    116: np.array([29.9969882965, 4.3849644661, -11.3092298508, -92.7258148193, 0.0, 1.0], dtype=np.float32),
}

def compare(name, cpp_val, py_val, tol=1e-4):
    """Compare values and report differences."""
    cpp_val = np.asarray(cpp_val)
    py_val = np.asarray(py_val)
    diff = np.abs(cpp_val - py_val)
    max_diff = np.max(diff)
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    if max_diff < tol:
        print(f"  {name}: MATCH (max_diff={max_diff:.2e})")
        return True
    else:
        print(f"  {name}: DIVERGE! max_diff={max_diff:.6f} at idx {max_idx}")
        print(f"    cpp={cpp_val.flat[np.argmax(diff)]:.8f} py={py_val.flat[np.argmax(diff)]:.8f}")
        return False

def main():
    from pyclnf import CLNF
    from pyclnf.core.pdm import PDM
    import cv2

    print("=" * 60)
    print("WS7 RIGID Iteration 0 - Full Comparison")
    print("=" * 60)

    # Load PDM directly
    model_dir = '/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf/pyclnf/models/exported_pdm'
    pdm = PDM(model_dir)
    pdm._params_global = CPP_GLOBAL.copy()

    # Load C++ local params (first 10 were: 21.5634, -8.0410, -12.1828, 1.6911, -30.2788, 6.6264, 4.1154, 2.0388, -4.5119, 1.1652)
    # Build full params array
    params = np.zeros(pdm.n_params, dtype=np.float32)
    params[:6] = CPP_GLOBAL
    # local params from C++ dump - need full 34
    cpp_local_first10 = np.array([21.5634403229, -8.0410032272, -12.1828241348, 1.6911402941, -30.2788028717,
                                   6.6264181137, 4.1153292656, 2.0388066769, -4.5119481087, 1.1651834249], dtype=np.float32)
    params[6:16] = cpp_local_first10  # Set first 10 local params

    # Get current shape (should match C++)
    current_shape_2d = pdm.params_to_landmarks_2d(params)  # (68, 2)

    # C++ landmarks for LM4
    cpp_lm4 = np.array([253.9139404297, 1139.2149658203])
    py_lm4 = current_shape_2d[4]
    print("\n--- LANDMARK COMPARISON (sample) ---")
    print(f"  LM4: cpp={cpp_lm4} py={py_lm4} diff={np.abs(cpp_lm4-py_lm4)}")

    # Compare 3D landmarks (used for Jacobian)
    landmarks_3d = pdm.params_to_landmarks_3d(params)
    print(f"\n  3D landmarks shape: {landmarks_3d.shape}")
    print(f"  LM4 3D: {landmarks_3d[4]}")

    # For now, let's just compute Jacobian and compare
    print("\n--- JACOBIAN COMPARISON ---")
    J = pdm.compute_jacobian_rigid(params)  # 136x6
    print(f"  Jacobian shape: {J.shape}")

    for row_idx, cpp_row in CPP_J_ROWS.items():
        py_row = J[row_idx, :]
        compare(f"J row {row_idx}", cpp_row, py_row, tol=1e-3)

    # Compute full Jacobian comparison
    print("\n--- COMPUTING GRADIENT AND HESSIAN ---")

    # Mean-shifts need to be converted to vector form [ms_x0..ms_x67, ms_y0..ms_y67]
    # CRITICAL: Mean-shifts are in REFERENCE coords, but Jacobian is for IMAGE coords
    # Must transform: ms_img = ms_ref @ sim_ref_to_img.T
    print("\n  Transform test: sim_ref_to_img scale =", np.linalg.norm(CPP_SIM_REF_TO_IMG[0]))

    # Transform mean-shifts from reference to image coordinates
    ms_img = CPP_MEAN_SHIFTS @ CPP_SIM_REF_TO_IMG.T  # (68, 2)
    print(f"  Mean-shift transform example (LM4):")
    print(f"    ref coords: {CPP_MEAN_SHIFTS[4]}")
    print(f"    img coords: {ms_img[4]}")

    ms_vector = np.zeros(136, dtype=np.float32)
    ms_vector[:68] = ms_img[:, 0]  # x values (image coords)
    ms_vector[68:] = ms_img[:, 1]  # y values (image coords)

    # Weight matrix is identity (all 1s from dump)
    W = np.eye(68, dtype=np.float32)
    W_full = np.zeros((136, 136), dtype=np.float32)
    W_full[:68, :68] = W
    W_full[68:, 68:] = W

    # J_w_t = J^T @ W_full
    J_w_t = J.T @ W_full  # 6x136

    # Gradient = J_w_t @ mean_shifts
    py_gradient = J_w_t @ ms_vector
    print("\nGradient comparison:")
    compare("gradient", CPP_GRADIENT, py_gradient, tol=1.0)

    print("\n  Component breakdown:")
    names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']
    for i, name in enumerate(names):
        diff = abs(CPP_GRADIENT[i] - py_gradient[i])
        print(f"    {name}: cpp={CPP_GRADIENT[i]:.4f} py={py_gradient[i]:.4f} diff={diff:.4f}")

    # Hessian = J_w_t @ J (for RIGID, no regularization)
    py_hessian = J_w_t @ J  # 6x6

    print("\nHessian comparison:")
    compare("hessian", CPP_HESSIAN, py_hessian, tol=10.0)

    # Solve for delta_p
    py_delta_p = np.linalg.solve(py_hessian, py_gradient)

    print("\nParameter update (delta_p) comparison:")
    compare("delta_p", CPP_DELTA_P, py_delta_p, tol=1e-4)

    print("\n  Component breakdown:")
    for i, name in enumerate(names):
        diff = abs(CPP_DELTA_P[i] - py_delta_p[i])
        pct = abs(diff / CPP_DELTA_P[i]) * 100 if CPP_DELTA_P[i] != 0 else 0
        print(f"    {name}: cpp={CPP_DELTA_P[i]:.8f} py={py_delta_p[i]:.8f} diff={diff:.8f} ({pct:.2f}%)")

    # Show what happens if we use C++ mean-shifts to get Python delta_p
    print("\n" + "=" * 60)
    print("ANALYSIS: Using C++ mean-shifts with Python Jacobian")
    print("=" * 60)

    # If gradient matches, Jacobian is correct
    grad_diff = np.max(np.abs(CPP_GRADIENT - py_gradient))
    if grad_diff < 1.0:
        print(f"Gradient matches well (max_diff={grad_diff:.4f})")
        print("=> Jacobian computation is CORRECT")
    else:
        print(f"Gradient DIFFERS (max_diff={grad_diff:.4f})")
        print("=> Jacobian computation may differ from C++")

    # If delta_p matches when using C++ mean-shifts, the solve is correct
    delta_diff = np.max(np.abs(CPP_DELTA_P - py_delta_p))
    if delta_diff < 1e-4:
        print(f"delta_p matches well (max_diff={delta_diff:.8f})")
        print("=> Solve step is CORRECT")
    else:
        print(f"delta_p DIFFERS (max_diff={delta_diff:.8f})")
        print("=> Solve step may differ from C++")

    # Now let's compute Python mean-shifts from response maps to find divergence
    print("\n" + "=" * 60)
    print("MEAN-SHIFT COMPARISON (Python computed)")
    print("=" * 60)

    # We need to run the actual optimization to get Python mean-shifts
    # For now, let's compare a few landmarks using the response maps from optimizer

    print("\nTo find root cause:")
    print("1. If Jacobian matches but gradient differs -> mean-shift computation differs")
    print("2. If gradient matches but delta_p differs -> solve/Hessian differs")
    print("3. If delta_p matches -> parameter update application differs")

if __name__ == '__main__':
    main()
