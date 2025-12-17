#!/usr/bin/env python3
"""
Test pyCLNF accuracy starting from IDENTICAL C++ init state.

This compares Python vs C++ at each window size transition,
using the exact C++ initialization parameters.
"""

import sys
import numpy as np
import cv2

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')
sys.path.insert(0, f'{BASE_DIR}/pymtcnn')

# C++ init params from /tmp/cpp_init_landmarks_68.txt
CPP_INIT_GLOBAL = np.array([
    2.758240,     # scale
    -0.053495,    # rot_x
    0.115318,     # rot_y
    -0.053900,    # rot_z
    1596.590088,  # tx
    918.211243,   # ty
], dtype=np.float64)

CPP_INIT_LOCAL = np.array([
    -3.052162, -3.480800, -18.864716, 10.844760, -2.395396,
    0.757997, -3.297845, 18.070860, -13.012974, 10.900621,
], dtype=np.float64)

# C++ init landmarks (68 points) from /tmp/cpp_init_landmarks_68.txt
CPP_INIT_LANDMARKS = np.array([
    [1417.624634, 850.622498], [1419.937988, 902.399719], [1428.790039, 954.908691],
    [1441.045166, 1004.392639], [1457.419189, 1046.352905], [1481.903442, 1078.390259],
    [1514.855713, 1098.176270], [1554.873413, 1111.982788], [1606.677490, 1114.687500],
    [1659.985229, 1111.812744], [1707.763428, 1097.284058], [1747.435791, 1073.035034],
    [1773.992432, 1037.143433], [1785.999023, 991.662048], [1789.932861, 942.858093],
    [1791.653809, 892.136414], [1790.141724, 842.750610], [1432.454224, 775.665466],
    [1454.162842, 751.844910], [1482.038940, 740.029419], [1511.855103, 737.630737],
    [1540.310669, 744.626587], [1622.251953, 736.401917], [1655.702515, 723.553223],
    [1690.012085, 721.769714], [1721.451538, 731.310730], [1744.161621, 754.065002],
    [1586.293945, 799.540283], [1586.331665, 833.083130], [1586.402588, 865.337158],
    [1586.511475, 898.342529], [1555.031860, 935.690002], [1573.747437, 939.253479],
    [1592.554688, 941.987854], [1612.354980, 935.938904], [1630.634399, 929.662842],
    [1469.245972, 826.613953], [1487.707397, 807.779297], [1514.507202, 806.256348],
    [1538.870117, 823.075012], [1515.285156, 830.136414], [1488.316528, 832.979370],
    [1637.697876, 815.871216], [1659.735962, 795.142273], [1687.377686, 793.328857],
    [1711.114868, 808.094116], [1691.258789, 816.794922], [1664.096069, 819.176575],
    [1525.999634, 1008.971558], [1549.480469, 987.101074], [1574.739990, 975.238953],
    [1594.233887, 978.592834], [1615.050171, 972.272095], [1646.551636, 981.294495],
    [1678.182861, 995.886108], [1650.153687, 1015.829285], [1621.197510, 1025.369141],
    [1598.256104, 1028.780640], [1577.574829, 1029.069336], [1551.169312, 1024.103271],
    [1537.088989, 1007.302368], [1577.178467, 997.498840], [1596.511841, 996.736938],
    [1618.095947, 993.778931], [1665.394165, 997.210815], [1618.322510, 998.645508],
    [1596.510254, 1001.442810], [1576.924561, 1001.661560],
], dtype=np.float64)

# C++ landmarks at each WS transition (from /tmp/cpp_iteration_landmarks.txt)
CPP_WS_LANDMARKS = {
    # Format: lm4, lm36, lm48, lm30, lm8
    'ITER0_WS11_NONRIGID': {
        4: (1457.4192, 1046.3529), 36: (1469.2460, 826.6140),
        48: (1525.9996, 1008.9716), 30: (1586.5115, 898.3425), 8: (1606.6775, 1114.6875)
    },
    'ITER1_WS9_NONRIGID': {
        4: (1459.5282, 1052.5911), 36: (1470.2797, 827.8734),
        48: (1525.7144, 1007.2557), 30: (1584.6882, 897.4670), 8: (1610.1534, 1113.4343)
    },
    'ITER2_WS7_NONRIGID': {
        4: (1459.7732, 1052.8688), 36: (1470.6417, 827.5240),
        48: (1529.4923, 1006.5079), 30: (1585.0779, 897.5417), 8: (1611.0175, 1111.1018)
    },
    'ITER3_WS5_NONRIGID': {
        4: (1459.7887, 1052.2972), 36: (1471.4178, 827.2687),
        48: (1528.9897, 1007.3041), 30: (1584.1096, 897.0301), 8: (1611.0514, 1111.6555)
    },
}


def main():
    from pyclnf import CLNF
    from pyclnf.core.pdm import PDM

    print("=" * 80)
    print("PYCLNF vs C++ ACCURACY TEST (FROM IDENTICAL INIT STATE)")
    print("=" * 80)

    # Load image
    img_path = f'{BASE_DIR}/comparison_frame_0030.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load {img_path}")
        return

    print(f"\nTest image: {img_path}")
    print(f"Image size: {img.shape}")

    # Initialize CLNF
    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    # Build init params from C++ values
    init_params = np.zeros(pdm.n_params, dtype=np.float64)
    init_params[:6] = CPP_INIT_GLOBAL
    init_params[6:16] = CPP_INIT_LOCAL  # Only first 10 local params

    # Verify Python init landmarks match C++ init landmarks
    print("\n" + "=" * 80)
    print("STEP 1: Verify Init Landmarks Match")
    print("=" * 80)

    py_init_landmarks = pdm.params_to_landmarks_2d(init_params)
    init_errors = np.linalg.norm(py_init_landmarks - CPP_INIT_LANDMARKS, axis=1)

    print(f"\nInit landmark error (Python vs C++ init):")
    print(f"  Mean: {init_errors.mean():.6f} px")
    print(f"  Max:  {init_errors.max():.6f} px")

    if init_errors.max() > 0.01:
        print("\n  WARNING: Init landmarks don't match exactly!")
        print("  Checking sample landmarks:")
        for idx in [4, 8, 30, 36, 48]:
            py_lm = py_init_landmarks[idx]
            cpp_lm = CPP_INIT_LANDMARKS[idx]
            err = np.linalg.norm(py_lm - cpp_lm)
            print(f"    LM{idx}: py=({py_lm[0]:.4f}, {py_lm[1]:.4f}) cpp=({cpp_lm[0]:.4f}, {cpp_lm[1]:.4f}) err={err:.6f}")

    # Run optimization from C++ init state
    print("\n" + "=" * 80)
    print("STEP 2: Run Python Optimization from C++ Init")
    print("=" * 80)

    # Get initial landmarks for optimizer
    initial_landmarks = pdm.params_to_landmarks_2d(init_params)

    # Access optimizer directly for more control
    optimizer = clnf.optimizer
    patch_experts = clnf.patch_experts

    # Run optimization with iteration tracking
    final_params, info = optimizer.optimize(
        image=img,
        landmarks_2d_initial=initial_landmarks,
        params=init_params.copy(),
        patch_experts=patch_experts,
        pdm=pdm,
        window_sizes=[11, 9, 7, 5],
        return_history=True
    )

    # Extract landmarks at each WS transition
    print("\n" + "=" * 80)
    print("STEP 3: Compare Python vs C++ at Each Window Size")
    print("=" * 80)

    # Track error at each WS
    ws_errors = {}

    if 'iteration_history' in info:
        # Find last iteration for each window size
        ws_final_params = {}
        for iter_info in info['iteration_history']:
            ws = iter_info['window_size']
            phase = iter_info['phase']
            if phase == 'nonrigid':  # Compare after nonrigid phase
                ws_final_params[ws] = iter_info['params'].copy()

        # Compare with C++ at each WS
        ws_mapping = {
            11: 'ITER0_WS11_NONRIGID',
            9: 'ITER1_WS9_NONRIGID',
            7: 'ITER2_WS7_NONRIGID',
            5: 'ITER3_WS5_NONRIGID',
        }

        for ws in [11, 9, 7, 5]:
            if ws not in ws_final_params:
                print(f"\n  WS{ws}: No data")
                continue

            params = ws_final_params[ws]
            py_landmarks = pdm.params_to_landmarks_2d(params)
            cpp_key = ws_mapping[ws]
            cpp_lms = CPP_WS_LANDMARKS[cpp_key]

            errors = []
            for lm_idx, (cpp_x, cpp_y) in cpp_lms.items():
                py_lm = py_landmarks[lm_idx]
                err = np.sqrt((py_lm[0] - cpp_x)**2 + (py_lm[1] - cpp_y)**2)
                errors.append(err)

            mean_err = np.mean(errors)
            max_err = np.max(errors)
            ws_errors[ws] = (mean_err, max_err)

            print(f"\n  After WS{ws}:")
            print(f"    Mean error: {mean_err:.6f} px")
            print(f"    Max error:  {max_err:.6f} px")

            # Per-landmark breakdown
            for lm_idx, (cpp_x, cpp_y) in cpp_lms.items():
                py_lm = py_landmarks[lm_idx]
                err = np.sqrt((py_lm[0] - cpp_x)**2 + (py_lm[1] - cpp_y)**2)
                print(f"      LM{lm_idx}: py=({py_lm[0]:.4f}, {py_lm[1]:.4f}) cpp=({cpp_x:.4f}, {cpp_y:.4f}) err={err:.6f}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Stage                | Mean Error  | Max Error   |")
    print("|----------------------|-------------|-------------|")
    print(f"| Init (C++ state)     | {init_errors.mean():>10.6f} px | {init_errors.max():>10.6f} px |")

    for ws in [11, 9, 7, 5]:
        if ws in ws_errors:
            mean_err, max_err = ws_errors[ws]
            print(f"| After WS{ws:<13}| {mean_err:>10.6f} px | {max_err:>10.6f} px |")

    # Check if we achieved sub-pixel accuracy
    final_ws5 = ws_errors.get(5, (999, 999))
    if final_ws5[0] < 0.01:
        print("\n✓ SUB-PIXEL ACCURACY ACHIEVED!")
    else:
        print(f"\n✗ Accuracy is {final_ws5[0]:.4f} px (target: <0.01 px)")


if __name__ == '__main__':
    main()
