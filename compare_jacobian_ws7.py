#!/usr/bin/env python3
"""
Compare Python vs C++ Jacobian values at WS7 entry.

Uses C++ dump from /tmp/cpp_ws7_rigid_iter0_dump.txt as reference.
"""

import sys
import numpy as np

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')

# C++ values from dump file
CPP_PARAMS = np.array([2.7569327354, -0.0625211820, 0.1299121678, -0.0631994084, 1597.0220947266, 919.0394897461])

# C++ Jacobian rows for LM 4, 36, 48
CPP_J = {
    4: {
        'x': np.array([-49.8720474243, 7.6807904243, 135.7669830322, -124.3045578003, 1.0, 0.0]),
        'y': np.array([48.4420738220, -108.3946533203, 1.7482521534, -152.5102691650, 0.0, 1.0])
    },
    36: {
        'x': np.array([-45.9722671509, -14.6224355698, 38.6286315918, 93.0252456665, 1.0, 0.0]),
        'y': np.array([-33.0679435730, -28.9343948364, 6.1036920547, -131.2921295166, 0.0, 1.0])
    },
    48: {
        'x': np.array([-25.8648815155, 11.6092357635, -1.8187720776, -87.8022842407, 1.0, 0.0]),
        'y': np.array([31.9979591370, 16.9117412567, 5.5344896317, -69.5224914551, 0.0, 1.0])
    }
}

CPP_J_COLUMN_NAMES = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']


def main():
    from pyclnf import CLNF

    print("=" * 80)
    print("JACOBIAN COMPARISON: Python vs C++ at WS7 Entry")
    print("=" * 80)

    # Initialize CLNF and get PDM
    clnf = CLNF(detector=None)
    pdm = clnf.pdm

    # Build full params vector
    n_local = pdm.n_modes
    params = np.zeros(6 + n_local, dtype=np.float64)
    params[:6] = CPP_PARAMS

    # Use C++ local params from dump
    cpp_local = np.array([
        -1.9738134146, 0.7379326224, -17.6084537506, 10.0534906387, -1.7963975668,
        1.7258944511, -5.4227809906, 17.0406837463, -12.6726417542, 11.6865844727
    ])
    params[6:6+len(cpp_local)] = cpp_local

    print(f"\nUsing C++ state:")
    print(f"  Global params: scale={params[0]:.6f}, rot=({params[1]:.6f}, {params[2]:.6f}, {params[3]:.6f})")
    print(f"  Translation: ({params[4]:.2f}, {params[5]:.2f})")

    # Compute Python Jacobian
    J_py = pdm.compute_jacobian_rigid(params)

    print(f"\nJacobian shape: {J_py.shape}")

    # Compare element by element
    print("\n" + "=" * 80)
    print("ELEMENT-BY-ELEMENT COMPARISON")
    print("=" * 80)

    total_diff = 0
    count = 0
    max_diff = 0
    max_diff_loc = ""

    for lm_idx in [4, 36, 48]:
        print(f"\n--- Landmark {lm_idx} ---")

        # X row
        py_x = J_py[lm_idx, :]
        cpp_x = CPP_J[lm_idx]['x']

        print(f"  X row (J[{lm_idx},:]):")
        print(f"    {'Column':8} {'C++':>14} {'Python':>14} {'Diff':>10} {'%Err':>8}")
        print(f"    {'-'*60}")

        for i, name in enumerate(CPP_J_COLUMN_NAMES):
            diff = abs(py_x[i] - cpp_x[i])
            pct = abs(diff / cpp_x[i] * 100) if abs(cpp_x[i]) > 1e-6 else 0
            print(f"    {name:8} {cpp_x[i]:>14.6f} {py_x[i]:>14.6f} {diff:>10.6f} {pct:>7.2f}%")
            total_diff += diff
            count += 1
            if diff > max_diff and abs(cpp_x[i]) > 0.1:  # ignore tx/ty which are 0/1
                max_diff = diff
                max_diff_loc = f"LM{lm_idx} X col {name}"

        # Y row
        py_y = J_py[lm_idx + 68, :]
        cpp_y = CPP_J[lm_idx]['y']

        print(f"\n  Y row (J[{lm_idx+68},:]):")
        print(f"    {'Column':8} {'C++':>14} {'Python':>14} {'Diff':>10} {'%Err':>8}")
        print(f"    {'-'*60}")

        for i, name in enumerate(CPP_J_COLUMN_NAMES):
            diff = abs(py_y[i] - cpp_y[i])
            pct = abs(diff / cpp_y[i] * 100) if abs(cpp_y[i]) > 1e-6 else 0
            print(f"    {name:8} {cpp_y[i]:>14.6f} {py_y[i]:>14.6f} {diff:>10.6f} {pct:>7.2f}%")
            total_diff += diff
            count += 1
            if diff > max_diff and abs(cpp_y[i]) > 0.1:
                max_diff = diff
                max_diff_loc = f"LM{lm_idx} Y col {name}"

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    avg_diff = total_diff / count
    print(f"\nAverage absolute difference: {avg_diff:.6f}")
    print(f"Maximum difference: {max_diff:.6f} at {max_diff_loc}")

    # Column norms comparison
    print("\n--- Column Norms Comparison ---")
    py_norms = np.linalg.norm(J_py, axis=0)

    # Compute C++ norms from available data (just 3 landmarks)
    # This won't be accurate, but gives an idea
    print(f"Python column norms: {py_norms}")

    # Check gradient computation
    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON (J^T * mean_shift)")
    print("=" * 80)

    # C++ gradient from dump
    cpp_gradient = np.array([-228.2102508545, -752.2742309570, 883.0863037109,
                             -1308.6041259766, 7.8047909737, -19.5743770599])

    # Use C++ mean-shifts to compute Python gradient
    cpp_mean_shifts = np.array([
        [-0.3923368454, -0.0035784245],
        [-0.1358723640, -0.2304949760],
        [0.8362510204, 0.4473071098],
        [0.1799793243, 0.7588558197],
        [-0.0995686054, 0.0932664871],
        [-0.1717143059, -0.1589248180],
        [-0.4336073399, -0.5263593197],
        [-0.3157839775, -0.2771933079],
        [0.1357686520, 0.1073720455],
        [0.7630708218, 0.2151556015],
        [-0.0442578793, -0.9223613739],
        [0.3506371975, -0.5303387642],
        [-0.0620727539, -0.0408258438],
        [0.0870656967, 0.2355306149],
        [0.1493687630, 0.4745690823],
        [0.0541350842, -0.0645592213],
        [0.3372635841, 0.2469000816],
        [0.4469633102, 0.2404651642],
        [0.2139203548, 0.1495678425],
        [-0.2790849209, 0.1037681103],
        [-0.3236601353, 0.0925207138],
        [0.1577470303, 0.0383949280],
        [-0.3371825218, -0.5620963573],
        [0.4343500137, -0.1777896881],
        [0.3397786617, -0.1575250626],
        [-0.2991318703, -0.4050562382],
        [-0.2174003124, -0.1634433270],
        [-0.1597421169, -0.7678749561],
        [-0.2500009537, 0.2632336617],
        [0.0196237564, -0.0149996281],
        [0.2648296356, -0.1016616821],
        [-0.5676023960, -0.3178822994],
        [-0.2027215958, 0.1913249493],
        [0.0079011917, -0.4704351425],
        [0.1673090458, 0.3046073914],
        [0.4435856342, 0.3251247406],
        [-0.5742721558, -0.3165628910],
        [0.1447155476, -0.1399550438],
        [0.5317039490, -0.2751550674],
        [0.6091377735, 0.1421668530],
        [0.3315958977, 0.3171713352],
        [0.0323958397, 0.4907701015],
        [-0.4358420372, -0.2356069088],
        [-0.2521517277, -0.4593942165],
        [0.0068020821, -0.6406733990],
        [0.4750044346, -0.5682890415],
        [-0.1441810131, 0.4301638603],
        [-0.2525124550, 0.2889804840],
        [0.5750837326, -0.8831069469],
        [0.8137519360, 0.0408599377],
        [0.1535084248, 0.1544229984],
        [0.1941878796, 0.2070169449],
        [-0.4385540485, 0.2378289700],
        [-0.6787736416, 0.4380528927],
        [-0.6161656380, 0.7014901638],
        [-0.1607358456, -0.2888495922],
        [0.0497636795, 0.1736273766],
        [0.3067016602, 0.1332724094],
        [0.2704925537, 0.1080448627],
        [0.3881399632, -0.5685079098],
        [0.4495601654, -0.5527386665],
        [-0.2319138050, -0.5937438011],
        [-0.0942063332, -0.6647291183],
        [-0.5926282406, -0.2694430351],
        [0.0498800278, 0.7448098660],
        [-0.5653536320, 0.0882730484],
        [0.1601679325, 0.0864660740],
        [0.0516452789, -0.1874561310]
    ])

    # Stack mean-shifts as [ms_x; ms_y]
    ms_vector = np.concatenate([cpp_mean_shifts[:, 0], cpp_mean_shifts[:, 1]])

    # Compute Python gradient
    py_gradient = J_py.T @ ms_vector

    print(f"\n{'Component':>10} {'C++':>14} {'Python':>14} {'Diff':>10} {'%Err':>8}")
    print("-" * 60)

    for i, name in enumerate(CPP_J_COLUMN_NAMES):
        diff = abs(py_gradient[i] - cpp_gradient[i])
        pct = abs(diff / cpp_gradient[i] * 100) if abs(cpp_gradient[i]) > 1e-6 else 0
        print(f"{name:>10} {cpp_gradient[i]:>14.4f} {py_gradient[i]:>14.4f} {diff:>10.4f} {pct:>7.2f}%")

    # Compute delta_p comparison
    print("\n" + "=" * 80)
    print("DELTA_P COMPARISON")
    print("=" * 80)

    cpp_delta_p = np.array([-0.0004941087, -0.0028528615, 0.0029550930,
                            -0.0009840260, 0.1147763506, -0.2878584564])

    # C++ Hessian from dump
    cpp_hessian = np.array([
        [199779.0937500000, 21710.1679687500, -22482.4414062500, 1142.0561523438, 0.0000209808, 0.0001144409],
        [21710.1679687500, 291216.8125000000, -907.6908569336, -93429.1484375000, -0.0000257492, 0.0006790161],
        [-22482.4414062500, -907.6908569336, 284955.0937500000, -27762.7460937500, -0.0004005432, -0.0000066757],
        [1142.0561523438, -93429.1484375000, -27762.7460937500, 1516767.6250000000, -0.0003814697, 0.0000381470],
        [0.0000209808, -0.0000257492, -0.0004005432, -0.0003814697, 68.0000000000, 0.0000000000],
        [0.0001144409, 0.0006790161, -0.0000066757, 0.0000381470, 0.0000000000, 68.0000000000]
    ])

    # Compute Python Hessian
    py_hessian = J_py.T @ J_py

    # Solve for delta_p
    try:
        py_delta_p = np.linalg.solve(py_hessian, py_gradient)
    except np.linalg.LinAlgError:
        py_delta_p = np.linalg.lstsq(py_hessian, py_gradient, rcond=None)[0]

    print(f"\n{'Component':>10} {'C++':>14} {'Python':>14} {'Diff':>10} {'%Err':>8}")
    print("-" * 60)

    for i, name in enumerate(CPP_J_COLUMN_NAMES):
        diff = abs(py_delta_p[i] - cpp_delta_p[i])
        pct = abs(diff / cpp_delta_p[i] * 100) if abs(cpp_delta_p[i]) > 1e-6 else 0
        print(f"{name:>10} {cpp_delta_p[i]:>14.8f} {py_delta_p[i]:>14.8f} {diff:>10.8f} {pct:>7.2f}%")

    # Hessian diagonal comparison
    print("\n--- Hessian Diagonal Comparison ---")
    print(f"{'Index':>6} {'C++':>18} {'Python':>18} {'%Err':>8}")
    print("-" * 60)
    for i in range(6):
        diff = abs(py_hessian[i,i] - cpp_hessian[i,i])
        pct = abs(diff / cpp_hessian[i,i] * 100) if abs(cpp_hessian[i,i]) > 1e-6 else 0
        print(f"{i:>6} {cpp_hessian[i,i]:>18.2f} {py_hessian[i,i]:>18.2f} {pct:>7.2f}%")


if __name__ == '__main__':
    main()
