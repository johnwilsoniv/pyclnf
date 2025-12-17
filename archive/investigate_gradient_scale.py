#!/usr/bin/env python3
"""
Investigate the 5x gradient scale difference between Python and C++.

Hypothesis: Mean-shifts need to be scaled by similarity transform.
"""

import sys
import numpy as np

BASE_DIR = '/Users/johnwilsoniv/Documents/SplitFace Open3'
sys.path.insert(0, f'{BASE_DIR}/pyclnf')

# C++ values from dump
CPP_SIM_REF_TO_IMG = np.array([[5.4744458199, 0.3617998958],
                               [-0.3617998958, 5.4744453430]])

CPP_GRADIENT = np.array([-228.2102508545, -752.2742309570, 883.0863037109,
                         -1308.6041259766, 7.8047909737, -19.5743770599])

# Scale factor in sim_ref_to_img
scale_factor = np.sqrt(CPP_SIM_REF_TO_IMG[0,0]**2 + CPP_SIM_REF_TO_IMG[0,1]**2)
print(f"Similarity transform scale: {scale_factor:.4f}")
print(f"Scale squared: {scale_factor**2:.4f}")

# The mean-shifts from C++ are in reference coords
# When Jacobian is computed in image coords, mean-shifts must be scaled

print("\n=== GRADIENT RATIO ANALYSIS ===")
print(f"C++ gradient (scale): {CPP_GRADIENT[0]:.4f}")
print(f"Expected Python gradient if 5x smaller: {CPP_GRADIENT[0]/5:.4f}")
print(f"Ratio C++/Python should be: {scale_factor:.4f}")

print("\n=== KEY INSIGHT ===")
print("The Jacobian computes d(landmark_img)/d(params)")
print("The mean-shifts from KDE are in REFERENCE coords")
print("So: gradient = J^T * mean_shift_img")
print("where mean_shift_img = sim_ref_to_img @ mean_shift_ref")
print()
print("If Python uses mean_shift_ref directly, gradient will be ~5x smaller!")

# Let's verify by transforming mean-shifts
print("\n=== VERIFICATION ===")

# Load Python components
from pyclnf import CLNF
clnf = CLNF(detector=None)
pdm = clnf.pdm

# C++ params
cpp_params = np.array([2.7569327354, -0.0625211820, 0.1299121678,
                       -0.0631994084, 1597.0220947266, 919.0394897461])
n_local = pdm.n_modes
params = np.zeros(6 + n_local, dtype=np.float64)
params[:6] = cpp_params

cpp_local = np.array([
    -1.9738134146, 0.7379326224, -17.6084537506, 10.0534906387, -1.7963975668,
    1.7258944511, -5.4227809906, 17.0406837463, -12.6726417542, 11.6865844727
])
params[6:6+len(cpp_local)] = cpp_local

# C++ mean-shifts (ref coords)
cpp_mean_shifts_ref = np.array([
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

# Transform mean-shifts to image coordinates
cpp_mean_shifts_img = cpp_mean_shifts_ref @ CPP_SIM_REF_TO_IMG.T

print("Sample mean-shift transformation:")
print(f"  LM36 ref: ({cpp_mean_shifts_ref[36,0]:.4f}, {cpp_mean_shifts_ref[36,1]:.4f})")
print(f"  LM36 img: ({cpp_mean_shifts_img[36,0]:.4f}, {cpp_mean_shifts_img[36,1]:.4f})")
print(f"  Magnitude ratio: {np.linalg.norm(cpp_mean_shifts_img[36])/np.linalg.norm(cpp_mean_shifts_ref[36]):.4f}")

# Compute Jacobian
J = pdm.compute_jacobian_rigid(params)

# Stack mean-shifts as [ms_x; ms_y] - using REFERENCE coords (wrong)
ms_ref_vector = np.concatenate([cpp_mean_shifts_ref[:, 0], cpp_mean_shifts_ref[:, 1]])

# Stack mean-shifts as [ms_x; ms_y] - using IMAGE coords (correct?)
ms_img_vector = np.concatenate([cpp_mean_shifts_img[:, 0], cpp_mean_shifts_img[:, 1]])

# Compute gradients both ways
gradient_with_ref = J.T @ ms_ref_vector
gradient_with_img = J.T @ ms_img_vector

print("\n=== GRADIENT COMPARISON ===")
print(f"{'Component':>10} {'C++':>14} {'Py(ref)':>14} {'Py(img)':>14}")
print("-" * 60)

names = ['scale', 'rot_x', 'rot_y', 'rot_z', 'tx', 'ty']
for i, name in enumerate(names):
    print(f"{name:>10} {CPP_GRADIENT[i]:>14.4f} {gradient_with_ref[i]:>14.4f} {gradient_with_img[i]:>14.4f}")

print("\n=== ERROR COMPARISON ===")
print(f"{'Component':>10} {'Err(ref)':>12} {'Err(img)':>12}")
print("-" * 40)

for i, name in enumerate(names):
    err_ref = abs(gradient_with_ref[i] - CPP_GRADIENT[i]) / abs(CPP_GRADIENT[i]) * 100
    err_img = abs(gradient_with_img[i] - CPP_GRADIENT[i]) / abs(CPP_GRADIENT[i]) * 100
    print(f"{name:>10} {err_ref:>11.2f}% {err_img:>11.2f}%")

print("\n=== CONCLUSION ===")
ref_err_avg = np.mean([abs(gradient_with_ref[i] - CPP_GRADIENT[i]) / abs(CPP_GRADIENT[i]) * 100 for i in range(6)])
img_err_avg = np.mean([abs(gradient_with_img[i] - CPP_GRADIENT[i]) / abs(CPP_GRADIENT[i]) * 100 for i in range(6)])

if img_err_avg < ref_err_avg:
    print(f"Using IMAGE coords reduces error from {ref_err_avg:.1f}% to {img_err_avg:.1f}%")
    if img_err_avg < 5:
        print("→ CONFIRMED: Mean-shifts should be in IMAGE coordinates!")
    else:
        print("→ Still have significant error - may be other issues")
else:
    print(f"Using IMAGE coords increases error from {ref_err_avg:.1f}% to {img_err_avg:.1f}%")
    print("→ The issue is NOT coordinate transform")
