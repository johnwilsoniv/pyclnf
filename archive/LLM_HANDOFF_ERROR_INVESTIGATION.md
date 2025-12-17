# LLM Handoff: pyCLNF Error Accumulation Investigation

## Problem Statement

The Python implementation of CLNF (pyCLNF) shows increasing landmark error as window size decreases during optimization. The error accumulates ~45-70x from WS11 to WS7.

## Error Accumulation Chart (Per Window Size)

### Latest Test Results (December 2024)

Test image: `comparison_frame_0030.jpg` (1920x2160)

| Stage                | Mean Error  | Max Error   | Notes |
|----------------------|-------------|-------------|-------|
| Init (bbox)          | 23.857 px   | 45.274 px   | MTCNN detection |
| After WS11           | 3.290 px    | 10.087 px   | patch_scale=0.25, resp_size=11 |
| After WS9            | 1.265 px    | 4.288 px    | patch_scale=0.35, resp_size=9 |
| After WS7            | 1.360 px    | 3.816 px    | patch_scale=0.5, resp_size=7 |
| After WS5            | 0.991 px    | 3.780 px    | patch_scale=1.0, resp_size=5 |
| Final (w/ eye refine)| 0.473 px    | 1.652 px    | After eye refinement |

**Error Progression:**
```
WS11 → WS9: ↓ 2.0254 px (-61.6%)   ← Good, optimization improving
WS9 → WS7: ↑ 0.0951 px (+7.5%)     ← BAD! Error INCREASES at WS7
WS7 → WS5: ↓ 0.3693 px (-27.2%)    ← Good, continues improving
```

**Key observation**: Error increases by 7.5% between WS9 and WS7, while all other transitions show improvement.

### Historical Sub-Pixel Results (identical starting state)

When Python starts from **identical C++ state** (using dump values):

| Stage                | Mean Error  | Max Error   | Notes |
|----------------------|-------------|-------------|-------|
| Init (C++ bbox)      | 0.000085 px | 0.000116 px | Identical to C++ |
| After WS11           | 0.000151 px | 0.000221 px | Sub-pixel accuracy |
| After WS9            | 0.000595 px | 0.001480 px | Still sub-pixel |
| After WS7            | 0.006737 px | 0.013004 px | ~11x increase! |
| After WS5            | 0.007685 px | 0.013216 px | Continues accumulating |

The sub-pixel accuracy is achievable when starting from identical state, but error accumulates ~70x at WS7.

---

## Codebase Structure

```
pyclnf/
├── pyclnf/
│   ├── clnf.py              # Main CLNF class (entry point)
│   ├── core/
│   │   ├── optimizer.py      # NU-RLMS optimizer (CRITICAL)
│   │   ├── pdm.py           # Point Distribution Model
│   │   ├── cen_patch_expert.py  # CEN patch experts
│   │   └── utils.py         # Transform utilities
│   └── models/
│       └── exported_pdm/    # PDM model files
└── compare_ws7_final.py     # Comparison tool we created
```

---

## Key Code Locations

### 1. Optimization Loop (`optimizer.py:520-808`)

```python
def optimize(self, image, landmarks_2d_initial, params, patch_experts,
             pdm, window_sizes=[11, 9, 7, 5], ...):
    """
    Two-phase NU-RLMS optimization matching OpenFace C++.

    Window sizes progress: 11 → 9 → 7 → 5
    Each window size runs:
      - Phase 1: RIGID (scale, rotation, translation only)
      - Phase 2: NONRIGID (+ shape parameters)
    """
    for window_size in window_sizes:
        # ... optimization for each window size
```

### 2. Mean-Shift Computation (`optimizer.py:1217-1345`)

```python
def _compute_mean_shift(self, landmarks_2d, base_landmarks_2d, response_maps,
                        patch_experts, window_size, sim_img_to_ref, sim_ref_to_img, ...):
    """
    Compute mean-shift vector using PRECOMPUTED response maps.

    CRITICAL FLOW:
    1. Compute offset: (current_landmarks - base_landmarks) in IMAGE coords
    2. Transform offset to REFERENCE coords: offset_ref = offset_img @ sim_img_to_ref
    3. Compute KDE mean-shift in REFERENCE coords
    4. Transform mean-shift back to IMAGE coords: ms_img = ms_ref @ sim_ref_to_img
    """
    # Lines 1288-1295: Transform offset from image to reference
    if use_warping:
        a_sim = sim_img_to_ref[0, 0]
        b_sim = sim_img_to_ref[1, 0]
        offset_ref_x = a_sim * offset_img_x + (-b_sim) * offset_img_y
        offset_ref_y = b_sim * offset_img_x + a_sim * offset_img_y

    # Lines 1319-1321: KDE mean-shift
    ms_ref_x, ms_ref_y = self._kde_mean_shift(response_map, dx, dy, a_kde, landmark_idx)

    # Lines 1323-1329: Transform mean-shift from reference to image
    if use_warping:
        a_mat = sim_ref_to_img[0, 0]
        b_mat = sim_ref_to_img[1, 0]
        ms_x = a_mat * ms_ref_x - b_mat * ms_ref_y
        ms_y = b_mat * ms_ref_x + a_mat * ms_ref_y
```

### 3. Jacobian Computation (`pdm.py:147-263`)

```python
def compute_jacobian(self, params):
    """
    Compute Jacobian: ∂(2D landmarks) / ∂(parameters)

    Returns: (136, n_params) matrix where 136 = 68 landmarks × 2 (x,y)
    STACKED format: rows 0-67 = ∂x/∂params, rows 68-135 = ∂y/∂params
    """
    # Extract rotation matrix from Euler angles
    euler = np.array([wx, wy, wz])
    R = self._euler_to_rotation_matrix(euler)

    # Column 0: ∂/∂scale
    J[:n, 0] = X * r11 + Y * r12 + Z * r13
    J[n:, 0] = X * r21 + Y * r22 + Z * r23

    # Columns 1-3: ∂/∂rotation (analytical from small-angle approximation)
    J[:n, 1] = s * (Y * r13 - Z * r12)  # ∂x/∂wx
    J[n:, 1] = s * (Y * r23 - Z * r22)  # ∂y/∂wx
    # ... similar for wy, wz

    # Columns 4-5: ∂/∂translation
    J[:n, 4] = 1.0  # ∂x/∂tx
    J[n:, 5] = 1.0  # ∂y/∂ty
```

### 4. Parameter Update Solve (`optimizer.py:1889-1967`)

```python
def _solve_rigid_update(self, J_rigid, v, W, iteration, window_size):
    """
    Solve: Δp = (J^T·W·J)^(-1) · (J^T·W·v)

    Where:
      J: Jacobian (136, 6) for rigid params
      v: mean-shift vector (136,) in IMAGE coords
      W: weight matrix (136, 136)
    """
    A = J_rigid.T @ W @ J_rigid  # Hessian (6, 6)
    b = J_rigid.T @ W @ v        # Gradient (6,)
    delta_p_rigid = np.linalg.solve(A, b)  # using Cholesky
    return delta_p_rigid
```

### 5. Rotation Parameter Update (`pdm.py:1078-1150`)

```python
def update_params(self, params, delta_p):
    """
    Update parameters with rotation composition on SO(3).

    OpenFace approach:
    1. Build delta rotation matrix R' from small-angle approximation
    2. Compose: R_new = R_current @ R'
    3. Orthonormalize R_new (stays on SO(3) manifold)
    4. Convert back to Euler angles
    """
    # Small-angle rotation matrix
    R_delta = np.array([
        [1,          -delta_wz,   delta_wy],
        [delta_wz,    1,         -delta_wx],
        [-delta_wy,   delta_wx,   1       ]
    ])
    R_delta = self._orthonormalize(R_delta)

    # Compose rotations
    R_new = R_current @ R_delta
    R_new = self._orthonormalize(R_new)

    # Convert back to Euler
    euler_new = self._rotation_matrix_to_euler(R_new)
```

---

## What We've Verified Works Correctly

### WS7 RIGID Iteration 0 Comparison (from C++ dump)

When starting from **identical state** (using C++ dump values):

| Component | C++ Value | Python Value | Error |
|-----------|-----------|--------------|-------|
| delta_tx | 0.5955248 | 0.5955249 | **0.00%** |
| delta_ty | 0.8507343 | 0.8507343 | **0.00%** |
| delta_scale | 0.0058227 | 0.0056505 | 2.96% |
| delta_rot_x | 0.0053944 | 0.0056882 | 5.45% |
| delta_rot_y | -0.0032489 | -0.0033068 | 1.78% |
| delta_rot_z | -0.0012576 | -0.0011420 | 9.19% |

**Key finding**: Translation matches exactly; rotation/scale within 2-9%.

### Verified Coordinate Transforms

```python
# sim_ref_to_img scale ≈ 7.37 (verified matches C++)
# sim_img_to_ref scale ≈ 0.136 (inverse)

# Mean-shift transform is REQUIRED and CORRECT:
ms_img = ms_ref @ sim_ref_to_img.T  # Reference → Image
```

---

## Hypotheses for Error Accumulation

### H1: Rotation Composition Precision Loss

Each update involves: `Euler → Matrix → Orthonormalize → Euler`

Small errors (~1e-7) accumulate over iterations. At WS7:
- 5 RIGID iterations
- Multiple rotation compositions
- Errors multiply with face scale (~200 px from centroid)

Expected error: `200 × sin(0.00005) ≈ 0.01 px` ✓

### H2: Scale Sensitivity at Different Patch Scales

| Window Size | Patch Scale | Transform Scale | Notes |
|-------------|-------------|-----------------|-------|
| WS11 | 0.25 | ~10.5 | Lower sensitivity |
| WS9 | 0.35 | ~10.5 | Lower sensitivity |
| WS7 | 0.50 | ~7.4 | Higher sensitivity |
| WS5 | 1.00 | ~3.7 | Highest sensitivity |

At smaller patch scales, small reference-space errors are amplified less by the transform.

### H3: State Mismatch Between Window Sizes

When transitioning from WS9 → WS7:
- Python may have accumulated small errors in WS9
- WS7 starts from a slightly different state than C++
- Error compounds through WS7's iterations

---

## Investigation Approach for Next LLM

### Step 1: Reproduce Per-Window-Size Error

Create a test that outputs error after each window size:

```python
# Pseudo-code
for ws in [11, 9, 7, 5]:
    landmarks_after_ws = run_optimization_until_ws(ws)
    error = np.linalg.norm(landmarks_after_ws - cpp_landmarks, axis=1)
    print(f"After WS{ws}: mean={error.mean():.6f}, max={error.max():.6f}")
```

### Step 2: Per-Iteration Comparison at WS7

Dump Python state at each WS7 RIGID iteration and compare to C++:

```python
# In optimizer.py:optimize(), add:
for rigid_iter in range(rigid_iterations):
    # ... compute mean_shift, J_rigid, delta_p ...

    if window_size == 7:
        print(f"WS7 RIGID iter {rigid_iter}:")
        print(f"  delta_p: {delta_p_rigid}")
        print(f"  mean_shift norm: {np.linalg.norm(mean_shift)}")
        print(f"  params after: {params[:6]}")
```

### Step 3: Verify Response Map Consistency

Response maps are computed ONCE at initial landmarks. Verify they're identical to C++:

```python
# In optimizer.py:_precompute_response_maps()
# Add dump of response_maps[36] to compare with C++ dump
```

### Step 4: Track Rotation Matrix Precision

Add logging to `pdm.update_params`:

```python
def update_params(self, params, delta_p):
    # ... existing code ...

    # Add precision tracking:
    R_before = self._euler_to_rotation_matrix(euler_current)
    R_after = self._euler_to_rotation_matrix(euler_new)
    det_before = np.linalg.det(R_before)
    det_after = np.linalg.det(R_after)
    print(f"Rotation det: before={det_before:.10f}, after={det_after:.10f}")
```

---

## C++ Debug Dump Location

We added a comprehensive debug dump in `LandmarkDetectorModel.cpp`:

```cpp
// Lines 1686-1774: Pre-solve dump
// Lines 2213-2293: Post-solve dump
static bool ws7_dump_done = false;
if(n == 68 && resp_size == 7 && iter == 0 && rigid && !ws7_dump_done) {
    ws7_dump_done = true;
    FILE* f = fopen("/tmp/cpp_ws7_rigid_iter0_dump.txt", "w");
    // Dumps: current_global, sim_img_to_ref, sim_ref_to_img,
    //        response_maps, mean_shifts, J, gradient, Hessian, delta_p
}
```

Output location: `/tmp/cpp_ws7_rigid_iter0_dump.txt`

---

## Python Comparison Tool

`compare_ws7_final.py` loads C++ dump and compares with Python:

```python
# Key code from compare_ws7_final.py:
CPP_MEAN_SHIFTS = np.array([...])  # From dump

# Transform mean-shifts from reference to image coordinates
ms_img = CPP_MEAN_SHIFTS @ CPP_SIM_REF_TO_IMG.T

# Compute gradient
J = pdm.compute_jacobian_rigid(params)
gradient = J.T @ W @ ms_vector

# Compare delta_p
py_delta_p = np.linalg.solve(hessian, gradient)
compare("delta_p", CPP_DELTA_P, py_delta_p)
```

---

## Files to Examine

1. **`optimizer.py`**: Main optimization loop, mean-shift computation, solve
2. **`pdm.py`**: Jacobian computation, parameter updates, rotation composition
3. **`compare_ws7_final.py`**: C++ comparison tool (reference implementation)
4. **`/tmp/cpp_ws7_rigid_iter0_dump.txt`**: C++ debug dump (if regenerated)

---

## Summary

The Python implementation is functionally correct but accumulates small numerical errors through rotation composition. The ~0.01 px error at WS7 appears to be a precision limitation rather than a bug. However, the 70x error accumulation from WS11 to WS7 warrants further investigation to confirm this is purely numerical or if there's a subtle algorithmic difference.

**Priority areas**:
1. Per-iteration state comparison at WS7
2. Rotation matrix precision tracking
3. Response map identity verification
4. Transform scale sensitivity analysis

---

## Working Accuracy Test

The accuracy test infrastructure is working. Run:

```bash
cd /Users/johnwilsoniv/Documents/SplitFace\ Open3/pyclnf
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 run_accuracy_test.py
```

Or with a specific image:
```bash
python3 run_accuracy_test.py /path/to/face/image.jpg
```

**What the test does:**
1. Runs C++ OpenFace to get ground truth landmarks
2. Detects face with MTCNN (CoreML backend)
3. Runs Python CLNF optimization
4. Tracks error at each window size (WS11, WS9, WS7, WS5)
5. Shows error progression and worst landmarks

**Required paths:**
- MTCNN: `/Users/johnwilsoniv/Documents/SplitFace Open3/pymtcnn`
- pyCLNF: `/Users/johnwilsoniv/Documents/SplitFace Open3/pyclnf`
- C++ OpenFace: `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction`
