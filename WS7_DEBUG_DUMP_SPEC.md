# WS7 RIGID Iteration 0 Debug Dump Specification

## Goal
Export ALL intermediate values from C++ WS7 RIGID iteration 0 to enable exact comparison with Python, finding the precise divergence point.

## Dump File Location
`/tmp/cpp_ws7_rigid_iter0_dump.bin` (binary) + `/tmp/cpp_ws7_rigid_iter0_dump.txt` (human-readable)

---

## Variables to Dump

### 1. Input State (Before Optimization)
| Variable | Shape | Description |
|----------|-------|-------------|
| `initial_global` | 6 | scale, rot_x, rot_y, rot_z, tx, ty |
| `initial_local` | 34 | Shape coefficients |
| `current_global` | 6 | Same as initial for iter 0 |
| `current_local` | 34 | Same as initial for iter 0 |

### 2. Shapes
| Variable | Shape | Description |
|----------|-------|-------------|
| `current_shape` | 136 | Landmarks as column vector [x0..x67, y0..y67] |
| `base_shape` | 136 | Base landmarks (fixed for this window size) |
| `current_shape_2D` | 68x2 | Reshaped landmarks |
| `base_shape_2D` | 68x2 | Reshaped base landmarks |

### 3. Similarity Transforms
| Variable | Shape | Description |
|----------|-------|-------------|
| `sim_img_to_ref` | 2x2 | Image to reference transform |
| `sim_ref_to_img` | 2x2 | Reference to image transform |

### 4. Offset Computation
| Variable | Shape | Description |
|----------|-------|-------------|
| `offsets` | 68x2 | `(current_shape_2D - base_shape_2D) * sim_img_to_ref.t()` |
| `dxs` | 68 | `offsets[:, 0] + (resp_size-1)/2` |
| `dys` | 68 | `offsets[:, 1] + (resp_size-1)/2` |

### 5. Response Maps (Select 3 landmarks)
| Variable | Shape | Description |
|----------|-------|-------------|
| `resp_4` | 7x7 | Response map for LM 4 (jaw) |
| `resp_36` | 7x7 | Response map for LM 36 (left eye outer) |
| `resp_48` | 7x7 | Response map for LM 48 (mouth left) |

### 6. Mean-Shift Computation
| Variable | Shape | Description |
|----------|-------|-------------|
| `sigma` | 1 | `parameters.sigma` value |
| `a` | 1 | `-0.5/(sigma*sigma)` |
| `mean_shifts` | 136 | All mean-shifts [ms_x0..ms_x67, ms_y0..ms_y67] |
| `mean_shifts_for_lm4` | 2 | Mean-shift for landmark 4 |
| `mean_shifts_for_lm36` | 2 | Mean-shift for landmark 36 |
| `mean_shifts_for_lm48` | 2 | Mean-shift for landmark 48 |

### 7. Jacobian Computation
| Variable | Shape | Description |
|----------|-------|-------------|
| `WeightMatrix_diag` | 68 | Diagonal of weight matrix |
| `J` | 136x6 | Full rigid Jacobian |
| `J_w_t` | 6x136 | Weighted transposed Jacobian |

### 8. Solve Step
| Variable | Shape | Description |
|----------|-------|-------------|
| `J_w_t_m` | 6 | `J_w_t * mean_shifts` (gradient) |
| `regTerm` | 6x6 | Regularization (all zeros for RIGID) |
| `Hessian` | 6x6 | `J_w_t * J + regTerm` |
| `param_update` | 6 | `solve(Hessian, J_w_t_m)` (delta_p) |

### 9. After Parameter Update
| Variable | Shape | Description |
|----------|-------|-------------|
| `new_global` | 6 | Updated global params |
| `new_local` | 34 | Updated local params (unchanged for RIGID) |
| `new_shape` | 136 | New landmarks after update |

---

## C++ Code Location for Dump

Add dump code in `LandmarkDetectorModel.cpp` at line ~1684 (after mean-shift) and ~2119 (after solve):

```cpp
// After line 1684 (mean_shifts computed)
if(n == 68 && resp_size == 7 && iter == 0 && rigid) {
    FILE* f = fopen("/tmp/cpp_ws7_rigid_iter0_dump.txt", "w");
    if(f) {
        // Dump all variables here
        fclose(f);
    }
}
```

---

## Python Comparison Tool

Create `compare_ws7_dump.py` that:
1. Loads C++ dump
2. Runs Python WS7 RIGID iter 0 from identical starting state
3. Compares each variable element-by-element
4. Reports first divergence point with magnitude

### Expected Output
```
=== WS7 RIGID Iter 0 Comparison ===

INPUT STATE:
  current_global: MATCH (max_diff=0.0000)
  current_local:  MATCH (max_diff=0.0000)

SHAPES:
  current_shape:  MATCH (max_diff=0.0000)
  base_shape:     MATCH (max_diff=0.0000)

TRANSFORMS:
  sim_img_to_ref: MATCH (max_diff=0.0000)

OFFSETS:
  offsets:        MATCH (max_diff=0.0000)
  dxs:            MATCH (max_diff=0.0000)
  dys:            MATCH (max_diff=0.0000)

MEAN-SHIFTS:
  LM4:            DIVERGE! cpp=(0.1234, 0.5678) py=(0.1230, 0.5680) diff=0.0005
  LM36:           ...

>>> First divergence at MEAN-SHIFTS for LM4
>>> Probable cause: Response map processing or KDE computation
```

---

## Files to Modify/Create

1. **C++ (modify)**: `LandmarkDetectorModel.cpp`
   - Add WS7 dump block at lines ~1684 and ~2119

2. **Python (create)**: `compare_ws7_dump.py`
   - Load C++ dump and run Python comparison

3. **C++ rebuild**: After modification, rebuild OpenFace binary

---

## Precision Notes

- All float values dumped with 10 decimal places
- Binary dump uses `double` for lossless transfer
- Python loads with `np.float64` then converts to `np.float32` for comparison
