# PyCLNF

**Pure Python implementation of CLNF (Constrained Local Neural Fields) facial landmark detector**

[![PyPI version](https://badge.fury.io/py/pyclnf.svg)](https://badge.fury.io/py/pyclnf)
[![Python](https://img.shields.io/pypi/pyversions/pyclnf.svg)](https://pypi.org/project/pyclnf/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A pure Python implementation of OpenFace's CLNF facial landmark detector with built-in PyMTCNN face detection. Uses exported OpenFace trained models with no C++ dependencies, making it perfect for cross-platform deployment and PyInstaller distribution.

## Features

- **100% Pure Python**: No C++ compilation required
- **Built-in Face Detection**: Integrated PyMTCNN for automatic face detection
- **OpenFace Compatible**: Uses original OpenFace CEN (Convolutional Expert Network) patch experts
- **Cross-Platform**: Works on Windows, macOS, Linux
- **68-Point Landmarks**: Full facial landmark detection
- **Sub-pixel Accuracy**: 0.06-0.17 px mean error vs C++ OpenFace (video mode)
- **No GPU Required**: CPU-based inference
- **Simple API**: One-line face detection and landmark fitting

## Installation

```bash
pip install pyclnf
```

This automatically installs PyMTCNN as a dependency for face detection.

## Quick Start

### Automatic Face Detection (Recommended)

```python
from pyclnf import CLNF
import cv2

# Initialize with built-in PyMTCNN detector
clnf = CLNF()

# Load image
image = cv2.imread("face.jpg")

# Detect face and fit landmarks in one call
landmarks, info = clnf.detect_and_fit(image)

# landmarks: (68, 2) array of (x, y) coordinates
# info: {'converged': bool, 'iterations': int, 'bbox': tuple, ...}

print(f"Detected {len(landmarks)} landmarks")
print(f"Face bbox: {info['bbox']}")
print(f"Converged: {info['converged']} in {info['iterations']} iterations")
```

### Manual Bounding Box with PyMTCNN

For maximum accuracy matching C++ OpenFace, use PyMTCNN with bbox calibration:

```python
from pymtcnn import MTCNN
from pyclnf import CLNF
import cv2

# Initialize detectors
mtcnn = MTCNN(backend='coreml')  # or 'onnx' for cross-platform
clnf = CLNF()

# Detect face with PyMTCNN
image = cv2.imread("face.jpg")
boxes, landmarks_5pt = mtcnn.detect(image)

if boxes is not None and len(boxes) > 0:
    # PyMTCNN returns (x, y, w, h, score) format
    raw_x, raw_y, raw_w, raw_h = boxes[0][:4]

    # Apply OpenFace bbox calibration (matches C++ FaceDetectorMTCNN.cpp)
    cal_x = raw_x + raw_w * (-0.0075)
    cal_y = raw_y + raw_h * 0.2459
    cal_w = raw_w * 1.0323
    cal_h = raw_h * 0.7751
    bbox = (cal_x, cal_y, cal_w, cal_h)

    # Fit CLNF landmarks
    landmarks_68, info = clnf.fit(image, face_bbox=bbox, detector_type=None)

    print(f"68 landmarks detected: {landmarks_68.shape}")
```

### Video Processing (Recommended for Videos)

**IMPORTANT**: For video processing, use `convergence_profile='video'` to enable template tracking and achieve sub-pixel accuracy matching C++ OpenFace:

```python
from pymtcnn import MTCNN
from pyclnf import CLNF
import cv2

# Initialize with VIDEO profile for temporal tracking
mtcnn = MTCNN(backend='coreml')
clnf = CLNF(convergence_profile='video')  # Enables template tracking!

# Open video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face
    boxes, _ = mtcnn.detect(frame)
    if boxes is None or len(boxes) == 0:
        continue

    # Apply bbox calibration
    raw_x, raw_y, raw_w, raw_h = boxes[0][:4]
    cal_x = raw_x + raw_w * (-0.0075)
    cal_y = raw_y + raw_h * 0.2459
    cal_w = raw_w * 1.0323
    cal_h = raw_h * 0.7751

    # Fit landmarks (template tracking happens automatically)
    landmarks, info = clnf.fit(frame, face_bbox=(cal_x, cal_y, cal_w, cal_h))

    # Use landmarks...

cap.release()

# Reset state when switching videos or faces
clnf.reset_temporal_state()
```

**Video mode features:**
- **Template tracking**: Uses face template from previous frame for translation correction
- **Adaptive windows**: Smaller search windows after first frame (faster)
- **Failure recovery**: Switches to larger windows after detection failure
- **Temporal warm-start**: Uses previous frame params as initialization

**Accuracy comparison (100 frames):**
| Mode | Mean Error | Jaw Error |
|------|------------|-----------|
| Default | 0.67 px | 1.50 px |
| **Video** | **0.07 px** | **0.10 px** |

## Architecture

```
Input Image
    ↓
Face Detection (PyMTCNN)
    ↓
Bounding Box Calibration (OpenFace coefficients)
    ↓
PDM Initialization (Mean Shape → Bbox)
    ↓
CLNF Optimization (4 window sizes: 11 → 9 → 7 → 5)
  ├── CEN Patch Experts (4 scales: 0.25, 0.35, 0.5, 1.0)
  ├── NU-RLMS Optimizer (rigid + non-rigid phases)
  ├── Sparse Response Maps with Interpolation
  └── Shape Constraints (PCA-based PDM)
    ↓
68 Facial Landmarks
```

### Pipeline Details

1. **Face Detection**: PyMTCNN detects faces and returns bounding boxes in `(x, y, w, h)` format

2. **Bbox Calibration**: OpenFace calibration coefficients adjust the raw MTCNN bbox:
   ```
   cal_x = x + w × (-0.0075)
   cal_y = y + h × 0.2459
   cal_w = w × 1.0323
   cal_h = h × 0.7751
   ```

3. **PDM Initialization**: Scale and translation computed from calibrated bbox

4. **Multi-Scale Optimization**: Coarse-to-fine refinement through 4 window sizes:
   - WS11 @ scale 0.25 (coarsest)
   - WS9 @ scale 0.35
   - WS7 @ scale 0.50
   - WS5 @ scale 1.00 (finest)

5. **Per-Scale Phases**:
   - **Rigid phase** (10 iterations): Optimizes global pose (scale, rotation, translation)
   - **Non-rigid phase** (5 iterations): Optimizes local shape deformations

## Components

### 1. PDM (Point Distribution Model)
Statistical shape model using PCA to represent plausible facial configurations.

- **68 landmarks** (3D coordinates)
- **34 shape parameters** (principal components)
- Rodrigues rotation for 3D → 2D projection

### 2. CEN Patch Experts
Convolutional Expert Network patch experts for landmark localization.

- **272 patch experts** (68 landmarks × 4 scales)
- **4 scales**: 0.25, 0.35, 0.5, 1.0
- Sparse response computation with interpolation (matches C++ exactly)
- Sigma components for spatial correlation modeling
- **Model size**: ~410MB

### 3. NU-RLMS Optimizer
Non-Uniform Regularized Landmark Mean-Shift optimization.

- **Rigid phase**: 10 iterations for global pose
- **Non-rigid phase**: 5 iterations for local shape (fewer iterations avoids jaw divergence)
- Precomputed KDE grids for mean-shift (0.1 pixel spacing)
- Scale-adaptive regularization and sigma
- Convergence threshold: 0.01 (shape change norm)

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.06-0.17 px mean error vs C++ OpenFace |
| **Eye landmarks** | 0.05-0.09 px error |
| **Jaw landmarks** | 0.08-0.44 px error |
| **Speed** | ~2-3 FPS (pure NumPy) |
| **Convergence** | 95%+ on frontal faces |
| **Model Size** | ~410MB (CEN patch experts) |

### Accuracy Validation

Tested on multiple images against C++ OpenFace (video mode, weight_factor=0):

| Image | Mean Error | Jaw Error | Eye Error |
|-------|------------|-----------|-----------|
| test_frame_mtcnn.png | 0.06 px | 0.08 px | 0.05 px |
| test_frame.png | 0.09 px | 0.17 px | 0.05 px |
| test_face_clean.png | 0.17 px | 0.44 px | 0.09 px |

## Use Cases

- **Facial Paralysis Research**: High-accuracy landmark tracking for AU extraction
- **Medical Applications**: Quantitative facial analysis
- **Cross-Platform Tools**: No C++ compilation hassle
- **PyInstaller Apps**: Bundle models with executable
- **Offline Processing**: Batch video analysis

## API Reference

### `CLNF(model_dir, detector, convergence_profile, ...)`

Initialize CLNF landmark detector.

**Parameters:**
- `model_dir` (str): Path to model directory (default: "pyclnf/models")
- `detector` (bool): Enable PyMTCNN face detector (default: True)
- `convergence_profile` (str): Optimization profile (default: None)
  - `None`: Default settings for single images
  - `'video'`: **Recommended for videos** - enables template tracking, adaptive windows, failure recovery
  - `'accurate'`: More iterations for maximum accuracy
  - `'fast'`: Fewer iterations for speed
- `regularization` (float): Shape constraint weight (default: 22.5)
- `sigma` (float): KDE kernel sigma (default: 2.25)
- `window_sizes` (list): Hierarchical window sizes (default: [11, 9, 7, 5])

**Optimizer Settings** (access via `clnf.optimizer`):
- `rigid_iterations` (int): Rigid phase iterations (default: 10)
- `nonrigid_iterations` (int): Non-rigid phase iterations (default: 5)
- `convergence_threshold` (float): Convergence threshold (default: 0.005)

### `fit(image, face_bbox, initial_params, return_params)`

Fit CLNF model to detect landmarks from a bounding box.

**Parameters:**
- `image` (ndarray): Input image (BGR or grayscale)
- `face_bbox` (tuple): Face bounding box [x, y, width, height]
- `initial_params` (ndarray, optional): Initial PDM parameters
- `return_params` (bool): Include optimized parameters in output

**Returns:**
- `landmarks` (ndarray): 68-point landmarks (68, 2)
- `info` (dict): Fitting information
  - `converged` (bool): Whether optimization converged
  - `iterations` (int): Number of iterations performed
  - `final_update` (float): Final parameter update magnitude

### `detect_and_fit(image, return_all_faces, return_params)`

Detect face and fit landmarks in one call (requires built-in detector).

**Parameters:**
- `image` (ndarray): Input image
- `return_all_faces` (bool): Return results for all faces (default: False)
- `return_params` (bool): Include optimized parameters

**Returns:**
- `landmarks` (ndarray): 68-point landmarks for first/largest face
- `info` (dict): Fitting information including 'bbox'

### `reset_temporal_state()`

Reset video mode tracking state. **Call this when:**
- Starting a new video
- Switching to a different face
- After extended tracking failure

This clears the template tracking, warm-start parameters, and failure counters.

## Model Files

PyCLNF uses OpenFace's trained CEN models in binary format:

```
pyclnf/models/
├── exported_pdm/               # Point Distribution Model
│   ├── mean_shape.npy
│   ├── eigenvectors.npy
│   └── eigenvalues.npy
├── cen_patches_0.25_of.dat     # Scale 0.25 patch experts (~100MB)
├── cen_patches_0.35_of.dat     # Scale 0.35 patch experts (~100MB)
├── cen_patches_0.5_of.dat      # Scale 0.50 patch experts (~100MB)
├── cen_patches_1.0_of.dat      # Scale 1.00 patch experts (~100MB)
└── sigma_components/           # Spatial correlation matrices for KDE
    ├── sigma_components_ws7.npy
    ├── sigma_components_ws9.npy
    ├── sigma_components_ws11.npy
    └── sigma_components_ws15.npy
```

## Requirements

- Python >= 3.8
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- PyMTCNN >= 1.0.0 (for face detection)

## Wheel Distribution

**Pure Python - Universal Wheel**

PyCLNF is 100% pure Python with no compiled extensions, so it can be distributed as a single universal wheel (`py3-none-any.whl`) that works on all platforms:

- Windows (x86, x64, ARM)
- macOS (Intel, Apple Silicon)
- Linux (x86_64, ARM64, etc.)

No platform-specific wheels needed!

## Citation

If you use PyCLNF in your research, please cite OpenFace:

```bibtex
@inproceedings{baltrusaitis2018openface,
  title={OpenFace 2.0: Facial behavior analysis toolkit},
  author={Baltru{\v{s}}aitis, Tadas and Zadeh, Amir and Lim, Yao Chong and Morency, Louis-Philippe},
  booktitle={2018 13th IEEE international conference on automatic face \& gesture recognition (FG 2018)},
  pages={59--66},
  year={2018},
  organization={IEEE}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenFace for the original C++ implementation and trained models
- Tadas Baltru{\v{s}}aitis et al. for the CLNF algorithm
- Multi-PIE dataset for patch expert training

## Links

- **PyPI**: https://pypi.org/project/pyclnf/
- **GitHub**: https://github.com/johnwilsoniv/pyclnf
- **Related**: [PyMTCNN](https://github.com/johnwilsoniv/pymtcnn) (companion face detector)
