# PyCLNF

**Pure Python implementation of CLNF (Constrained Local Neural Fields) facial landmark detector**

[![PyPI version](https://badge.fury.io/py/pyclnf.svg)](https://badge.fury.io/py/pyclnf)
[![Python](https://img.shields.io/pypi/pyversions/pyclnf.svg)](https://pypi.org/project/pyclnf/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A pure Python implementation of OpenFace's CLNF facial landmark detector. Uses exported OpenFace trained models with no C++ dependencies, making it perfect for cross-platform deployment and PyInstaller distribution.

## Features

- **100% Pure Python**: No C++ compilation required
- **OpenFace Compatible**: Uses original OpenFace trained models (CCNF patch experts)
- **Cross-Platform**: Works on Windows, macOS, Linux
- **68-Point Landmarks**: Full facial landmark detection
- **8.23px Accuracy**: Validated against C++ OpenFace
- **No GPU Required**: CPU-based inference
- **Easy Integration**: Simple API, works with any face detector

## Installation

```bash
pip install pyclnf
```

## Quick Start

```python
from pyclnf import CLNF
import cv2

# Initialize detector
clnf = CLNF()

# Load image
image = cv2.imread("face.jpg")

# Detect landmarks (requires face bounding box)
face_bbox = (100, 100, 200, 250)  # [x, y, width, height]
landmarks, info = clnf.fit(image, face_bbox)

# landmarks: (68, 2) array of (x, y) coordinates
# info: {'converged': bool, 'iterations': int, ...}

print(f"Detected {len(landmarks)} landmarks")
print(f"Converged: {info['converged']} in {info['iterations']} iterations")
```

## With Automatic Face Detection

PyCLNF includes optional RetinaFace integration for automatic face detection:

```python
from pyclnf import CLNF

# Initialize with built-in face detector
clnf = CLNF(detector="retinaface", use_coreml=True)  # CoreML for ARM Mac speedup

# Detect face and landmarks in one call
landmarks, info = clnf.detect_and_fit(image)

print(f"Face bbox: {info['bbox']}")
print(f"Landmarks: {landmarks.shape}")
```

## Integration with PyMTCNN

For best results with facial paralysis research, use with [PyMTCNN](https://github.com/johnwilsoniv/pymtcnn):

```python
from pymtcnn import PyMTCNN
from pyclnf import CLNF
import cv2

# Initialize detectors
mtcnn = PyMTCNN()
clnf = CLNF(detector=None)  # No built-in detector

# Detect face with MTCNN
image = cv2.imread("face.jpg")
bboxes, landmarks_5 = mtcnn.detect(image, return_landmarks=True)

if len(bboxes) > 0:
    bbox = bboxes[0]  # Use first face

    # Refine with CLNF
    landmarks_68, info = clnf.fit(image, bbox)

    print(f"68 landmarks detected: {landmarks_68.shape}")
```

## Architecture

```
Input Image
    ↓
Face Detection (MTCNN / RetinaFace / Manual)
    ↓
Face Bounding Box
    ↓
PDM Initialization (Mean Shape → Bbox)
    ↓
CLNF Optimization
  ├── CCNF Patch Experts (3 scales: 0.25, 0.35, 0.5)
  ├── NU-RLMS Optimizer
  ├── Hierarchical Refinement (3 window sizes)
  └── Shape Constraints (PCA-based PDM)
    ↓
68 Facial Landmarks
```

## Components

### 1. PDM (Point Distribution Model)
Statistical shape model using PCA to represent plausible facial configurations.

- **68 landmarks** (3D coordinates)
- **34 shape parameters** (principal components)
- Rodrigues rotation for 3D → 2D projection

### 2. CCNF Patch Experts
Likelihood-based landmark position estimators trained on Multi-PIE dataset.

- **1,032 patch experts** (344 per scale × 3 scales)
- **3 scales**: 0.25, 0.35, 0.5
- **7 views**: Multi-angle face support
- **Model size**: 33MB (NumPy .npz files)

### 3. NU-RLMS Optimizer
Non-Uniform Regularized Landmark Mean-Shift optimization.

- Iterative refinement in PDM parameter space
- Hierarchical coarse-to-fine optimization
- Patch confidence weighting
- Converges in ~5-10 iterations

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 8.23px mean error vs C++ OpenFace |
| **Speed** | ~2-3 FPS (pure NumPy) |
| **Convergence** | 95%+ on frontal faces |
| **Model Size** | 47MB |

## Use Cases

- **Facial Paralysis Research**: High-accuracy landmark tracking for AU extraction
- **Medical Applications**: Quantitative facial analysis
- **Cross-Platform Tools**: No C++ compilation hassle
- **PyInstaller Apps**: Bundle models with executable
- **Offline Processing**: Batch video analysis

## API Reference

### `CLNF(model_dir, detector, use_coreml, ...)`

Initialize CLNF landmark detector.

**Parameters:**
- `model_dir` (str): Path to model directory (default: "pyclnf/models")
- `detector` (str): Face detector ("retinaface" or None)
- `use_coreml` (bool): Enable CoreML acceleration for RetinaFace
- `regularization` (float): Shape constraint weight (default: 25.0)
- `max_iterations` (int): Max optimization iterations (default: 10)
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

## Model Files

PyCLNF uses OpenFace's trained CCNF models, exported to NumPy format:

```
pyclnf/models/
├── exported_pdm/               # Point Distribution Model (36KB)
│   ├── mean_shape.npy
│   ├── eigenvectors.npy
│   └── eigenvalues.npy
├── exported_ccnf_0.25/         # Coarse scale (11MB)
├── exported_ccnf_0.35/         # Medium scale (11MB)
├── exported_ccnf_0.5/          # Fine scale (11MB)
└── sigma_components/           # KDE kernels for mean-shift
```

## Requirements

- Python >= 3.8
- NumPy >= 1.19.0
- OpenCV >= 4.5.0

Optional:
- PyMTCNN (for face detection)

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
