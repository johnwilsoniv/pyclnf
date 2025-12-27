"""
PyCLNF - Pure Python CLNF (Constrained Local Neural Fields) implementation

A pure Python implementation of OpenFace's CLNF facial landmark detector.
Uses exported OpenFace models with no C++ dependencies, making it perfect
for PyInstaller distribution and cross-platform deployment.

Usage:
    from pyclnf import CLNF

    # Initialize model
    clnf = CLNF(model_dir="pyclnf/models")

    # Detect landmarks
    landmarks, info = clnf.fit(image, face_bbox)

Components:
    - PDM: Point Distribution Model (statistical shape model)
    - CCNF: Patch experts for landmark detection
    - NU-RLMS: Optimization algorithm for fitting
    - CLNF: Complete pipeline
"""

from .clnf import CLNF
from .core import PDM, CCNFModel, CCNFPatchExpert, NURLMSOptimizer

__version__ = "0.3.2"
__all__ = [
    'CLNF',
    'PDM',
    'CCNFModel',
    'CCNFPatchExpert',
    'NURLMSOptimizer',
]
