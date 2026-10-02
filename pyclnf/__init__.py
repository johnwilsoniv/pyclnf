"""
PyCLNF - Pure Python CLNF (Constrained Local Neural Fields) implementation

A pure Python implementation of OpenFace's CLNF facial landmark detector.

OpenFace's model files are not included: run `pyclnf-download-models` once (or
call pyclnf.models.ensure_models(accept_license=True)) to download them from
OpenFace's official sources and prepare them on this computer.

Usage:
    from pyclnf import CLNF

    # Initialize model (uses the installed OpenFace model files)
    clnf = CLNF()

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

__version__ = "0.3.4"
__all__ = [
    'CLNF',
    'PDM',
    'CCNFModel',
    'CCNFPatchExpert',
    'NURLMSOptimizer',
]
