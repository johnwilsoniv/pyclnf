# cpp_warp - C++ warpAffine extension for exact OpenCV matching
try:
    from .cpp_warp import extract_aoi, warp_affine
except ImportError:
    # Fallback if .so not available for this platform
    extract_aoi = None
    warp_affine = None
