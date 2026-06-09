# cpp_warp - C++ warpAffine extension for exact OpenCV matching
#
# The C++ extension binds cv::warpAffine from the OpenCV the wheel was built
# against (Homebrew 4.12 on macOS, vendored 4.12 on Windows), giving bit-exact
# patch extraction against C++ OpenFace. Bit-exactness is the whole reason
# this module exists -- a pure-Python cv2.warpAffine fallback would silently
# change numerics and is intentionally NOT provided.
#
# If the platform-specific extension isn't packaged in the installed wheel
# (e.g. you're on Windows but only the macOS .so shipped), every function
# below raises ImportError on first use with a clear remediation message.
# This is preferable to extract_aoi = None (the prior behavior) which
# crashed deep inside the CLNF optimizer with a confusing "'NoneType'
# object is not callable" later on.

# On Windows the cpp_warp .pyd depends on opencv_world<ver>.dll which we
# ship alongside the .pyd. Windows' default DLL search order checks the
# process's exe directory, not the .pyd's directory, so we have to add this
# package directory explicitly before importing the extension.
import os as _os
import sys as _sys

if _sys.platform == "win32":
    _here = _os.path.dirname(_os.path.abspath(__file__))
    try:
        _os.add_dll_directory(_here)
    except (AttributeError, OSError):
        # add_dll_directory is 3.8+; OSError fires if the path is missing
        pass

try:
    from .cpp_warp import (
        extract_aoi,
        warp_affine,
        get_opencv_version,
        get_opencv_build_info,
    )
    BACKEND = "cpp"
    _IMPORT_ERROR = None
except ImportError as _e:
    BACKEND = "missing"
    _IMPORT_ERROR = _e

    def _missing_cpp_warp(*args, **kwargs):
        raise ImportError(
            "pyclnf.cpp_warp C++ extension not available on this platform. "
            f"Underlying error: {_IMPORT_ERROR!r}. "
            "Install a platform-matched pyclnf wheel that includes "
            "cpp_warp.<tag>.{so,pyd}, or build the extension locally via "
            "cpp_warp/build.sh (macOS/Linux) or cpp_warp/build.ps1 (Windows). "
            "CLNF only needs this on the CPU path (CLNF_CONFIG['use_gpu']=False); "
            "use_gpu=True routes around cpp_warp."
        )

    extract_aoi = _missing_cpp_warp
    warp_affine = _missing_cpp_warp
    get_opencv_version = _missing_cpp_warp
    get_opencv_build_info = _missing_cpp_warp
