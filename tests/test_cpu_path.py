"""The CPU path (use_gpu=False, or torch not installed) must work on any machine.

The CPU path used to extract patches with a compiled cpp_warp extension linked
against Homebrew's OpenCV 4.12, so every fit failed on other machines.
It now uses cv2.warpAffine through pyclnf.core.utils.extract_aoi, the same
function as the batched (GPU) path.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("numba")

REPO = Path(__file__).resolve().parents[1]


def _random_calls(n, w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        scale = float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))
        angle = rng.uniform(-np.pi, np.pi)
        a, b = scale * np.cos(angle), scale * np.sin(angle)
        sim = np.array([[a, -b, 0.0], [b, a, 0.0]])
        cx, cy = rng.uniform(-20, w + 20), rng.uniform(-20, h + 20)
        yield cx, cy, sim, int(rng.choice([15, 17, 19, 21]))


def test_extract_aoi_is_openfaces_warp():
    """OpenFace's warp (Patch_experts.cpp): float 2x3 matrix, inverse map, bilinear, border 0."""
    from pyclnf.core.utils import extract_aoi
    rng = np.random.default_rng(1)
    image = (rng.random((240, 320)) * 255).astype(np.float32)
    for cx, cy, sim, aoi in _random_calls(200):
        a1, b1, c = sim[0, 0], -sim[0, 1], (aoi - 1.0) / 2.0
        matrix = np.array([[a1, -b1, cx - a1 * c + b1 * c],
                           [b1, a1, cy - a1 * c - b1 * c]], dtype=np.float32)
        expected = cv2.warpAffine(image, matrix, (aoi, aoi),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        got = extract_aoi(image, cx, cy, sim, aoi)
        assert got.dtype == np.float32 and got.shape == (aoi, aoi)
        assert np.array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_cpu_and_batched_paths_extract_identical_patches(dtype):
    from pyclnf.core.batched_cen import BatchedCEN
    from pyclnf.core.optimizer import NURLMSOptimizer
    optimizer = NURLMSOptimizer(use_gpu=False)
    batched = BatchedCEN.__new__(BatchedCEN)  # AOI extraction does not need the patch experts
    rng = np.random.default_rng(2)
    image = (rng.random((240, 320)) * 255).astype(dtype)
    for cx, cy, sim, aoi in _random_calls(50, seed=3):
        landmarks = np.tile([cx, cy], (68, 1))
        aois, valid = batched._batch_extract_aoi_warped(image, landmarks, aoi, sim)
        cpu = optimizer._extract_aoi(image, cx, cy, sim, aoi)
        assert valid.all() and np.array_equal(aois[0], cpu)


def _run_without_torch(code):
    header = "import sys\nsys.modules['torch'] = None  # behave as if torch were not installed\n"
    code = header + textwrap.dedent(code)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO / "tests", env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr
    return r


def test_optimizer_without_torch_falls_back_to_the_cpu_path():
    _run_without_torch("""
        import warnings
        from pyclnf import NURLMSOptimizer
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = NURLMSOptimizer(use_gpu=True, gpu_device="auto")
        assert not opt.use_gpu and opt.gpu_device == "cpu"
        assert any("torch" in str(w.message) for w in caught), [str(w.message) for w in caught]
        assert not NURLMSOptimizer(use_gpu=True, gpu_device="mps").use_gpu
        assert NURLMSOptimizer(use_gpu=False, gpu_device="auto").gpu_device == "cpu"
        print("ok")
    """)


def test_cpu_path_fits_without_torch():
    from pyclnf import models
    if not models.models_ready():
        pytest.skip("OpenFace model files are not installed (run pyclnf-download-models)")
    _run_without_torch("""
        import warnings
        import numpy as np
        import pyclnf.core.optimizer as optimizer
        from pyclnf import CLNF

        dtypes = set()
        real_extract = optimizer.extract_aoi
        def recording_extract(image, *args):
            dtypes.add(image.dtype)
            return real_extract(image, *args)
        optimizer.extract_aoi = recording_extract

        rng = np.random.default_rng(0)
        gray = np.clip(rng.normal(128, 40, (240, 320)), 0, 255).astype(np.uint8)
        for kwargs in ({}, {"use_gpu": False}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clnf = CLNF(detector=None, use_validator=False, **kwargs)
            assert not clnf.optimizer.use_gpu
            assert clnf.use_eye_refinement and not clnf.eye_model.use_gpu
            if kwargs.get("use_gpu", True):
                assert any("torch" in str(w.message) for w in caught)
            landmarks, info = clnf.fit(gray, (100, 60, 120, 120))
            assert landmarks.shape == (68, 2) and np.isfinite(landmarks).all()
        # patches come from the float32 image, like OpenFace
        assert dtypes == {np.dtype(np.float32)}, dtypes
        print("ok")
    """)
