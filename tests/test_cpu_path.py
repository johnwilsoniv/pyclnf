"""The CPU path (use_gpu=False) must work on any machine.

The CPU path used to extract patches with a compiled cpp_warp extension linked
against Homebrew's OpenCV 4.12, so every fit failed on other machines.
It now uses cv2.warpAffine through pyclnf.core.utils.extract_aoi, the same
function as the batched (GPU) path.
"""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("numba")


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
