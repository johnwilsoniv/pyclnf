# Changelog

## 0.4.1 (2026-10-03)

### What changes for you

- **pyclnf works without PyTorch.** PyTorch (`torch`) is optional. With it, pyclnf uses your
  graphics processor (Apple Silicon or NVIDIA); without it, pyclnf now runs on the processor
  (CPU) instead of stopping with an error.
- **CPU mode works on every computer.** `CLNF(use_gpu=False)` failed on every frame unless a
  compiled add-on loaded, and that add-on loaded on only one particular Mac setup. The add-on is
  gone, so pyclnf is pure Python again. CPU-mode landmarks are practically unchanged (0.006
  pixels on average on OpenFace's sample images), and CPU mode is about a third faster on large
  images.
- **No more debug files in your temporary folder.** In CPU mode, pyclnf wrote a small debug
  file into your computer's temporary folder for every face it fitted (about four per video
  frame). Debug files are now written only when you ask for them with `CLNF(debug_mode=True)`.
  Landmarks are unchanged.
- **pyclnf stays on OpenCV 4.** Since OpenCV 5 came out, a new installation of pyclnf got
  OpenCV 5. OpenCV 5 resamples images slightly differently, and that changes results: on
  OpenFace's sample videos, run through pyfaceau, landmarks move by up to 2.45 pixels and AU
  intensities by up to 0.54, and 0.4–0.6% of AU presence flags flip. Results would then depend
  on the day pyclnf was installed. pyclnf now asks for OpenCV 4; if you have OpenCV 5,
  `pip install --upgrade pyclnf` puts OpenCV 4 back.
- With PyTorch on a graphics processor and the same OpenCV 4 version, landmarks are exactly
  the same as with 0.4.0.

### Details

- Fixed: the CPU path (`CLNF(use_gpu=False)`, e.g. pyfaceau with `CLNF_CONFIG['use_gpu'] = False`)
  failed on every frame unless pyclnf's compiled `cpp_warp` extension loaded, which needed
  Homebrew's OpenCV 4.12 on a Mac with Python 3.10 (the Windows build was never in the wheel). The
  extension is removed: the CPU path extracts patches with `cv2.warpAffine`, through the same
  function as the GPU path (`pyclnf.core.utils.extract_aoi`). With the same matrix,
  `cv2.warpAffine` (OpenCV 4.10 to 4.14) gives exactly the patches `cpp_warp` gave. The matrix is
  now float32, as in OpenFace's C++ code and the GPU path (cpp_warp used float64), which moves
  about 0.1% of patch pixels by one 1/32-pixel interpolation step (on OpenFace's sample images,
  landmarks move by 0.006 px on average, 0.07 px at most). The CPU path still warps the image as
  float32, like OpenFace, and no longer converts the whole image for every landmark.
  `NURLMSOptimizer(use_cpp_warp=...)` is accepted and ignored.
- Fixed: without torch, `CLNF()` and `CLNF(use_gpu=False)` raised `ModuleNotFoundError: torch`,
  because device detection for `gpu_device='auto'` (the default) imported torch, in the optimizer
  and in the eye model. torch is now imported only for the GPU path, and `use_gpu=True` without
  torch warns and uses the CPU path (before, an explicit `gpu_device` kept the GPU path on and
  `fit` failed with `NameError: torch`).
- Debug output is opt-in everywhere. The eye model's CCNF debug file
  (`python_ccnf_neuron_debug_call<N>.txt`, CPU path) is written only with
  `CLNF(debug_mode=True)` (new `HierarchicalEyeModel(debug_mode=...)`), and the optimizer's debug
  dumps, which were already written only with `debug_mode=True`, go to the system temporary folder
  instead of a hard-coded `/tmp`. `tests/test_debug_output.py` checks that CPU-path and GPU-path
  fits leave the temporary folder empty.
- `opencv-python>=4.5.0,<5` (was `>=4.5.0`). OpenCV 5's `warpAffine` and `remap` no longer snap
  sample positions to 1/32 pixel, so every patch changes on both paths, and no `cv2` flag
  restores the 4.x behaviour.
- `tests/test_accuracy_vs_cpp.py` (the comparison with OpenFace's C++ `FeatureExtraction`) reads
  the recordings folder and the binary from `PYCLNF_TEST_VIDEO_DIR` and
  `OPENFACE_FEATURE_EXTRACTION`, and is skipped with the reason when either, pandas or the model
  files are missing (pytest used to fail on its helper function).

## 0.4.0 (2026-10-02)

OpenFace's model files are no longer included in pyclnf.

- **Breaking:** the package no longer contains OpenFace's model files (until 0.3.4 the wheel
  shipped NumPy exports of them, and the CEN patch experts were downloaded from this project's
  GitHub release). OpenFace's license allows academic or non-profit non-commercial research use
  only and no redistribution, so each user now downloads OpenFace's original files once, from
  OpenFace's official sources, after accepting the license:
  - `pyclnf-download-models` (or `python -m pyclnf.download_models`): shows a license summary,
    asks you to type YES (`--accept-license` for scripts), shows progress, checks the files;
  - `pyclnf.models.ensure_models(accept_license=True, progress=...)` for apps that show their own
    license dialog;
  - `OPENFACE_MODELS_ACCEPT_LICENSE=1` also counts as acceptance.
- Every file is checked against its SHA-256 and kept in a per-user cache shared with pyfaceau and
  pymtcnn (`<user data dir>/OpenFaceModels/2.2.0`, or `$OPENFACE_MODELS_DIR/2.2.0`). After that
  pyclnf works offline.
- The NumPy files pyclnf loads are generated on your computer from the originals, by pyclnf's own
  exporters (`pyclnf/models/_derive.py`). They are byte-for-byte identical to the files shipped in
  0.3.4, and `CLNF.fit` gives identical landmarks.
- `CLNF()` never downloads anything by itself: if the files are missing it raises
  `pyclnf.models.ModelsNotInstalledError` (a `FileNotFoundError`) that explains how to install
  them. `CLNF(model_dir=...)` works as before.
- Removed the download from this project's GitHub release; `python -m pyclnf.model_downloader`
  now runs the new installer.
- No longer produced: exports that pyclnf never loads at runtime (the legacy CCNF patch experts
  `exported_ccnf_*`, their duplicated nested copy, and `exported_pdm_old`). The legacy
  `CCNFModel` class is unchanged but needs exports you create with `pyclnf.models.openface_loader`.
- Fixed: a clean `pip install pyclnf` could not be imported unless numba and torch happened to
  be installed. numba (imported unconditionally) is now a declared dependency, and torch is
  optional again (an eagerly evaluated `torch.Tensor` annotation broke the import without it).
- Fixed: pyclnf no longer changes `sys.path`. It used to import `models.openface_loader` as a
  top-level module (which broke when another module named `models` existed) and to put the
  installed pymtcnn and pyfaceau package folders on `sys.path`.
- `CCNFPatchExpertLoader` has a `verbose` flag and keeps the view centers as stored in the file
  (`raw_centers`); `save_numpy` writes explicit dtypes and no empty sigma folders.
- Also since 0.3.4: Windows build support for the `cpp_warp` extension, with a clear
  `ImportError` when the compiled extension is unavailable; the eye-model debug file uses the
  system temporary folder.
- Repository: model files and Git LFS were removed from the history.
