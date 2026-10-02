# Changelog

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
