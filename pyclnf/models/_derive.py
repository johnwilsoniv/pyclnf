"""Build pyclnf's model folder from OpenFace 2.2.0's original files.

pyclnf <= 0.3.4 shipped NumPy exports of OpenFace's models. Since 0.4.0 the same
files are produced on the user's computer, from the originals downloaded from
OpenFace, by the exporters below (they reproduce the 0.3.4 files byte for byte).

The model folder (what ``CLNF(model_dir=...)`` loads) is
<models_root>/derived/pyclnf/<CONVERTER_VERSION>/:

    exported_pdm/                 68-point PDM (In-the-wild_aligned_PDM_68.txt)
    sigma_components/             CCNF sigma components (ccnf_patches_0.25_general.txt)
    exported_eye_pdm_{left,right}/, exported_eye_ccnf_{left,right}/   eye models
    exported_inner_pdm/, exported_inner_ccnf/   inner-face model (use_inner_refinement)
    patch_experts/cen_patches_*.dat             the original CEN files (linked)
    detection_validation/validator_cnn_68.txt   the original validator (linked)
    pyclnf-models.json                          stamp, written last
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import _store
from .openface_loader import CCNFPatchExpertLoader, PDMLoader

# Bump when the content or layout of derived/pyclnf/<version>/ changes.
CONVERTER_VERSION = "1"
STAMP_FILE = "pyclnf-models.json"

_M = "lib/local/LandmarkDetector/model/"
FACE_PDM = _M + "pdms/In-the-wild_aligned_PDM_68.txt"
SIGMA_SOURCE = _M + "patch_experts/ccnf_patches_0.25_general.txt"
EYE_SOURCES = {
    "left": (_M + "model_eye/pdms/pdm_28_l_eye_3D_closed.txt",
             _M + "model_eye/patch_experts/left_ccnf_patches_{scale}_synth_lid_.txt"),
    "right": (_M + "model_eye/pdms/pdm_28_eye_3D_closed.txt",
              _M + "model_eye/patch_experts/ccnf_patches_{scale}_synth_lid_.txt"),
}
EYE_SCALES = ("1.00", "1.50")
INNER_PDM = _M + "model_inner/pdms/pdm_51_inner.txt"
INNER_CCNF = _M + "model_inner/patch_experts/ccnf_patches_1.00_inner.txt"
LINKED = {
    **{f"patch_experts/cen_patches_{s}_of.dat": _M + f"patch_experts/cen_patches_{s}_of.dat"
       for s in ("0.25", "0.35", "0.50", "1.00")},
    "detection_validation/validator_cnn_68.txt": _M + "detection_validation/validator_cnn_68.txt",
}


# --------------------------------------------------------------------------
# Exporters (OpenFace files -> the NumPy files pyclnf loads)
# --------------------------------------------------------------------------

def export_face_pdm(pdm_txt: Path, out_dir: Path) -> None:
    """exported_pdm/: mean shape, principal components, eigenvalues (float64)."""
    pdm = PDMLoader(str(pdm_txt))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mean_shape.npy", pdm.mean_shape)                     # (204, 1)
    np.save(out_dir / "princ_comp.npy", pdm.princ_comp)                     # (204, 34)
    np.save(out_dir / "eigen_values.npy", pdm.eigen_values.reshape(-1, 1))  # (34, 1)


def export_sigma_components(ccnf_txt: Path, out_dir: Path) -> None:
    """sigma_components/: CCNF sigma components per window size (float32)."""
    ccnf = CCNFPatchExpertLoader(str(ccnf_txt), num_landmarks=68, verbose=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "window_sizes.npy", np.array(ccnf.window_sizes, dtype=np.int64))
    for w_idx, window_size in enumerate(ccnf.window_sizes):
        for c_idx, sigma in enumerate(ccnf.sigma_components[w_idx]):
            np.save(out_dir / f"sigma_w{window_size}_c{c_idx}.npy", np.asarray(sigma, dtype=np.float32))


def export_eye_models(originals: Path, out_root: Path) -> None:
    """exported_eye_pdm_{side}/ and exported_eye_ccnf_{side}/scale_{s}/ for both eyes."""
    for side, (pdm_rel, ccnf_rel) in EYE_SOURCES.items():
        pdm = PDMLoader(str(originals / pdm_rel))
        pdm_dir = out_root / f"exported_eye_pdm_{side}"
        pdm_dir.mkdir(parents=True, exist_ok=True)
        np.save(pdm_dir / "mean_shape.npy", pdm.mean_shape)
        np.save(pdm_dir / "eigenvectors.npy", pdm.princ_comp)
        np.save(pdm_dir / "eigenvalues.npy", pdm.eigen_values)
        for scale in EYE_SCALES:
            ccnf = CCNFPatchExpertLoader(str(originals / ccnf_rel.format(scale=scale)),
                                         num_landmarks=28, verbose=False)
            ccnf.save_numpy(str(out_root / f"exported_eye_ccnf_{side}" / f"scale_{scale}"))


def _text_sections(path: Path) -> List[List[float]]:
    """Numbers of an OpenFace text model, grouped by the '#' comment before them."""
    sections: List[List[float]] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                sections.append([])
            elif sections:
                sections[-1].extend(float(x) for x in s.split())
    return sections


def export_inner_model(originals: Path, out_root: Path) -> None:
    """exported_inner_pdm/ and exported_inner_ccnf/ (experimental inner-face model).

    Reproduces the files shipped in pyclnf 0.3.x bit for bit. The 0.3.x inner PDM
    export read each matrix starting at its OpenCV type code, so mean_shape and
    eigenvectors begin with 6.0 and lose their last value; this is kept as-is so
    use_inner_refinement=True behaves exactly as before.
    """
    mean_sec, vec_sec, val_sec = _text_sections(originals / INNER_PDM)
    rows, cols = int(mean_sec[0]), int(mean_sec[1])
    vrows, vcols = int(vec_sec[0]), int(vec_sec[1])
    erows, ecols = int(val_sec[0]), int(val_sec[1])
    pdm_dir = out_root / "exported_inner_pdm"
    pdm_dir.mkdir(parents=True, exist_ok=True)
    np.save(pdm_dir / "mean_shape.npy",
            np.array(mean_sec[2:2 + rows * cols], dtype=np.float64).reshape(rows // 3, 3))
    np.save(pdm_dir / "eigenvectors.npy",
            np.array(vec_sec[2:2 + vrows * vcols], dtype=np.float64).reshape(vrows, vcols))
    np.save(pdm_dir / "eigenvalues.npy", np.array(val_sec[3:3 + erows * ecols], dtype=np.float64))

    ccnf = CCNFPatchExpertLoader(str(originals / INNER_CCNF), num_landmarks=51, verbose=False)
    ccnf_dir = out_root / "exported_inner_ccnf"
    ccnf_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ccnf_dir / "metadata.npz",
             patch_scaling=np.float64(ccnf.patch_scaling),
             num_views=np.int64(ccnf.num_views),
             num_points=np.int64(ccnf.num_landmarks),
             windows=np.array(ccnf.window_sizes, dtype=np.int64),
             n_betas=np.int64(len(ccnf.sigma_components[0])),
             centers=np.array([c.ravel() for c in ccnf.raw_centers], dtype=np.float64),
             visibilities=np.array([v.ravel() for v in ccnf.visibilities]))
    for w_idx, window_size in enumerate(ccnf.window_sizes):
        sigma_dir = ccnf_dir / f"sigma_ws{window_size}"
        sigma_dir.mkdir(exist_ok=True)
        for c_idx, sigma in enumerate(ccnf.sigma_components[w_idx]):
            np.save(sigma_dir / f"sigma_{c_idx}.npy", np.asarray(sigma, dtype=np.float32))
    for idx, patch in enumerate(ccnf.patches[0]):  # frontal view only
        if patch["empty"]:
            continue
        patch_dir = ccnf_dir / "view_0" / f"patch_{idx}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        np.savez(patch_dir / "info.npz", width=np.int64(patch["width"]),
                 height=np.int64(patch["height"]), confidence=np.float64(patch["patch_confidence"]))
        # One row per neuron: [bias, weights * norm_weights in column-major order]
        rows_ = [np.concatenate([[n["bias"]], (np.asarray(n["weights"], dtype=np.float32)
                                               * np.float32(n["norm_weights"])).ravel(order="F")])
                 for n in patch["neurons"]]
        np.save(patch_dir / "weight_matrix.npy", np.array(rows_).astype(np.float32))
        np.save(patch_dir / "alphas.npy", np.array([n["alpha"] for n in patch["neurons"]], dtype=np.float64))
        np.save(patch_dir / "betas.npy", np.array(patch["betas"], dtype=np.float64))


def derive_all(originals: Path, out_root: Path) -> Dict[str, str]:
    """Write every derived file into out_root and link the originals used as they are."""
    export_face_pdm(originals / FACE_PDM, out_root / "exported_pdm")
    export_sigma_components(originals / SIGMA_SOURCE, out_root / "sigma_components")
    export_eye_models(originals, out_root)
    export_inner_model(originals, out_root)
    return {rel: _store.link_or_copy(originals / src, out_root / rel) for rel, src in LINKED.items()}


# --------------------------------------------------------------------------
# The ready folder
# --------------------------------------------------------------------------

def models_dir(cache_dir: Optional[os.PathLike] = None) -> Path:
    """The folder pyclnf loads its model files from (it may not exist yet)."""
    return _store.models_root(cache_dir) / "derived" / _store.PACKAGE_NAME / CONVERTER_VERSION


def _derived_ok(ready: Path, manifest: dict) -> bool:
    try:
        stamp = json.loads((ready / STAMP_FILE).read_text(encoding="utf-8"))
        if stamp.get("manifest_key") != _store.manifest_key(manifest):
            return False
        if stamp.get("converter_version") != CONVERTER_VERSION:
            return False
        for rel, size in stamp["files"].items():
            if ready.joinpath(*rel.split("/")).stat().st_size != size:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def models_ready(cache_dir: Optional[os.PathLike] = None) -> bool:
    """True if pyclnf's model folder is complete (nothing to download or prepare)."""
    return _derived_ok(models_dir(cache_dir), _store.load_manifest())


def _derive(root: Path, ready: Path, manifest: dict) -> None:
    """Build the model folder in a temporary folder, then move it into place atomically."""
    parent = ready.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=parent))
    except OSError as e:
        raise _store.ModelDownloadError(_store.write_error(parent, e)) from e
    try:
        links = derive_all(root / "originals", tmp)
        sources = {e["path"]: e for e in manifest["files"]}
        for rel, kind in links.items():
            entry = sources[LINKED[rel]]
            dst = tmp.joinpath(*rel.split("/"))
            if kind == "copy" and _store.sha256_of(dst) != entry["sha256"]:
                raise _store.ModelDownloadError(f"{entry['path']}: copy does not match its checksum")
        files = {p.relative_to(tmp).as_posix(): p.stat().st_size
                 for p in sorted(tmp.rglob("*")) if p.is_file()}
        stamp = {
            "package": _store.PACKAGE_NAME,
            "converter_version": CONVERTER_VERSION,
            "openface_version": _store.OPENFACE_VERSION,
            "manifest_key": _store.manifest_key(manifest),
            "links": links,
            "files": files,
        }
        (tmp / STAMP_FILE).write_text(json.dumps(stamp, indent=1) + "\n", encoding="utf-8")
        if ready.exists():
            shutil.rmtree(ready)  # incomplete or outdated; we hold the lock
        os.replace(tmp, ready)
    except OSError as e:
        raise _store.ModelDownloadError(_store.write_error(parent, e)) from e
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def ensure_models(accept_license: bool = False, *, cache_dir=None,
                  progress: Optional[_store.ProgressCallback] = None) -> Path:
    manifest = _store.load_manifest()
    root = _store.models_root(cache_dir)
    ready = models_dir(cache_dir)
    if _derived_ok(ready, manifest):
        return ready

    def present(e):  # cheap check here; full SHA-256 check under the lock
        try:
            return _store.original_path(root, e).stat().st_size == e["size"]
        except OSError:
            return False

    missing = [e for e in manifest["files"] if not present(e)]
    if missing and not _store.license_accepted(accept_license):
        raise _store.ModelsNotInstalledError(_store.not_installed_message(root, missing))

    with _store._thread_lock, _store.FileLock(root / ".lock"):
        # Another process may have finished while we waited for the lock.
        if _derived_ok(ready, manifest):
            return ready
        missing = [e for e in manifest["files"] if not _store.original_ok(root, e)]
        if missing:
            if not _store.license_accepted(accept_license):
                raise _store.ModelsNotInstalledError(_store.not_installed_message(root, missing))
            _store.download_all(root, missing, progress or _store.stderr_progress())
        _derive(root, ready, manifest)
    return ready
