"""
OpenFace model files for pyclnf.

pyclnf does not include any OpenFace model files. OpenFace's files are licensed
by Carnegie Mellon University for academic or non-profit, non-commercial
research only, and may not be redistributed. pyclnf therefore downloads them,
only after you accept the OpenFace license, from OpenFace's official sources
(GitHub tag OpenFace_2.2.0, and the Dropbox/OneDrive links in OpenFace's own
download_models script for the four large CEN files). Every file is checked
against the SHA-256 checksum listed in ``openface_models.json``, then converted
on your computer into the files pyclnf loads.

The files are kept in a cache shared with pyfaceau and pymtcnn::

    macOS    ~/Library/Application Support/OpenFaceModels/2.2.0/
    Windows  %LOCALAPPDATA%\\OpenFaceModels\\2.2.0\\
    Linux    $XDG_DATA_HOME/OpenFaceModels/2.2.0/   (default ~/.local/share)

Set the environment variable ``OPENFACE_MODELS_DIR`` to use another folder in
place of ``<user data dir>/OpenFaceModels`` (the ``2.2.0`` sub-folder is always
added). Inside the version folder:

    originals/<path in OpenFace's repository>   the downloaded files, unchanged
    derived/pyclnf/<converter version>/         the folder pyclnf loads from
    .lock                                       held while files are written

Typical use::

    from pyclnf.models import ensure_models
    model_dir = ensure_models()                       # raises ModelsNotInstalledError if missing
    model_dir = ensure_models(accept_license=True)    # downloads what is missing

Setting ``OPENFACE_MODELS_ACCEPT_LICENSE=1`` counts as accepting the license.
``CLNF()`` calls ``ensure_models()`` without accepting, so it never downloads
anything unless the license was accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ._store import (
    ENV_ACCEPT_LICENSE,
    ENV_MODELS_DIR,
    LICENSE_URL,
    OPENFACE_VERSION,
    ModelDownloadError,
    ModelsNotInstalledError,
    ProgressCallback,
    license_accepted,
    load_manifest,
    models_root,
    user_data_dir,
)

__all__ = [
    "ensure_models",
    "ModelsNotInstalledError",
    "ModelDownloadError",
    "models_root",
    "models_dir",
    "models_ready",
    "license_accepted",
    "load_manifest",
    "user_data_dir",
    "OPENFACE_VERSION",
    "LICENSE_URL",
]


def models_dir(cache_dir: Optional[os.PathLike] = None) -> Path:
    """The folder pyclnf loads its model files from (it may not exist yet)."""
    from ._derive import models_dir as _models_dir
    return _models_dir(cache_dir)


def models_ready(cache_dir: Optional[os.PathLike] = None) -> bool:
    """True if pyclnf's model folder is complete (nothing to download or prepare)."""
    from ._derive import models_ready as _models_ready
    return _models_ready(cache_dir)


def ensure_models(accept_license: bool = False, *, cache_dir=None,
                  progress: Optional[ProgressCallback] = None) -> Path:
    """Return the folder with pyclnf's model files, preparing it if needed.

    Missing OpenFace files are downloaded from OpenFace's official sources only
    when the OpenFace license is accepted (``accept_license=True`` or the
    environment variable ``OPENFACE_MODELS_ACCEPT_LICENSE=1``). Every file is
    checked against its SHA-256 checksum, then converted on this computer into
    the files pyclnf loads. Files already downloaded for another package are
    reused (and converted without asking again).

    Args:
        accept_license: True to accept the OpenFace license and allow downloads.
        cache_dir: Folder to use in place of ``<user data dir>/OpenFaceModels``
            (default: ``OPENFACE_MODELS_DIR`` or the per-user data folder).
        progress: Optional ``progress(done_bytes, total_bytes, name)`` callback,
            called while downloading; ``name`` is the OpenFace path of the file
            being downloaded (empty at the start and the end). Without it,
            download progress is printed to stderr.

    Returns:
        The ready folder (``.../OpenFaceModels/2.2.0/derived/pyclnf/1``). Pass
        it as ``CLNF(model_dir=...)``, or let ``CLNF()`` find it.

    Raises:
        ModelsNotInstalledError: files are missing and the license was not accepted.
        ModelDownloadError: a download failed, a checksum did not match, or the
            folder could not be written.
    """
    from ._derive import ensure_models as _ensure_models
    return _ensure_models(accept_license, cache_dir=cache_dir, progress=progress)
