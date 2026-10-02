"""Shared OpenFace 2.2.0 model cache: locations, license, lock, verified downloads.

Same conventions as pyfaceau.models and pymtcnn, so the three packages share one
cache and never download a file twice:

    <cache_dir | $OPENFACE_MODELS_DIR | <user data dir>/OpenFaceModels>/2.2.0/
        originals/<path in OpenFace's repository>   downloaded files, unchanged
        derived/<package>/<converter version>/      what each package loads
        .lock                                       held while files are written

<user data dir> is ~/Library/Application Support (macOS), %LOCALAPPDATA%
(Windows) or $XDG_DATA_HOME, default ~/.local/share (Linux).

Downloads use only the standard library (urllib). Each attempt writes to
<file>.part-<pid>-<random>; the file is renamed into place only after its size
and SHA-256 match the manifest, so a partial or corrupted file is never used.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

OPENFACE_VERSION = "2.2.0"
LICENSE_URL = "https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt"
ENV_MODELS_DIR = "OPENFACE_MODELS_DIR"
ENV_ACCEPT_LICENSE = "OPENFACE_MODELS_ACCEPT_LICENSE"
PACKAGE_NAME = "pyclnf"
DOWNLOAD_COMMAND = "pyclnf-download-models"
DOWNLOAD_MODULE = "pyclnf.download_models"
MANIFEST_FILE = "openface_models.json"

_RETRIES = 4            # attempts per URL
_TIMEOUT_S = 30         # socket timeout per request
_LOCK_TIMEOUT_S = 15 * 60
_CHUNK = 1 << 16

ProgressCallback = Callable[[int, int, str], None]
"""progress(done_bytes, total_bytes, name): bytes downloaded so far in this
call, total bytes this call will download, and the OpenFace path of the file
being downloaded (empty string at the start and the end)."""

_thread_lock = threading.Lock()


class ModelsNotInstalledError(FileNotFoundError):
    """The OpenFace model files are not installed and the license was not accepted.

    Subclass of FileNotFoundError, which older pyclnf versions raised for
    missing model files.
    """


class ModelDownloadError(RuntimeError):
    """The model files could not be downloaded, checked or saved."""


# --------------------------------------------------------------------------
# Locations, manifest, license
# --------------------------------------------------------------------------

def user_data_dir() -> Path:
    """The per-user data folder of this operating system."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    xdg = os.environ.get("XDG_DATA_HOME", "")
    if xdg and os.path.isabs(xdg):
        return Path(xdg)
    return Path.home() / ".local" / "share"


def models_root(cache_dir: Optional[os.PathLike] = None) -> Path:
    """The shared cache folder for this OpenFace version.

    ``cache_dir`` (or else ``OPENFACE_MODELS_DIR``) replaces
    ``<user data dir>/OpenFaceModels``; ``2.2.0`` is always appended.
    """
    if cache_dir is not None:
        base = Path(cache_dir).expanduser()
    elif os.environ.get(ENV_MODELS_DIR, "").strip():
        base = Path(os.environ[ENV_MODELS_DIR].strip()).expanduser()
    else:
        base = user_data_dir() / "OpenFaceModels"
    return base / OPENFACE_VERSION


def load_manifest() -> dict:
    """The OpenFace files pyclnf needs (path, URLs, SHA-256, size)."""
    with open(Path(__file__).with_name(MANIFEST_FILE), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("openface_version") != OPENFACE_VERSION:
        raise ValueError(f"{MANIFEST_FILE} is not for OpenFace {OPENFACE_VERSION}")
    for entry in manifest["files"]:
        parts = entry["path"].split("/")
        if entry["path"].startswith("/") or ".." in parts or "" in parts:
            raise ValueError(f"unsafe path in {MANIFEST_FILE}: {entry['path']!r}")
    return manifest


def license_accepted(accept_license: bool = False) -> bool:
    """True if the caller accepted the license or OPENFACE_MODELS_ACCEPT_LICENSE=1."""
    return bool(accept_license) or os.environ.get(ENV_ACCEPT_LICENSE, "").strip() == "1"


def manifest_key(manifest: dict) -> str:
    rows = sorted((e["path"], e["sha256"], e["size"]) for e in manifest["files"])
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def original_path(root: Path, entry: dict) -> Path:
    return root.joinpath("originals", *entry["path"].split("/"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def original_ok(root: Path, entry: dict) -> bool:
    path = original_path(root, entry)
    try:
        if path.stat().st_size != entry["size"]:
            return False
        return sha256_of(path) == entry["sha256"]
    except OSError:
        return False


def human_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{max(1, round(n / 1000))} KB"


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def not_installed_message(root: Path, missing: List[dict]) -> str:
    size = human_size(sum(e["size"] for e in missing))
    python = sys.executable or "python"
    return (
        f"The OpenFace model files that {PACKAGE_NAME} needs are not installed yet.\n"
        "\n"
        f"{PACKAGE_NAME} does not include them: they belong to OpenFace (Carnegie Mellon\n"
        "University) and may only be used for academic or non-profit, non-commercial\n"
        f"research. License: {LICENSE_URL}\n"
        "\n"
        f"To install them ({size}, one time), do ONE of these:\n"
        "\n"
        "  1. Open a terminal and run:\n"
        f"         {DOWNLOAD_COMMAND}\n"
        "     If that command is not found, run this instead:\n"
        f"         \"{python}\" -m {DOWNLOAD_MODULE}\n"
        "\n"
        f"  2. Set the environment variable {ENV_ACCEPT_LICENSE}=1 (this accepts the\n"
        "     license) and run your program again.\n"
        "\n"
        "  3. In Python, after reading the license:\n"
        f"         from {PACKAGE_NAME}.models import ensure_models\n"
        "         ensure_models(accept_license=True)\n"
        "\n"
        "The files will be saved in:\n"
        f"    {root}\n"
    )


def ssl_hint(exc: BaseException) -> str:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
        if sys.platform == "darwin":
            return ("\nYour Python cannot check the website's security certificate. If you\n"
                    "installed Python from python.org, open the Python folder in\n"
                    "Applications, double-click 'Install Certificates.command', then try again.")
        return ("\nYour Python cannot check the website's security certificate. Update your\n"
                "system's certificates (or set SSL_CERT_FILE), then try again.")
    return ""


def write_error(folder: Path, exc: BaseException) -> str:
    return (f"Could not write to {folder}: {exc}\n"
            f"Free some disk space, or set {ENV_MODELS_DIR} to a folder you can write to.")


def stderr_progress() -> ProgressCallback:
    """Plain progress lines on stderr, used when the caller passes no callback."""
    state = {"name": None, "pct": -10}

    def report(done: int, total: int, name: str) -> None:
        if total <= 0:
            return
        if name and name != state["name"]:
            state["name"] = name
            print(f"{PACKAGE_NAME}: downloading {name.rsplit('/', 1)[-1]}", file=sys.stderr, flush=True)
        pct = int(done * 100 / total)
        if pct >= state["pct"] + 10 or (done >= total and state["pct"] < 100):
            state["pct"] = pct
            print(f"{PACKAGE_NAME}:   {pct}% of {human_size(total)}", file=sys.stderr, flush=True)
    return report


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

class _ChecksumError(Exception):
    pass


class _WrongContent(Exception):
    """The URL serves something else (for example a web page): try the next URL."""


def _user_agent() -> str:
    try:
        from .. import __version__ as version
    except Exception:  # pragma: no cover
        version = "unknown"
    return f"{PACKAGE_NAME}/{version} (OpenFace model downloader)"


def download_all(root: Path, entries: List[dict], progress: Optional[ProgressCallback]) -> None:
    """Download the given originals into originals/ (caller holds the lock)."""
    total = sum(e["size"] for e in entries)
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
    except OSError as e:
        raise ModelDownloadError(write_error(root, e)) from e
    if free < total + 100_000_000:  # the downloads, the converted files and some headroom
        raise ModelDownloadError(
            f"Not enough free disk space in {root}: about {human_size(total + 100_000_000)} "
            f"is needed and {human_size(free)} is free.\n"
            f"Free some disk space, or set {ENV_MODELS_DIR} to a folder on another drive.")
    done = 0
    if progress:
        progress(0, total, "")
    for entry in entries:
        dest = original_path(root, entry)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ModelDownloadError(write_error(dest.parent, e)) from e
        base = done

        def file_progress(n: int, _base=base, _name=entry["path"]) -> None:
            if progress:
                progress(_base + n, total, _name)

        download_one(entry, dest, file_progress)
        done += entry["size"]
    if progress:
        progress(total, total, "")


def download_one(entry: dict, dest: Path, file_progress: Callable[[int], None]) -> None:
    last_error: Optional[BaseException] = None
    for url in entry["urls"]:
        for attempt in range(_RETRIES):
            if attempt:
                time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            tmp = dest.with_name(f"{dest.name}.part-{os.getpid()}-{random.randrange(1 << 30):x}")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
                h = hashlib.sha256()
                n = 0
                with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response, open(tmp, "wb") as out:
                    # Fail fast when a mirror answers with something else than the file
                    # (e.g. a sign-in or preview web page), instead of retrying it.
                    if "text/html" in response.headers.get("Content-Type", "").lower():
                        raise _WrongContent(f"{url} returned a web page instead of the file")
                    length = response.headers.get("Content-Length")
                    if length is not None and length.isdigit() and int(length) != entry["size"]:
                        raise _WrongContent(f"{url} offers {length} bytes, expected {entry['size']}")
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        n += len(chunk)
                        if n > entry["size"]:
                            raise _ChecksumError(f"{entry['path']}: larger than expected")
                        out.write(chunk)
                        h.update(chunk)
                        file_progress(n)
                    out.flush()
                    os.fsync(out.fileno())
                if n != entry["size"] or h.hexdigest() != entry["sha256"]:
                    raise _ChecksumError(
                        f"{entry['path']}: checksum mismatch (got {n} bytes, sha256 {h.hexdigest()})")
                os.replace(tmp, dest)
                return
            except _WrongContent as e:
                last_error = e
                break  # this URL does not serve the file; try the next one
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code in (400, 401, 403, 404, 410):
                    break  # permanent for this URL; try the next one
            except PermissionError as e:
                raise ModelDownloadError(write_error(dest.parent, e)) from e
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EROFS, errno.EACCES):
                    raise ModelDownloadError(write_error(dest.parent, e)) from e
                last_error = e  # URLError, timeouts, resets, SSL errors
                if ssl_hint(e):
                    break  # certificate problems do not go away by retrying
            except (_ChecksumError, ValueError, EOFError) as e:
                last_error = e
            except Exception as e:  # http.client.IncompleteRead and similar
                last_error = e
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
    raise ModelDownloadError(
        f"Could not download {entry['path']} from OpenFace's official sources.\n"
        f"Last error: {last_error}\n"
        "Check your internet connection and try again (files already downloaded are kept).\n"
        "If you are behind a firewall or proxy, make sure these sites can be reached:\n"
        "  " + "\n  ".join(sorted({u.split('/')[2] for u in entry["urls"]}))
        + ssl_hint(last_error if last_error is not None else Exception())
    )


# --------------------------------------------------------------------------
# Lock and atomic folders
# --------------------------------------------------------------------------

class FileLock:
    """Inter-process lock on a file (released by the OS if the process dies)."""

    def __init__(self, path: Path, timeout: float = _LOCK_TIMEOUT_S):
        self.path = Path(path)
        self.timeout = timeout
        self._fh = None

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+b")
        except OSError as e:
            raise ModelDownloadError(write_error(self.path.parent, e)) from e
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > deadline:
                    self._fh.close()
                    raise ModelDownloadError(
                        f"Another program has been preparing the model files in {self.path.parent}\n"
                        "for a long time. Close other programs using pyfaceau, pyclnf or pymtcnn and try again.")
                time.sleep(0.5)

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
        return False


def link_or_copy(src: Path, dst: Path) -> str:
    """Make dst refer to src without duplicating the data when the file system allows it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
        if dst.stat().st_size == src.stat().st_size:
            return "symlink"
        dst.unlink()
    except OSError:
        if os.path.lexists(dst):
            os.unlink(dst)
    shutil.copyfile(src, dst)
    return "copy"
