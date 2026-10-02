"""Offline tests for pyclnf.models and the pyclnf-download-models command.

A small fake manifest with file:// URLs stands in for OpenFace's files, and the
conversion step is replaced by a stub, so no network access and no OpenFace
files are needed.
"""

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from pyclnf import download_models, models
from pyclnf.models import _derive, _store

M = "lib/local/LandmarkDetector/model/"


def _fake_manifest(tmp_path, corrupt=None, first_url_broken=False):
    src = tmp_path / "server"
    files = {
        M + "pdms/In-the-wild_aligned_PDM_68.txt": b"pdm " * 50,
        M + "detection_validation/validator_cnn_68.txt": b"validator " * 20,
        M + "patch_experts/cen_patches_0.25_of.dat": bytes(range(256)) * 40,
    }
    entries = []
    for path, data in files.items():
        f = src / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if path != corrupt else data + b"tampered")
        urls = [f.as_uri()]
        if first_url_broken:
            urls.insert(0, (src / "missing" / Path(path).name).as_uri())
        entries.append({"path": path, "urls": urls, "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data)})
    return {"schema": 1, "openface_version": "2.2.0", "files": entries}


def _stub_derive_all(originals, out_root):
    out = out_root / "exported_pdm"
    out.mkdir(parents=True)
    (out / "mean_shape.npy").write_bytes((originals / M / "pdms/In-the-wild_aligned_PDM_68.txt").read_bytes())
    return {}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(_store.ENV_ACCEPT_LICENSE, raising=False)
    monkeypatch.delenv(_store.ENV_MODELS_DIR, raising=False)
    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(_store, "load_manifest", lambda: json.loads(json.dumps(manifest)))
    monkeypatch.setattr(_store, "_RETRIES", 1)
    monkeypatch.setattr(_derive, "derive_all", _stub_derive_all)
    return tmp_path, manifest


def test_real_manifest_lists_openface_originals():
    manifest = models.load_manifest()
    files = manifest["files"]
    assert manifest["openface_version"] == "2.2.0" and len(files) == 15
    paths = {e["path"] for e in files}
    needed = {_derive.FACE_PDM, _derive.SIGMA_SOURCE, _derive.INNER_PDM, _derive.INNER_CCNF,
              *_derive.LINKED.values()}
    for pdm, ccnf in _derive.EYE_SOURCES.values():
        needed |= {pdm} | {ccnf.format(scale=s) for s in _derive.EYE_SCALES}
    assert needed == paths
    for e in files:
        assert len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0 and e["size"] > 0
        if e["path"].endswith(".dat"):
            assert e["urls"][0].startswith("https://www.dropbox.com/s/") and e["urls"][0].endswith("?dl=1")
            assert e["urls"][1].startswith("https://onedrive.live.com/download?")
        else:
            assert e["urls"][0] == ("https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/"
                                    "OpenFace_2.2.0/" + e["path"])
    assert 430e6 < sum(e["size"] for e in files) < 440e6


def test_package_contains_no_model_files():
    pkg = Path(models.__file__).parents[1]
    bad = [p for p in pkg.rglob("*") if p.suffix.lower() in
           {".dat", ".onnx", ".pth", ".pt", ".mlmodel", ".h5", ".npz", ".npy", ".pkl"}
           or (p.suffix == ".txt" and "models" in p.relative_to(pkg).parts)]
    assert bad == []
    shipped = sorted(p.name for p in (pkg / "models").iterdir() if p.is_file())
    assert [n for n in shipped if not n.endswith((".py", ".pyc"))] == ["openface_models.json"]


def test_cache_location(monkeypatch, tmp_path):
    monkeypatch.delenv(_store.ENV_MODELS_DIR, raising=False)
    monkeypatch.setattr(_store.sys, "platform", "darwin")
    assert models.models_root() == Path.home() / "Library" / "Application Support" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setattr(_store.sys, "platform", "linux")
    monkeypatch.setattr(_store.os, "name", "posix")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert models.models_root() == tmp_path / "xdg" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/ignored")
    assert models.models_root() == Path.home() / ".local" / "share" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setenv(_store.ENV_MODELS_DIR, str(tmp_path / "custom"))
    assert models.models_root() == tmp_path / "custom" / "2.2.0"
    assert models.models_root(tmp_path / "arg") == tmp_path / "arg" / "2.2.0"
    assert models.models_dir(tmp_path / "arg") == tmp_path / "arg" / "2.2.0" / "derived" / "pyclnf" / "1"


def test_missing_without_license_raises_plain_message(env):
    tmp, _ = env
    with pytest.raises(models.ModelsNotInstalledError) as info:
        models.ensure_models(cache_dir=tmp / "cache")
    assert isinstance(info.value, FileNotFoundError)
    assert issubclass(models.ModelDownloadError, RuntimeError)
    text = str(info.value)
    for needle in ("pyclnf-download-models", "-m pyclnf.download_models",
                   "OPENFACE_MODELS_ACCEPT_LICENSE=1", "ensure_models(accept_license=True)",
                   str(tmp / "cache" / "2.2.0"), models.LICENSE_URL):
        assert needle in text
    assert not (tmp / "cache").exists()


def test_only_exact_env_value_one_accepts(env, monkeypatch):
    tmp, _ = env
    for value in ("yes", "true", "TRUE", "0", ""):
        monkeypatch.setenv(_store.ENV_ACCEPT_LICENSE, value)
        with pytest.raises(models.ModelsNotInstalledError):
            models.ensure_models(cache_dir=tmp / "cache")
    monkeypatch.setenv(_store.ENV_ACCEPT_LICENSE, "1")
    assert models.ensure_models(cache_dir=tmp / "cache", progress=lambda *a: None).is_dir()


def test_download_verify_derive_and_reuse_offline(env):
    tmp, manifest = env
    events = []
    ready = models.ensure_models(True, cache_dir=tmp / "cache", progress=lambda d, t, n: events.append((d, t, n)))
    assert ready == models.models_dir(tmp / "cache")
    for e in manifest["files"]:
        data = (tmp / "cache" / "2.2.0" / "originals" / e["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e["sha256"]
    assert (ready / "pyclnf-models.json").is_file() and (ready / "exported_pdm" / "mean_shape.npy").is_file()
    total = sum(e["size"] for e in manifest["files"])
    assert events[0] == (0, total, "") and events[-1] == (total, total, "")
    assert {n for _, _, n in events} - {""} == {e["path"] for e in manifest["files"]}
    leftovers = [p for p in (tmp / "cache").rglob("*") if ".part-" in p.name or p.name.startswith(".tmp-")]
    assert leftovers == []
    assert models.models_ready(tmp / "cache")
    # Ready now: works without the license flag and without network.
    for e in manifest["files"]:
        e["urls"] = ["file:///nonexistent"]
    assert models.ensure_models(cache_dir=tmp / "cache") == ready


def test_checksum_mismatch_leaves_nothing(tmp_path, monkeypatch, env):
    corrupt = M + "patch_experts/cen_patches_0.25_of.dat"
    manifest = _fake_manifest(tmp_path / "bad", corrupt=corrupt)
    monkeypatch.setattr(_store, "load_manifest", lambda: manifest)
    with pytest.raises(models.ModelDownloadError):
        models.ensure_models(True, cache_dir=tmp_path / "cache", progress=lambda *a: None)
    root = tmp_path / "cache" / "2.2.0"
    assert not (root / "originals" / corrupt).exists()
    assert [p for p in root.rglob("*") if ".part-" in p.name] == []
    assert not models.models_dir(tmp_path / "cache").exists()


def test_falls_back_to_the_next_url(tmp_path, monkeypatch, env):
    manifest = _fake_manifest(tmp_path / "m", first_url_broken=True)
    monkeypatch.setattr(_store, "load_manifest", lambda: manifest)
    assert models.ensure_models(True, cache_dir=tmp_path / "cache", progress=lambda *a: None).is_dir()


def test_originals_from_another_package_are_converted_without_asking(env):
    tmp, manifest = env
    root = tmp / "cache" / "2.2.0"
    for e in manifest["files"]:
        dest = root / "originals" / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((tmp / "server" / e["path"]).read_bytes())
    assert models.ensure_models(cache_dir=tmp / "cache") == models.models_dir(tmp / "cache")


def test_stamp_from_another_manifest_is_rebuilt(env):
    tmp, manifest = env
    ready = models.ensure_models(True, cache_dir=tmp / "cache", progress=lambda *a: None)
    stamp = json.loads((ready / "pyclnf-models.json").read_text())
    stamp["manifest_key"] = "0" * 64
    (ready / "pyclnf-models.json").write_text(json.dumps(stamp))
    assert not models.models_ready(tmp / "cache")
    assert models.ensure_models(cache_dir=tmp / "cache") == ready  # originals present: no license needed
    assert models.models_ready(tmp / "cache")


def test_command_needs_yes(env, monkeypatch, capsys):
    tmp, _ = env
    monkeypatch.setattr(builtins, "input", lambda prompt="": "no")
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 1
    assert "Nothing was downloaded" in capsys.readouterr().out
    assert not (tmp / "cache" / "2.2.0" / "originals").exists()
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 0
    assert models.models_ready(tmp / "cache")
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 0
    assert "already installed" in capsys.readouterr().out


def test_command_accept_license_flag(env, capsys):
    tmp, _ = env
    assert download_models.main(["--accept-license", "--cache-dir", str(tmp / "cache")]) == 0
    out = capsys.readouterr().out
    assert models.LICENSE_URL in out and "ready in" in out


@pytest.mark.skipif(_store.os.name == "nt", reason="uses flock semantics")
def test_lock_times_out_with_a_plain_message(tmp_path):
    with _store.FileLock(tmp_path / ".lock"):
        # flock locks belong to an open file, so a second open in this process must wait.
        with pytest.raises(models.ModelDownloadError) as info:
            with _store.FileLock(tmp_path / ".lock", timeout=0.6):
                pass
    assert "Another program" in str(info.value)
    with _store.FileLock(tmp_path / ".lock", timeout=0.6):
        pass  # free again once released
