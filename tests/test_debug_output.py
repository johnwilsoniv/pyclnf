"""Fitting must not leave files in the temporary folder unless debug output is asked for.

The eye model's CPU path used to write a CCNF debug file
(python_ccnf_neuron_debug_call<N>.txt) into the temporary folder for every fit.
Debug files are now written only with CLNF(debug_mode=True). Each check runs in
a fresh Python process whose TMPDIR is an empty folder, so that
tempfile.gettempdir() points there, and whose working folder is checked too.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("numba")

REPO = Path(__file__).resolve().parents[1]

FIT = """
    import importlib.util, os, tempfile, warnings
    import numpy as np
    from pyclnf import CLNF

    assert tempfile.gettempdir() == os.environ["TMPDIR"], tempfile.gettempdir()
    rng = np.random.default_rng(0)
    gray = np.clip(rng.normal(128, 40, (240, 320)), 0, 255).astype(np.uint8)
    paths = [p for p in PATHS if p == "cpu" or importlib.util.find_spec("torch")]
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clnf = CLNF(detector=None, use_validator=False, use_gpu=(path == "gpu"),
                        debug_mode=DEBUG)
        assert clnf.optimizer.use_gpu == (path == "gpu") and clnf.use_eye_refinement
        landmarks, info = clnf.fit(gray, (100, 60, 120, 120))
        assert landmarks.shape == (68, 2) and np.isfinite(landmarks).all()
        print("fitted on the", path, "path, device", clnf.optimizer.gpu_device)
    print("ok")
"""


def _fit_in_fresh_process(tmp_path, debug, paths):
    from pyclnf import models
    if not models.models_ready():
        pytest.skip("OpenFace model files are not installed (run pyclnf-download-models)")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    code = textwrap.dedent(FIT).replace("PATHS", repr(paths)).replace("DEBUG", repr(debug))
    env = dict(os.environ, PYTHONPATH=str(REPO), TMPDIR=str(tmpdir))
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-3000:] + r.stderr[-3000:]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tmp"]  # nothing in the working folder
    return sorted(p.name for p in tmpdir.iterdir()), r.stdout


def test_cpu_and_gpu_fits_leave_the_temp_folder_empty(tmp_path):
    files, out = _fit_in_fresh_process(tmp_path, debug=False, paths=["cpu", "gpu"])
    assert "fitted on the cpu path" in out
    if importlib.util.find_spec("torch") is not None:
        assert "fitted on the gpu path" in out
    assert files == [], files


def test_debug_mode_writes_its_files_to_the_temp_folder(tmp_path):
    files, out = _fit_in_fresh_process(tmp_path, debug=True, paths=["cpu"])
    assert any(f.startswith("python_ccnf_neuron_debug_call") for f in files), files
