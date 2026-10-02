"""pyclnf must import its own modules by package name only.

pyclnf 0.3.4 put its package folder on sys.path and imported
"models.openface_loader" as a top-level module. That broke whenever the user
had a module named "models" (or pyfaceau's pyfaceau.models was reachable that
way), and it let pyclnf's folders shadow other packages' modules.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "pyclnf"
# Names that exist at the top of the pyclnf package (models, core, utils, ...)
INTERNAL = {p.stem for p in PKG.iterdir()
            if (p.is_dir() and not p.name.startswith(("_", "."))) or (p.suffix == ".py" and p.stem != "__init__")}


def test_sources_use_package_imports_and_leave_sys_path_alone():
    assert {"models", "core", "utils", "cpp_warp"} <= INTERNAL
    problems = []
    for f in sorted(PKG.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            where = f"{f.relative_to(REPO)}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Import):
                problems += [f"{where} import {a.name}" for a in node.names
                             if a.name.split(".")[0] in INTERNAL]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] in INTERNAL:
                    problems.append(f"{where} from {node.module} import ...")
            elif (isinstance(node, ast.Attribute) and node.attr == "path"
                  and isinstance(node.value, ast.Name) and node.value.id == "sys"):
                problems.append(f"{where} uses sys.path")
    assert problems == []


def test_user_modules_named_models_and_core_are_not_shadowed(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    (tmp_path / "models.py").write_text("SENTINEL = 'user models'\n")
    (tmp_path / "core.py").write_text("SENTINEL = 'user core'\n")
    code = textwrap.dedent("""
        import os, sys
        import models, core                      # the user's own modules, imported first
        import pyclnf
        import pyclnf.core.patch_expert          # 0.3.4 imported "models.openface_loader" here
        import pyclnf.models, pyclnf.download_models
        from pyclnf.models import ensure_models
        from pyclnf.models.openface_loader import load_sigma_components
        assert models.SENTINEL == "user models", models
        assert core.SENTINEL == "user core", core
        assert sys.modules["models"] is models and sys.modules["core"] is core
        pkg = os.path.dirname(os.path.abspath(pyclnf.__file__))
        assert not any(os.path.abspath(p) == pkg for p in sys.path if p), sys.path
        print("ok")
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]))
    env.pop("OPENFACE_MODELS_ACCEPT_LICENSE", None)
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr
