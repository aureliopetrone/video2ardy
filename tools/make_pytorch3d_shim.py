"""Install a minimal ``pytorch3d`` into a venv: only ``pytorch3d.transforms`` (pure Python, copied
from a checkout of facebookresearch/pytorch3d, BSD licence) plus import-only placeholders for the
rendering modules GVHMR imports but this pipeline never calls. Avoids building pytorch3d's CUDA
extensions on Windows / new GPUs.

    git clone --depth 1 https://github.com/facebookresearch/pytorch3d third_party_pytorch3d
    venv_gvhmr\\Scripts\\python.exe tools\\make_pytorch3d_shim.py third_party_pytorch3d
"""

import shutil
import site
import sys
from pathlib import Path

PLACEHOLDER = '''"""pytorch3d shim: rendering is not available (import-only placeholders)."""


class _Missing:
    def __init__(self, *a, **k):
        raise NotImplementedError("pytorch3d shim: rendering not available")


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _Missing
'''


def main(src: str):
    src = Path(src) / "pytorch3d"
    dst = Path(site.getsitepackages()[-1]) / "pytorch3d"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "common" / "workaround").mkdir(parents=True)
    shutil.copytree(src / "transforms", dst / "transforms")
    shutil.copy(src / "common" / "datatypes.py", dst / "common" / "datatypes.py")
    shutil.copy(src / "common" / "workaround" / "utils.py", dst / "common" / "workaround" / "utils.py")
    (dst / "common" / "__init__.py").write_text("")
    (dst / "common" / "workaround" / "__init__.py").write_text("from .utils import _safe_det_3x3\n")
    (dst / "__init__.py").write_text('"""Minimal shim: pytorch3d.transforms only."""\n__version__ = "0.7.8-shim"\n')
    (dst / "ops").mkdir()
    (dst / "ops" / "__init__.py").write_text("")
    (dst / "ops" / "knn.py").write_text(
        'def knn_points(*a, **k):\n    raise NotImplementedError("pytorch3d shim: knn_points not available")\n')
    for sub in ("renderer", "structures"):
        (dst / sub).mkdir()
    for f in ("renderer/__init__.py", "renderer/cameras.py", "structures/__init__.py", "structures/meshes.py"):
        (dst / f).write_text(PLACEHOLDER)
    print("pytorch3d shim installed in", dst)


if __name__ == "__main__":
    main(sys.argv[1])
