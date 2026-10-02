"""Deprecated: use ``pyclnf-download-models`` or ``pyclnf.models.ensure_models``.

Kept so that ``python -m pyclnf.model_downloader`` still works. Since pyclnf
0.4.0, OpenFace's model files are downloaded from OpenFace's own sources after
the user accepts OpenFace's license; pyclnf no longer hosts copies of them.
"""

import sys


def download_models(force: bool = False) -> bool:
    """Deprecated. Runs the interactive installer (asks to accept OpenFace's license)."""
    from .download_models import main
    return main([]) == 0


if __name__ == "__main__":
    from .download_models import main
    sys.exit(main(sys.argv[1:]))
