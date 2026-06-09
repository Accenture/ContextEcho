"""Compatibility launcher for users who run from inside tools/donate.

The public quick-start says:

    cd tools/donate
    python -m donate

Python normally expects the parent directory of a package on sys.path, so that
command would miss the package. This tiny launcher redirects it to the real
package entry point, preserving the one-command donor experience.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    parent = here.parent
    sys.path.insert(0, str(parent))
    # This module was loaded as top-level "donate"; remove it so import
    # resolution can find the package at ../donate instead of this wrapper.
    sys.modules.pop("donate", None)
    runpy.run_module("donate", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
