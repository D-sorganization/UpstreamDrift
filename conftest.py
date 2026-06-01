"""Session conftest: import MuJoCo early to avoid Windows DLL collection crashes."""

import contextlib
import sys
from pathlib import Path

# Prioritize local packages over installed site-packages to prevent package shadowing (e.g. tools)
root_path = str(Path(__file__).resolve().parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

with contextlib.suppress(ImportError):
    import mujoco  # noqa: F401


# Import MuJoCo early to avoid Windows DLL initialization conflicts (Access Violation)
# that occur when MuJoCo is loaded during pytest collection with certain plugins.
