"""Lab - Experimental data readers and laboratory tools.

Subpackages:
    bio: Biomechanics data readers (C3D motion capture format)
"""

from pathlib import Path

from src.shared.python._seam_redirect import vendor_search_paths

# Lab is a split boundary: retain UD readers while resolving Tools-owned
# subpackages (including mocap contracts) from the pinned authority.
for _search_root in vendor_search_paths():
    _lab_root = Path(_search_root) / "sidekick" / "lab"
    if _lab_root.is_dir() and str(_lab_root) not in __path__:
        __path__.append(str(_lab_root))

__all__: list[str] = []
