"""Conftest for humanoid_character_builder in-package tests.

Ensures UpstreamDrift's src/tools is first on sys.path so that
the local humanoid_character_builder package is imported rather
than any externally installed copy.
"""

import sys
from pathlib import Path

# Bootstrap: add repo root + src/tools to sys.path
_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_tools_dir = str(Path(__file__).resolve().parent.parent.parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
