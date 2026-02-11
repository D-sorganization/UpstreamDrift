"""Conftest for humanoid_character_builder in-package tests.

Ensures UpstreamDrift's src/tools is first on sys.path so that
the local humanoid_character_builder package is imported rather
than any externally installed copy.
"""

import sys
from pathlib import Path

_tools_dir = str(Path(__file__).resolve().parent.parent.parent)
if _tools_dir not in sys.path:
