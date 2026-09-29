"""Local conftest for wave10 gap audit tests.

Isolated from the repo-wide conftest because the global conftest pulls in
heavy deps (tensorflow, PyQt6) that are broken/slow on this machine. Each
test only imports the small, pure-Python module it covers.
"""

from __future__ import annotations
