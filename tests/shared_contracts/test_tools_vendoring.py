"""Contract tests to verify Tools consumer provider-path resolution.

This ensures that we can strictly validate whether shared modules are loaded
from the local UpstreamDrift `src/shared/python` or the vendored
`vendor/ud-tools/src/shared/python` directory depending on `--tools-mode`.
"""

from __future__ import annotations
