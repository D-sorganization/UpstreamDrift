"""Contract tests to verify Tools consumer provider-path resolution.

This ensures that we can strictly validate whether shared modules are loaded
from the local UpstreamDrift `src/shared/python` or the vendored
`vendor/ud-tools/src/shared/python` directory depending on `--tools-mode`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_tools_vendoring_provider_path(pytestconfig: pytest.Config) -> None:
    """Proves that shared modules resolve to the configured tooling mode path."""
    mode = pytestconfig.getoption("--tools-mode")
    root_dir = Path(__file__).resolve().parent.parent.parent

    # We import a module known to exist in both local and vendored locations
    # "common_utils" or "logging_pkg" is typically shared. Let's try to import
    # `upstream_drift_tools` package since it's the core vendored framework.
    local_path = (root_dir / "src/shared/python").resolve()
    vendored_path = (root_dir / "vendor/ud-tools/src/shared/python").resolve()

    import importlib
    import sys

    importlib.invalidate_caches()
    # Clear out conflicting paths to enforce test's mode, including other repos
    sys.path = [p for p in sys.path if "shared/python" not in p.replace("\\", "/")]
    if mode == "vendored":
        sys.path.insert(0, str(vendored_path))
    else:
        sys.path.insert(0, str(local_path))

    try:
        # Clear out existing imports
        keys_to_pop = [
            m for m in list(sys.modules) if m.startswith(("upstream_drift_tools", "common_utils"))
        ]
        for k in keys_to_pop:
            sys.modules.pop(k)

        import upstream_drift_tools
    except ImportError:
        pytest.skip("upstream_drift_tools package is not available to test vendoring.")

    provider_file = Path(upstream_drift_tools.__file__).resolve()

    if mode == "vendored":
        assert "ud-tools" in provider_file.parts or "ud-tools" in str(
            provider_file
        ), f"Expected a vendored tools path containing 'ud-tools', got {provider_file}"
        # Ensure it is definitely not resolving from our local UpstreamDrift/src/shared/python
        assert "upstreamdrift/src/shared/python" not in str(provider_file).lower().replace(
            "\\", "/"
        ), f"Resolved to local path instead of vendored: {provider_file}"
    elif mode == "local":
        assert str(local_path) in str(
            provider_file
        ), f"Expected local tools path {local_path}, got {provider_file}"
    else:
        pytest.fail(f"Unknown tools mode configured: {mode}")
