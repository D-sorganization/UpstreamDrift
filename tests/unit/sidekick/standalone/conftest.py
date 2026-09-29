"""Path and fixture setup for standalone Sidekick unit tests."""

from __future__ import annotations

import os
import importlib
import sys
import types
from pathlib import Path

import pytest


def _ensure_numpy_stub() -> None:
    """Install a minimal numpy stub when numpy is not available.

    core.contracts.exceptions imports numpy only for dtype/shape introspection
    in PostconditionError — functionality that is never exercised by standalone
    session-store tests.  A lightweight stub lets StateError be imported
    without pulling in the full numpy C-extension stack.
    """
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401

        return
    except ImportError:
        pass

    # Build a minimal stub just large enough to satisfy the import chain.
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = type("ndarray", (), {})  # type: ignore[attr-defined]
    sys.modules["numpy"] = np_stub
    sys.modules["numpy._core"] = types.ModuleType("numpy._core")


def _install_canonical_tools_paths() -> None:
    """Force standalone behavior tests through the selected canonical Tools."""
    root = Path(__file__).resolve().parents[4]
    tools_root = Path(
        os.environ.get("TOOLS_REPO_PATH", root / "vendor/ud-tools")
    ).resolve()
    canonical_paths = (
        tools_root / "src/shared/python",
        tools_root / "src",
        tools_root / "src/python/src",
    )
    missing = [path for path in canonical_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"canonical Tools test paths are missing: {missing}")

    canonical_text = [str(path.resolve()) for path in canonical_paths]
    local_shared = str((root / "src/shared/python").resolve())
    rejected = {local_shared.casefold(), *(path.casefold() for path in canonical_text)}
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry).resolve()).casefold() not in rejected
    ]
    for entry in reversed(canonical_text):
        sys.path.insert(0, entry)

    shared_paths = {
        "shared": tools_root / "src/shared",
        "shared.python": tools_root / "src/shared/python",
    }
    for name, package_path in shared_paths.items():
        module = sys.modules.get(name)
        if module is not None:
            module.__path__ = [str(package_path.resolve())]  # type: ignore[attr-defined]
    for name in tuple(sys.modules):
        if (
            name == "sidekick"
            or name.startswith(
                (
                    "sidekick.",
                    "shared.python.sidekick",
                    "src.shared.python.sidekick",
                )
            )
            or name
            in {
                "shared.python.contracts",
                "shared.python.import_aliases",
                "src.shared.python.import_aliases",
            }
        ):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    parent_contracts = importlib.import_module("shared.python.contracts")
    contract_origin = Path(parent_contracts.__file__).resolve()
    if not contract_origin.is_relative_to(tools_root):
        raise RuntimeError(
            f"canonical Tools contracts did not resolve: {contract_origin}"
        )
    sys.modules["contracts"] = parent_contracts


_ensure_numpy_stub()
_install_canonical_tools_paths()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Reassert parent precedence after the repository-wide path hook."""
    del config
    _install_canonical_tools_paths()
