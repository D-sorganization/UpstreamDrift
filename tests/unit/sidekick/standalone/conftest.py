"""Path and fixture setup for standalone Sidekick unit tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path


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


_ensure_numpy_stub()


def pytest_configure(config: object) -> None:
    """Ensure src/shared/python precedes src/ so core.contracts resolves correctly.

    The project conftest adjusts sys.path when vendor/ud-tools is present.
    In bare checkout environments (no submodule init) it exits early, leaving
    src/ ahead of src/shared/python/ so src/core/ shadows src/shared/python/core/.
    This hook is a targeted fix limited to the standalone test directory.
    """
    root = Path(__file__).resolve().parents[4]
    shared_python = str((root / "src" / "shared" / "python").resolve())
    src_dir = str((root / "src").resolve())

    resolved_paths = [str(Path(p).resolve()) for p in sys.path]

    # Insert shared/python if missing
    if shared_python not in resolved_paths:
        sys.path.insert(0, shared_python)
        return

    # Move shared/python before src/ if src/ precedes it
    try:
        sp_idx = resolved_paths.index(shared_python)
        src_idx = resolved_paths.index(src_dir)
        if src_idx < sp_idx:
            original = next(
                p for p in sys.path if str(Path(p).resolve()) == shared_python
            )
            sys.path.remove(original)
            sys.path.insert(src_idx, original)
    except (ValueError, StopIteration):
        pass
