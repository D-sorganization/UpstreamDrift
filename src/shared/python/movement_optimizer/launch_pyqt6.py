#!/usr/bin/env python3
"""Repository launcher for the canonical Movement Optimizer PyQt6 app."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast


class _BootstrapModule(Protocol):
    """Protocol for the dynamically loaded repository bootstrap module."""

    def bootstrap(self, caller_file: str) -> None:
        """Install repository-local import paths for a script entry point."""


def _load_bootstrap() -> _BootstrapModule:
    """Load the repository bootstrap module when launched as a script."""
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap_path = repo_root / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_tools_bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bootstrap module from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_BootstrapModule, module)


bootstrap = _load_bootstrap().bootstrap
bootstrap(__file__)

from movement_optimizer.__main__ import main as movement_optimizer_main  # noqa: E402


def main() -> int:
    """Launch the Movement Optimizer GUI."""
    return int(movement_optimizer_main(["--gui"]))


if __name__ == "__main__":
    sys.exit(main())
