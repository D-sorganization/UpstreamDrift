#!/usr/bin/env python3
"""Publish fresh job-owned Rust homes before toolchain installation (#11977)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


def prepare_rust_environment(temp_root: Path, environment_file: Path) -> dict[str, str]:
    """Allocate empty Rust homes under runner temp and append their bindings.

    Existing installations are never inspected, reused or deleted. Opening the
    environment file precedes allocation so an invalid publication destination
    cannot create job state. The runner owns the temporary directory lifecycle.
    """
    for path in (temp_root, environment_file):
        if any(character in str(path) for character in ("\r", "\n", "\0")):
            raise ValueError("Rust environment paths must be single-line paths")
    resolved_root = temp_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("Rust temporary root must be an existing directory")

    with environment_file.open("a", encoding="utf-8") as destination:
        job_root = Path(tempfile.mkdtemp(prefix="ud-rust-", dir=resolved_root))
        bindings = {
            "RUSTUP_HOME": str(job_root / "rustup"),
            "CARGO_HOME": str(job_root / "cargo"),
        }
        for value in bindings.values():
            Path(value).mkdir()
        destination.write("\n")
        destination.write(
            "".join(f"{name}={value}\n" for name, value in bindings.items())
        )
    return bindings


def main() -> int:
    """Prepare the current Actions job or report an explicit setup failure."""
    try:
        temp_root = os.environ.get("RUNNER_TEMP", "")
        environment_file = os.environ.get("GITHUB_ENV", "")
        if not temp_root.strip() or not environment_file.strip():
            raise ValueError("RUNNER_TEMP and GITHUB_ENV must be nonempty")
        prepare_rust_environment(Path(temp_root), Path(environment_file))
    except (OSError, ValueError) as error:
        logger.error("Rust environment preparation failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
