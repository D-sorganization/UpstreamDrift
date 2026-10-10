"""Common derived-model preconditions reject stale source and occupied outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_mtp_reduction import (
    require_fresh_reduction_paths,
)

pytestmark = pytest.mark.unit


def test_reduction_paths_require_exact_source_and_fresh_output(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_text("model", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "derived.osim"

    require_fresh_reduction_paths(source, digest, output)
    with pytest.raises(ValueError, match="source model hash mismatch"):
        require_fresh_reduction_paths(source, "0" * 64, output)
    with pytest.raises(ValueError, match="source missing"):
        require_fresh_reduction_paths(tmp_path / "missing.osim", digest, output)
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_fresh_reduction_paths(source, digest, output)
    with pytest.raises(ValueError, match="parent directory missing"):
        require_fresh_reduction_paths(source, digest, tmp_path / "missing" / "new.osim")
