"""Corrective RED contracts for anthropometry provenance on authority (#13775)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[2]
_ANTHROPOMETRY_PREFIX = "src/shared/python/anthropometrics/"
_BODY_SEGMENT_PREFIX = "src/shared/python/body_segment_parameters/"


def _fingerprint(sources: dict[str, str]) -> str:
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_source_paths_cover_canonical_anthropometry_inputs() -> None:
    """Anthropometry inputs that feed the manufactured model must be governed."""

    paths = set(runner.SOURCE_PATHS)
    anthropometry = {path for path in paths if path.startswith(_ANTHROPOMETRY_PREFIX)}
    body_segments = {path for path in paths if path.startswith(_BODY_SEGMENT_PREFIX)}

    assert anthropometry, "SOURCE_PATHS must include anthropometrics package inputs"
    assert body_segments, "SOURCE_PATHS must include body_segment_parameters inputs"


def test_governed_source_fingerprint_changes_when_anthropometry_input_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tracked anthropometry edit must move the aggregate source fingerprint."""

    baseline = runner.governed_source_fingerprint()
    tracked = next(
        path
        for path in runner.SOURCE_PATHS
        if path.startswith((_ANTHROPOMETRY_PREFIX, _BODY_SEGMENT_PREFIX))
    )
    original = runner.governed_source_sha256()[tracked]
    real_sha256 = runner._sha256

    def _patched_sha256(path: Path) -> str:
        relative = path.relative_to(ROOT).as_posix()
        if relative == tracked:
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(runner, "_sha256", _patched_sha256)
    assert runner.governed_source_fingerprint() != baseline
    assert runner.governed_source_sha256()[tracked] == "0" * 64
    assert runner.governed_source_sha256()[tracked] != original
