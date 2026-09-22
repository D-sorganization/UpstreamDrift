"""NM-00 (#10615): DbC identity gates that survive python -O."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SNIPPET = r"""
from src.shared.python.neural_motion.types import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactRole,
    ClaimStatus,
    Disposition,
)

try:
    ArtifactIdentity(
        artifact_id="bad.synthetic_native",
        kind=ArtifactKind.FIXTURE,
        role=ArtifactRole.SWEEP_CORPUS,
        path=None,
        exists=False,
        content_sha256=None,
        schema_version=None,
        model_ids=(),
        engine=None,
        source_revision=None,
        units=None,
        seeds=(),
        trial_ancestry=None,
        required_channels=(),
        label_availability={},
        is_synthetic_fixture=True,
        disposition=Disposition.RETAIN,
        claim_status=ClaimStatus.NATIVE_QUALIFIED,
        blockers=(),
        retrieval_instructions="n/a",
    )
except ValueError as exc:
    assert "synthetic" in str(exc).lower()
    print("OK")
else:
    raise SystemExit("expected ValueError for synthetic native claim")
"""


def test_artifact_identity_rejects_synthetic_native_under_optimize() -> None:
    """__post_init__ raises even when assert statements are stripped."""
    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-O", "-c", _SNIPPET],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
