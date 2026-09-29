"""Machine-readable evidence for the two clean CRBA requalification replays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_crba_replay_report import (
    FIGURE_PATHS,
    PRIMARY_ARTIFACTS,
    build_replay_report,
)


pytestmark = pytest.mark.unit


def _article(path: Path) -> Path:
    data = path / "data"
    figures = path / "figures"
    data.mkdir(parents=True)
    figures.mkdir(parents=True)
    for index, name in enumerate(PRIMARY_ARTIFACTS):
        (data / f"{name}.json").write_text(
            json.dumps({"name": name, "index": index}) + "\n",
            encoding="utf-8",
        )
        np.savez(data / f"{name}.npz", value=np.array([index, np.nan]))
    for name in FIGURE_PATHS:
        (figures / name).write_bytes(f"figure:{name}".encode())
    (data / "claim_audit_registry.json").write_text(
        json.dumps(
            {
                "claims": [{"claim_id": "C-1", "candidate_ids": ["K-1"]}],
                "candidate_reviews": [{"candidate_id": "K-1", "claim_ids": ["C-1"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data / "articulated_crba_requalification.json").write_text(
        '{"schema_version":"fixture"}\n', encoding="utf-8"
    )
    return path


def _environment() -> dict[str, object]:
    return {
        "python_version": "3.11.14",
        "platform": "Linux",
        "machine": "x86_64",
        "numpy_version": "2.3.5",
        "mujoco_version": "3.8.0",
        "pinocchio_version": "3.8.0",
        "distribution_record_sha256": {"pin": "a" * 64, "mujoco": "b" * 64},
        "thread_limits": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def test_identical_replays_produce_a_qualified_hash_inventory(tmp_path: Path) -> None:
    first = _article(tmp_path / "first")
    second = _article(tmp_path / "second")

    report = build_replay_report(
        first,
        second,
        source_revision="1" * 40,
        environment=_environment(),
    )

    assert report["all_replay_gates_passed"] is True
    assert report["promotion_status"] == (
        "replay_qualified_pending_claim_pdf_release_and_protected_ci"
    )
    assert report["summary"] == {
        "primary_artifact_count": 6,
        "npz_member_count": 6,
        "figure_count": 11,
        "claim_registry_exact": True,
        "claim_registry_reciprocal": True,
    }


def test_replay_report_fails_closed_on_numeric_or_figure_drift(tmp_path: Path) -> None:
    first = _article(tmp_path / "first")
    second = _article(tmp_path / "second")
    np.savez(second / "data" / f"{PRIMARY_ARTIFACTS[0]}.npz", value=[99.0])

    with pytest.raises(ValueError, match="NPZ replay mismatch"):
        build_replay_report(
            first,
            second,
            source_revision="1" * 40,
            environment=_environment(),
        )


def test_replay_report_fails_closed_on_nonreciprocal_claim_registry(
    tmp_path: Path,
) -> None:
    first = _article(tmp_path / "first")
    second = _article(tmp_path / "second")
    broken = {
        "claims": [{"claim_id": "C-1", "candidate_ids": ["K-1"]}],
        "candidate_reviews": [{"candidate_id": "K-1", "claim_ids": []}],
    }
    for article in (first, second):
        (article / "data" / "claim_audit_registry.json").write_text(
            json.dumps(broken) + "\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="claim registry is not reciprocal"):
        build_replay_report(
            first,
            second,
            source_revision="1" * 40,
            environment=_environment(),
        )

    second = _article(tmp_path / "third")
    (second / "figures" / FIGURE_PATHS[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="figure replay mismatch"):
        build_replay_report(
            first,
            second,
            source_revision="1" * 40,
            environment=_environment(),
        )
