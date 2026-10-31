"""The docker size gates measure an engine-independent image size (#14146)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import image_content_size as ics

REPO_ROOT = Path(__file__).resolve().parents[3]
SIZE_GATES = REPO_ROOT / ".github" / "workflows" / "docker-size-gates.yml"

pytestmark = pytest.mark.unit


def _manifest(
    arch: str, content: int, total: int, *, kind: str = "image", available=True
):
    return {
        "Kind": kind,
        "Available": available,
        "Descriptor": {"platform": {"architecture": arch, "os": "linux"}},
        "Size": {"Content": content, "Total": total},
    }


def test_measures_the_compressed_content_not_the_unpacked_total() -> None:
    payload = {
        "Size": 667_750_720,
        "Manifests": [
            _manifest("amd64", 169_627_968, 667_750_720),
            _manifest("unknown", 2_474, 2_474, kind="attestation"),
            _manifest("arm64", 0, 0, available=False),
        ],
    }

    assert ics.content_size_bytes(payload, "amd64") == 169_627_968


def test_picks_the_daemon_architecture_when_several_platforms_are_present() -> None:
    payload = {
        "Manifests": [
            _manifest("arm64", 50, 500),
            _manifest("amd64", 70, 700),
        ]
    }

    assert ics.content_size_bytes(payload, "amd64") == 70


def test_fails_closed_without_per_manifest_sizes() -> None:
    # A daemon without the containerd image store returns no Manifests; falling
    # back to .Size would silently change what the budgets measure.
    with pytest.raises(ics.MeasurementError, match="Manifests"):
        ics.content_size_bytes({"Size": 123, "Manifests": None}, "amd64")


def test_fails_closed_when_no_available_manifest_matches_the_daemon() -> None:
    payload = {"Manifests": [_manifest("arm64", 50, 500)]}

    with pytest.raises(ics.MeasurementError, match="amd64"):
        ics.content_size_bytes(payload, "amd64")


def test_size_gates_never_measure_with_the_engine_dependent_size_field() -> None:
    workflow = SIZE_GATES.read_text(encoding="utf-8")

    assert "{{.Size}}" not in workflow
    assert workflow.count("python3 scripts/ci/image_content_size.py") == 2
