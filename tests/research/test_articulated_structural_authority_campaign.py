"""Contracts for the registered structural authority corner campaign."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_structural_authority_campaign as campaign,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def test_registered_corners_are_nominal_plus_six_oat_bounds() -> None:
    corners = campaign.registered_corners()

    assert [corner.corner_id for corner in corners] == [
        "nominal",
        "height_scale-low",
        "height_scale-high",
        "body_mass_scale-low",
        "body_mass_scale-high",
        "joint_limit_scale-low",
        "joint_limit_scale-high",
    ]
    assert [(corner.axis_name, corner.value) for corner in corners[1:]] == [
        ("height_scale", 0.90),
        ("height_scale", 1.10),
        ("body_mass_scale", 0.85),
        ("body_mass_scale", 1.15),
        ("joint_limit_scale", 0.85),
        ("joint_limit_scale", 1.15),
    ]


def test_campaign_checkpoints_every_corner_and_retains_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nominal = load_scaled_authority(
        DATA / "articulated_scaled_authority_nominal.json",
        DATA / "articulated_scaled_authority_nominal.npz",
    )
    calls: list[str] = []
    saved = {}

    def fake_build(config):
        corner = next(
            item
            for item in campaign.registered_corners()
            if item.configuration == config
        )
        calls.append(corner.corner_id)
        feasible = nominal.feasible.copy()
        failure_class = nominal.selected_failure_class.copy()
        if corner.corner_id == "joint_limit_scale-low":
            feasible[0, 4] = False
            failure_class[0, 4] = "joint_limit_failure"
        return replace(
            nominal,
            configuration=config,
            feasible=feasible,
            selected_failure_class=failure_class,
        )

    def fake_save(authority, record_path, arrays_path):
        record_path.write_text("{}", encoding="utf-8")
        arrays_path.write_bytes(b"npz")
        saved[(record_path.name, arrays_path.name)] = authority
        return {"authority_sha256": authority.authority_sha256}

    def fake_load(record_path, arrays_path):
        return saved[(record_path.name, arrays_path.name)]

    monkeypatch.setattr(campaign, "build_scaled_authority", fake_build)
    monkeypatch.setattr(campaign, "save_scaled_authority", fake_save)
    monkeypatch.setattr(campaign, "load_scaled_authority", fake_load)
    checkpoint = tmp_path / "campaign.json"

    first = campaign.run_campaign(checkpoint, artifact_directory=tmp_path)
    second = campaign.run_campaign(checkpoint, artifact_directory=tmp_path)

    assert calls == [corner.corner_id for corner in campaign.registered_corners()]
    assert first == second
    assert first["status"] == "complete"
    assert len(first["corners"]) == 7
    failed = next(
        row for row in first["corners"] if row["corner_id"] == "joint_limit_scale-low"
    )
    assert failed["status"] == "infeasible_retained"
    assert failed["failure_count"] == 1
    assert failed["failure_distribution"] == {"joint_limit_failure": 1}
    assert checkpoint.is_file()


def test_campaign_rejects_checkpoint_design_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "campaign.json"
    checkpoint.write_text(
        '{"schema_version":"articulated-structural-authority-campaign/v1",'
        '"design_sha256":"wrong","corners":[]}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="design digest"):
        campaign.run_campaign(checkpoint, artifact_directory=tmp_path)


def test_campaign_does_not_resume_terminal_rows_after_source_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "campaign.json"
    record = {
        "schema_version": "articulated-structural-authority-campaign/v1",
        "design_sha256": campaign._design_digest(),
        "source_sha256": {"source.py": "old"},
        "corners": [{"corner_id": "nominal", "status": "complete"}],
    }
    checkpoint.write_text(json.dumps(record), encoding="utf-8")

    assert campaign._existing_rows(checkpoint, {"source.py": "new"}) == {}
    assert campaign._existing_rows(checkpoint, {"source.py": "old"}) == {
        "nominal": record["corners"][0]
    }


def test_campaign_does_not_resume_an_invalid_authority_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    previous = {
        "corner_id": "nominal",
        "status": "feasible",
        "record_artifact": "authority.json",
        "array_artifact": "authority.npz",
        "authority_sha256": "expected",
    }

    def reject_artifact(*_args):
        raise RuntimeError("content digest does not match")

    monkeypatch.setattr(campaign, "load_scaled_authority", reject_artifact)

    assert (
        campaign._reusable_row(previous, campaign.registered_corners()[0], tmp_path)
        is None
    )
