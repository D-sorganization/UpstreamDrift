"""The F03 native marker receipt must fail closed on incomplete evidence."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _junit(path: Path, evidence: dict[str, object], *, skipped: bool = False) -> None:
    property_value = escape(json.dumps(evidence), quote=True)
    skip_xml = "<skipped/>" if skipped else ""
    path.write_text(
        "<testsuite>"
        '<testcase name="test_native_marker_cost_has_correct_tangent_gradient_and_exact_clock"/>'
        '<testcase name="test_native_marker_fit_improves_reachable_observations_and_replays">'
        f'{skip_xml}<properties><property name="f03_marker_fit_evidence" '
        f'value="{property_value}"/></properties></testcase></testsuite>',
        encoding="utf-8",
    )


def _evidence() -> dict[str, object]:
    hashes = (
        "observation_sha256",
        "source_model_sha256",
        "loaded_native_model_sha256",
        "initial_state_sha256",
        "policy_sha256",
        "time_grid_sha256",
        "applied_input_sha256",
    )
    return {
        **dict.fromkeys(hashes, "a" * 64),
        "masked_site_samples": 1,
        "observed_site_samples": 15,
        "accepted_commands": 4,
        "steps": 4,
        "full_trajectory_accepted": True,
        "statuses": ["optimized"] * 4,
        "fit_marker_rmse_m": 0.001,
        "zero_marker_rmse_m": 0.01,
        "max_full_state_replay_error": 0.0,
        "total_wall_s": 0.5,
        "total_cpu_s": 0.2,
    }


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"accepted_commands": 3}, "accepted"),
        ({"full_trajectory_accepted": False}, "trajectory"),
        ({"fit_marker_rmse_m": 0.02}, "improvement"),
        ({"masked_site_samples": 0}, "mask"),
        ({"initial_state_sha256": "unknown"}, "SHA-256"),
        ({"total_wall_s": -1.0}, "timing"),
        ({"invented": 1}, "fields"),
    ],
)
def test_receipt_rejects_contradictory_native_evidence(
    tmp_path: Path, change: dict[str, object], error: str
) -> None:
    from scripts.f03_native_marker_receipt import extract_native_evidence

    evidence = _evidence() | change
    path = tmp_path / "native.xml"
    _junit(path, evidence)
    with pytest.raises(ValueError, match=error):
        extract_native_evidence(path)


def test_receipt_rejects_skipped_native_test(tmp_path: Path) -> None:
    from scripts.f03_native_marker_receipt import extract_native_evidence

    path = tmp_path / "native.xml"
    _junit(path, _evidence(), skipped=True)
    with pytest.raises(ValueError, match="passed"):
        extract_native_evidence(path)


def test_receipt_rejects_missing_gradient_companion(tmp_path: Path) -> None:
    from scripts.f03_native_marker_receipt import extract_native_evidence

    path = tmp_path / "native.xml"
    _junit(path, _evidence())
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            '<testcase name="test_native_marker_cost_has_correct_tangent_gradient_and_exact_clock"/>',
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inventory"):
        extract_native_evidence(path)
