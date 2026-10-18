"""Guard default addopts exclusions and capture-contract CI coverage (#13000)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.pytest_addopts_guard import (
    REPO_ROOT,
    file_selects_marker_positively,
    load_exclusion_lane_config,
    markers_negated_in_default_addopts,
)

pytestmark = pytest.mark.unit

_CAPTURE_CONTRACT_TESTS = (
    REPO_ROOT / "tests/unit/motion_matching/test_capture_contract_generic.py"
)


def test_default_addopts_exclusions_have_documented_lanes() -> None:
    """Every marker negated in pyproject addopts has a CI lane or manual registry."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    negated = markers_negated_in_default_addopts(pyproject)
    lanes = load_exclusion_lane_config()

    missing_docs = sorted(set(negated) - set(lanes))
    assert not missing_docs, (
        "addopts excludes markers without scripts/config/pytest_addopts_exclusion_lanes.json "
        f"entries: {missing_docs}"
    )

    undocumented = sorted(set(lanes) - set(negated))
    assert not undocumented, (
        "pytest_addopts_exclusion_lanes.json lists markers not negated in addopts: "
        f"{undocumented}"
    )

    offenders: list[str] = []
    for marker, spec in lanes.items():
        if spec.get("manual_only"):
            if not spec.get("reason"):
                offenders.append(f"{marker}: manual_only requires reason")
            continue
        paths = spec.get("positive_lane_paths") or []
        if not paths:
            offenders.append(f"{marker}: missing positive_lane_paths")
            continue
        if not any(
            file_selects_marker_positively(REPO_ROOT / rel, marker) for rel in paths
        ):
            offenders.append(
                f"{marker}: none of {paths} positively select the marker in -m expressions"
            )

    assert not offenders, (
        "addopts exclusion lane registry is incomplete:\n" + "\n".join(offenders)
    )


def test_capture_contract_generic_runs_in_default_unit_lane() -> None:
    """Tracked C3D capture-contract tests must not use requires_mocap_fixtures."""
    source = _CAPTURE_CONTRACT_TESTS.read_text(encoding="utf-8")
    assert "requires_mocap_fixtures" not in source, (
        "tests/unit/motion_matching/test_capture_contract_generic.py must not carry "
        "requires_mocap_fixtures; fixtures are tracked and skipif guards absence (#13000)."
    )
