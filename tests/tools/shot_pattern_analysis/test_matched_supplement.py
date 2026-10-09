"""Predeclared supplements select existing matrix cells without copying physics."""

from __future__ import annotations

import pytest

from src.tools.shot_pattern_analysis import matched_supplement

pytestmark = pytest.mark.unit


def test_three_protocols_have_exact_reviewed_distinct_cell_inventories():
    primary = matched_supplement.supplement_cell_names("pw_primary_face")
    pw_control = matched_supplement.supplement_cell_names("pw_control_face")
    driver_control = matched_supplement.supplement_cell_names("driver_control_speed")
    assert len(primary) == 18
    assert len(pw_control) == len(driver_control) == 4
    assert len({*primary, *pw_control, *driver_control}) == 26
    assert all(name.startswith("pitching_wedge_lean0_curve2_") for name in primary)
    assert all(name.startswith("pitching_wedge_lean5_curve2_") for name in pw_control)
    assert all(name.startswith("driver_lean5_curve2_") for name in driver_control)
    assert {name.rsplit("_", 1)[-1] for name in primary} >= {"only", "combined2"}
    aliases = matched_supplement.pw_control_alias_candidates()
    assert len(aliases) == 4
    assert set(aliases) <= set(primary)
    assert all(source == destination for source, destination in aliases.items())


def test_supplement_runner_reuses_matrix_pipeline_and_separates_speed_protocol(
    monkeypatch, tmp_path
):
    captured = []
    monkeypatch.setattr(
        matched_supplement,
        "run_matched_matrix",
        lambda *args, **kwargs: captured.append((args, kwargs)) or [],
    )
    matched_supplement.run_supplement_matrix(
        "pw_primary_face", tmp_path / "primary", n_shots=32
    )
    matched_supplement.run_supplement_matrix(
        "pw_control_face", tmp_path / "control", n_shots=32
    )
    matched_supplement.run_supplement_matrix(
        "driver_control_speed", tmp_path / "speed", n_shots=32
    )
    assert [len(item[1]["selected_cell_names"]) for item in captured] == [18, 4, 4]
    assert all(item[1]["match_face_bounds_deg"] == (-6.0, 6.0) for item in captured)
    assert [item[1]["match_speed_factors"] for item in captured] == [
        (0.75, 1.25),
        (0.75, 1.25),
        (0.75, 1.30),
    ]
    assert captured[0][1]["common_targets"] is None
    assert captured[1][1]["common_targets"] == {
        "pitching_wedge": pytest.approx(77.40247724866299)
    }
    assert captured[2][1]["common_targets"] == {
        "driver": pytest.approx(178.94875782337786)
    }
    assert len({item[1]["supplement_protocol"] for item in captured}) == 3
    with pytest.raises(ValueError, match="protocol"):
        matched_supplement.supplement_cell_names("unknown")
