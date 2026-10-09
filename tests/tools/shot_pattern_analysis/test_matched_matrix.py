"""The study inventory preserves every club/lean/noise condition."""

import pytest
from src.tools.shot_pattern_analysis.matched_matrix import (
    matched_matrix_cells,
    noise_profile,
    validate_final_acceptance,
)

pytestmark = pytest.mark.unit


def test_complete_factorial_catalog_preserves_infeasible_candidates():
    cells = matched_matrix_cells(n_shots=32)
    assert len(cells) == 324
    assert len({name for name, _ in cells}) == 324
    assert {r.club.shaft_lean_deg for _, r in cells} == {0, 5, 10}
    assert any(
        r.club.club_id == "driver" and r.club.shaft_lean_deg == 10 for _, r in cells
    )
    assert all(r.n_shots == 32 for _, r in cells)
    assert {r.noise.standard_deviations[0] for _, r in cells} == {1, 2}


def test_correlated_profiles_have_explicit_opposite_rotation_path_signs():
    positive = noise_profile("cov_facepath_positive", 1)
    negative = noise_profile("cov_facepath_negative", 1)
    assert positive.correlation_matrix()[0, 1] == pytest.approx(-0.6)
    assert negative.correlation_matrix()[0, 1] == pytest.approx(0.6)
    assert positive.standard_deviations[3:5] == pytest.approx((0.00635, 0.00635))
    assert positive.standard_deviations[2] == pytest.approx(0.44704)


def test_common_target_is_explicit_input():
    targets = {"driver": 180.0, "seven_iron": 140.0, "pitching_wedge": 90.0}
    cells = matched_matrix_cells(n_shots=32, common_targets=targets)
    assert all(r.target_x_m == targets[r.club.club_id] for _, r in cells)


def test_profile_subset_is_explicit_complete_factorial_and_validated():
    selected = matched_matrix_cells(n_shots=32, profiles=("combined2", "shaft_only"))
    assert len(selected) == 72
    assert len({name for name, _ in selected}) == 72
    assert selected == matched_matrix_cells(
        n_shots=32, profiles=("shaft_only", "combined2")
    )
    assert all(name.endswith(("_shaft_only", "_combined2")) for name, _ in selected)
    for invalid in ((), ("shaft_only", "shaft_only"), ("not_a_profile",)):
        with pytest.raises(ValueError, match="profiles"):
            matched_matrix_cells(profiles=invalid)


def test_selected_matrix_plan_only_contains_requested_cells_and_resume_is_exact(
    tmp_path, monkeypatch
):
    import json
    import src.tools.shot_pattern_analysis.matched_matrix as module

    monkeypatch.setattr(
        module, "matched_source_snapshot", lambda: {"source_sha256": {"a": "b"}}
    )
    monkeypatch.setattr(
        module,
        "execute_and_export",
        lambda *a, **kw: pytest.fail("no cell should run in this empty shard"),
    )
    selected = ("shaft_only", "combined2")
    result = module.run_matched_matrix(
        tmp_path, profiles=selected, shard_index=72, shard_count=73
    )
    assert result == []
    plan = json.loads((tmp_path / "matrix_plan.json").read_text())
    assert len(plan["cells"]) == 72
    assert all(name.endswith(("_shaft_only", "_combined2")) for name in plan["cells"])
    assert (
        module.run_matched_matrix(
            tmp_path,
            profiles=("combined2", "shaft_only"),
            shard_index=72,
            shard_count=73,
        )
        == []
    )
    with pytest.raises(RuntimeError, match="plan differs"):
        module.run_matched_matrix(
            tmp_path, profiles=("shaft_only",), shard_index=72, shard_count=73
        )


def test_cli_profiles_are_forwarded(tmp_path, monkeypatch):
    import sys
    import src.tools.shot_pattern_analysis.matched_matrix as module

    captured = []
    monkeypatch.setattr(
        module, "run_matched_matrix", lambda *a, **kw: captured.append(kw)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["matched-matrix", str(tmp_path), "--profiles", "shaft_only", "combined2"],
    )
    module.main()
    assert captured[0]["profiles"] == ("shaft_only", "combined2")


def test_final_run_requires_source_bound_accepted_review(tmp_path):
    with pytest.raises(ValueError, match="acceptance"):
        validate_final_acceptance(None, {"source_sha256": {"a": "b"}})
    path = tmp_path / "acceptance.json"
    path.write_text(
        '{"status":"accepted_for_conditional_simulation","execution":{"source_sha256":{"a":"old"}},"checks":{"pilot":true,"numerical_refinement":true,"independent_review":true,"rendered_figures":true,"software_tests":true}}'
    )
    with pytest.raises(RuntimeError, match="source changed"):
        validate_final_acceptance(path, {"source_sha256": {"a": "b"}})


def test_source_change_before_cell_prevents_execution(tmp_path, monkeypatch):
    import src.tools.shot_pattern_analysis.matched_matrix as module

    snapshots = iter([{"source_sha256": {"a": "old"}}, {"source_sha256": {"a": "new"}}])
    monkeypatch.setattr(module, "matched_source_snapshot", lambda: next(snapshots))
    called = []
    monkeypatch.setattr(module, "execute_and_export", lambda *a, **k: called.append(a))
    with pytest.raises(RuntimeError, match="source changed"):
        module.run_matched_matrix(tmp_path)
    assert called == []
    assert not (tmp_path / "shard_0_receipt.json").exists()
