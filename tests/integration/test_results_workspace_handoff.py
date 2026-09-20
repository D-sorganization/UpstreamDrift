"""Integration tests for Results Workspace handoff (ORG-13, #10521).

Acceptance criteria:
- RED: opening selected run in viewer/data tool must use that run, not most recent global state.
- RED: identical filenames in different runs, missing assets, and mismatched units cannot silently compare.
- GREEN: two fixture runs -> compare/replay/export/reimport with IDs/provenance preserved; existing #10353 model reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.workspace.results_workspace import (
    ComparisonResult,
    HandoffDispatchPayload,
    MissingAssetDiagnosticError,
    ResultArtifactItem,
    ResultCategory,
    ResultsWorkspaceCoordinator,
    UnitMismatchDiagnosticError,
    WorkspaceActionType,
)
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)


@pytest.fixture
def workspace_env(tmp_path: Path) -> dict[str, Any]:
    """Create a temporary workspace environment with fixture runs and ledger."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    # Fixture Run 1: MuJoCo dynamic run
    run1_dir = runs_dir / "run_mujoco_01"
    run1_dir.mkdir()
    npz_path1 = run1_dir / "trajectory.npz"
    np.savez(
        npz_path1,
        time_s=np.linspace(0.0, 1.0, 11),
        coordinates=np.zeros((11, 6)),
        units_position="m",
        units_angles="rad",
    )
    receipt1_path = run1_dir / "receipt.json"
    receipt1_data = {
        "schema_version": "pipeline.receipt/1",
        "engine": "mujoco",
        "run_id": "run_mujoco_01",
        "status": "PASSED",
        "is_physically_accepted": True,
        "metrics": {"whole_rmse": 0.012, "club_rmse": 0.015},
    }
    receipt1_path.write_text(json.dumps(receipt1_data), encoding="utf-8")

    csv1_path = run1_dir / "measurements.csv"
    csv1_path.write_text(
        "time,speed,force\n0.0,10.0,50.0\n1.0,15.0,75.0\n", encoding="utf-8"
    )

    # Fixture Run 2: Drake dynamic run
    run2_dir = runs_dir / "run_drake_02"
    run2_dir.mkdir()
    npz_path2 = (
        run2_dir / "trajectory.npz"
    )  # Identical filename in different run directory
    np.savez(
        npz_path2,
        time_s=np.linspace(0.0, 1.0, 11),
        coordinates=np.ones((11, 6)) * 0.05,
        units_position="m",
        units_angles="rad",
    )
    receipt2_path = run2_dir / "receipt.json"
    receipt2_data = {
        "schema_version": "pipeline.receipt/1",
        "engine": "drake",
        "run_id": "run_drake_02",
        "status": "PASSED",
        "is_physically_accepted": True,
        "metrics": {"whole_rmse": 0.018, "club_rmse": 0.022},
    }
    receipt2_path.write_text(json.dumps(receipt2_data), encoding="utf-8")

    csv2_path = run2_dir / "measurements.csv"
    csv2_path.write_text(
        "time,speed,force\n0.0,11.0,52.0\n1.0,16.0,78.0\n", encoding="utf-8"
    )

    # Ledger pointing to both runs
    ledger_path = tmp_path / "reports" / "matched_swing_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_data = {
        "schema_version": "1.0.0",
        "generated_at": "2026-09-19T22:00:00Z",
        "total_receipts": 2,
        "rows": [
            {
                "receipt_path": str(receipt1_path.relative_to(tmp_path)).replace(
                    "\\", "/"
                ),
                "sha256": "1111222233334444555566667777888899990000aaaaabbbbccccddddeeeefff",
                "engine": "mujoco",
                "lane": "candidate",
                "capture": "driver",
                "artefacts": {
                    "npz": str(npz_path1.relative_to(tmp_path)).replace("\\", "/"),
                    "gif": None,
                },
                "acceptance": {"status": "PASSED", "is_physically_accepted": True},
                "metrics": {"whole_marker_rmse_m": 0.012},
            },
            {
                "receipt_path": str(receipt2_path.relative_to(tmp_path)).replace(
                    "\\", "/"
                ),
                "sha256": "222233334444555566667777888899990000aaaaabbbbccccddddeeeeffff111",
                "engine": "drake",
                "lane": "candidate",
                "capture": "driver",
                "artefacts": {
                    "npz": str(npz_path2.relative_to(tmp_path)).replace("\\", "/"),
                    "gif": None,
                },
                "acceptance": {"status": "PASSED", "is_physically_accepted": True},
                "metrics": {"whole_marker_rmse_m": 0.018},
            },
        ],
    }
    ledger_path.write_text(json.dumps(ledger_data), encoding="utf-8")

    return {
        "root": tmp_path,
        "ledger_path": ledger_path,
        "run1": {
            "run_id": "run_mujoco_01",
            "npz": npz_path1,
            "receipt": receipt1_path,
            "csv": csv1_path,
        },
        "run2": {
            "run_id": "run_drake_02",
            "npz": npz_path2,
            "receipt": receipt2_path,
            "csv": csv2_path,
        },
    }


@pytest.mark.integration
def test_red_opening_selected_run_in_viewer_or_data_tool_must_use_that_run_not_most_recent_global_state(
    workspace_env: dict[str, Any],
) -> None:
    """RED test: Opening selected run in viewer/data tool must use that run, not global state."""
    coordinator = ResultsWorkspaceCoordinator(repo_root=workspace_env["root"])

    # Register two runs in coordinator
    run1 = ResultArtifactItem(
        run_id="run_mujoco_01",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["run1"]["npz"]),
        engine="mujoco",
        units={"position": "m", "angles": "rad"},
        provenance={
            "engine_name": "mujoco",
            "run_id": "run_mujoco_01",
            "model_file_hash": "abc1234",
        },
        qualification_verdict="PASSED",
    )
    run2 = ResultArtifactItem(
        run_id="run_drake_02",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["run2"]["npz"]),
        engine="drake",
        units={"position": "m", "angles": "rad"},
        provenance={
            "engine_name": "drake",
            "run_id": "run_drake_02",
            "model_file_hash": "def5678",
        },
        qualification_verdict="PASSED",
    )

    coordinator.register_result(run1)
    coordinator.register_result(run2)

    # Set external/global state to point to run2 as the most recent run
    coordinator.set_global_active_run_id("run_drake_02")

    # Explicitly select run1 for REPLAY handoff
    payload: HandoffDispatchPayload = coordinator.prepare_tool_handoff(
        action=WorkspaceActionType.REPLAY,
        item=run1,
    )

    # Must carry run1 references, NOT the global active run2
    assert payload.run_id == "run_mujoco_01"
    assert payload.engine == "mujoco"
    assert payload.artifact_path == str(workspace_env["run1"]["npz"])
    assert payload.run_id != coordinator.get_global_active_run_id()

    # Mock tool receiving handoff
    mock_viewer_state: dict[str, Any] = {"loaded_run_id": None}

    def dummy_load(received_payload: HandoffDispatchPayload) -> None:
        mock_viewer_state["loaded_run_id"] = received_payload.run_id

    coordinator.dispatch_to_tool(payload, loader_fn=dummy_load)
    assert mock_viewer_state["loaded_run_id"] == "run_mujoco_01"


@pytest.mark.integration
def test_red_identical_filenames_different_runs_missing_assets_and_mismatched_units_cannot_silently_compare(
    workspace_env: dict[str, Any],
) -> None:
    """RED test: Identical filenames, missing assets, and mismatched units cannot silently compare."""
    coordinator = ResultsWorkspaceCoordinator(repo_root=workspace_env["root"])

    # Case A: Identical filenames in different run directories
    # Both run1 and run2 have a file named 'trajectory.npz'
    run1 = ResultArtifactItem(
        run_id="run_mujoco_01",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["run1"]["npz"]),
        engine="mujoco",
        units={"position": "m", "angles": "rad"},
        provenance={"engine_name": "mujoco", "run_id": "run_mujoco_01"},
    )
    run2 = ResultArtifactItem(
        run_id="run_drake_02",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["run2"]["npz"]),
        engine="drake",
        units={"position": "m", "angles": "rad"},
        provenance={"engine_name": "drake", "run_id": "run_drake_02"},
    )

    # Compare identical filenames without overwriting either run
    comparison: ComparisonResult = coordinator.compare_runs(run1, run2)
    assert comparison.run_a_id == "run_mujoco_01"
    assert comparison.run_b_id == "run_drake_02"
    assert comparison.path_a == str(workspace_env["run1"]["npz"])
    assert comparison.path_b == str(workspace_env["run2"]["npz"])
    assert comparison.path_a != comparison.path_b
    assert comparison.units_matched is True
    assert comparison.diff_metrics["coordinates_rmse"] > 0.0

    # Case B: Missing asset raises MissingAssetDiagnosticError without guessing substitutes
    missing_item = ResultArtifactItem(
        run_id="run_missing_03",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["root"] / "nonexistent" / "trajectory.npz"),
        engine="pinocchio",
        units={"position": "m", "angles": "rad"},
        provenance={"engine_name": "pinocchio", "run_id": "run_missing_03"},
    )

    with pytest.raises(MissingAssetDiagnosticError) as exc_missing:
        coordinator.compare_runs(run1, missing_item)

    assert "run_missing_03" in str(exc_missing.value)
    assert "Cannot resolve by guessing similarly named files" in str(exc_missing.value)

    # Case C: Mismatched units raises UnitMismatchDiagnosticError and refuses silent comparison
    mismatched_units_item = ResultArtifactItem(
        run_id="run_mismatched_04",
        project_id="proj_alpha",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(workspace_env["run2"]["npz"]),
        engine="drake",
        units={"position": "mm", "angles": "deg"},  # mm vs m!
        provenance={"engine_name": "drake", "run_id": "run_mismatched_04"},
    )

    with pytest.raises(UnitMismatchDiagnosticError) as exc_units:
        coordinator.compare_runs(run1, mismatched_units_item)

    assert "mismatched units" in str(exc_units.value).lower()
    assert "run_mujoco_01" in str(exc_units.value)
    assert "run_mismatched_04" in str(exc_units.value)


@pytest.mark.integration
def test_green_two_fixture_runs_compare_replay_export_reimport_with_ids_provenance_preserved_existing_10353_model_reused(
    workspace_env: dict[str, Any],
) -> None:
    """GREEN test: Two fixture runs compare/replay/export/reimport with IDs/provenance preserved, reusing #10353 model."""
    root = workspace_env["root"]

    # 1. Reuse existing #10353 model
    browser_model = MatchedSwingBrowserModel(repo_root=root)
    ledger_rows = browser_model.load_ledger(workspace_env["ledger_path"])
    assert len(ledger_rows) == 2

    # Filter with MatchedSwingFilter delegating to ResultFilter lineage
    swing_filter = MatchedSwingFilter(engine="mujoco")
    filtered = browser_model.filter_rows(ledger_rows, swing_filter)
    assert len(filtered) == 1
    assert filtered[0].engine == "mujoco"

    # 2. Coordinator consumes #10353 model ledger rows and classifies categories
    coordinator = ResultsWorkspaceCoordinator(repo_root=root)
    items = coordinator.index_ledger_rows(ledger_rows)
    assert len(items) >= 2

    # Verify separation of categories:
    categories = {item.category for item in items}
    assert ResultCategory.KINEMATIC_REPLAY in categories
    assert ResultCategory.QUALIFICATION_VERDICT in categories

    # 3. Artifact-type-aware action checks
    replay_item = next(
        it
        for it in items
        if it.category == ResultCategory.KINEMATIC_REPLAY and it.engine == "mujoco"
    )
    verdict_item = next(
        it
        for it in items
        if it.category == ResultCategory.QUALIFICATION_VERDICT and it.engine == "mujoco"
    )

    # Replay is enabled for kinematic replay, but disabled for qualification verdict
    replay_avail = coordinator.get_action_availability(
        WorkspaceActionType.REPLAY, replay_item
    )
    assert replay_avail.enabled is True

    verdict_replay_avail = coordinator.get_action_availability(
        WorkspaceActionType.REPLAY, verdict_item
    )
    assert verdict_replay_avail.enabled is False
    assert (
        "requires kinematic replay or dynamic run"
        in (verdict_replay_avail.reason or "").lower()
    )

    # 4. Compare two fixture runs
    run1_replay = next(
        it
        for it in items
        if it.category == ResultCategory.KINEMATIC_REPLAY and it.engine == "mujoco"
    )
    run2_replay = next(
        it
        for it in items
        if it.category == ResultCategory.KINEMATIC_REPLAY and it.engine == "drake"
    )
    comp_result = coordinator.compare_runs(run1_replay, run2_replay)
    assert comp_result.run_a_id == "run_mujoco_01"
    assert comp_result.run_b_id == "run_drake_02"
    assert comp_result.diff_metrics["coordinates_rmse"] > 0.0

    # 5. Export run with full provenance stamping (revalidating #8820)
    export_dir = root / "exports"
    export_dir.mkdir()
    exported_files = coordinator.export_result_with_provenance(
        item=run1_replay,
        export_dir=export_dir,
        formats=["csv", "json"],
    )
    assert "csv" in exported_files
    assert "json" in exported_files

    # 6. Reimport representative formats and verify round-trip
    reimported_json = coordinator.reimport_result_artifact(exported_files["json"])
    assert reimported_json.run_id == "run_mujoco_01"
    assert reimported_json.engine == "mujoco"
    assert reimported_json.units == run1_replay.units
    assert reimported_json.provenance["run_id"] == "run_mujoco_01"
    assert reimported_json.provenance["engine_name"] == "mujoco"

    reimported_csv = coordinator.reimport_result_artifact(exported_files["csv"])
    assert reimported_csv.run_id == "run_mujoco_01"
    assert reimported_csv.engine == "mujoco"


@pytest.mark.integration
def test_results_browser_hdf5_indexing_and_categorization(tmp_path: Path) -> None:
    """Test consuming canonical ResultsBrowser for HDF5 result indexing without duplicate scanning."""
    from src.shared.python.simulation_backends import (
        ProvenanceStamp,
        Trace,
        attach_provenance_to_trace,
    )
    from src.shared.python.simulation_backends.trace_io import write_trace

    stamp = ProvenanceStamp(
        engine="mujoco",
        engine_version="3.0.0",
        model_hash="hash123",
        param_hash="param456",
        git_commit="git789",
        solver_settings={"dt": 0.001},
        seed=42,
        created_at="2026-09-19T22:00:00Z",
        convention="canonical-core",
        frame="world",
        units={"length": "m", "time": "s"},
    )
    trace = attach_provenance_to_trace(
        Trace(
            t=np.array([0.0, 0.05, 0.1]),
            q=np.zeros((3, 2)),
            v=np.ones((3, 2)),
            dt=0.05,
            backend="mujoco",
            meta={
                "project_id": "proj_omega",
                "run_id": "sim_run_01",
                "kind": "dynamic_simulation",
            },
        ),
        stamp,
    )
    h5_file = tmp_path / "sim_run_01.h5"
    write_trace(trace, h5_file)

    coordinator = ResultsWorkspaceCoordinator(repo_root=tmp_path)
    indexed_items = coordinator.index_workspace_results()

    assert len(indexed_items) == 1
    item = indexed_items[0]
    assert item.run_id == "sim_run_01"
    assert item.project_id == "proj_omega"
    assert item.category == ResultCategory.DYNAMIC_RUN
    assert item.engine == "mujoco"
    assert item.provenance.get("engine") == "mujoco"


@pytest.mark.integration
def test_action_availability_rules_and_diagnostics(tmp_path: Path) -> None:
    """Test action availability rules across categories with clear diagnostic reasons."""
    coordinator = ResultsWorkspaceCoordinator(repo_root=tmp_path)

    # 1. Nonexistent file produces disabled status with clear reason
    missing_item = ResultArtifactItem(
        run_id="run_missing",
        category=ResultCategory.KINEMATIC_REPLAY,
        path=str(tmp_path / "ghost.npz"),
    )
    avail = coordinator.get_action_availability(
        WorkspaceActionType.REPLAY, missing_item
    )
    assert avail.enabled is False
    assert "does not exist on disk" in str(avail.reason)

    # 2. Existing CSV measurements file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1,2\n", encoding="utf-8")
    csv_item = ResultArtifactItem(
        run_id="run_csv",
        category=ResultCategory.MEASUREMENTS,
        path=str(csv_file),
    )

    # Data Explorer is enabled for CSV
    avail_de = coordinator.get_action_availability(
        WorkspaceActionType.DATA_EXPLORER, csv_item
    )
    assert avail_de.enabled is True

    # Replay is disabled for CSV measurements
    avail_replay = coordinator.get_action_availability(
        WorkspaceActionType.REPLAY, csv_item
    )
    assert avail_replay.enabled is False
    assert "requires kinematic replay or dynamic run" in str(avail_replay.reason)
