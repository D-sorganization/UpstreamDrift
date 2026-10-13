"""Synthetic matrix checkpoints preserve immutable resume and failure boundaries."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import matched_matrix as module
from src.tools.shot_pattern_analysis.core import AnalysisCancelled

pytestmark = pytest.mark.unit
REFERENCE = Path(__file__).with_name("fixtures") / "matrix_orchestration_reference.json"


def _run_fixture(root, monkeypatch, mode):
    events = []
    snapshots = 0
    execution = {"source_sha256": {"synthetic": "fixed"}}
    cells = module.matched_matrix_cells(n_shots=2)[:2]

    def snapshot():
        nonlocal snapshots
        snapshots += 1
        events.append("snapshot")
        if mode == "source_change" and snapshots == 3:
            return {"source_sha256": {"synthetic": "changed"}}
        return execution

    def export(request, destination, *, figures):
        events.append(["export", destination.name, figures])
        destination.mkdir()
        (destination / "run_start.json").write_text(
            json.dumps({"config": asdict(request), "execution": execution})
        )
        if mode == "cancel":
            raise AnalysisCancelled("synthetic cancellation")
        status = "unavailable" if mode == "unavailable" else "completed"
        (destination / "summary.json").write_text(
            json.dumps({"status": status, "reason": "synthetic"})
        )
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.iterdir())
        }
        (destination / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))

    monkeypatch.setattr(module, "matched_source_snapshot", snapshot)
    monkeypatch.setattr(module, "matched_matrix_cells", lambda **kwargs: cells)
    monkeypatch.setattr(module, "execute_and_export", export)
    results = []
    try:
        results.append(module.run_matched_matrix(root, n_shots=2, figures="all"))
        if mode == "resume":
            results.append(module.run_matched_matrix(root, n_shots=2, figures="all"))
        error = None
    except (AnalysisCancelled, RuntimeError) as exc:
        error = [type(exc).__name__, str(exc)]
    files = {
        str(path.relative_to(root)): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {"events": events, "results": results, "error": error, "files": files}


@pytest.mark.parametrize(
    "mode", ["fresh", "resume", "unavailable", "cancel", "source_change"]
)
def test_matrix_checkpoint_receipts_remain_exact(tmp_path, monkeypatch, mode):
    expected = json.loads(REFERENCE.read_text())["receipts"][mode]
    assert _run_fixture(tmp_path, monkeypatch, mode) == expected


def test_selection_defaults_preserve_complete_catalog():
    selected = module.MatrixSelection()
    assert asdict(selected) == {
        "common_targets": None,
        "profiles": None,
        "selected_cell_names": None,
        "match_face_bounds_deg": None,
        "match_speed_factors": (0.75, 1.25),
        "supplement_protocol": None,
    }


def test_future_matrix_snapshot_binds_selection_and_supplement_caller():
    from src.tools.shot_pattern_analysis import matched_supplement
    from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot

    hashes = matched_source_snapshot()["source_sha256"]
    for source in (module, matched_supplement):
        path = Path(source.__file__).resolve()
        key = f"src/tools/shot_pattern_analysis/{path.name}"
        assert hashes[key] == hashlib.sha256(path.read_bytes()).hexdigest()
