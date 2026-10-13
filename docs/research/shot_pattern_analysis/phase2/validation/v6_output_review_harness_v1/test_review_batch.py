import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "review_batch", Path(__file__).with_name("review_batch.py")
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load review_batch spec or loader")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PAIR = {
    "schema": "matched-explicit-source-pair/1",
    "comparison_kind": "across_lean",
    "source_protocols": ["original_control/1", "original_control/1"],
    "sources": [
        {"protocol_id": "original_control/1"},
        {"protocol_id": "original_control/1"},
    ],
    "target_x_m": 77.40247724866299,
    "common_supported_n": 9999,
    "total_n_per_population": 10000,
}
TEXT = "Protocol: Lean 0°: original_control/1 · Lean 10°: original_control/1 | Matched Target: 77.40 m downrange | Population: Same Shot Indices Supported in All Six Populations | Valid/Total: Lean 0° 9,999/10,000 valid; Lean 10° 9,999/10,000 valid"


def test_actual_shaped_caption_passes():
    m.verify_social_caption(PAIR, "pairs", TEXT)


def test_curve_labels_adjacent_to_correct_counts_pass():
    record = {
        **PAIR,
        "comparison_kind": "curve_strength",
        "common_supported_n": 10000,
        "sources": [
            {
                "protocol_id": "original_control/1",
                "cell": "driver_lean0_curve1_rotation1_combined1",
            },
            {
                "protocol_id": "original_control/1",
                "cell": "driver_lean0_curve2_rotation1_combined1",
            },
        ],
    }
    caption = TEXT.replace(
        "Lean 0° 9,999/10,000 valid; Lean 10° 9,999/10,000 valid",
        "Curve ×1 10,000/10,000 valid; Curve ×2 10,000/10,000 valid",
    )
    m.verify_social_caption(record, "pairs", caption)
    with pytest.raises(ValueError, match="common-six counts"):
        m.verify_social_caption(
            record, "pairs", caption.replace("Curve ×2 10,000", "Curve ×2 9,999")
        )


@pytest.mark.parametrize(
    "text",
    [
        TEXT.replace("original_control/1", "original_control/10"),
        TEXT.replace("77.40 m", "177.40 m"),
        TEXT.replace("9,999/10,000", "9,998/10,000"),
        TEXT.replace("All Six", "Three"),
    ],
)
def test_wrong_caption_is_rejected(text):
    with pytest.raises(ValueError):
        m.verify_social_caption(PAIR, "pairs", text)


def test_no_completion_boundary_is_pending_without_reading_live_assets(tmp_path):
    assert m.completion_boundary(tmp_path, tmp_path / "absent_exit.json", {}) is None


def test_nonterminal_or_failed_boundary_is_pending(tmp_path):
    p = tmp_path / "exit.json"
    for record in (
        {"status": "running"},
        {"status": "failed_not_accepted", "process_exit_code": 143},
    ):
        p.write_text(json.dumps(record))
        assert m.completion_boundary(tmp_path, p, {}) is None


@pytest.fixture
def completed(tmp_path):
    ledger = tmp_path / "REPORT_SUPPLEMENT_FIGURES.json"
    ledger.write_text('{"figures":[]}')
    pipeline = tmp_path / "REPORT_SUPPLEMENT_PIPELINE.json"
    pipeline.write_text(
        json.dumps(
            {
                "postprocessing_source_sha256": {"x": "hash"},
                "artifact_sha256": {"figure_ledger": m.sha(ledger)},
            }
        )
    )
    boundary = tmp_path / "exit.json"
    boundary.write_text(
        json.dumps(
            {
                "status": "completed_pending_full_independent_acceptance",
                "process_exit_code": 0,
                "producer_sha256": {"x": "hash"},
                "pipeline_receipt_sha256": m.sha(pipeline),
                "command": ["python", "--output-dir", str(tmp_path)],
            }
        )
    )
    start = tmp_path / "start.json"
    record = json.loads(boundary.read_text())
    record.pop("pipeline_receipt_sha256")
    record.pop("process_exit_code")
    record["status"] = "running"
    start.write_text(json.dumps(record))
    return tmp_path, boundary, start, m.sha(start)


def test_completed_source_bound_boundary_passes(completed):
    root, boundary, start, start_sha = completed
    assert (
        m.completion_boundary(root, boundary, {"x": "hash"}, (start, start_sha))
        is not None
    )


@pytest.mark.parametrize("mutation", ["source", "receipt", "ledger", "output"])
def test_boundary_tampering_rejected(completed, mutation):
    root, boundary, start, start_sha = completed
    sources = {"x": "other"} if mutation == "source" else {"x": "hash"}
    if mutation == "receipt":
        (root / "REPORT_SUPPLEMENT_PIPELINE.json").write_text("{}")
    if mutation == "ledger":
        (root / "REPORT_SUPPLEMENT_FIGURES.json").write_text("{}")
    if mutation == "output":
        record = json.loads(boundary.read_text())
        record["command"][-1] = str(root / "other")
        boundary.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        m.completion_boundary(root, boundary, sources, (start, start_sha))


@pytest.mark.parametrize("defect", ["missing", "changed", "unbound"])
def test_assets_fail_closed_after_completion(tmp_path, defect):
    p = tmp_path / "figures" / "test.svg"
    p.parent.mkdir()
    p.write_text("<svg/>")
    bound = {"figures/test.svg": m.sha(p)}
    if defect == "missing":
        p.unlink()
    if defect == "changed":
        p.write_text("<svg>changed</svg>")
    if defect == "unbound":
        bound = {}
    with pytest.raises(ValueError):
        m.verify_files(tmp_path, [p], bound)


def test_expected_numeric_hash_change_rejected(tmp_path):
    p = tmp_path / "one.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        m.verify_files(tmp_path, [p], {"one.json": "0" * 64})


def test_main_defers_live_writes_before_any_numeric_or_asset_read(
    tmp_path, monkeypatch
):
    import sys
    import types

    figure = types.ModuleType(
        "src.tools.shot_pattern_analysis.matched_supplement_figures"
    )
    for name in ("_slug", "pair_comparison_identity", "pair_comparison_slug"):
        monkeypatch.setattr(figure, name, lambda *a: None, raising=False)
    pipeline = types.ModuleType(
        "src.tools.shot_pattern_analysis.matched_supplement_pipeline"
    )
    monkeypatch.setattr(pipeline, "_report_source_hashes", dict, raising=False)

    def forbidden(*args: object) -> None:
        raise AssertionError("live outputs must not be read")

    monkeypatch.setattr(
        pipeline, "_check_numeric_parity_reference", forbidden, raising=False
    )
    monkeypatch.setattr(
        pipeline, "_load_numeric_parity_reference", forbidden, raising=False
    )
    monkeypatch.setitem(sys.modules, figure.__name__, figure)
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_batch.py",
            "--repo",
            str(tmp_path),
            "--report",
            str(tmp_path / "live"),
            "--completion-receipt",
            str(tmp_path / "missing_exit"),
            "--start-receipt",
            str(tmp_path / "start"),
            "--start-sha256",
            "unused_while_pending",
            "--expected-numeric",
            str(tmp_path / "missing_hashes"),
            "--output",
            str(tmp_path / "evidence"),
            "--group",
            "pairs",
        ],
    )
    m.main()
    result = json.loads((tmp_path / "evidence/batch_inventory.json").read_text())
    assert result["status"] == "pending_completed_producer_boundary"
    assert result["assets_read"] == 0 and not result["accepted"]


def test_six_source_bound_saved_caption_fixtures():
    rows = json.loads(
        Path(__file__).with_name("saved_caption_fixtures.json").read_text()
    )
    assert len(rows) == 6
    for row in rows:
        m.verify_social_caption(row["record"], row["group"], row["text"])


def test_wrong_cell_population_count_rejected():
    rows = json.loads(
        Path(__file__).with_name("saved_caption_fixtures.json").read_text()
    )
    row = next(r for r in rows if r["group"] == "cells")
    with pytest.raises(ValueError):
        m.verify_social_caption(
            row["record"], "cells", row["text"].replace("9,999/10,000", "9,998/10,000")
        )


@pytest.mark.parametrize("mutation", ["command", "start_hash"])
def test_terminal_claim_cannot_override_pinned_start(completed, mutation):
    root, boundary, start, start_sha = completed
    if mutation == "command":
        record = json.loads(boundary.read_text())
        record["command"].extend(["--inventory", "forged.json"])
        boundary.write_text(json.dumps(record))
    else:
        start.write_text(start.read_text() + " ")
    with pytest.raises(ValueError):
        m.completion_boundary(root, boundary, {"x": "hash"}, (start, start_sha))


def test_actual_wrapper_start_schema_and_complete_command_binding(tmp_path):
    actual = json.loads(
        Path(__file__).with_name("actual_v6_start_fixture.json").read_text()
    )
    assert "--output-dir" in actual["command"] and "--output" not in actual["command"]
    assert actual["candidate_commit"] == "4023111d2aaf8b20449a36f8691a295f4a94ceff"
    actual["command"][actual["command"].index("--output-dir") + 1] = str(tmp_path)
    start = tmp_path / "start.json"
    start.write_text(json.dumps(actual))
    pin = (start, m.sha(start))
    ledger = tmp_path / "REPORT_SUPPLEMENT_FIGURES.json"
    ledger.write_text('{"figures":[]}')
    pipeline = tmp_path / "REPORT_SUPPLEMENT_PIPELINE.json"
    pipeline.write_text(
        json.dumps(
            {
                "postprocessing_source_sha256": actual["producer_sha256"],
                "artifact_sha256": {"figure_ledger": m.sha(ledger)},
            }
        )
    )
    complete = {
        **actual,
        "status": "completed_pending_full_independent_acceptance",
        "process_exit_code": 0,
        "finished_utc": "fixture",
        "pipeline_receipt_sha256": m.sha(pipeline),
    }
    exit_path = tmp_path / "exit.json"
    exit_path.write_text(json.dumps(complete))
    assert m.completion_boundary(tmp_path, exit_path, actual["producer_sha256"], pin)
    complete["command"][complete["command"].index("--inventory") + 1] = (
        "forged-inventory.json"
    )
    exit_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="pinned producer start"):
        m.completion_boundary(tmp_path, exit_path, actual["producer_sha256"], pin)
