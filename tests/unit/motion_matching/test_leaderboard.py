"""Unit tests for the cross-engine leaderboard helper.

Mirrors issue #4097 acceptance: empty-results case, single-engine case,
multi-engine sorting, schema validation. No physics-engine dependencies
are required - the table-generation logic is pure Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.motion_matching.leaderboard import (
    COLUMNS,
    SCHEMA_FIELDS,
    SUPPORTED_ENGINES,
    FitResult,
    LeaderboardError,
    generate_report,
    load_results,
    render_markdown,
)

# --- Fixtures ----------------------------------------------------------------


def _good_payload(**overrides) -> dict:
    base = {
        "engine": "simscape",
        "solver": "fmincon-sqp+ms8",
        "trial": "TW_ProV1",
        "grip_rmse_mm": 1.85,
        "clubhead_rmse_mm": 2.31,
        "total_work_J": 284.0,
        "wall_clock_s": 252.4,
        "commit": "7a3f1c2",
        "run_at": "2026-05-05T17:34:21Z",
    }
    base.update(overrides)
    return base


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- Schema --------------------------------------------------------------


class TestFitResultSchema:
    def test_constructs_with_valid_fields(self) -> None:
        r = FitResult(**_good_payload())
        assert r.engine == "simscape"
        assert r.trial == "TW_ProV1"
        assert r.grip_rmse_mm == pytest.approx(1.85)

    def test_rejects_unknown_engine(self) -> None:
        with pytest.raises(LeaderboardError, match="engine must be one of"):
            FitResult(**_good_payload(engine="bullet"))

    def test_rejects_negative_rmse(self) -> None:
        with pytest.raises(LeaderboardError, match="grip_rmse_mm"):
            FitResult(**_good_payload(grip_rmse_mm=-1.0))

    def test_rejects_negative_wall_clock(self) -> None:
        with pytest.raises(LeaderboardError, match="wall_clock_s"):
            FitResult(**_good_payload(wall_clock_s=-0.001))

    def test_rejects_non_hex_commit(self) -> None:
        with pytest.raises(LeaderboardError, match="commit"):
            FitResult(**_good_payload(commit="not-a-sha"))

    def test_rejects_too_short_commit(self) -> None:
        with pytest.raises(LeaderboardError, match="commit"):
            FitResult(**_good_payload(commit="abc"))

    def test_rejects_non_iso8601_timestamp(self) -> None:
        with pytest.raises(LeaderboardError, match="run_at"):
            FitResult(**_good_payload(run_at="yesterday"))

    def test_rejects_naive_iso8601_timestamp(self) -> None:
        with pytest.raises(LeaderboardError, match="run_at"):
            FitResult(**_good_payload(run_at="2026-05-05T17:34:21"))  # missing Z

    def test_accepts_iso8601_with_microseconds(self) -> None:
        r = FitResult(**_good_payload(run_at="2026-05-05T17:34:21.123456Z"))
        assert r.run_at.endswith("Z")

    def test_rejects_empty_solver(self) -> None:
        with pytest.raises(LeaderboardError, match="solver"):
            FitResult(**_good_payload(solver=""))

    def test_rejects_empty_trial(self) -> None:
        with pytest.raises(LeaderboardError, match="trial"):
            FitResult(**_good_payload(trial=""))

    def test_leaderboard_frozen(self) -> None:
        import dataclasses

        r = FitResult(**_good_payload())
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.engine = "mujoco"  # type: ignore[misc]

    def test_supported_engines_set(self) -> None:
        assert (
            frozenset({"simscape", "mujoco", "drake", "pinocchio", "opensim"})
            == SUPPORTED_ENGINES
        )

    def test_columns_match_issue_spec(self) -> None:
        # Per #4097: engine, solver, grip_rmse_mm, clubhead_rmse_mm,
        # total_work_J, wall_clock_s, commit, run_at.
        assert COLUMNS == (
            "engine",
            "solver",
            "grip_rmse_mm",
            "clubhead_rmse_mm",
            "total_work_J",
            "wall_clock_s",
            "commit",
            "run_at",
        )

    def test_schema_fields_documented(self) -> None:
        # SCHEMA_FIELDS exposes the dataclass field names; useful for
        # downstream introspection and stable for tests.
        assert "trial" in SCHEMA_FIELDS
        for col in COLUMNS:
            assert col in SCHEMA_FIELDS


class TestFitResultFromDict:
    def test_from_dict_roundtrip(self) -> None:
        r = FitResult.from_dict(_good_payload(), trial="TW_ProV1")
        assert r.engine == "simscape"

    def test_from_dict_trial_mismatch_raises(self) -> None:
        with pytest.raises(LeaderboardError, match="trial mismatch"):
            FitResult.from_dict(_good_payload(trial="TW_ProV1"), trial="GW_wiffle")

    def test_from_dict_missing_field_raises(self) -> None:
        payload = _good_payload()
        del payload["solver"]
        with pytest.raises(LeaderboardError, match="missing required field"):
            FitResult.from_dict(payload, trial="TW_ProV1")

    def test_from_dict_extra_fields_ignored(self) -> None:
        payload = _good_payload(coefficients=[[1, 2, 3]], n_iterations=42)
        r = FitResult.from_dict(payload, trial="TW_ProV1")
        assert r.engine == "simscape"

    def test_from_dict_rejects_non_object(self) -> None:
        with pytest.raises(LeaderboardError, match="expected JSON object"):
            FitResult.from_dict([1, 2, 3], trial="TW_ProV1")  # type: ignore[arg-type]


# --- load_results ------------------------------------------------------------


class TestLoadResults:
    def test_empty_results_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        assert load_results(tmp_path) == {}

    def test_nonexistent_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        assert load_results(tmp_path / "missing") == {}

    def test_path_to_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hi")
        with pytest.raises(LeaderboardError, match="not a directory"):
            load_results(f)

    def test_single_engine_single_trial(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "TW_ProV1" / "simscape.json", _good_payload())
        results = load_results(tmp_path)
        assert set(results) == {"TW_ProV1"}
        assert len(results["TW_ProV1"]) == 1
        assert results["TW_ProV1"][0].engine == "simscape"

    def test_multi_engine_multi_trial(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / "TW_ProV1" / "simscape.json",
            _good_payload(engine="simscape", grip_rmse_mm=2.3),
        )
        _write_json(
            tmp_path / "TW_ProV1" / "mujoco.json",
            _good_payload(engine="mujoco", grip_rmse_mm=3.7, solver="cma-es"),
        )
        _write_json(
            tmp_path / "GW_wiffle" / "drake.json",
            _good_payload(
                engine="drake",
                trial="GW_wiffle",
                grip_rmse_mm=2.4,
                solver="ipopt",
            ),
        )
        results = load_results(tmp_path)
        assert set(results) == {"TW_ProV1", "GW_wiffle"}
        assert {r.engine for r in results["TW_ProV1"]} == {"simscape", "mujoco"}
        assert {r.engine for r in results["GW_wiffle"]} == {"drake"}

    def test_unrecognised_filename_skipped(self, tmp_path: Path) -> None:
        _write_json(tmp_path / "TW_ProV1" / "simscape.json", _good_payload())
        # Stray sibling file: not a recognised engine, ignored.
        (tmp_path / "TW_ProV1" / "notes.json").write_text("{}", encoding="utf-8")
        (tmp_path / "TW_ProV1" / "bullet.json").write_text("{}", encoding="utf-8")
        results = load_results(tmp_path)
        assert len(results["TW_ProV1"]) == 1

    def test_missing_engine_field_filled_from_filename(self, tmp_path: Path) -> None:
        # Engines may be lazy and not include their own name; the loader
        # injects it from the file stem.
        payload = _good_payload()
        del payload["engine"]
        _write_json(tmp_path / "TW_ProV1" / "mujoco.json", payload)
        results = load_results(tmp_path)
        assert results["TW_ProV1"][0].engine == "mujoco"

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "TW_ProV1" / "simscape.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(LeaderboardError, match="could not parse"):
            load_results(tmp_path)

    def test_non_path_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="results_dir"):
            load_results("some/path")  # type: ignore[arg-type]


# --- render_markdown ---------------------------------------------------------


class TestRenderMarkdown:
    def test_empty_renders_skip_message(self) -> None:
        md = render_markdown({})
        assert "No FitResult JSON files found" in md
        assert "honestly skipped" in md

    def test_single_engine_renders_table(self) -> None:
        rows = {"TW_ProV1": [FitResult(**_good_payload())]}
        md = render_markdown(rows)
        # Header + columns + at least one row.
        assert "## TW_ProV1" in md
        assert "engine" in md and "solver" in md and "grip_rmse_mm" in md
        assert "simscape" in md
        # 1.85 mm should round-trip with 3 decimals.
        assert "1.850" in md

    def test_multi_engine_sorted_by_grip_rmse_ascending(self) -> None:
        rows = {
            "TW_ProV1": [
                FitResult(**_good_payload(engine="simscape", grip_rmse_mm=2.3)),
                FitResult(**_good_payload(engine="mujoco", grip_rmse_mm=1.1)),
                FitResult(**_good_payload(engine="drake", grip_rmse_mm=3.7)),
            ]
        }
        md = render_markdown(rows)
        # The most accurate engine should appear first within the trial.
        body = md.split("## TW_ProV1", 1)[1]
        idx_mujoco = body.index("mujoco")
        idx_simscape = body.index("simscape")
        idx_drake = body.index("drake")
        assert idx_mujoco < idx_simscape < idx_drake

    def test_multi_trial_sorted_alphabetically(self) -> None:
        rows = {
            "TW_ProV1": [FitResult(**_good_payload(engine="simscape"))],
            "GW_wiffle": [
                FitResult(**_good_payload(engine="drake", trial="GW_wiffle"))
            ],
        }
        md = render_markdown(rows)
        idx_gw = md.index("## GW_wiffle")
        idx_tw = md.index("## TW_ProV1")
        assert idx_gw < idx_tw

    def test_render_is_deterministic(self) -> None:
        rows = {
            "TW_ProV1": [
                FitResult(**_good_payload(engine="mujoco", grip_rmse_mm=1.1)),
                FitResult(**_good_payload(engine="simscape", grip_rmse_mm=2.3)),
            ]
        }
        # Same input must produce byte-identical output across calls.
        assert render_markdown(rows) == render_markdown(rows)

    def test_columns_appear_in_canonical_order(self) -> None:
        rows = {"TW_ProV1": [FitResult(**_good_payload())]}
        md = render_markdown(rows)
        header_line = next(
            line for line in md.splitlines() if "engine" in line and "solver" in line
        )
        positions = [header_line.index(col) for col in COLUMNS]
        assert positions == sorted(positions)


# --- generate_report end-to-end ---------------------------------------------


class TestGenerateReport:
    def test_writes_file_for_empty_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "out" / "LB.md"
        result = generate_report(tmp_path / "results", out)
        assert result == out.resolve()
        assert out.exists()
        assert "No FitResult JSON files found" in out.read_text(encoding="utf-8")

    def test_writes_file_for_populated_dir(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        _write_json(
            results_dir / "TW_ProV1" / "simscape.json",
            _good_payload(engine="simscape", grip_rmse_mm=2.3),
        )
        _write_json(
            results_dir / "TW_ProV1" / "pinocchio.json",
            _good_payload(engine="pinocchio", grip_rmse_mm=1.7, solver="lm"),
        )
        out = tmp_path / "LB.md"
        generate_report(results_dir, out)
        text = out.read_text(encoding="utf-8")
        assert "## TW_ProV1" in text
        assert "simscape" in text and "pinocchio" in text
        assert text.endswith("\n")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "deeply" / "nested" / "missing" / "LB.md"
        generate_report(tmp_path / "results", out)
        assert out.exists()

    def test_non_path_args_raise_typeerror(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            generate_report("results", tmp_path / "out.md")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            generate_report(tmp_path, "out.md")  # type: ignore[arg-type]
