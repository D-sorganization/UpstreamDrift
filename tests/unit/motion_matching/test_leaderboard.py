"""Unit tests for the cross-engine leaderboard helper.

Mirrors issue #4097 acceptance: empty-results case, single-engine case,
multi-engine sorting, schema validation. No physics-engine dependencies
are required - the table-generation logic is pure Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.contracts import PreconditionError
from src.shared.python.motion_matching.leaderboard import (
    COLUMNS,
    SCHEMA_FIELDS,
    SUPPORTED_ENGINES,
    FitResult,
    Leaderboard,
    LeaderboardError,
    LeaderboardRow,
    generate_report,
    group_by_capture,
    load_results,
    render_markdown,
    render_per_capture_markdown,
    render_side_by_side_table,
)


# --- Fixtures ----------------------------------------------------------------


def _good_payload(**overrides) -> dict:
    base = {
        "engine": "simscape",
        "solver": "fmincon-sqp+ms8",
        "trial": "TW_ProV1",
        "grip_rmse_mm": 1.85,
        "clubhead_rmse_mm": 2.31,
        "body_marker_rmse_mm": 2.31,
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
            frozenset(
                {
                    "simscape",
                    "mujoco",
                    "drake",
                    "pinocchio",
                    "opensim",
                    "myosuite",
                    "pendulum",
                }
            )
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
            "body_marker_rmse_mm",
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


# --- Per-Capture Leaderboard (Issue #11170, Epic #11161) ---------------------


class TestPerCaptureLeaderboard:
    def test_capture_field_defaults_to_driver(self) -> None:
        r = FitResult(**_good_payload())
        assert r.capture == "driver"
        assert r.verdict is None
        # No verdict was ever given, so none is invented (issue #11170: a
        # PASS/FAIL must never be fabricated from an RMSE value alone).
        assert r.display_verdict == "-"

    def test_display_verdict_never_invents_a_verdict(self) -> None:
        # A row with a "good" RMSE and no verdict still reads "-", never PASS.
        good = FitResult(**_good_payload(grip_rmse_mm=0.01))
        assert good.verdict is None
        assert good.display_verdict == "-"

        # An explicit verdict is echoed back verbatim.
        explicit = FitResult(**_good_payload(verdict="FAIL"))
        assert explicit.display_verdict == "FAIL"

        # An unavailable solver reads "not run" regardless of verdict.
        unavailable = FitResult(
            **_good_payload(
                solver="unavailable: no python provider",
                grip_rmse_mm=None,
                clubhead_rmse_mm=None,
                body_marker_rmse_mm=None,
                total_work_J=None,
                wall_clock_s=None,
            )
        )
        assert unavailable.display_verdict == "not run"

    def test_capture_and_verdict_can_be_customized(self) -> None:
        r = FitResult(**_good_payload(capture="owner", verdict="QUALIFIED"))
        assert r.capture == "owner"
        assert r.verdict == "QUALIFIED"
        assert r.display_verdict == "QUALIFIED"

    def test_empty_capture_name_raises_dbc_error(self) -> None:
        with pytest.raises(PreconditionError, match="capture"):
            FitResult(**_good_payload(capture=""))
        with pytest.raises(PreconditionError, match="capture"):
            FitResult(**_good_payload(capture="   "))

    def test_group_by_capture_groups_rows(self) -> None:
        r1 = FitResult(**_good_payload(engine="simscape", capture="driver"))
        r2 = FitResult(**_good_payload(engine="mujoco", capture="driver"))
        r3 = FitResult(**_good_payload(engine="simscape", capture="owner"))
        grouped = group_by_capture([r1, r2, r3])
        assert set(grouped.keys()) == {"driver", "owner"}
        assert len(grouped["driver"]) == 2
        assert len(grouped["owner"]) == 1
        assert grouped["owner"][0].engine == "simscape"

    def test_duplicate_engine_capture_raises_dbc_error(self) -> None:
        r1 = FitResult(**_good_payload(engine="simscape", capture="driver"))
        r2 = FitResult(
            **_good_payload(engine="simscape", capture="driver", grip_rmse_mm=3.0)
        )
        with pytest.raises(PreconditionError, match="duplicate"):
            group_by_capture([r1, r2])

    def test_two_trials_same_engine_same_capture_is_not_a_duplicate(self) -> None:
        # The leaderboard is per trial: two distinct trials fit by the same
        # engine on the same capture are legitimate, not duplicates.
        r1 = FitResult(
            **_good_payload(engine="simscape", capture="driver", trial="TW_ProV1")
        )
        r2 = FitResult(
            **_good_payload(engine="simscape", capture="driver", trial="GW_wiffle")
        )
        grouped = group_by_capture([r1, r2])
        assert len(grouped["driver"]) == 2
        assert {r.trial for r in grouped["driver"]} == {"TW_ProV1", "GW_wiffle"}

    def test_duplicate_trial_engine_capture_raises_dbc_error(self) -> None:
        r1 = FitResult(
            **_good_payload(engine="simscape", capture="driver", trial="TW_ProV1")
        )
        r2 = FitResult(
            **_good_payload(
                engine="simscape",
                capture="driver",
                trial="TW_ProV1",
                grip_rmse_mm=3.0,
            )
        )
        with pytest.raises(PreconditionError, match="duplicate"):
            group_by_capture([r1, r2])

    def test_leaderboard_data_model_stores_and_groups_captures(self) -> None:
        r1 = FitResult(**_good_payload(engine="simscape", capture="driver"))
        r2 = FitResult(**_good_payload(engine="simscape", capture="owner"))
        lb = Leaderboard([r1, r2])
        assert lb.captures == ["driver", "owner"]
        assert len(lb.get_capture_rows("driver")) == 1
        assert len(lb.get_capture_rows("owner")) == 1
        filtered = lb.filter_by_capture("owner")
        assert filtered.captures == ["owner"]
        assert len(filtered.rows) == 1

    def test_two_captures_renders_per_capture_and_side_by_side_tables(self) -> None:
        r_sim_driver = FitResult(
            **_good_payload(
                engine="simscape", capture="driver", grip_rmse_mm=1.5, verdict="PASS"
            )
        )
        r_muj_driver = FitResult(
            **_good_payload(
                engine="mujoco", capture="driver", grip_rmse_mm=2.0, verdict="PASS"
            )
        )
        r_sim_owner = FitResult(
            **_good_payload(
                engine="simscape",
                capture="owner",
                grip_rmse_mm=2.5,
                verdict="QUALIFIED",
            )
        )
        r_muj_owner = FitResult(
            **_good_payload(
                engine="mujoco", capture="owner", grip_rmse_mm=1.8, verdict="PASS"
            )
        )

        rows = [r_sim_driver, r_muj_driver, r_sim_owner, r_muj_owner]
        md = render_per_capture_markdown(rows)

        # 1. One Markdown table per capture
        assert "## Capture: driver" in md
        assert "## Capture: owner" in md

        # Headers and match metrics in per-capture tables
        for cap in ("driver", "owner"):
            cap_section = md.split(f"## Capture: {cap}")[1].split("##")[0]
            assert "engine" in cap_section
            assert "grip_rmse_mm" in cap_section
            assert "clubhead_rmse_mm" in cap_section
            assert "verdict" in cap_section
            assert "simscape" in cap_section
            assert "mujoco" in cap_section

        # 2. Side-by-side table
        assert "## Side-by-Side Comparison" in md
        sbs_section = md.split("## Side-by-Side Comparison")[1]
        assert "driver grip_rmse_mm" in sbs_section
        assert "owner grip_rmse_mm" in sbs_section
        assert "driver verdict" in sbs_section
        assert "owner verdict" in sbs_section

    def test_engine_missing_on_one_capture_shows_not_run_never_zero(self) -> None:
        r_sim_driver = FitResult(
            **_good_payload(engine="simscape", capture="driver", grip_rmse_mm=1.5)
        )
        r_muj_driver = FitResult(
            **_good_payload(engine="mujoco", capture="driver", grip_rmse_mm=2.0)
        )
        r_sim_owner = FitResult(
            **_good_payload(engine="simscape", capture="owner", grip_rmse_mm=2.5)
        )
        # mujoco is missing on "owner"
        rows = [r_sim_driver, r_muj_driver, r_sim_owner]
        md = render_per_capture_markdown(rows)

        # In owner table, mujoco should show "not run"
        owner_section = md.split("## Capture: owner")[1].split("##")[0]
        assert "mujoco" in owner_section
        assert "not run" in owner_section

        # In side-by-side table under owner, mujoco should show "not run"
        sbs_section = md.split("## Side-by-Side Comparison")[1]
        mujoco_line = next(
            line
            for line in sbs_section.splitlines()
            if line.startswith("| mujoco") or "| mujoco " in line
        )
        assert "not run" in mujoco_line

        # Ensure missing values NEVER display as zero
        cells = [c.strip() for c in mujoco_line.split("|")[1:-1]]
        for cell in cells:
            assert cell != "0"
            assert cell != "0.0"
            assert cell != "0.000"

    def test_two_trials_same_engine_same_capture_both_shown_in_render(self) -> None:
        # Regression for a dict-keyed-by-engine bug that silently dropped
        # one of two trials fit by the same engine on the same capture.
        r_trial_a = FitResult(
            **_good_payload(
                engine="simscape",
                capture="driver",
                trial="TW_ProV1",
                grip_rmse_mm=1.5,
            )
        )
        r_trial_b = FitResult(
            **_good_payload(
                engine="simscape",
                capture="driver",
                trial="GW_wiffle",
                grip_rmse_mm=2.5,
            )
        )
        md = render_per_capture_markdown([r_trial_a, r_trial_b])
        driver_section = md.split("## Capture: driver")[1].split("## Capture:")[0]
        assert "TW_ProV1" in driver_section
        assert "GW_wiffle" in driver_section

        sbs_section = md.split("## Side-by-Side Comparison")[1]
        assert "TW_ProV1" in sbs_section
        assert "GW_wiffle" in sbs_section

    def test_engine_missing_on_one_capture_for_one_trial_shows_not_run(self) -> None:
        # simscape ran "owner" for TW_ProV1 only; GW_wiffle/simscape/owner
        # must show "not run" there, never be silently omitted.
        r_a_driver = FitResult(
            **_good_payload(engine="simscape", capture="driver", trial="TW_ProV1")
        )
        r_b_driver = FitResult(
            **_good_payload(engine="simscape", capture="driver", trial="GW_wiffle")
        )
        r_a_owner = FitResult(
            **_good_payload(engine="simscape", capture="owner", trial="TW_ProV1")
        )
        md = render_per_capture_markdown([r_a_driver, r_b_driver, r_a_owner])
        owner_section = md.split("## Capture: owner")[1].split("##")[0]
        assert "GW_wiffle" in owner_section
        gw_line = next(
            line for line in owner_section.splitlines() if "GW_wiffle" in line
        )
        assert "not run" in gw_line

    def test_single_capture_driver_output_identical_to_legacy(self) -> None:
        results = {
            "TW_ProV1": [
                FitResult(**_good_payload(engine="simscape", grip_rmse_mm=1.85)),
                FitResult(**_good_payload(engine="mujoco", grip_rmse_mm=2.10)),
            ]
        }
        # Calling render_markdown on single-capture driver rows gives exact legacy output
        md = render_markdown(results)
        assert "# Cross-engine leaderboard\n" in md
        assert "## TW_ProV1" in md
        assert (
            "Sorted by `grip_rmse_mm` ascending within each trial; lower is better."
            in md
        )
        assert "## Side-by-Side Comparison" not in md
        assert "1.850" in md
        assert "2.100" in md
