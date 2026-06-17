# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the CLI module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from conftest import make_test_result

from movement_optimizer import cli
from movement_optimizer.cli import (
    _build_parser,
    _emit_cli_summary,
    _result_to_full_dict,
    _result_to_summary,
)


class TestCLIParser:
    def test_required_exercise(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_valid_exercise(self):
        parser = _build_parser()
        args = parser.parse_args(["--exercise", "squat"])
        assert args.exercise == "squat"

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["--exercise", "deadlift"])
        assert args.body_mass == 75.0
        assert args.height == 1.75
        assert args.bar_mass == 60.0
        assert args.duration == 2.0
        assert args.smoothness == 1.0
        assert args.output is None

    def test_custom_args(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--exercise",
                "bench_press",
                "--body-mass",
                "90",
                "--height",
                "1.85",
                "--bar-mass",
                "100",
                "--duration",
                "3.0",
                "--output",
                "out.json",
            ]
        )
        assert args.exercise == "bench_press"
        assert args.body_mass == 90.0
        assert args.bar_mass == 100.0
        assert args.output == "out.json"


class TestCLIRangeValidation:
    """Issue #400: CLI must reject out-of-range numeric arguments."""

    @pytest.mark.parametrize(
        ("flag", "value", "match"),
        [
            ("--body-mass", "-1", "body_mass"),
            ("--body-mass", "0", "body_mass"),
            ("--body-mass", "500", "body_mass"),
            ("--height", "0", "height"),
            ("--height", "5", "height"),
            ("--bar-mass", "-10", "bar_mass"),
            ("--bar-mass", "1000", "bar_mass"),
            ("--duration", "0.1", "duration"),
            ("--duration", "100", "duration"),
            ("--smoothness", "-1", "smoothness"),
            ("--smoothness", "0", "smoothness"),
            ("--smoothness", "999", "smoothness"),
        ],
    )
    def test_rejects_out_of_range(self, capsys, flag: str, value: str, match: str):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--exercise", "squat", flag, value])
        assert excinfo.value.code == 2  # argparse error exit code
        err = capsys.readouterr().err
        assert match in err
        # Helpful message includes the valid range so the user knows what to do.
        assert "[" in err and "]" in err

    def test_rejects_nan_body_mass(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--exercise", "squat", "--body-mass", "nan"])
        assert "finite" in capsys.readouterr().err


class TestResultSummary:
    def test_summary_keys(self):
        result = make_test_result()
        summary = _result_to_summary(result, "squat")
        assert summary["exercise"] == "squat"
        assert "success" in summary
        assert "cost" in summary
        assert "peak_torques_Nm" in summary
        assert "ankle" in summary["peak_torques_Nm"]

    def test_full_dict_includes_arrays(self):
        result = make_test_result()
        full = _result_to_full_dict(result, "deadlift")
        assert full["exercise"] == "deadlift"
        assert full["arrays"]["t"] == result.t.tolist()
        assert full["arrays"]["bar"] == result.bar.tolist()

    def test_emit_cli_summary_writes_json(self, monkeypatch: pytest.MonkeyPatch):
        lines: list[str] = []
        monkeypatch.setattr(cli.sys.stdout, "write", lambda text: lines.append(text))
        _emit_cli_summary({"exercise": "squat", "success": True})
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"exercise": "squat", "success": True}


class _FakeOptimizer:
    init_calls: ClassVar[list[dict[str, Any]]] = []
    next_result: ClassVar = make_test_result()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        type(self).init_calls.append({"args": args, "kwargs": kwargs})

    def optimize(self):
        return type(self).next_result


class TestMain:
    def test_main_writes_output_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        result = make_test_result(cost=12.3)
        _FakeOptimizer.init_calls = []
        _FakeOptimizer.next_result = result

        fake_config = (
            object(),
            np.zeros(3),
            np.ones(3),
            np.zeros((3, 2)),
        )
        monkeypatch.setitem(cli.EXERCISE_FACTORIES, "squat", lambda body, bar_mass: fake_config)
        monkeypatch.setattr(cli, "TrajectoryOptimizer", _FakeOptimizer)

        output_path = tmp_path / "result.json"
        exit_code = cli.main(["--exercise", "squat", "--output", str(output_path)])

        assert exit_code == 0
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written["exercise"] == "squat"
        assert written["cost"] == pytest.approx(12.3)
        assert written["arrays"]["q"] == result.q.tolist()
        assert _FakeOptimizer.init_calls[-1]["kwargs"]["duration"] == 2.0
        assert _FakeOptimizer.init_calls[-1]["kwargs"]["q_via"] is None

    def test_main_emits_summary_for_multiphase_lift(self, monkeypatch: pytest.MonkeyPatch):
        result = make_test_result(cost=7.5)
        result.success = False
        _FakeOptimizer.init_calls = []
        _FakeOptimizer.next_result = result

        q_via = np.full(3, 0.25)
        fake_config = (
            object(),
            np.zeros(3),
            np.ones(3),
            np.zeros((3, 2)),
            q_via,
        )
        emitted: list[dict[str, Any]] = []

        monkeypatch.setitem(cli.EXERCISE_FACTORIES, "clean", lambda body, bar_mass: fake_config)
        monkeypatch.setattr(cli, "TrajectoryOptimizer", _FakeOptimizer)
        monkeypatch.setattr(cli, "_emit_cli_summary", lambda summary: emitted.append(summary))

        exit_code = cli.main(["--exercise", "clean", "--duration", "1.0", "--verbose"])

        assert exit_code == 1
        assert emitted[0]["exercise"] == "clean"
        assert emitted[0]["cost"] == pytest.approx(7.5)
        assert _FakeOptimizer.init_calls[-1]["kwargs"]["duration"] == 2.0
        assert np.array_equal(_FakeOptimizer.init_calls[-1]["kwargs"]["q_via"], q_via)
