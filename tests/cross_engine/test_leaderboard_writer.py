"""Schema and behaviour tests for ``leaderboard.append_row`` (issue #4713).

The shared writer is the single point of truth for the
``reports/cross_engine_leaderboard.json`` file produced by CI. These
tests validate the exact column set called out in the issue's
acceptance criteria, plus the append-only file behaviour the workflow
relies on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.shared.python.motion_matching.leaderboard import (
    JSON_LEADERBOARD_COLUMNS,
    LeaderboardError,
    append_row,
    default_json_path,
    maybe_append_row,
)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "engine",
        "engine_version",
        "target_id",
        "theta",
        "residual_rms",
        "wallclock",
        "commit_sha",
    }
)


@dataclass
class _FakeFitResult:
    """Minimal duck-typed FitResult covering both naming conventions."""

    theta_optimal: list[float]
    final_rmse_m: float
    wall_clock_s: float
    git_commit: str
    target_hash: str = "TW_ProV1"
    timestamp_utc: str = "2026-05-08T12:00:00Z"
    method: str = "lm"
    iterations: int = 7


def _stub() -> _FakeFitResult:
    return _FakeFitResult(
        theta_optimal=[0.1, 0.2, -0.3],
        final_rmse_m=0.012,
        wall_clock_s=1.25,
        git_commit="deadbeefcafe",
    )


# --- Schema --------------------------------------------------------------


class TestSchema:
    def test_required_columns_match_issue_4713(self) -> None:
        assert REQUIRED_COLUMNS.issubset(set(JSON_LEADERBOARD_COLUMNS))
        assert JSON_LEADERBOARD_COLUMNS[0] == "engine"
        assert JSON_LEADERBOARD_COLUMNS[1] == "engine_version"
        assert "commit_sha" in JSON_LEADERBOARD_COLUMNS

    def test_writes_all_required_columns(self, tmp_path: Path) -> None:
        out = append_row(
            "pinocchio",
            _stub(),
            engine_version="3.4.0",
            json_path=tmp_path / "lb.json",
        )
        rows = json.loads(out.read_text(encoding="utf-8"))
        assert len(rows) == 1
        row = rows[0]
        for col in REQUIRED_COLUMNS:
            assert col in row, f"missing required column: {col}"

    def test_engine_must_be_known(self, tmp_path: Path) -> None:
        with pytest.raises(LeaderboardError, match="engine must be one of"):
            append_row(
                "bullet",
                _stub(),
                engine_version="1.0",
                json_path=tmp_path / "lb.json",
            )

    def test_residual_rms_required(self, tmp_path: Path) -> None:
        @dataclass
        class _NoRmse:
            theta_optimal: list[float]
            wall_clock_s: float

        with pytest.raises(LeaderboardError, match="residual_rms"):
            append_row(
                "drake",
                _NoRmse([0.0], 1.0),
                engine_version="1.0",
                json_path=tmp_path / "lb.json",
            )

    def test_residual_rms_must_be_nonnegative(self, tmp_path: Path) -> None:
        bad = _stub()
        bad.final_rmse_m = -0.001
        with pytest.raises(LeaderboardError, match="non-negative"):
            append_row(
                "drake",
                bad,
                engine_version="1.0",
                json_path=tmp_path / "lb.json",
            )

    def test_theta_must_be_vector_like(self, tmp_path: Path) -> None:
        @dataclass
        class _BadTheta:
            theta_optimal: object
            final_rmse_m: float
            wall_clock_s: float
            git_commit: str

        with pytest.raises(LeaderboardError, match="theta"):
            append_row(
                "mujoco",
                _BadTheta("not-a-vector", 0.0, 0.0, "abcdef0"),
                engine_version="1.0",
                json_path=tmp_path / "lb.json",
            )


# --- Behaviour -----------------------------------------------------------


class TestAppendBehaviour:
    def test_append_creates_parent_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "lb.json"
        append_row("drake", _stub(), "1.0", json_path=target)
        assert target.exists()

    def test_append_accumulates(self, tmp_path: Path) -> None:
        path = tmp_path / "lb.json"
        append_row("drake", _stub(), "1.0", json_path=path)
        append_row("mujoco", _stub(), "3.2", json_path=path)
        append_row("pinocchio", _stub(), "2.7", json_path=path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == 3
        assert [r["engine"] for r in rows] == ["drake", "mujoco", "pinocchio"]

    def test_target_id_override_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "lb.json"
        append_row(
            "drake",
            _stub(),
            "1.0",
            json_path=path,
            target_id="GW_wiffle",
        )
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert rows[0]["target_id"] == "GW_wiffle"

    def test_engine_version_falsy_normalised(self, tmp_path: Path) -> None:
        path = tmp_path / "lb.json"
        append_row("drake", _stub(), "", json_path=path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert rows[0]["engine_version"] == "unknown"

    def test_rejects_corrupt_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "lb.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(LeaderboardError, match="not valid JSON"):
            append_row("drake", _stub(), "1.0", json_path=path)

    def test_rejects_existing_non_list(self, tmp_path: Path) -> None:
        path = tmp_path / "lb.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(LeaderboardError, match="must be a list"):
            append_row("drake", _stub(), "1.0", json_path=path)


# --- Env-gated wrapper ---------------------------------------------------


class TestMaybeAppendRow:
    def test_no_op_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UD_LEADERBOARD_PUBLISH", raising=False)
        path = tmp_path / "lb.json"
        result = maybe_append_row("drake", _stub(), "1.0", json_path=path)
        assert result is None
        assert not path.exists()

    def test_writes_when_env_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UD_LEADERBOARD_PUBLISH", "1")
        path = tmp_path / "lb.json"
        result = maybe_append_row("drake", _stub(), "1.0", json_path=path)
        assert result == path.resolve()
        assert path.exists()

    def test_swallows_internal_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UD_LEADERBOARD_PUBLISH", "1")
        path = tmp_path / "lb.json"
        result = maybe_append_row("drake", object(), "1.0", json_path=path)
        assert result is None  # logged-and-swallowed


# --- Default path resolution --------------------------------------------


class TestDefaultJsonPath:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "UD_LEADERBOARD_JSON_PATH", "/tmp/some/path/leaderboard.json"
        )
        assert default_json_path() == Path("/tmp/some/path/leaderboard.json")

    def test_falls_back_to_repo_reports_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UD_LEADERBOARD_JSON_PATH", raising=False)
        path = default_json_path()
        assert path.name == "cross_engine_leaderboard.json"
        assert path.parent.name == "reports"
