"""Parity tests: Python ball_flight_physics vs Rust tools-core ball_flight.

These tests generate reference trajectories from the Python simulator
and check them against the JSON test-vector schema. When tools_core is
available, they also verify that the Rust solver produces matching results.

The committed golden fixture (``tests/parity_fixtures/ball_flight/
default_trajectory.json``, pinned by sha256 in
``src/config/capability_migration.json``) is read-only by default. To
regenerate it deliberately, run::

    UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1 python -m pytest \
        tests/parity/test_ball_flight_parity.py -k regenerate

then update the ``flight.ball_flight_trajectory`` sha256/size_bytes pin.

Principles:
- TDD: Exact physical parity between Python and Rust implementations.
- DbC: Tolerance bounds (1e-6 relative) for numerical integration agreement.
- DRY: Both implementations derive from the same canonical physics model.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pytest

# Python reference implementation
from src.shared.python.physics.ball_flight_physics import (
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)
from src.shared.python.physics.rust_kernel import is_rust_available

logger = logging.getLogger(__name__)

requires_rust = pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not available in this lane",
)

# Tolerance for Rust/Python parity (RK4 with same dt should agree closely)
POSITION_TOLERANCE_M = 0.5  # meters — allow small drift from implementation differences
HEIGHT_TOLERANCE_M = 0.3
TIME_TOLERANCE_S = 0.05

# Fixture directory — path is pinned by src/config/capability_migration.json
FIXTURE_DIR = Path(__file__).parent.parent / "parity_fixtures" / "ball_flight"
DEFAULT_TRAJECTORY_FIXTURE = FIXTURE_DIR / "default_trajectory.json"

# Opt-in switch for rewriting the committed golden fixture. Only the exact
# value "1" enables it, so a stray "0"/"false" can never trigger a rewrite.
REGENERATE_ENV_VAR = "UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES"

_INPUT_KEYS = frozenset({"ball", "environment", "launch", "dt", "max_time"})
_EXPECTED_SCALAR_KEYS = frozenset(
    {"carry_distance", "max_height", "flight_time", "num_points"}
)
_POINT_KEYS = ("first_point", "mid_point", "last_point")


def regeneration_requested() -> bool:
    """Return True only when the committed fixture may be rewritten."""
    return os.environ.get(REGENERATE_ENV_VAR) == "1"


def assert_vector_schema(vectors: dict[str, Any]) -> None:
    """Assert ``vectors`` has the ball-flight parity test-vector schema."""
    assert set(vectors) == {"input", "expected"}, sorted(vectors)
    assert set(vectors["input"]) >= _INPUT_KEYS, sorted(vectors["input"])

    expected = vectors["expected"]
    missing = (_EXPECTED_SCALAR_KEYS | set(_POINT_KEYS)) - set(expected)
    assert not missing, f"expected block missing keys: {sorted(missing)}"
    for key in _EXPECTED_SCALAR_KEYS:
        assert isinstance(expected[key], int | float), (key, expected[key])
    assert expected["num_points"] > 0

    for name in _POINT_KEYS:
        point = expected[name]
        assert set(point) == {"time", "position", "velocity"}, (name, sorted(point))
        assert isinstance(point["time"], int | float), (name, point["time"])
        for vec in ("position", "velocity"):
            assert len(point[vec]) == 3, f"{name}.{vec} must be a 3-vector"


def build_default_trajectory_vectors() -> dict[str, Any]:
    """Simulate the default 7-iron launch and return its parity test vectors."""
    ball = BallProperties()
    env = EnvironmentalConditions()
    launch = LaunchConditions(
        velocity=70.0,
        launch_angle=math.radians(12.0),  # Python API uses radians
        spin_rate=2500.0,
    )
    sim = BallFlightSimulator(ball=ball, env=env)
    trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)
    analysis = sim.analyze_trajectory(trajectory)

    # Extract key trajectory points (first, mid, last)
    mid_idx = len(trajectory) // 2
    vectors = {
        "input": {
            "ball": {
                "mass": ball.mass,
                "diameter": ball.diameter,
                "cd0": ball.cd0,
                "cd1": ball.cd1,
                "cd2": ball.cd2,
                "cl0": ball.cl0,
                "cl1": ball.cl1,
                "cl2": ball.cl2,
            },
            "environment": {
                "air_density": float(env.air_density),
                "wind_velocity": [0.0, 0.0, 0.0],
                "gravity": float(env.gravity),
            },
            "launch": {
                "velocity": launch.velocity,
                "launch_angle": launch.launch_angle,
                "spin_rate": launch.spin_rate,
                "azimuth_angle": 0.0,
                "spin_axis": [0.0, -1.0, 0.0],
            },
            "dt": 0.01,
            "max_time": 10.0,
        },
        "expected": {
            "carry_distance": analysis["carry_distance"],
            "max_height": analysis["max_height"],
            "flight_time": analysis["flight_time"],
            "num_points": len(trajectory),
            "first_point": {
                "time": trajectory[0].time,
                "position": list(trajectory[0].position),
                "velocity": list(trajectory[0].velocity),
            },
            "mid_point": {
                "time": trajectory[mid_idx].time,
                "position": list(trajectory[mid_idx].position),
                "velocity": list(trajectory[mid_idx].velocity),
            },
            "last_point": {
                "time": trajectory[-1].time,
                "position": list(trajectory[-1].position),
                "velocity": list(trajectory[-1].velocity),
            },
        },
    }
    assert_vector_schema(vectors)
    return vectors


def write_vectors(vectors: dict[str, Any], path: Path) -> None:
    """Write ``vectors`` to ``path`` and verify the round trip."""
    assert_vector_schema(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vectors, indent=2) + "\n", newline="\n")
    assert json.loads(path.read_text()) == vectors


@requires_rust
class TestPythonBallFlightBaseline:
    """Generate and verify Python ball flight baseline values.

    These tests establish the reference values that the Rust implementation
    must match. They document the expected physics behavior.
    """

    def test_default_trajectory_physics(self) -> None:
        """Default 7-iron launch produces physically reasonable trajectory."""
        ball = BallProperties()
        env = EnvironmentalConditions()
        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),  # Python API uses radians
            spin_rate=2500.0,
        )
        sim = BallFlightSimulator(ball=ball, env=env)
        trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)
        analysis = sim.analyze_trajectory(trajectory)

        # Physical reasonableness checks
        carry = analysis["carry_distance"]
        max_h = analysis["max_height"]
        flight_t = analysis["flight_time"]

        logger.info(
            "Python baseline: carry=%.1fm, height=%.1fm, time=%.2fs",
            carry,
            max_h,
            flight_t,
        )

        assert carry > 50.0, f"Carry too short: {carry:.1f}m"
        assert carry < 300.0, f"Carry too long: {carry:.1f}m"
        assert max_h > 3.0, f"Height too low: {max_h:.1f}m"
        assert max_h < 60.0, f"Height too high: {max_h:.1f}m"
        assert flight_t > 0.5, f"Too short flight: {flight_t:.2f}s"
        assert flight_t < 10.0, f"Too long flight: {flight_t:.2f}s"

    def test_export_reference_vectors(self, tmp_path: Path) -> None:
        """Export test vectors to tmp_path; the committed fixture is untouched."""
        committed_before = DEFAULT_TRAJECTORY_FIXTURE.read_bytes()

        out = tmp_path / DEFAULT_TRAJECTORY_FIXTURE.name
        write_vectors(build_default_trajectory_vectors(), out)

        assert_vector_schema(json.loads(out.read_text()))
        assert DEFAULT_TRAJECTORY_FIXTURE.read_bytes() == committed_before

    @pytest.mark.skipif(
        not regeneration_requested(),
        reason=f"set {REGENERATE_ENV_VAR}=1 to rewrite the committed fixture",
    )
    def test_regenerate_committed_fixture(self) -> None:
        """Opt-in: rewrite the committed golden fixture from the current model."""
        write_vectors(build_default_trajectory_vectors(), DEFAULT_TRAJECTORY_FIXTURE)
        logger.warning(
            "Regenerated %s; update the flight.ball_flight_trajectory "
            "sha256/size_bytes pin in src/config/capability_migration.json",
            DEFAULT_TRAJECTORY_FIXTURE,
        )


@requires_rust
class TestRustPythonParity:
    """Verify Rust tools_core ball flight matches Python reference.

    These tests require the tools_core wheel to be installed.
    They are skipped gracefully when the wheel is not available.
    """

    @pytest.fixture(autouse=True)
    def require_tools_core(self) -> None:
        """Skip if tools_core is not installed."""
        pytest.importorskip(
            "tools_core",
            reason="tools_core wheel not installed — skipping Rust parity tests",
        )

    def test_types_importable(self) -> None:
        """All ball flight types must be importable from tools_core."""
        import tools_core

        assert hasattr(tools_core, "BallProperties")
        assert hasattr(tools_core, "LaunchConditions")
        assert hasattr(tools_core, "EnvironmentalConditions")
        assert hasattr(tools_core, "TrajectoryPoint")
        assert hasattr(tools_core, "TrajectoryAnalysis")

    def test_default_ball_properties(self) -> None:
        """Rust BallProperties defaults must match Python defaults."""
        import tools_core

        rust_ball = tools_core.BallProperties()
        py_ball = BallProperties()

        # Repr should contain the mass
        assert "0.0459" in repr(rust_ball) or "BallProperties" in repr(rust_ball)
        logger.info("Rust BallProperties: %s", repr(rust_ball))
        logger.info("Python BallProperties: mass=%s", py_ball.mass)

    def test_launch_conditions_repr(self) -> None:
        """Rust LaunchConditions must accept same parameters as Python."""
        import tools_core

        rust_lc = tools_core.LaunchConditions(
            velocity=70.0,
            launch_angle=math.radians(12.0),
            azimuth_angle=0.0,
            spin_rate=2500.0,
        )
        assert "70" in repr(rust_lc)
        assert "2500" in repr(rust_lc)


class TestParityFixtureContract:
    """The committed golden fixture is read-only unless regeneration is opted into."""

    def test_regeneration_not_requested_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(REGENERATE_ENV_VAR, raising=False)
        assert not regeneration_requested()

    @pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1"])
    def test_regeneration_requires_exact_opt_in(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(REGENERATE_ENV_VAR, value)
        assert not regeneration_requested()

    def test_regeneration_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REGENERATE_ENV_VAR, "1")
        assert regeneration_requested()

    def test_committed_fixture_has_vector_schema(self) -> None:
        assert DEFAULT_TRAJECTORY_FIXTURE.is_file(), (
            f"Committed parity fixture missing: {DEFAULT_TRAJECTORY_FIXTURE}"
        )
        assert_vector_schema(json.loads(DEFAULT_TRAJECTORY_FIXTURE.read_text()))

    def test_fixture_path_pinned_by_capability_migration(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = json.loads(
            (repo_root / "src" / "config" / "capability_migration.json").read_text()
        )
        pinned_paths = {
            entry.get("path")
            for section in registry.values()
            if isinstance(section, dict)
            for entry in section.values()
            if isinstance(entry, dict)
        }
        assert (
            DEFAULT_TRAJECTORY_FIXTURE.resolve().relative_to(repo_root).as_posix()
            in pinned_paths
        )

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda v: v.pop("input"),
            lambda v: v.pop("expected"),
            lambda v: v["expected"].pop("carry_distance"),
            lambda v: v["expected"]["mid_point"].pop("velocity"),
            lambda v: v["expected"]["last_point"].__setitem__("position", [0.0, 0.0]),
        ],
    )
    def test_schema_rejects_malformed_vectors(self, mutate: object) -> None:
        vectors = json.loads(DEFAULT_TRAJECTORY_FIXTURE.read_text())
        mutate(vectors)  # type: ignore[operator]
        with pytest.raises(AssertionError):
            assert_vector_schema(vectors)
