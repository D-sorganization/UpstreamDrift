"""Frozen-source contracts for supplemental matched-shot trajectories."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.research.shot_pattern_analysis.phase2.presentation_v2 import replay

pytestmark = pytest.mark.unit


def _path(carry: float) -> list[list[float]]:
    return [[carry * i / 40, 0] for i in range(41)]


def _runtime() -> dict:
    return {
        "native_binary_sha256": "native",
        "numerical_runtime": {"python": "3.12", "numpy": "2.2"},
        "contact_backend": {"active_backend": "numba"},
        "source_sha256": dict.fromkeys(replay.PHYSICS_SOURCES, "source-hash"),
    }


def _write_cell(root: Path, supported: set[int], *, offset: float = 0) -> Path:
    root.mkdir(parents=True)
    fields = [
        "pattern",
        "shot_index",
        "status",
        "shaft_rotation_deg",
        "path_deg",
        "speed_mps",
        "contact_horizontal_m",
        "contact_vertical_m",
        "shaft_lean_deg",
        "attack_angle_deg",
        "carry_x_m",
        "carry_y_m",
        "spin_rpm",
        "launch_azimuth_deg",
        "spin_axis_tilt_deg",
        "launch_elevation_deg",
        "ball_speed_mps",
        "face_deg",
        "loft_deg",
        "contact_face_deg",
        "contact_loft_deg",
    ]
    with gzip.open(root / "shots.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pattern in replay.PATTERNS:
            for index in range(110):
                writer.writerow(
                    {
                        "pattern": pattern,
                        "shot_index": index,
                        "status": "supported" if index in supported else "off_face",
                        "shaft_rotation_deg": index,
                        "path_deg": 0.2,
                        "speed_mps": 45,
                        "contact_horizontal_m": 0,
                        "contact_vertical_m": 0,
                        "shaft_lean_deg": 0,
                        "attack_angle_deg": -1,
                        "carry_x_m": 100 + index + offset,
                        "carry_y_m": 0,
                        "spin_rpm": 2500,
                        "launch_azimuth_deg": 0,
                        "spin_axis_tilt_deg": 0,
                        "launch_elevation_deg": 10,
                        "ball_speed_mps": 60,
                        "face_deg": 0,
                        "loft_deg": 10,
                        "contact_face_deg": 0,
                        "contact_loft_deg": 10,
                    }
                )
    (root / "flight_samples.json").write_text(
        json.dumps(
            {
                "target_x_m": 100,
                "trajectories": {
                    p: [_path(100)] + [_path(100 + i + offset) for i in range(5)]
                    for p in replay.PATTERNS
                },
            }
        )
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "nominals": {
                    p: {"face_deg": 0, "path_deg": 0, "speed_mps": 45}
                    for p in replay.PATTERNS
                },
            }
        )
    )
    (root / "run_start.json").write_text(
        json.dumps(
            {
                "config": {
                    "club": {
                        "club_id": "driver",
                        "club_speed_mps": 45,
                        "shaft_lean_deg": 0,
                        "attack_angle_deg": -1,
                        "dt_s": 0.02,
                        "max_time_s": 12,
                    }
                },
                "execution": _runtime(),
            }
        )
    )
    files = ("shots.csv.gz", "flight_samples.json", "summary.json", "run_start.json")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in files
                }
            }
        )
    )
    return root


def test_plan_uses_same_supported_shot_ids_in_every_population(tmp_path):
    a = _write_cell(tmp_path / "a", set(range(110)))
    b = _write_cell(tmp_path / "b", set(range(110)) - {50})
    plan = replay.plan_replay({"lean0": a, "lean5": b}, target=100)
    assert len(plan.shot_indices) == 100
    assert 50 not in plan.shot_indices
    assert not set(range(5)) & set(plan.shot_indices)
    assert (
        plan.shot_indices == replay.plan_replay({"lean0": a, "lean5": b}).shot_indices
    )


def test_plan_rejects_changed_frozen_input(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    (cell / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        replay.plan_replay({"baseline": cell})


def test_execution_checks_saved_paths_before_new_shots(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    plan = replay.plan_replay({"baseline": cell}, target=10)

    class Fake:
        def __init__(self):
            self.calls = []

        def simulate(self, **kw):
            self.calls.append("nominal")
            return SimpleNamespace(trajectory_xy_m=_path(100))

        def simulate_delivery(self, **kw):
            self.calls.append("shot")
            return SimpleNamespace(
                delivered_face_deg=0,
                delivered_loft_deg=10,
                contact_face_deg=0,
                contact_loft_deg=10,
                outcome=SimpleNamespace(
                    trajectory_xy_m=((0, 0), (999, 0)),
                    carry_x_m=999,
                    carry_y_m=0,
                    spin_rpm=2500,
                    launch_azimuth_deg=0,
                    spin_axis_tilt_deg=0,
                    launch_elevation_deg=10,
                    ball_speed_mps=60,
                ),
            )

    fake = Fake()
    with pytest.raises(ValueError, match="saved path"):
        replay.execute_replay(
            plan,
            fake,
            tmp_path / "supplement",
            runtime=_runtime(),
        )
    assert not (tmp_path / "supplement").exists()
    assert fake.calls.count("shot") == 1


def test_execution_rejects_unqualified_runtime_without_simulation(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    plan = replay.plan_replay({"baseline": cell}, target=10)

    class NeverSimulate:
        def simulate(self, **kw):
            pytest.fail("simulation began before runtime qualification")

        def simulate_delivery(self, **kw):
            pytest.fail("simulation began before runtime qualification")

    with pytest.raises(ValueError, match="runtime differs"):
        replay.execute_replay(
            plan,
            NeverSimulate(),
            tmp_path / "supplement",
            runtime={"native_binary_sha256": "wrong"},
        )
    assert not (tmp_path / "supplement").exists()


def test_missing_physics_source_identity_blocks_replay(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    plan = replay.plan_replay({"baseline": cell}, target=10)
    runtime = _runtime()
    del runtime["source_sha256"][replay.PHYSICS_SOURCES[0]]
    with pytest.raises(ValueError, match="physics source"):
        replay.execute_replay(plan, object(), tmp_path / "supplement", runtime=runtime)
    assert not (tmp_path / "supplement").exists()


def test_changed_input_after_planning_blocks_replay(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    plan = replay.plan_replay({"baseline": cell}, target=10)
    (cell / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        replay.execute_replay(
            plan, object(), tmp_path / "supplement", runtime=_runtime()
        )
    assert not (tmp_path / "supplement").exists()


def test_saved_path_parity_rejects_nonfinite_values():
    assert not replay._same_path(_path(100), [[float("nan"), 0]] * 41)


def test_replay_rejects_noninteger_target(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    with pytest.raises(ValueError, match="positive integer"):
        replay.plan_replay({"baseline": cell}, target=2.5)


def test_execution_writes_separate_traces_only_after_parity(tmp_path):
    cell = _write_cell(tmp_path / "a", set(range(110)))
    plan = replay.plan_replay({"baseline": cell}, target=10)

    class Fake:
        def simulate(self, **kw):
            return SimpleNamespace(trajectory_xy_m=_path(100))

        def simulate_delivery(self, **kw):
            index = int(kw["shaft_rotation_deg"])
            carry = 100 + index
            return SimpleNamespace(
                delivered_face_deg=0,
                delivered_loft_deg=10,
                contact_face_deg=0,
                contact_loft_deg=10,
                outcome=SimpleNamespace(
                    trajectory_xy_m=_path(carry),
                    carry_x_m=carry,
                    carry_y_m=0,
                    spin_rpm=2500,
                    launch_azimuth_deg=0,
                    spin_axis_tilt_deg=0,
                    launch_elevation_deg=10,
                    ball_speed_mps=60,
                ),
            )

    destination = tmp_path / "supplement"
    runtime = _runtime()
    receipt = replay.execute_replay(plan, Fake(), destination, runtime=runtime)
    traces = json.loads((destination / "trajectories.json").read_text())
    assert receipt["selected_paths_per_pattern"] == 10
    assert [x["shot_index"] for x in traces["baseline"]["Draw"]] == list(
        plan.shot_indices
    )
    assert len(traces["baseline"]["Draw"][0]["path_xy_m"]) == 41
    assert (
        receipt["replay_source_sha256"]
        == hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest()
    )
    assert receipt["tolerances"]["trajectory_m"] == 1e-6
    assert (cell / "manifest.json").exists()
