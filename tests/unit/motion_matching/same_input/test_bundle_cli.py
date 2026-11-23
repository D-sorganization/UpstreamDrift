"""The same-input CLI accepts every registered engine (#11607) and chain bundles (#12691)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.same_input_bundle import (
    build_parser,
    padded_track,
    shaft_inputs,
    target_tracking_summary,
)
from src.shared.python.motion_matching.same_input import (
    ALL_ENGINES,
    OPTIONAL_ENGINES,
    PARITY_ENGINES,
)

pytestmark = pytest.mark.unit


def test_target_tracking_is_measured_against_interpolated_controller_reference() -> (
    None
):
    times = np.array([10.0, 11.0, 12.0])
    target = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    actual = np.array([[0.0, 0.0], [1.0, 2.5], [2.0, 4.0]])
    summary = target_tracking_summary(times, target, np.array([0.0, 0.5, 1.0]), actual)
    assert summary["rms_by_coordinate"] == pytest.approx([0.0, np.sqrt(0.25 / 3)])
    assert summary["max_abs_by_coordinate"] == pytest.approx([0.0, 0.5])
    assert summary["reference"] == "controller_position_target"


@pytest.mark.parametrize("run_times", [np.array([0.0, 2.1]), np.array([0.0, np.nan])])
def test_target_tracking_rejects_uncovered_or_nonfinite_rollout_times(
    run_times,
) -> None:
    with pytest.raises(ValueError):
        target_tracking_summary(
            np.array([4.0, 6.0]), np.zeros((2, 1)), run_times, np.zeros((2, 1))
        )


def test_closed_loop_refuses_an_unbound_tracking_record(monkeypatch, tmp_path) -> None:
    from scripts import same_input_bundle as cli

    bundle = SimpleNamespace(provenance={"dynamics_record_sha256": "not-the-record"})
    monkeypatch.setattr(cli.InputBundle, "load", lambda _: bundle)
    record = tmp_path / "record.npz"
    record.write_bytes(b"different record")
    monkeypatch.setattr(
        cli, "_track", lambda _: (record, np.arange(2.0), np.zeros((2, 1)))
    )
    monkeypatch.setattr(
        cli, "closed_loop", lambda *a, **k: pytest.fail("must reject before simulating")
    )
    with pytest.raises(ValueError, match="tracking record"):
        cli.closed_loop_receipt(tmp_path, tmp_path / "bundle.npz", "mujoco")


def test_closed_loop_reports_target_and_rollout_scores_separately(
    monkeypatch, tmp_path
) -> None:
    from scripts import same_input_bundle as cli

    record = tmp_path / "record.npz"
    record.write_bytes(b"bound reference")
    bundle_path = tmp_path / "bundle.npz"
    bundle_path.write_bytes(b"bundle")
    bundle = SimpleNamespace(
        provenance={"dynamics_record_sha256": cli._sha256(record)},
        spec_bytes=b"{}",
        steps=1,
        dt_s=1.0,
        coordinate_order=("joint",),
        ball_impact=lambda: None,
        step_policy=lambda: None,
        manifest=dict,
        replay_efforts=lambda: np.zeros((1, 1)),
    )
    rollout = SimpleNamespace(
        time_s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        efforts=np.zeros((1, 1)),
        pose_drift=np.zeros(1),
    )
    monkeypatch.setattr(cli.InputBundle, "load", lambda _: bundle)
    monkeypatch.setattr(
        cli,
        "_track",
        lambda _: (record, np.array([5.0, 6.0]), np.array([[0.0], [2.0]])),
    )
    monkeypatch.setattr(cli, "padded_track", lambda _, q: q)
    monkeypatch.setattr(cli, "closed_loop", lambda *a, **k: rollout)
    monkeypatch.setattr(cli, "VectorPlant", lambda *a: None)
    monkeypatch.setattr(
        cli,
        "score_replay",
        lambda *a: SimpleNamespace(summary=lambda: {"max_coordinate_error_rad": 0.0}),
    )
    result = cli.closed_loop_receipt(tmp_path, bundle_path, "mujoco")
    assert result["full_horizon"]["max_coordinate_error_rad"] == 0.0
    assert result["target_tracking"]["max_abs_by_coordinate"] == [1.0]
    assert result["target_tracking"]["coordinate_order"] == ("joint",)


def test_target_tracking_rms_does_not_overflow_for_finite_values() -> None:
    result = target_tracking_summary(
        np.array([0.0, 1.0]),
        np.zeros((2, 1)),
        np.array([0.0, 1.0]),
        np.full((2, 1), 1e200),
    )
    assert result["rms_by_coordinate"] == pytest.approx([1e200])


def test_all_engines_lists_core_then_optional() -> None:
    assert (*PARITY_ENGINES, *OPTIONAL_ENGINES) == ALL_ENGINES


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("command", ["replay", "closed-loop"])
def test_parser_accepts_every_engine(command: str, engine: str) -> None:
    argv = [command, "--bundle", "b.npz", "--engine", engine, "--out", "r.json"]
    if command == "closed-loop":
        argv += ["--run-dir", "run"]
    assert build_parser().parse_args(argv).engine == engine


def test_parser_rejects_unknown_engine() -> None:
    argv = ["replay", "--bundle", "b.npz", "--engine", "bullet", "--out", "r.json"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# --- chain-shaft bundles (R11, #12691) -------------------------------------------

SEED_SPEC = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)


def test_export_shaft_model_defaults_to_rigid() -> None:
    argv = ["export", "--run-dir", "run", "--out", "b.npz"]
    assert build_parser().parse_args(argv).shaft_model == "rigid"
    chain = build_parser().parse_args([*argv, "--shaft-model", "chain"])
    assert chain.shaft_model == "chain"
    with pytest.raises(SystemExit):
        build_parser().parse_args([*argv, "--shaft-model", "noodle"])


def test_rigid_shaft_inputs_are_returned_unchanged() -> None:
    spec = SEED_SPEC.read_bytes()
    track = np.ones((3, len(json.loads(spec)["coordinate_order"])))
    out_spec, out_track = shaft_inputs(spec, track, "rigid")
    assert out_spec is spec and out_track is track


def test_chain_shaft_inputs_append_an_undeflected_shaft() -> None:
    spec = SEED_SPEC.read_bytes()
    nv = len(json.loads(spec)["coordinate_order"])
    track = np.ones((3, nv))
    out_spec, out_track = shaft_inputs(spec, track, "chain")
    document = json.loads(out_spec)
    shaft = document["club"]["shaft"]["coordinates"]
    assert document["club"]["shaft"]["model"] == "chain"
    assert document["coordinate_order"][-len(shaft) :] == shaft
    assert out_track.shape == (3, nv + len(shaft))
    np.testing.assert_array_equal(out_track[:, :nv], track)
    assert not out_track[:, nv:].any()
    # A chain bundle's own spec pads the run's rigid-width track the same way.
    np.testing.assert_array_equal(padded_track(out_spec, track), out_track)
    np.testing.assert_array_equal(padded_track(out_spec, out_track), out_track)


def test_padded_track_rejects_a_mismatched_width() -> None:
    spec = SEED_SPEC.read_bytes()
    with pytest.raises(ValueError, match="q_track"):
        padded_track(spec, np.ones((3, 5)))


def test_target_tracking_uses_clamped_endpoint_for_rounded_last_step() -> None:
    times = np.array([7.0, 8.81388888888889])
    target = np.array([[0.0], [2.0]])
    summary = target_tracking_summary(
        times,
        target,
        np.array([0.0, 1.814]),
        np.array([[0.0], [2.5]]),
        max_endpoint_extension_s=0.0005,
    )
    assert summary["max_abs_by_coordinate"] == pytest.approx([0.5])
    assert summary["rms_by_coordinate"] == pytest.approx([np.sqrt(0.125)])
    with pytest.raises(ValueError):
        target_tracking_summary(
            times,
            target,
            np.array([0.0, 1.815]),
            np.zeros((2, 1)),
            max_endpoint_extension_s=0.0005,
        )
