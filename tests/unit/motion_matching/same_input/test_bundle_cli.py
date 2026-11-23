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
    (tmp_path / "full_body_spec_hipcal_scaled.json").write_text("{}", encoding="utf-8")
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
        v=np.array([[2.0], [1.0]]),
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
    assert result["velocity_target_tracking"]["max_abs_by_coordinate"] == [1.0]


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


def test_velocity_target_uses_split_rates_and_clamped_endpoint() -> None:
    from scripts.same_input_bundle import target_velocity_tracking_summary

    times = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # Pre-impact slope 1, post-impact slope 10; neither side is smeared.
    q = np.array([[0.0], [1.0], [2.0], [20.0], [30.0], [40.0]])
    run_t = np.array([0.0, 2.5, 3.0, 5.0001])
    run_v = np.array([[1.0], [1.0], [8.0], [10.0]])
    result = target_velocity_tracking_summary(
        times, q, run_t, run_v, split_time_s=2.5, max_endpoint_extension_s=0.0005
    )
    assert result["reference"] == "controller_derived_velocity_target"
    assert result["max_abs_by_coordinate"] == pytest.approx([2.0])
    assert result["rms_by_coordinate"] == pytest.approx([1.0])
    assert result["endpoint_clamped_duration_s"] == pytest.approx(0.0001)
    with pytest.raises(ValueError):
        target_velocity_tracking_summary(
            times,
            q,
            np.array([0.0, 5.001]),
            np.zeros((2, 1)),
            split_time_s=2.5,
            max_endpoint_extension_s=0.0005,
        )


def test_velocity_target_uses_derived_rates_not_interpolant_derivative() -> None:
    from scripts.same_input_bundle import target_velocity_tracking_summary

    # Finite differences are [1, 2, 3]; the first segment slope is 1.
    times = np.array([0.0, 1.0, 2.0])
    q = np.array([[0.0], [1.0], [4.0]])
    summary = target_velocity_tracking_summary(
        times, q, np.array([0.0, 0.5, 1.0]), np.array([[1.0], [1.5], [2.0]])
    )
    assert summary["max_abs_by_coordinate"] == pytest.approx([0.0])
    with pytest.raises(ValueError, match="split_time_s"):
        target_velocity_tracking_summary(
            times, q, np.array([0.0, 1.0]), np.zeros((2, 1)), split_time_s=0.5
        )


def test_closed_loop_rejects_same_width_but_different_run_model(
    monkeypatch, tmp_path
) -> None:
    from scripts import same_input_bundle as cli

    record = tmp_path / "record.npz"
    record.write_bytes(b"bound reference")
    (tmp_path / "full_body_spec_hipcal_scaled.json").write_text(
        json.dumps({"coordinate_order": ["joint"], "mass": 80.0}), encoding="utf-8"
    )
    bundle = SimpleNamespace(
        provenance={"dynamics_record_sha256": cli._sha256(record)},
        spec_bytes=json.dumps({"coordinate_order": ["joint"], "mass": 70.0}).encode(),
        ball_impact=lambda: None,
        steps=1,
        dt_s=1.0,
        step_policy=lambda: None,
    )
    monkeypatch.setattr(cli.InputBundle, "load", lambda _: bundle)
    monkeypatch.setattr(
        cli, "_track", lambda _: (record, np.arange(2.0), np.zeros((2, 1)))
    )
    monkeypatch.setattr(
        cli, "closed_loop", lambda *a, **k: pytest.fail("must reject before simulating")
    )
    with pytest.raises(ValueError, match="model.*match"):
        cli.closed_loop_receipt(tmp_path, tmp_path / "bundle.npz", "mujoco")


def test_closed_loop_reconstructs_recorded_shaft_before_model_check(
    monkeypatch, tmp_path
) -> None:
    from scripts import same_input_bundle as cli

    record = tmp_path / "record.npz"
    record.write_bytes(b"bound reference")
    (tmp_path / "full_body_spec_hipcal_scaled.json").write_text(
        '{"rigid": true}', encoding="utf-8"
    )
    bundle = SimpleNamespace(
        provenance={
            "dynamics_record_sha256": cli._sha256(record),
            "shaft_model": "chain",
            "shaft_options": {"scale": 2},
        },
        spec_bytes=b'{"chain": 2}',
        ball_impact=lambda: None,
        steps=1,
        dt_s=1.0,
        step_policy=lambda: None,
    )
    monkeypatch.setattr(cli.InputBundle, "load", lambda _: bundle)
    monkeypatch.setattr(
        cli, "_track", lambda _: (record, np.arange(2.0), np.zeros((2, 1)))
    )

    def transformed(spec, q, model, options):
        assert json.loads(spec) == {"rigid": True}
        assert model == "chain" and options == {"scale": 2}
        return b'{ "chain": 2 }', q

    monkeypatch.setattr(cli, "shaft_inputs", transformed)
    monkeypatch.setattr(cli, "padded_track", lambda _, q: q)
    monkeypatch.setattr(
        cli,
        "closed_loop",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulation reached")),
    )
    with pytest.raises(RuntimeError, match="simulation reached"):
        cli.closed_loop_receipt(tmp_path, tmp_path / "bundle.npz", "mujoco")
