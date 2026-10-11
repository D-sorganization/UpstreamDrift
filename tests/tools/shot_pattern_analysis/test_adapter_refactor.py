"""Synthetic adapter receipts retain dispatch, population, and callback contracts."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _json_value(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist()))


class _SyntheticPhysics:
    def __init__(self):
        self.calls = []

    def simulate(self, **kwargs):
        from src.tools.shot_pattern_analysis.physics import ShotOutcome

        self.calls.append(
            {
                key: asdict(val) if key == "config" else val
                for key, val in kwargs.items()
            }
        )
        face, path = kwargs["face_deg"], kwargs["path_deg"]
        lateral = face - 0.5 * path
        return ShotOutcome(
            100.0,
            lateral,
            math.hypot(100.0, lateral),
            face,
            1000.0,
            face - path,
            ((0.0, 0.0), (100.0, lateral)),
        )

    def simulate_delivery(self, **kwargs):
        from src.tools.shot_pattern_analysis.matched_physics import (
            MatchedDeliveryOutcome,
            UnsupportedShot,
        )

        self.calls.append(
            {
                "delivery": {
                    key: asdict(val) if key == "config" else val
                    for key, val in kwargs.items()
                }
            }
        )
        if abs(kwargs["contact_horizontal_m"]) > 0.005:
            raise UnsupportedShot("off_face")
        face = kwargs["nominal_face_deg"] - kwargs["shaft_rotation_deg"]
        outcome = self.simulate(
            face_deg=face, path_deg=kwargs["path_deg"], config=kwargs["config"]
        )
        if kwargs["shaft_rotation_deg"] < 0:
            outcome = replace(outcome, spin_axis_defined=False)
        loft = kwargs["config"].loft_deg - kwargs["config"].shaft_lean_deg
        return MatchedDeliveryOutcome(outcome, face, loft, face, loft)


def _core_receipt():
    from src.tools.shot_pattern_analysis.core import AnalysisConfig, run_analysis

    results = {}
    for mode in ("fixed_loft", "shaft_rotation"):
        engine = _SyntheticPhysics()
        progress = []
        result = run_analysis(
            AnalysisConfig(n_shots=3, seed=87, delivery_mode=mode),
            physics=engine,
            progress_callback=lambda a, b, target=progress: target.append((a, b)),
        )
        results[mode] = {
            "result": asdict(result),
            "calls": engine.calls,
            "progress": progress,
        }
    return results


def _matched_receipt(monkeypatch):
    from src.tools.shot_pattern_analysis import matched_runner as runner
    from src.tools.shot_pattern_analysis.matched_config import matched_preset
    from src.tools.shot_pattern_analysis.noise import NoiseSpec
    from src.tools.shot_pattern_analysis.nominal_matching import MatchedNominal
    from src.tools.shot_pattern_analysis.physics import ShotOutcome

    errors = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0.5, 0.25, 0, 0, 0, 0, 0],
            [0.1, 0, 0, 0.01, 0, 0, 0],
            [0, 0, -100, 0, 0, 0, 0],
            [-0.25, -0.25, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    nominals = {
        p.name: MatchedNominal(
            p.face_deg,
            p.path_deg,
            36.0,
            ShotOutcome(100.0, 0.0, 100.0, 0.0, 1000.0, 0.0),
            0.0,
            1,
        )
        for p in runner.PATTERNS
    }
    monkeypatch.setattr(runner, "calibrate_patterns", lambda *a: (100.0, nominals))
    monkeypatch.setattr(runner, "sample_paired_noise", lambda *a, **k: errors.copy())
    request = runner.ExperimentConfig(
        matched_preset("seven_iron"),
        NoiseSpec((1, 0.5, 0, 0.00635, 0, 0, 0)),
        n_shots=6,
        seed=87,
    )
    engine, progress = _SyntheticPhysics(), []
    result = runner.run_matched_analysis(
        request,
        physics=engine,
        include_scoring=False,
        progress=lambda a, b: progress.append((a, b)),
    )
    return {"result": asdict(result), "calls": engine.calls, "progress": progress}


def _delivery_receipt(monkeypatch):
    from src.tools.shot_pattern_analysis import matched_physics as module
    from src.tools.shot_pattern_analysis.matched_config import matched_preset
    from src.tools.shot_pattern_analysis.physics import ShotOutcome

    calls = []

    def contact(state, **kwargs):
        calls.append({"contact": asdict(state), "options": kwargs})
        return SimpleNamespace(
            ball_velocity_mps=[50.0, -1.0, 10.0],
            ball_angular_velocity_rad_s=[1.0, 2.0, 3.0],
        )

    def flight(velocity, angular, **kwargs):
        calls.append(
            {
                "velocity": velocity,
                "angular": angular,
                "options": {k: v for k, v in kwargs.items() if k != "flight"},
            }
        )
        return ShotOutcome(100.0, 2.0, 101.0, 3.0, 1000.0, 4.0)

    monkeypatch.setattr(module, "solve_incremental_contact", contact)
    monkeypatch.setattr(module, "solve_rigid_contact", contact)
    monkeypatch.setattr(module, "outcome_from_launch", flight)
    engine = object.__new__(module.MatchedShotPhysics)
    engine._flight = object()
    results = {}
    for law in ("incremental_coulomb_energetic", "maximum_dissipation_newton"):
        calls.clear()
        result = engine.simulate_delivery(
            1.5,
            3.0,
            replace(matched_preset("driver"), contact_law=law),
            shaft_rotation_deg=0.5,
            contact_horizontal_m=0.00635,
            contact_vertical_m=0.002,
            sample_trajectory=True,
        )
        results[law] = {"result": asdict(result), "calls": list(calls)}
    return results


def _adapter_receipt():
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _json_value(
            {
                "core": _core_receipt(),
                "matched": _matched_receipt(monkeypatch),
                "delivery": _delivery_receipt(monkeypatch),
            }
        )


def test_adapters_match_fixed_pre_refactor_synthetic_receipt():
    reference = json.loads(
        (Path(__file__).parent / "fixtures/adapter_refactor_reference.json").read_text()
    )
    assert json.dumps(_adapter_receipt()) == json.dumps(reference["receipt"])


@pytest.mark.parametrize(
    "mode,nominal_present", [("fixed_loft", False), ("shaft_rotation", True)]
)
def test_pattern_dispatch_preserves_optional_nominal_argument(mode, nominal_present):
    from src.tools.shot_pattern_analysis.core import (
        AnalysisConfig,
        PatternConfig,
        _simulate_pattern,
    )

    engine = _SyntheticPhysics()
    config = AnalysisConfig(n_shots=2, delivery_mode=mode)
    pattern = PatternConfig("Draw", 1.5, 3.0)
    _simulate_pattern(engine, config, pattern, 2.0, sample_trajectory=True)
    assert ("nominal_face_deg" in engine.calls[0]) is nominal_present
    if nominal_present:
        assert engine.calls[0]["nominal_face_deg"] == 1.5


def test_delivery_validation_preserves_rejection_precedence():
    from src.tools.shot_pattern_analysis.matched_physics import (
        _validate_delivery_inputs,
        UnsupportedShot,
    )
    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    with pytest.raises(TypeError, match="config must be MatchedConfig"):
        _validate_delivery_inputs(None, float("nan"), 35.0, 0.0, 1.0, 1.0)
    with pytest.raises(UnsupportedShot, match="delivery_nonfinite"):
        _validate_delivery_inputs(
            matched_preset("driver"), float("nan"), 35.0, 0.0, 1.0, 1.0
        )
    with pytest.raises(UnsupportedShot, match="delivery_out_of_range"):
        _validate_delivery_inputs(matched_preset("driver"), 35.0, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(UnsupportedShot, match="off_face"):
        _validate_delivery_inputs(matched_preset("driver"), 0.0, 0.0, 0.0, 1.0, 1.0)


def test_supported_record_keeps_undefined_spin_axis_and_numeric_failure():
    from src.tools.shot_pattern_analysis.matched_runner import _supported_record
    from src.tools.shot_pattern_analysis.matched_physics import MatchedDeliveryOutcome
    from src.tools.shot_pattern_analysis.physics import ShotOutcome

    inputs = {
        "pattern": "Draw",
        "shot_index": 2,
        "status": "supported",
        "shaft_rotation_deg": 0.0,
        "path_deg": 3.0,
        "speed_mps": 36.0,
        "contact_horizontal_m": 0.0,
        "contact_vertical_m": 0.0,
        "shaft_lean_deg": 0.0,
        "attack_angle_deg": 0.0,
    }
    delivered = MatchedDeliveryOutcome(
        ShotOutcome(100.0, 1.0, 100.0, 1.0, 0.0, 0.0, spin_axis_defined=False),
        1.5,
        28.0,
        1.6,
        28.1,
    )
    record = _supported_record(inputs, delivered)
    assert record.spin_axis_tilt_deg is None
    assert record.status == "supported"
    assert record.contact_face_deg == 1.6
    assert inputs["status"] == "supported"
    with pytest.raises(RuntimeError, match="matched flight returned nonfinite values"):
        _supported_record(
            inputs,
            replace(
                delivered, outcome=replace(delivered.outcome, carry_x_m=float("nan"))
            ),
        )


@pytest.mark.parametrize(
    "velocity,error,reason",
    [
        ([1.0, 0.0, 0.0], ValueError, "nonairborne_launch"),
        ([1.0, 0.0, -1.0], ValueError, "nonairborne_launch"),
        (
            [float("nan"), 0.0, 1.0],
            RuntimeError,
            "contact produced nonfinite ball velocity",
        ),
    ],
)
def test_contact_launch_validation_retains_errors(monkeypatch, velocity, error, reason):
    from src.tools.shot_pattern_analysis import matched_physics as module
    from src.tools.shot_pattern_analysis.matched_config import matched_preset

    monkeypatch.setattr(
        module,
        "solve_incremental_contact",
        lambda *a, **k: SimpleNamespace(ball_velocity_mps=velocity),
    )
    with pytest.raises(error, match=reason):
        module._solve_ball_contact(object(), matched_preset("driver"))


def test_future_receipts_bind_current_adapters_and_contact_source():
    import hashlib
    from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot

    root = Path(__file__).resolve().parents[3]
    hashes = matched_source_snapshot()["source_sha256"]
    for name in ("core.py", "matched_runner.py", "matched_physics.py"):
        relative = "src/tools/shot_pattern_analysis/" + name
        assert (
            hashes[relative]
            == hashlib.sha256((root / relative).read_bytes()).hexdigest()
        )
    relative = "src/shared/python/physics/rigid_contact.py"
    assert (
        hashes[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    )
