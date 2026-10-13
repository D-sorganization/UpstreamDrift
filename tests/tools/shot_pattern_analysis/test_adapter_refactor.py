"""Synthetic adapter receipts retain dispatch, population, and callback contracts."""

import json
import math
import hashlib
import platform
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy

from src.tools.shot_pattern_analysis.uncertainty import HISTORICAL_RESAMPLES

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_HISTORICAL_FIXTURE = "adapter_refactor_reference.json"
_HISTORICAL_SHA256 = "0403629f476ae9f01b50afe9060fd2e6d0b41b4f463aad213fec032c6af245fa"
_CI_LOCK_FIXTURE = "adapter_refactor_reference_scipy_1_15_3.json"
_CI_LOCK_SHA256 = "2340fb22609bc17eac72ccd4f621bbe187edeff9be0bd26d062f89b1bce90d93"
_PY311_FIXTURE = "adapter_refactor_reference_py311_scipy_1_15_3.json"
_PY311_SHA256 = "8922892320bea6ea9f10bedb0f6793fcd3d972d88149ffa48650cde06e8219fe"
_SUPPORTED_RUNTIMES = {
    ("3.11", "2.2.6", "1.15.3"): (_PY311_FIXTURE, _PY311_SHA256),
    ("3.12", "2.2.6", "1.15.3"): (_CI_LOCK_FIXTURE, _CI_LOCK_SHA256),
    ("3.12", "2.2.6", "1.18.1"): (_HISTORICAL_FIXTURE, _HISTORICAL_SHA256),
}


def _reference_for_runtime(
    python_version: str, numpy_version: str, scipy_version: str
) -> tuple[str, str]:
    python_minor = ".".join(python_version.split(".")[:2])
    versions = (python_minor, numpy_version, scipy_version)
    try:
        return _SUPPORTED_RUNTIMES[versions]
    except KeyError as exc:
        raise ValueError(f"Unsupported adapter golden runtime: {versions}") from exc


def _read_hash_bound_reference(path: Path, expected_sha256: str) -> dict:
    contents = path.read_bytes()
    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Adapter golden fixture hash mismatch: {path.name}")
    return json.loads(contents)


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
        # The golden receipt predates the 2,000-resample publication default.
        resamples=HISTORICAL_RESAMPLES,
    )
    receipt = asdict(result)
    for section in ("course_scoring", "paired_comparisons"):
        # Recorded since #12409; the frozen receipt predates the field.
        assert receipt[section].pop("bootstrap_resamples") == HISTORICAL_RESAMPLES
    return {"result": receipt, "calls": engine.calls, "progress": progress}


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
    filename, sha256 = _reference_for_runtime(
        platform.python_version(), np.__version__, scipy.__version__
    )
    reference = _read_hash_bound_reference(_FIXTURE_DIR / filename, sha256)
    assert json.dumps(_adapter_receipt()) == json.dumps(reference["receipt"])


def test_adapter_golden_provenance_is_bound_to_original_fixture():
    original = _read_hash_bound_reference(
        _FIXTURE_DIR / _HISTORICAL_FIXTURE, _HISTORICAL_SHA256
    )
    ci_lock = _read_hash_bound_reference(
        _FIXTURE_DIR / _CI_LOCK_FIXTURE, _CI_LOCK_SHA256
    )
    py311 = _read_hash_bound_reference(_FIXTURE_DIR / _PY311_FIXTURE, _PY311_SHA256)
    provenance = ci_lock["provenance"]
    assert provenance["source_commit"] == "de9ece2b99121e6f0c5a18f85e9331012d80a7b0"
    assert provenance["source_tree"] == "d0fca9395abdaca6d1075ba5ce2343ed0b361ba7"
    assert provenance["harness_sha256"] == (
        "53a3ba0ef97e120afa778d0eb8e1e38849d8264e2f1bbcdbf1bd18d584fa6d4e"
    )
    assert provenance["historical_fixture_sha256"] == _HISTORICAL_SHA256
    assert provenance["source_sha256"] == original["provenance"]["source_sha256"]
    assert len(provenance["loaded_historical_source_sha256"]) == 65
    assert (
        py311["provenance"]["loaded_historical_source_sha256"]
        == provenance["loaded_historical_source_sha256"]
    )
    assert (
        py311["provenance"]["source_sha256"] == original["provenance"]["source_sha256"]
    )
    assert py311["provenance"]["harness_sha256"] == provenance["harness_sha256"]
    assert py311["receipt"] == ci_lock["receipt"]
    assert (
        provenance["python"],
        provenance["numpy"],
        provenance["scipy"],
    ) == ("3.12.13", "2.2.6", "1.15.3")
    assert (
        py311["provenance"]["python"],
        py311["provenance"]["numpy"],
        py311["provenance"]["scipy"],
    ) == ("3.11.15", "2.2.6", "1.15.3")


def test_adapter_golden_rejects_unreviewed_runtime_and_modified_fixture(tmp_path):
    with pytest.raises(ValueError, match="Unsupported adapter golden runtime"):
        _reference_for_runtime("3.12.13", "2.2.6", "1.99.0")
    with pytest.raises(ValueError, match="Unsupported adapter golden runtime"):
        _reference_for_runtime("3.13.0", "2.2.6", "1.15.3")
    modified = tmp_path / _CI_LOCK_FIXTURE
    modified.write_bytes((_FIXTURE_DIR / _CI_LOCK_FIXTURE).read_bytes() + b" ")
    with pytest.raises(ValueError, match="golden fixture hash mismatch"):
        _read_hash_bound_reference(modified, _CI_LOCK_SHA256)


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
