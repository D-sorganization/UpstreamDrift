"""Corrective RED contracts for hybrid evidence semantics and writes (#9236)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)

pytestmark = [pytest.mark.scientific]

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "articulated_manufactured_solution.json"
)


def _profiled_records() -> tuple[dict[str, Any], dict[str, Any]]:
    authority = json.loads(COMMITTED.read_text(encoding="utf-8"))
    authority["execution_profile"] = {
        "id": runner.AUTHORITY_PROFILE,
        "publication_authority": "authoritative",
        "publication_eligible": True,
    }
    rolling = copy.deepcopy(authority)
    rolling["execution_profile"] = {
        "id": runner.ROLLING_PROFILE,
        "publication_authority": "non_authoritative_compatibility_only",
        "publication_eligible": False,
    }
    return authority, rolling


def _set_nested(record: dict[str, Any], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    target: dict[str, Any] = record
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[parts[-1]] = value


def _install_compatibility_tolerance(
    authority: dict[str, Any],
    rolling: dict[str, Any],
    dotted_path: str,
    tolerance: float,
) -> None:
    field = "rolling_compatibility_absolute_tolerance_by_field"
    authority["design"][field] = {dotted_path: tolerance}
    rolling["design"][field] = {dotted_path: tolerance}


def test_rolling_comparison_accepts_difference_inside_declared_tolerance() -> None:
    """The drift test has a legitimate within-tolerance control."""

    authority, rolling = _profiled_records()
    field = "free_body.inverse_dynamics_relative_error.lagrange_mujoco"
    tolerance = 1.0e-8
    _install_compatibility_tolerance(authority, rolling, field, tolerance)
    baseline = authority["free_body"]["inverse_dynamics_relative_error"][
        "lagrange_mujoco"
    ]
    _set_nested(rolling, field, baseline + tolerance / 2.0)

    assert runner.compare_semantic_evidence(authority, rolling)[
        "all_registered_gates_pass"
    ]


def test_rolling_comparison_rejects_difference_outside_declared_tolerance() -> None:
    """Rolling results must remain numerically compatible with authority."""

    authority, rolling = _profiled_records()
    field = "free_body.inverse_dynamics_relative_error.lagrange_mujoco"
    tolerance = 1.0e-8
    _install_compatibility_tolerance(authority, rolling, field, tolerance)
    baseline = authority["free_body"]["inverse_dynamics_relative_error"][
        "lagrange_mujoco"
    ]
    _set_nested(rolling, field, baseline + tolerance * 2.0)

    with pytest.raises(ValueError, match="compatib|semantic|tolerance"):
        runner.compare_semantic_evidence(authority, rolling)


@pytest.mark.parametrize(
    "field",
    (
        "free_body.inverse_dynamics_relative_error.lagrange_mujoco",
        "free_body.gravity_free_zero_torque_relative_drift.linear_momentum",
        "constrained_motion.multiplier_relative_residual",
        "constrained_motion.cross_engine_multiplier_relative_residual",
    ),
)
def test_rolling_comparison_rejects_negative_error_or_residual(field: str) -> None:
    """Norms, errors, drift magnitudes, and residuals cannot be negative."""

    authority, rolling = _profiled_records()
    _set_nested(rolling, field, -1.0)

    with pytest.raises(ValueError, match="negative|nonnegative|semantic|gate"):
        runner.compare_semantic_evidence(authority, rolling)


def test_rolling_comparison_rejects_inconsistent_maximum() -> None:
    """The recorded maximum must be derived from the component residuals."""

    authority, rolling = _profiled_records()
    rolling["free_body"]["inverse_dynamics_relative_error"]["maximum"] = 0.0

    with pytest.raises(ValueError, match="maximum|semantic|gate"):
        runner.compare_semantic_evidence(authority, rolling)


def test_rolling_comparison_rejects_nonconvergent_step_errors() -> None:
    """Refining the integration step must not increase the registered error."""

    authority, rolling = _profiled_records()
    rolling["free_body"]["integration_step_error_rad"] = {
        "0.0005": 4.0,
        "0.001": 2.0,
        "0.002": 1.0,
    }

    with pytest.raises(ValueError, match="integration|monotonic|semantic|gate"):
        runner.compare_semantic_evidence(authority, rolling)


def test_rolling_comparison_uses_declared_constraint_tolerance() -> None:
    """The comparator cannot substitute a module constant for record policy."""

    authority, rolling = _profiled_records()
    authority["design"]["constraint_position_tolerance_m"] = 1.0e-13
    rolling["design"]["constraint_position_tolerance_m"] = 1.0e-13
    rolling["constrained_motion"]["position_residual_m"] = 1.0e-12

    with pytest.raises(ValueError, match="position|tolerance|semantic|gate"):
        runner.compare_semantic_evidence(authority, rolling)


def test_execution_profile_rejects_unknown_runtime_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Literal typing is not a runtime DbC boundary."""

    monkeypatch.setattr(runner, "_distribution_version", lambda _name: "test")

    with pytest.raises(ValueError, match="profile"):
        runner._execution_profile(cast(Any, "typo"))


def test_unknown_profile_cannot_replace_canonical_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-authority profile must be barred from canonical output."""

    canonical = tmp_path / "articulated_manufactured_solution.json"
    original = b"authoritative-before\n"
    canonical.write_bytes(original)
    monkeypatch.setattr(runner, "OUTPUT", canonical)
    monkeypatch.setattr(runner, "build_record", lambda _profile: {"value": 1.0})

    with pytest.raises(ValueError, match="profile|authorit"):
        runner.write_record(canonical, profile=cast(Any, "typo"))
    assert canonical.read_bytes() == original


def test_interrupted_write_leaves_complete_old_or_new_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write interruption may expose old or new bytes, never a partial file."""

    target = tmp_path / "articulated.json"
    original = b"authoritative-before\n"
    record = {"schema_version": "test", "value": 1.0}
    expected = runner.canonical_record_bytes(record)
    target.write_bytes(original)
    real_write_bytes = Path.write_bytes

    def interrupt_direct_destination(path: Path, data: bytes) -> int:
        if path.resolve() == target.resolve():
            real_write_bytes(path, b"partial")
            raise OSError("simulated interrupted destination write")
        return real_write_bytes(path, data)

    monkeypatch.setattr(runner, "build_record", lambda _profile: record)
    monkeypatch.setattr(Path, "write_bytes", interrupt_direct_destination)

    try:
        runner.write_record(target, profile="authority")
    except OSError:
        pass

    assert target.read_bytes() in {original, expected}
    assert set(tmp_path.iterdir()) == {target}
