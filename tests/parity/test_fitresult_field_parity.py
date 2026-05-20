"""Cross-engine FitResult field parity (issue #4250).

Asserts that every physics engine's ``FitResult`` exposes the same
canonical field set defined in
``src/shared/python/motion_matching/fit_result.py`` and that the
deprecated old-name accessors still work but emit
``DeprecationWarning``.

Per CROSS_ENGINE_PARITY_SPEC.md, all four Python engines (MuJoCo,
Drake, Pinocchio, OpenSim) must surface the SAME ``FitResult`` schema
so that callers can write engine-agnostic code.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from src.shared.python.motion_matching.fit_result import CanonicalFitResult

# The canonical, engine-agnostic field set every engine must surface.
# These are the names callers should use going forward.
CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "theta_optimal",
        "final_cost",
        "final_rmse_m",
        "solver_status",
        "iterations",
        "n_evaluations",
        "wall_clock_s",
        "message",
        "history",
        "method",
        "git_commit",
        "engine_version",
        "target_hash",
        "timestamp_utc",
    }
)

# Old names kept around for backward compatibility. Each must resolve via
# a ``@property`` shim that emits a ``DeprecationWarning``.
DEPRECATED_ALIASES: dict[str, str] = {
    "coefficients": "theta_optimal",
    "n_iter": "iterations",
    "n_eval": "n_evaluations",
    "n_evals": "n_evaluations",
}


def _all_engine_fitresult_classes() -> list[tuple[str, type]]:
    """Import each engine's re-exported FitResult and return (name, cls)."""
    classes: list[tuple[str, type]] = []
    from src.engines.physics_engines.drake.python.motion_matching.fit_swing import (
        FitResult as DrakeFit,
    )
    from src.engines.physics_engines.mujoco.python.motion_matching.fit_swing import (
        FitResult as MujocoFit,
    )
    from src.engines.physics_engines.opensim.python.motion_matching.fit_swing import (
        FitResult as OpenSimFit,
    )
    from src.engines.physics_engines.pinocchio.python.motion_matching.fit_swing import (
        FitResult as PinocchioFit,
    )

    classes.append(("mujoco", MujocoFit))
    classes.append(("drake", DrakeFit))
    classes.append(("pinocchio", PinocchioFit))
    classes.append(("opensim", OpenSimFit))
    return classes


@pytest.mark.unit
def test_all_engines_reexport_canonical_fitresult() -> None:
    """Every engine's ``FitResult`` is the shared ``CanonicalFitResult``.

    Re-exporting the same class is the simplest way to guarantee schema
    parity; if anyone forks back to a per-engine dataclass this test
    fails immediately.
    """
    for name, cls in _all_engine_fitresult_classes():
        assert cls is CanonicalFitResult, (
            f"{name}.FitResult must be CanonicalFitResult, got {cls!r}"
        )


@pytest.mark.unit
def test_canonical_fields_present_on_every_engine() -> None:
    """Every canonical field exists on every engine's ``FitResult``."""
    for name, cls in _all_engine_fitresult_classes():
        field_names = {f.name for f in dataclasses.fields(cls)}
        missing = CANONICAL_FIELDS - field_names
        assert not missing, f"{name}.FitResult missing canonical fields: {missing}"


def _make_canonical_instance() -> CanonicalFitResult:
    """Construct a minimal ``CanonicalFitResult`` for shim verification."""
    return CanonicalFitResult(
        theta_optimal=np.zeros(7, dtype=np.float64),
        final_cost=1.0,
        final_rmse_m=0.5,
        solver_status="success",
        iterations=3,
        n_evaluations=7,
        wall_clock_s=0.01,
        message="ok",
        history=(1.0, 0.5, 0.25),
        method="SLSQP",
        git_commit="deadbeef",
        engine_version="0.0.0",
        target_hash="0" * 8,
        timestamp_utc="1970-01-01T00:00:00Z",
    )


@pytest.mark.unit
@pytest.mark.parametrize(("alias", "canonical"), sorted(DEPRECATED_ALIASES.items()))
def test_deprecated_alias_emits_warning_and_returns_canonical(
    alias: str, canonical: str
) -> None:
    """Old names still work but emit ``DeprecationWarning`` (issue #4250).

    The shim must:
      * Return the same value as the canonical attribute.
      * Emit exactly one ``DeprecationWarning`` per access.
      * Mention the canonical replacement name in the warning text.
    """
    fr = _make_canonical_instance()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        aliased = getattr(fr, alias)

    expected = getattr(fr, canonical)
    if isinstance(expected, np.ndarray):
        assert np.array_equal(aliased, expected)
    else:
        assert aliased == expected

    dep = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert dep, f"accessing {alias!r} did not emit DeprecationWarning"
    assert canonical in str(dep[0].message), (
        f"DeprecationWarning for {alias!r} must mention canonical name "
        f"{canonical!r}; got: {dep[0].message!r}"
    )
