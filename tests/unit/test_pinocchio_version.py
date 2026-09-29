"""Smoke test for the Pinocchio version pin (PIN-DEPS-AUDIT, issue #4107).

`pyproject.toml` declares `pinocchio>=2.6.0` so that the analytical
ABA derivatives required by PIN-FIT-DRIVER are available. This test
locks that contract in: if a future bump accidentally drops the floor
below 2.6 (or substitutes a build without the derivative bindings),
this test fails loudly instead of letting downstream optimiser code
crash at fit time.

The test is skipped in environments where Pinocchio is not installed
(e.g. the lightweight default CI lane). It runs whenever the
`pinocchio` extra is selected.
"""

from __future__ import annotations

import sys

import pytest
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

# Module-level marker keeps lightweight unit-test lanes happy. The
# Pinocchio-dependent tests below carry the skip decorator individually
# so the parser-only helper tests still run on the default lane.
pytestmark = [pytest.mark.unit]

# `tests/unit/conftest.py` injects a `MagicMock` into ``sys.modules["pinocchio"]``
# so that other unit tests can import-and-mock without triggering the real
# C-extension load. ``engine_availability.skip_if_unavailable`` is fooled by
# that stub (the probe only does ``hasattr`` and MagicMock answers Yes to
# everything) so we add a second guard here that detects the mock and skips.
_pin_mod = sys.modules.get("pinocchio")
_pinocchio_is_mock = (
    _pin_mod is not None and type(_pin_mod).__module__ == "unittest.mock"
)

_skip_if_mock = pytest.mark.skipif(
    _pinocchio_is_mock,
    reason="pinocchio is mocked by tests/unit/conftest.py; this smoke "
    "test needs the real C++ bindings (run with the `pinocchio` extra installed)",
)
_skip_if_unavailable = skip_if_unavailable("pinocchio")


def _requires_pinocchio(func):
    """Apply both skip guards: real-install check and mock-detection check."""
    return _skip_if_unavailable(_skip_if_mock(func))


# Minimum Pinocchio version that ships `computeABADerivatives` in the
# Python bindings. Bumping this floor must be a deliberate, reviewed
# change (PIN-FIT-DRIVER depends on it).
MIN_PINOCCHIO_VERSION = (2, 6, 0)


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints.

    Tolerates trailing pre-release / build metadata (e.g. ``"2.7.1.dev0"``
    or ``"2.6.0+local"``) by splitting on the first non-numeric component.
    """
    if not isinstance(raw, str):
        raise TypeError(f"version must be a string, got {type(raw).__name__}")
    if not raw:
        raise ValueError("version string is empty")

    parts: list[int] = []
    for chunk in raw.replace("+", ".").replace("-", ".").split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
            continue
        # First non-numeric chunk terminates the numeric prefix.
        break
    if not parts:
        raise ValueError(f"could not parse version: {raw!r}")
    return tuple(parts)


@_requires_pinocchio
def test_pinocchio_version_floor() -> None:
    """Pinocchio must be at least the version pinned in pyproject.toml."""
    import pinocchio as pin

    raw_version = getattr(pin, "__version__", None)
    assert raw_version, "pinocchio.__version__ is missing or empty"

    parsed = _parse_version(raw_version)
    # Pad to length-3 for comparison stability.
    padded = parsed + (0,) * (3 - len(parsed)) if len(parsed) < 3 else parsed
    assert padded[:3] >= MIN_PINOCCHIO_VERSION, (
        f"pinocchio {raw_version} is older than the required floor "
        f"{'.'.join(str(p) for p in MIN_PINOCCHIO_VERSION)} "
        "(see PIN-DEPS-AUDIT / pyproject.toml)"
    )


@_requires_pinocchio
def test_pinocchio_aba_derivatives_available() -> None:
    """`pin.computeABADerivatives` must be importable and callable.

    PIN-FIT-DRIVER relies on analytical `∂qdd/∂q`, `∂qdd/∂qd`, `∂qdd/∂tau`
    via `computeABADerivatives` to drive the LM optimiser without finite
    differences. If the binding is missing, the optimiser cannot run.
    """
    import pinocchio as pin

    assert hasattr(pin, "computeABADerivatives"), (
        "pin.computeABADerivatives is missing - PIN-FIT-DRIVER will fail"
    )
    assert callable(pin.computeABADerivatives), (
        "pin.computeABADerivatives is not callable"
    )


@_requires_pinocchio
def test_pinocchio_forward_kinematics_available() -> None:
    """`pin.computeForwardKinematics` must be importable and callable.

    Forward kinematics is the bedrock of every Pinocchio code path in
    this repo (grip-frame placement, club trajectory, IK seeding). A
    missing binding here is a packaging regression.
    """
    import pinocchio as pin

    assert hasattr(pin, "computeForwardKinematics"), (
        "pin.computeForwardKinematics is missing"
    )
    assert callable(pin.computeForwardKinematics), (
        "pin.computeForwardKinematics is not callable"
    )


@_requires_pinocchio
def test_pinocchio_rnea_derivatives_available() -> None:
    """`pin.computeRNEADerivatives` must be importable.

    Used alongside ABA derivatives for the inverse-dynamics Jacobian
    pieces of PIN-FIT-DRIVER.
    """
    import pinocchio as pin

    assert hasattr(pin, "computeRNEADerivatives"), (
        "pin.computeRNEADerivatives is missing - PIN-FIT-DRIVER will fail"
    )
    assert callable(pin.computeRNEADerivatives)


@_requires_pinocchio
def test_pinocchio_joint_jacobians_available() -> None:
    """`pin.computeJointJacobians` must be importable.

    Required for `∂grip/∂q` analytical Jacobians per PINOCCHIO_PARITY_SPEC §2.
    """
    import pinocchio as pin

    assert hasattr(pin, "computeJointJacobians"), "pin.computeJointJacobians is missing"
    assert callable(pin.computeJointJacobians)


@pytest.mark.parametrize(
    "raw, expected_prefix",
    [
        ("2.6.0", (2, 6, 0)),
        ("2.7.1", (2, 7, 1)),
        ("3.0.0", (3, 0, 0)),
        ("2.6.0.dev0", (2, 6, 0)),
        ("2.6.0+local", (2, 6, 0)),
        ("2.6", (2, 6)),
    ],
)
def test_parse_version_helper(raw: str, expected_prefix: tuple[int, ...]) -> None:
    """The internal version parser handles the shapes Pinocchio reports."""
    assert _parse_version(raw) == expected_prefix


def test_parse_version_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_version("")
    with pytest.raises(ValueError):
        _parse_version("not-a-version")
    with pytest.raises(TypeError):
        _parse_version(None)  # type: ignore[arg-type]
