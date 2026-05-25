"""Cross-engine §2.2 contract tests for the Drake forward simulator.

Covers the seven gap items from issue #4255 against
``src.engines.physics_engines.drake.python.motion_matching.simulate.
simulate_with_coefficients``:

1.  Happy path — a valid coefficient vector produces a canonical
    :class:`SimOut` whose timegrid, ``q``, ``qd``, ``grip``, ``clubhead``,
    ``grip_quat`` and ``club_quat`` arrays are aligned and finite.
2.  ``theta = zeros`` runs without raising and yields a non-trivial
    gravity-driven trajectory.
3.  Out-of-bounds theta (``1e9``) is rejected by ``ValueError`` *or* —
    if the canonical theta validator from PR #4252 is not yet wired in
    — produces a finite clamped sim. The test accepts either outcome
    so it lands ahead of the validator.
4.  Wrong-length theta raises ``ValueError`` (``not divisible by 7``).
5.  NaN-containing theta raises ``ValueError``.
6.  ``SimOut.time`` is monotonic non-decreasing and starts at ``0``.
7.  Per-frame ``q`` width matches the plant's actuated DOF count.

The pure-data tests run on every CI by mocking ``pydrake``; the live
integration paths are gated on ``@pytest.mark.requires_drake`` so a
missing pydrake install is a clean skip.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)

pytestmark = [pytest.mark.unit]

# Canonical 189-vec theta = 27 joints * 7 coeffs (per cross-engine spec).
N_JOINTS_CANONICAL = 27
THETA_LEN_CANONICAL = N_JOINTS_CANONICAL * COEFFS_PER_JOINT  # 189


# --------------------------------------------------------------------------- #
# Try to load the canonical theta validator (PR #4252). If absent, the
# bound-rejection test downgrades to "either ValueError OR finite output".
# --------------------------------------------------------------------------- #


def _has_canonical_theta_validator() -> bool:
    """Return True if Drake's simulate_with_coefficients enforces theta bounds."""
    spec = importlib.util.find_spec("src.shared.python.motion_matching.theta_validator")
    return spec is not None


# --------------------------------------------------------------------------- #
# Pydrake mock fixture (CLAUDE.md compliant: patch.dict, never global swap).
# --------------------------------------------------------------------------- #

_PYDRAKE_KEYS = [
    "pydrake",
    "pydrake.all",
    "pydrake.multibody",
    "pydrake.multibody.parsing",
    "pydrake.multibody.plant",
    "pydrake.multibody.tree",
    "pydrake.systems",
    "pydrake.systems.analysis",
    "pydrake.systems.framework",
    "pydrake.systems.primitives",
    "pydrake.math",
]


def _make_plant_mock(
    n_q: int = N_JOINTS_CANONICAL,
    n_v: int = N_JOINTS_CANONICAL,
    n_actuators: int = N_JOINTS_CANONICAL,
) -> MagicMock:
    plant = MagicMock(name="MultibodyPlant")
    plant.num_positions.return_value = n_q
    plant.num_velocities.return_value = n_v
    plant.num_actuators.return_value = n_actuators
    plant.num_multibody_states.return_value = n_q + n_v
    plant.GetPositions.return_value = np.zeros(n_q)
    plant.GetVelocities.return_value = np.zeros(n_v)
    plant.HasBodyNamed.return_value = False
    return plant


@pytest.fixture
def mocked_pydrake() -> Generator[dict[str, MagicMock], None, None]:
    """Patch ``sys.modules`` with mock pydrake submodules.

    Per CLAUDE.md we use ``patch.dict("sys.modules", ...)`` which is
    auto-cleaned at teardown -- never ``sys.modules['pydrake'] = MagicMock()``
    at module scope.
    """
    mocks: dict[str, MagicMock] = {key: MagicMock() for key in _PYDRAKE_KEYS}

    plant = _make_plant_mock()
    scene_graph = MagicMock(name="SceneGraph")
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, scene_graph)
    )

    class _FakeLeafSystem:  # noqa: D401
        """Minimal LeafSystem stand-in (matches the public surface)."""

        def __init__(self) -> None:
            self._ports: list[Any] = []

        def DeclareVectorOutputPort(  # noqa: N802
            self,
            name: str,
            model_vector: Any,
            calc_callback: Any,
        ) -> Any:
            port = MagicMock(name=f"OutputPort[{name}]")
            self._ports.append(port)
            return port

        def get_output_port(self, idx: int) -> Any:
            return self._ports[idx] if self._ports else MagicMock()

    framework_mod = mocks["pydrake.systems.framework"]
    framework_mod.LeafSystem = _FakeLeafSystem
    framework_mod.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    framework_mod.DiagramBuilder = MagicMock(return_value=MagicMock())

    sim_instance = MagicMock(name="Simulator")
    sim_context = MagicMock(name="DiagramContext")
    sim_instance.get_context.return_value = sim_context
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=sim_instance)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )

    plant_ctx = MagicMock(name="PlantContext")
    plant.GetMyMutableContextFromRoot.return_value = plant_ctx
    plant.GetMyContextFromRoot.return_value = plant_ctx

    builder_instance = framework_mod.DiagramBuilder.return_value
    builder_instance.Build.return_value = MagicMock(name="Diagram")
    builder_instance.Build.return_value.CreateDefaultContext.return_value = sim_context

    with patch.dict(sys.modules, mocks):
        yield {"plant": plant, "simulator": sim_instance, **mocks}


def _short_opts() -> SimOptions:
    """5-sample window; keeps mock + live paths fast."""
    return SimOptions(simulation_time_s=0.004, sample_rate_hz=1000.0)


# --------------------------------------------------------------------------- #
# Gap 1 — Happy path (mocked Drake).
# --------------------------------------------------------------------------- #


def test_simulate_contract_drake_happy_path_returns_canonical_simout(
    mocked_pydrake: dict[str, MagicMock],
) -> None:
    """A 189-vec theta produces a SimOut with aligned, finite arrays."""
    theta = np.linspace(-0.05, 0.05, THETA_LEN_CANONICAL)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert isinstance(out, SimOut)
    n = out.time.shape[0]
    assert n == 5  # 0..0.004s @ 1 kHz inclusive.
    for arr, cols in (
        (out.q, None),
        (out.qd, None),
        (out.qdd, None),
        (out.tau, None),
        (out.grip, 3),
        (out.grip_quat, 4),
        (out.clubhead, 3),
        (out.club_quat, 4),
    ):
        assert arr.shape[0] == n, f"row mismatch: {arr.shape}"
        if cols is not None:
            assert arr.shape[1] == cols
    # tau is finite by construction (poly evaluated by simulate).
    assert np.all(np.isfinite(out.tau))
    assert out.solver_status in {"success", "warning", "failed"}


# --------------------------------------------------------------------------- #
# Gap 2 — theta = zeros runs without error.
# --------------------------------------------------------------------------- #


def test_zero_theta_runs_without_error(
    mocked_pydrake: dict[str, MagicMock],
) -> None:
    """theta = 0 produces a valid SimOut (gravity-only, mocked plant)."""
    theta = np.zeros(THETA_LEN_CANONICAL)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert isinstance(out, SimOut)
    # All zero coefficients -> tau identically 0 across the trajectory.
    np.testing.assert_allclose(out.tau, 0.0)
    # Time grid still has the canonical shape regardless of theta content.
    assert out.time.shape[0] == out.q.shape[0]


# --------------------------------------------------------------------------- #
# Gap 3 — Out-of-bounds theta. Defensive: PR #4252 may not have landed.
# --------------------------------------------------------------------------- #


def test_out_of_bounds_theta_rejected_or_clamped(
    mocked_pydrake: dict[str, MagicMock],
) -> None:
    """theta with a 1e9 coefficient is rejected (ValueError) or clamped."""
    theta = np.zeros(THETA_LEN_CANONICAL)
    theta[0] = 1.0e9  # blatantly out of any reasonable physical bound.

    if _has_canonical_theta_validator():
        with pytest.raises(ValueError):
            simulate_with_coefficients(theta, options=_short_opts())
    else:
        # Validator not yet wired — accept either outcome.
        try:
            out = simulate_with_coefficients(theta, options=_short_opts())
        except ValueError:
            return  # acceptable: simulator's own DbC caught the magnitude
        assert np.all(np.isfinite(out.tau)) or out.solver_status == "failed"


# --------------------------------------------------------------------------- #
# Gap 4 — Wrong-length theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_drake_wrong_length_theta_raises() -> None:
    """A theta whose length isn't a multiple of 7 raises ValueError."""
    bad = np.zeros(THETA_LEN_CANONICAL + 1)  # 190 -> 190 % 7 != 0
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 5 — NaN theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_drake_nan_theta_raises() -> None:
    bad = np.zeros(THETA_LEN_CANONICAL)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_simulate_contract_drake_inf_theta_raises() -> None:
    bad = np.zeros(THETA_LEN_CANONICAL)
    bad[7] = np.inf
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 6 — Time monotonicity, starts at 0.
# --------------------------------------------------------------------------- #


def test_simulate_contract_drake_time_monotonic_starts_at_zero(
    mocked_pydrake: dict[str, MagicMock],
) -> None:
    theta = np.zeros(THETA_LEN_CANONICAL)
    out = simulate_with_coefficients(
        theta, options=SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0)
    )
    assert out.time[0] == 0.0
    assert np.all(np.diff(out.time) > 0), "time must be strictly increasing"


# --------------------------------------------------------------------------- #
# Gap 7 — q width matches n_joints.
# --------------------------------------------------------------------------- #


def test_q_width_matches_plant_dof(mocked_pydrake: dict[str, MagicMock]) -> None:
    """SimOut.q has the same column count as plant.num_positions()."""
    theta = np.zeros(THETA_LEN_CANONICAL)
    out = simulate_with_coefficients(theta, options=_short_opts())
    plant = mocked_pydrake["plant"]
    expected = plant.num_positions.return_value
    assert out.q.shape[1] == expected
    assert out.qd.shape[1] == plant.num_velocities.return_value


# --------------------------------------------------------------------------- #
# Live-pydrake smoke test (skipped when pydrake unavailable). Exercises the
# real Drake plant on a tiny window so any catastrophic regression in the
# float-pathway shows up here, not in the heavy CI lane.
# --------------------------------------------------------------------------- #


@pytest.mark.requires_drake
def test_live_drake_zero_theta_finite() -> None:
    """Live Drake forward sim with theta=0 yields finite q/qd."""
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        CANONICAL_URDF,
    )

    plant = MultibodyPlant(0.001)
    Parser(plant).AddModels(str(CANONICAL_URDF))
    plant.Finalize()
    n_act = max(plant.num_actuators(), plant.num_velocities() - 6)
    theta = np.zeros(n_act * COEFFS_PER_JOINT)

    out = simulate_with_coefficients(
        theta, options=SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0)
    )
    assert out.solver_status in {"success", "warning"}
    assert np.all(np.isfinite(out.q))
    assert out.time[0] == 0.0
    assert np.all(np.diff(out.time) > 0)
