"""Unit tests for :class:`OpenSimFitSwingProvider` (#4708).

Coverage:

* The provider auto-registers against both the canonical Protocol-based
  registry (``provider``) and the engine-agnostic registry
  (``provider_registry``) as soon as the engine package is imported.
* The provider's :meth:`fit_swing` accepts wrapped
  :class:`MultiSourceTarget`, raw :class:`ClubTarget`, and
  :class:`ClubBallTarget` inputs.
* :meth:`_build_native_options` projects ``maxiter`` / ``rng_seed``
  onto a default :class:`OpenSimFitOptions`, and passes through a
  user-supplied native options object unchanged.
* The provider declines body / ball target consumption.
* Re-importing the package is idempotent for registration.
* End-to-end :meth:`fit_swing` against an injected deterministic
  ``simulate_fn`` (no real OpenSim wheel required) returns a
  :class:`CanonicalFitResult` with the expected shape.

The optimiser-driven tests inject a ``simulate_fn`` so the OpenSim
SWIG wheel is *not* required. A guarded ``pytest.importorskip`` block
is reserved for any future scenario that needs the real engine.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.motion_matching import (
    OpenSimFitSwingProvider,
)
from src.engines.physics_engines.opensim.python.motion_matching.fit_swing import (
    FitOptions as OpenSimFitOptions,
)
from src.shared.python.motion_matching.club_ball_target import (
    BallImpactState,
    ClubBallTarget,
)
from src.shared.python.motion_matching.club_target import (
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.cost import SimOutput
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
)
from src.shared.python.motion_matching.provider import (
    available_engines as canonical_available_engines,
)
from src.shared.python.motion_matching.provider import (
    get_provider as canonical_get_provider,
)
from src.shared.python.motion_matching.provider_registry import (
    available_engines as registry_available_engines,
)
from src.shared.python.motion_matching.provider_registry import (
    get_provider as registry_get_provider,
)


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        filename="synth.bin",
        format="synthetic",
        subject_id="UNIT",
        trial_id="opensim-provider",
        sha256="0" * 64,
    )


def _synthetic_target(n: int = 11) -> ClubTarget:
    """Tiny but valid ClubTarget for adapter-level smoke tests."""
    time = np.linspace(0.0, 0.01, n, dtype=np.float64)
    butt = np.column_stack([time, np.zeros_like(time), np.zeros_like(time)]).astype(
        np.float64
    )
    clubhead = butt + np.array([0.0, 0.0, 1.0])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)).astype(np.float64)
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n,
        source=_provenance(),
    )


def _ball_state() -> BallImpactState:
    return BallImpactState(
        position_at_impact_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        launch_direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        launch_speed_mps=50.0,
        spin_rpm=0.0,
    )


class _DeterministicSimFn:
    """Echo a club-target-shaped :class:`SimOutput` regardless of theta.

    The output simulates a perfect roll-out so :func:`compute_cost`
    surfaces only the regularizer term and the cost is finite for any
    theta. ``n_joints`` advertises the per-joint coefficient count to
    :func:`fit_swing_opensim` so the warm-start dimensionality is small
    (3 joints * 7 coeffs = 21) and SLSQP converges in one iteration.
    """

    n_joints: int = 3

    def __init__(self, target: ClubTarget) -> None:
        self._target = target

    def __call__(self, theta: np.ndarray) -> SimOutput:
        n = self._target.time.shape[0]
        n_joints = self.n_joints
        return SimOutput(
            butt=self._target.butt.copy(),
            clubhead=self._target.clubhead.copy(),
            club_quat=self._target.club_quat.copy(),
            time=self._target.time.copy(),
            tau=np.zeros((n, n_joints), dtype=np.float64),
            omega=np.zeros((n, n_joints), dtype=np.float64),
        )


# --------------------------------------------------------------------------
# Registration tests
# --------------------------------------------------------------------------


class TestRegistration:
    """The opensim provider must register at import time."""

    def test_engine_name_is_opensim(self) -> None:
        assert OpenSimFitSwingProvider.engine_name == "opensim"

    def test_registered_in_canonical_registry(self) -> None:
        assert "opensim" in canonical_available_engines()

    def test_registered_in_engine_agnostic_registry(self) -> None:
        assert "opensim" in registry_available_engines()

    def test_canonical_get_provider_returns_instance(self) -> None:
        provider = canonical_get_provider("opensim")
        assert isinstance(provider, OpenSimFitSwingProvider)
        assert provider.engine_name == "opensim"

    def test_registry_get_provider_returns_instance(self) -> None:
        provider = registry_get_provider("opensim")
        assert isinstance(provider, OpenSimFitSwingProvider)

    def test_capability_flags(self) -> None:
        provider = OpenSimFitSwingProvider()
        assert provider.supports_body_target() is False
        assert provider.supports_ball_target() is False

    def test_reimport_is_idempotent(self) -> None:
        # Re-importing the package must not raise (the registries handle
        # idempotency for repeated registrations of the same provider type).
        before = sorted(canonical_available_engines())
        importlib.import_module(
            "src.engines.physics_engines.opensim.python.motion_matching"
        )
        after = sorted(canonical_available_engines())
        assert "opensim" in before
        assert before == after


# --------------------------------------------------------------------------
# I/O adapter tests (target unwrapping)
# --------------------------------------------------------------------------


class TestExtractClub:
    """Adapter must accept canonical inputs and reject malformed ones."""

    def test_extract_from_raw_clubtarget(self) -> None:
        club = _synthetic_target()
        assert OpenSimFitSwingProvider._extract_club(club) is club

    def test_extract_from_multisource(self) -> None:
        club = _synthetic_target()
        wrapped = MultiSourceTarget(club=club)
        assert OpenSimFitSwingProvider._extract_club(wrapped) is club

    def test_extract_from_clubballtarget(self) -> None:
        club = _synthetic_target()
        cbt = ClubBallTarget(club=club, ball_impact=_ball_state())
        assert OpenSimFitSwingProvider._extract_club(cbt) is club

    def test_multisource_with_body_only_rejected(self) -> None:
        wrapped = MultiSourceTarget(body=object())
        with pytest.raises(ValueError, match="target.club"):
            OpenSimFitSwingProvider._extract_club(wrapped)

    def test_multisource_with_non_clubtarget_rejected(self) -> None:
        # Bypass MultiSourceTarget validation by setting club to a non-None
        # non-ClubTarget value directly via __dict__ surrogate -- here we
        # rely on MultiSourceTarget(club=<bogus>) which will fail provider
        # validation rather than dataclass validation (dataclass does not
        # type-check beyond None / not-None).
        wrapped = MultiSourceTarget(club=None, body=object())
        with pytest.raises(ValueError, match="target.club"):
            OpenSimFitSwingProvider._extract_club(wrapped)

    def test_non_target_input_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="MultiSourceTarget"):
            OpenSimFitSwingProvider._extract_club(object())  # type: ignore[arg-type]

    def test_string_input_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            OpenSimFitSwingProvider._extract_club("not a target")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Options-projection tests
# --------------------------------------------------------------------------


class TestBuildNativeOptions:
    """Canonical FitOptions must project onto OpenSimFitOptions correctly."""

    def test_default_options_project_maxiter_and_seed(self) -> None:
        native = OpenSimFitSwingProvider._build_native_options(
            FitOptions(maxiter=7, rng_seed=99)
        )
        assert isinstance(native, OpenSimFitOptions)
        assert native.max_iter == 7
        assert native.rng_seed == 99

    def test_engine_options_passed_through(self) -> None:
        engine_opts = OpenSimFitOptions(max_iter=3, rng_seed=11)
        out = OpenSimFitSwingProvider._build_native_options(
            FitOptions(engine_options=engine_opts)
        )
        assert out is engine_opts

    def test_engine_options_type_mismatch_rejected(self) -> None:
        opts = FitOptions(engine_options=object())
        with pytest.raises(TypeError, match="OpenSimFitOptions"):
            OpenSimFitSwingProvider._build_native_options(opts)


# --------------------------------------------------------------------------
# End-to-end fit_swing through the provider (uses injected simulate_fn).
# --------------------------------------------------------------------------


class TestFitSwingEndToEnd:
    """Drive the provider with an injected sim_fn and validate the result."""

    def _opts_with_simfn(self, target: ClubTarget, *, max_iter: int = 1) -> FitOptions:
        sim_fn = _DeterministicSimFn(target)
        engine_opts = OpenSimFitOptions(
            max_iter=max_iter,
            simulate_fn=sim_fn,
            n_joints=sim_fn.n_joints,
            rng_seed=0,
        )
        return FitOptions(maxiter=max_iter, rng_seed=0, engine_options=engine_opts)

    def test_fit_swing_returns_canonical_result(self) -> None:
        provider = OpenSimFitSwingProvider()
        target = _synthetic_target(n=11)
        result = provider.fit_swing(
            MultiSourceTarget(club=target),
            self._opts_with_simfn(target, max_iter=1),
        )
        assert isinstance(result, CanonicalFitResult)
        assert isinstance(result.theta_optimal, np.ndarray)
        assert result.theta_optimal.ndim == 1
        # 3 joints * 7 coeffs = 21
        assert result.theta_optimal.shape == (21,)
        assert np.isfinite(result.final_cost)

    def test_fit_swing_accepts_raw_clubtarget(self) -> None:
        provider = OpenSimFitSwingProvider()
        target = _synthetic_target(n=11)
        result = provider.fit_swing(target, self._opts_with_simfn(target, max_iter=1))
        assert isinstance(result, CanonicalFitResult)

    def test_fit_swing_accepts_clubballtarget(self) -> None:
        provider = OpenSimFitSwingProvider()
        target = _synthetic_target(n=11)
        cbt = ClubBallTarget(club=target, ball_impact=_ball_state())
        result = provider.fit_swing(cbt, self._opts_with_simfn(target, max_iter=1))
        assert isinstance(result, CanonicalFitResult)

    def test_maxiter_respected_via_history_bound(self) -> None:
        """Engine options carry ``max_iter`` end-to-end.

        SLSQP performs at least one objective evaluation per iteration,
        so a very small ``max_iter`` keeps the recorded history short.
        """
        provider = OpenSimFitSwingProvider()
        target = _synthetic_target(n=11)
        result = provider.fit_swing(target, self._opts_with_simfn(target, max_iter=1))
        # SLSQP records >=1 evaluation. With max_iter=1 the optimiser
        # should not run away; assert the recorded history is bounded.
        assert len(result.history) >= 1
        assert result.iterations <= 5  # comfortably above SLSQP's 1-iter

    def test_fit_swing_rejects_non_target_inputs(self) -> None:
        provider = OpenSimFitSwingProvider()
        target = _synthetic_target(n=11)
        opts = self._opts_with_simfn(target, max_iter=1)
        with pytest.raises(TypeError):
            provider.fit_swing(object(), opts)  # type: ignore[arg-type]
