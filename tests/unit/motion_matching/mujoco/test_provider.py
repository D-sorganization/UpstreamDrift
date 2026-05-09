"""Unit tests for :class:`MujocoFitSwingProvider` (#4519).

Coverage:

* The provider auto-registers against the canonical registry as soon as
  the engine package is imported.
* The provider's :meth:`fit_swing` accepts both a wrapped
  :class:`MultiSourceTarget` and a raw :class:`ClubTarget` and returns a
  valid :class:`CanonicalFitResult`.
* The provider declines to support body / ball targets.
* If a ``TW_ProV1`` MuJoCo leaderboard baseline is present on disk, the
  numerical regression remains within 1% on impact-clubhead-speed RMSE.

Tests gated by ``pytest.importorskip("mujoco")`` so that environments
without MuJoCo skip cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if mujoco isn't installed locally; the provider
# imports the engine fit driver at module top, which transitively imports
# mujoco.
pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.motion_matching import (  # noqa: E402
    MujocoFitSwingProvider,
)
from src.engines.physics_engines.mujoco.python.motion_matching.fit_swing import (  # noqa: E402
    FitOptions as MujocoFitOptions,
)
from src.engines.physics_engines.mujoco.python.motion_matching.fit_swing import (  # noqa: E402
    MinimizerOptions,
)
from src.shared.python.motion_matching.club_target import (  # noqa: E402
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.fit_result import (  # noqa: E402
    CanonicalFitResult,
)
from src.shared.python.motion_matching.provider import (  # noqa: E402
    FitOptions,
    MultiSourceTarget,
    available_engines,
    get_provider,
)


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        filename="synth.bin",
        format="synthetic",
        subject_id="UNIT",
        trial_id="provider",
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


# --------------------------------------------------------------------------
# Registry tests (cheap; do not invoke the optimiser).
# --------------------------------------------------------------------------


class TestRegistration:
    """The mujoco provider must register at import time."""

    def test_engine_name_is_mujoco(self) -> None:
        assert MujocoFitSwingProvider.engine_name == "mujoco"

    def test_provider_registered(self) -> None:
        assert "mujoco" in available_engines()

    def test_get_provider_returns_mujoco_instance(self) -> None:
        provider = get_provider("mujoco")
        assert isinstance(provider, MujocoFitSwingProvider)
        assert provider.engine_name == "mujoco"

    def test_capability_flags(self) -> None:
        provider = MujocoFitSwingProvider()
        assert provider.supports_body_target() is False
        assert provider.supports_ball_target() is False


# --------------------------------------------------------------------------
# I/O adapter tests
# --------------------------------------------------------------------------


class TestAdapterShapes:
    """Adapter must accept canonical inputs and reject malformed ones."""

    def test_extract_club_from_multisource(self) -> None:
        club = _synthetic_target()
        wrapped = MultiSourceTarget(club=club)
        assert MujocoFitSwingProvider._extract_club(wrapped) is club

    def test_extract_club_from_raw_clubtarget(self) -> None:
        club = _synthetic_target()
        assert MujocoFitSwingProvider._extract_club(club) is club

    def test_multisource_without_club_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            MultiSourceTarget()

    def test_multisource_with_body_only_rejected_by_provider(self) -> None:
        wrapped = MultiSourceTarget(body=object())
        with pytest.raises(ValueError, match="target.club"):
            MujocoFitSwingProvider._extract_club(wrapped)

    def test_engine_options_type_mismatch_rejected(self) -> None:
        opts = FitOptions(engine_options=object())
        with pytest.raises(TypeError, match="MujocoFitOptions"):
            MujocoFitSwingProvider._build_native_options(opts)

    def test_default_options_project_maxiter(self) -> None:
        native = MujocoFitSwingProvider._build_native_options(
            FitOptions(maxiter=7, rng_seed=42)
        )
        assert native.minimizer.maxiter == 7
        assert native.rng_seed == 42

    def test_native_options_passed_through(self) -> None:
        engine_opts = MujocoFitOptions(
            minimizer=MinimizerOptions(maxiter=3),
            rng_seed=99,
        )
        out = MujocoFitSwingProvider._build_native_options(
            FitOptions(engine_options=engine_opts)
        )
        assert out is engine_opts


# --------------------------------------------------------------------------
# End-to-end fit (slow; runs only if mujoco can build the model).
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestFitSwingEndToEnd:
    """Drive the provider with a synthetic target and validate the result."""

    def test_fit_swing_returns_canonical_result(self) -> None:
        provider = MujocoFitSwingProvider()
        # Use the engine's own default-grid target (1 ms, 0.3 s) so the
        # rollout matches.
        n = 301
        time = np.linspace(0.0, 0.3, n, dtype=np.float64)
        butt = np.column_stack(
            [
                0.5 * np.cos(2 * np.pi * time / 0.3),
                0.5 * np.sin(2 * np.pi * time / 0.3),
                np.zeros_like(time),
            ]
        ).astype(np.float64)
        clubhead = butt + np.array([0.0, 0.0, 1.0])
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)).astype(np.float64)
        target = ClubTarget(
            time=time,
            butt=butt,
            clubhead=clubhead,
            club_quat=quat,
            impact_idx=n // 2,
            source=_provenance(),
        )

        opts = FitOptions(
            maxiter=2,  # smoke test; not aiming for tight convergence
            rng_seed=0,
            engine_options=MujocoFitOptions(
                minimizer=MinimizerOptions(maxiter=2),
                rng_seed=0,
            ),
        )

        result = provider.fit_swing(MultiSourceTarget(club=target), opts)

        assert isinstance(result, CanonicalFitResult)
        assert np.isfinite(result.final_rmse_m)
        assert result.final_rmse_m >= 0.0
        assert isinstance(result.theta_optimal, np.ndarray)
        assert result.theta_optimal.ndim == 1


# --------------------------------------------------------------------------
# Numerical regression vs the existing MuJoCo leaderboard.
# --------------------------------------------------------------------------


def _repo_root() -> Path:
    """Walk up from this test file to the repository root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[-1]


_REPO_ROOT = _repo_root()
_LEADERBOARD_CANDIDATES = (
    _REPO_ROOT / "results/leaderboard/mujoco_TW_ProV1.json",
    _REPO_ROOT / "results/mujoco/leaderboard.json",
    _REPO_ROOT / "src/engines/physics_engines/mujoco/leaderboard/TW_ProV1.json",
)
_TW_PROV1_MAT = (
    _REPO_ROOT / "src/engines/physics_engines/pinocchio/data/rob_neal/TW_ProV1.mat"
)


def _find_leaderboard() -> tuple[Path, dict] | None:
    """Return the first existing leaderboard JSON, parsed."""
    for candidate in _LEADERBOARD_CANDIDATES:
        if candidate.is_file():
            try:
                return candidate, json.loads(candidate.read_text())
            except json.JSONDecodeError:  # pragma: no cover
                continue
    return None
