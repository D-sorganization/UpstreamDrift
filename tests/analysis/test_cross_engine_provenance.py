"""Provenance tests for the cross-engine comparison service (#8816, #8817).

Contract under test:

- Requesting engine X either runs real X or degrades EXPLICITLY: every
  result carries the actual backend identity (``real`` vs ``stub_2dof``).
- ``run_cross_engine_study`` declares stub substitution in its payload and
  can refuse it outright (``allow_stub_substitution=False``).
- Per-engine robustness comes from genuine per-engine statistics, never a
  single aggregate value replicated per engine.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.analysis import cross_engine as ce
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineRunResult,
    CrossEngineSimConfig,
)

pytestmark = pytest.mark.unit

_FAST_CONFIG = CrossEngineSimConfig(t_end=0.1, dt=0.01, n_trials=2, seed=1)


def _no_real_engine(_name: str) -> Any | None:
    """Injected builder simulating a machine without real engine wheels."""
    return None


class _FakeRealEngine:
    """Duck-typed steppable engine standing in for a real backend."""

    def __init__(self) -> None:
        self._stub = ce.StubEngine("fake_real", n_dof=2)

    def reset(self) -> None:
        self._stub.reset()

    def set_control(self, u: Any) -> None:
        self._stub.set_control(u)

    def step(self, dt: float | None = None) -> None:
        self._stub.step(dt)

    def get_state(self) -> Any:
        return self._stub.get_state()


class TestBuildEngineWithBackend:
    """build_engine_with_backend must report the actual backend identity."""

    def test_unavailable_real_engine_reports_stub_backend(self) -> None:
        engine, backend = ce.build_engine_with_backend(
            "drake", try_real=_no_real_engine
        )
        assert backend == ce.BACKEND_STUB
        assert isinstance(engine, ce.StubEngine)

    def test_available_real_engine_reports_real_backend(self) -> None:
        real = _FakeRealEngine()
        engine, backend = ce.build_engine_with_backend(
            "mujoco", try_real=lambda _: real
        )
        assert backend == ce.BACKEND_REAL
        assert engine is real

    def test_pendulum_stub_is_stub_backend(self) -> None:
        engine, backend = ce.build_engine_with_backend("pendulum_stub")
        assert backend == ce.BACKEND_STUB
        assert isinstance(engine, ce.StubEngine)


class TestSubstitutedEngines:
    """pendulum_stub is requested-as-stub, never a silent substitution."""

    def test_substituted_excludes_pendulum_stub(self) -> None:
        backends = {
            "pendulum_stub": ce.BACKEND_STUB,
            "drake": ce.BACKEND_STUB,
            "mujoco": ce.BACKEND_REAL,
        }
        assert ce.substituted_engines(backends) == ["drake"]


class TestRunComparisonWithProvenance:
    def test_returns_backend_per_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ce, "try_build_real_engine", _no_real_engine)
        results, cv_summary, backends = ce.run_comparison_with_provenance(
            ["drake", "pendulum_stub"], _FAST_CONFIG
        )
        assert set(backends) == {"drake", "pendulum_stub"}
        assert backends["drake"] == ce.BACKEND_STUB
        assert set(results) == {"drake", "pendulum_stub"}
        assert cv_summary  # non-empty CV summary


class TestPerEngineRobustness:
    def test_differing_engines_produce_differing_scores(self) -> None:
        results = {
            "a": CrossEngineRunResult(
                engine_name="a",
                mean_total_energy_final=1.0,
                std_total_energy_final=0.5,
                mean_end_effector_speed_final=1.0,
                std_end_effector_speed_final=0.5,
                mean_peak_end_effector_speed=1.0,
                std_peak_end_effector_speed=0.5,
            ),
            "b": CrossEngineRunResult(
                engine_name="b",
                mean_total_energy_final=1.0,
                std_total_energy_final=0.05,
                mean_end_effector_speed_final=1.0,
                std_end_effector_speed_final=0.05,
                mean_peak_end_effector_speed=1.0,
                std_peak_end_effector_speed=0.05,
            ),
        }
        scores = ce.per_engine_robustness(results)
        assert set(scores) == {"a", "b"}
        assert scores["a"] != scores["b"]
        for value in scores.values():
            assert 0.0 <= value <= 1.0


class TestRunCrossEngineStudyProvenance:
    def test_payload_declares_backend_and_substitution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ce, "try_build_real_engine", _no_real_engine)
        payload = ce.run_cross_engine_study(["drake", "pendulum_stub"], _FAST_CONFIG)
        assert payload["engines"]["drake"]["backend"] == ce.BACKEND_STUB
        assert payload["engines"]["pendulum_stub"]["backend"] == ce.BACKEND_STUB
        assert payload["stubbed_engines"] == ["drake"]

    def test_refuses_substitution_when_disallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ce, "try_build_real_engine", _no_real_engine)
        with pytest.raises(ValueError, match="drake"):
            ce.run_cross_engine_study(
                ["drake"], _FAST_CONFIG, allow_stub_substitution=False
            )

    def test_pendulum_stub_alone_is_allowed_without_substitution(self) -> None:
        payload = ce.run_cross_engine_study(
            ["pendulum_stub"], _FAST_CONFIG, allow_stub_substitution=False
        )
        assert payload["stubbed_engines"] == []
