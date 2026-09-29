"""CC-7 conformance harness tests for adapter merge gating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.shared.python.engine_core.cross_engine_validator import (
    ConformanceReference,
    CrossEngineValidator,
    DivergenceRegistry,
)


@dataclass(frozen=True)
class _Capabilities:
    supported: frozenset[str]

    def supports(self, capability: str) -> bool:
        return capability in self.supported


class _MockAdapter:
    def __init__(
        self,
        engine_name: str = "mock-engine",
        *,
        capabilities: frozenset[str] | None = None,
        differential_offset: np.ndarray | None = None,
    ) -> None:
        self.engine_name = engine_name
        self.capabilities = _Capabilities(
            capabilities
            or frozenset({"forward_sim", "inverse_dynamics", "mass_matrix"})
        )
        self._differential_offset = (
            np.zeros(3) if differential_offset is None else differential_offset
        )

    def from_canonical(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def to_canonical(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def forward_kinematics(
        self,
        state: dict[str, np.ndarray],
    ) -> dict[str, dict[str, np.ndarray]]:
        del state
        return _reference().fk_poses

    def inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray:
        del q, v
        return a.copy()

    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        del q, v
        return tau.copy()

    def exported_mass_properties(self) -> dict[str, np.ndarray]:
        return {
            key: value.copy() for key, value in _reference().mass_properties.items()
        }

    def differential_state(self, state: np.ndarray) -> np.ndarray:
        return state + self._differential_offset


def _reference() -> ConformanceReference:
    return ConformanceReference(
        canonical_state={
            "pelvis_xyz": np.array([0.1, 0.2, 0.3]),
            "pelvis_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "lead_wrist": np.array([0.25]),
        },
        rigid_dofs=("pelvis_xyz", "lead_wrist"),
        quaternion_dofs=("pelvis_quat",),
        fk_poses={
            "address": {"clubhead": np.array([1.0, 2.0, 3.0])},
            "impact": {"clubhead": np.array([1.2, 2.1, 2.9])},
        },
        inverse_dynamics_q=np.array([0.1, 0.2, 0.3]),
        inverse_dynamics_v=np.array([0.0, 0.1, 0.0]),
        inverse_dynamics_a=np.array([0.5, -0.25, 0.125]),
        mass_properties={
            "mass": np.array([72.0]),
            "center_of_mass": np.array([0.0, 0.1, 0.9]),
            "inertia": np.diag([1.0, 1.5, 2.0]),
        },
        differential_state=np.array([0.4, 0.5, 0.6]),
    )


def test_five_conformance_checks_pass_against_stub_adapter() -> None:
    validator = CrossEngineValidator()
    adapter = _MockAdapter()
    reference = _reference()

    results = [
        validator.validate_round_trip_state_remap(adapter, reference),
        validator.validate_forward_kinematics(adapter, reference),
        validator.validate_inverse_forward_dynamics_consistency(adapter, reference),
        *validator.validate_post_export_mass_properties(adapter, reference),
        validator.validate_differential_against_reference(
            adapter,
            _MockAdapter(engine_name="mock-reference"),
            reference,
        ),
    ]

    assert {result.check_name for result in results} == {
        "round_trip_state_remap",
        "forward_kinematics_reference_pose",
        "inverse_forward_dynamics_consistency",
        "post_export_mass_properties",
        "differential_cross_engine_reference",
    }
    assert all(result.passed for result in results)
    assert not any(result.skipped for result in results)


class _RaisingCapabilities:
    """A capability descriptor whose ``supports()`` always raises."""

    def supports(self, capability: str) -> bool:
        raise RuntimeError(f"capability backend offline for {capability}")


class _AdvertisesButMissingMethodAdapter:
    """Advertises every capability but implements none of the methods."""

    engine_name = "mock-half-implemented"
    capabilities = _Capabilities(
        frozenset({"forward_sim", "inverse_dynamics", "mass_matrix"})
    )


class _SupportsRaisesAdapter:
    """Advertises nothing because ``supports()`` raises for every query."""

    engine_name = "mock-supports-raises"
    capabilities = _RaisingCapabilities()


def test_missing_method_for_advertised_capability_is_failure_not_skip() -> None:
    """A half-implemented adapter must FAIL, not clear the gate via a skip."""
    validator = CrossEngineValidator()
    adapter = _AdvertisesButMissingMethodAdapter()
    reference = _reference()

    fk = validator.validate_forward_kinematics(adapter, reference)
    idfd = validator.validate_inverse_forward_dynamics_consistency(adapter, reference)
    mass = validator.validate_post_export_mass_properties(adapter, reference)
    round_trip = validator.validate_round_trip_state_remap(adapter, reference)

    for result in (fk, idfd, *mass, round_trip):
        assert not result.passed, result.check_name
        assert not result.skipped, result.check_name
        assert "missing adapter method" in result.message


def test_supports_exception_is_failure_not_free_pass() -> None:
    """A throwing ``supports()`` must NOT route to a passing skip (#6891)."""
    validator = CrossEngineValidator()
    adapter = _SupportsRaisesAdapter()
    reference = _reference()

    result = validator.validate_inverse_forward_dynamics_consistency(adapter, reference)

    assert not result.passed
    assert not result.skipped
    assert "raised" in result.message


def test_capability_aware_skip_for_missing_inverse_dynamics() -> None:
    validator = CrossEngineValidator()
    adapter = _MockAdapter(capabilities=frozenset({"forward_sim", "mass_matrix"}))

    result = validator.validate_inverse_forward_dynamics_consistency(
        adapter,
        _reference(),
    )

    assert result.passed
    assert result.skipped
    assert "inverse_dynamics" in result.message


def test_divergence_registry_accepts_registered_difference() -> None:
    validator = CrossEngineValidator()
    registry = DivergenceRegistry.from_yaml(
        Path(__file__).with_name("divergence_registry.yaml")
    )
    adapter = _MockAdapter(
        engine_name="mock-soft-contact",
        differential_offset=np.array([0.002, 0.0, 0.0]),
    )

    result = validator.validate_differential_against_reference(
        adapter,
        _MockAdapter(engine_name="mock-reference"),
        _reference(),
        registry,
    )

    assert result.passed
    assert result.divergence is not None
    assert result.divergence.id == "soft-vs-hard-contact-smoke"


def test_unregistered_divergence_is_hard_failure() -> None:
    validator = CrossEngineValidator()
    adapter = _MockAdapter(
        engine_name="mock-unregistered",
        differential_offset=np.array([0.002, 0.0, 0.0]),
    )

    result = validator.validate_differential_against_reference(
        adapter,
        _MockAdapter(engine_name="mock-reference"),
        _reference(),
        DivergenceRegistry(),
    )

    assert not result.passed
    assert "unregistered divergence" in result.message
