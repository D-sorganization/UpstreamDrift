"""Tests for the frozen Impact Model Capability & Passivity Matrix (#9550)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.physics.impact_model_capabilities import (
    ALLOWED_DURATION_MODES,
    ALLOWED_SPIN_MODES,
    IMPACT_MODEL_CAPABILITY_MATRIX,
    get_impact_model_capability,
    qualify_spin_prediction,
    verify_impact_passivity,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPACT_ACCEPTANCE_PATH = _REPO_ROOT / "src" / "config" / "impact_acceptance.json"

_CANONICAL_MODEL_IDS = (
    "rigid_body",
    "spring_damper",
    "finite_time",
    "contact_interval",
)


class TestMatrixCoverage:
    def test_all_canonical_models_present(self) -> None:
        for model_id in _CANONICAL_MODEL_IDS:
            assert model_id in IMPACT_MODEL_CAPABILITY_MATRIX

    @pytest.mark.parametrize("model_id", _CANONICAL_MODEL_IDS)
    def test_modes_are_valid(self, model_id: str) -> None:
        capability = IMPACT_MODEL_CAPABILITY_MATRIX[model_id]
        assert capability.model_id == model_id
        assert capability.duration_mode in ALLOWED_DURATION_MODES
        assert capability.tangential_spin_mode in ALLOWED_SPIN_MODES

    def test_spring_damper_never_implies_tangential_spin(self) -> None:
        capability = IMPACT_MODEL_CAPABILITY_MATRIX["spring_damper"]
        assert capability.tangential_spin_mode == "unavailable_normal_only"

    def test_impact_acceptance_models_are_all_registered(self) -> None:
        if not _IMPACT_ACCEPTANCE_PATH.exists():
            pytest.skip("src/config/impact_acceptance.json not present")
        data = json.loads(_IMPACT_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
        models = data.get("model_capabilities", {}).get("models")
        if not models:
            pytest.skip("no models listed in impact_acceptance.json")
        for model in models:
            model_id = model["id"]
            assert model_id in IMPACT_MODEL_CAPABILITY_MATRIX, (
                f"{model_id!r} from impact_acceptance.json is not registered "
                "in IMPACT_MODEL_CAPABILITY_MATRIX"
            )


class TestGetImpactModelCapability:
    def test_returns_known_model(self) -> None:
        capability = get_impact_model_capability("rigid_body")
        assert capability.model_id == "rigid_body"

    def test_raises_type_error_on_non_str(self) -> None:
        with pytest.raises(TypeError):
            get_impact_model_capability(123)  # type: ignore[arg-type]

    def test_raises_value_error_on_empty(self) -> None:
        with pytest.raises(ValueError):
            get_impact_model_capability("")

    def test_raises_value_error_on_unknown(self) -> None:
        with pytest.raises(ValueError):
            get_impact_model_capability("not_a_real_model")


class TestQualifySpinPrediction:
    def test_spring_damper_refuses_zero_rpm_as_prediction(self) -> None:
        result = qualify_spin_prediction("spring_damper", 0.0)
        assert result["spin_status"] == "unavailable_normal_only"
        assert result["predicted_spin_rpm"] is None
        assert result["claim_qualified"] is False

    def test_spring_damper_refuses_any_value(self) -> None:
        result = qualify_spin_prediction("spring_damper", 3000.0)
        assert result["spin_status"] == "unavailable_normal_only"
        assert result["predicted_spin_rpm"] is None
        assert result["claim_qualified"] is False

    def test_contact_interval_qualifies_finite_spin(self) -> None:
        result = qualify_spin_prediction("contact_interval", 2650.0)
        assert result["claim_qualified"] is True
        assert result["predicted_spin_rpm"] == 2650.0
        assert result["spin_status"] == "supported_3d_tangential"

    def test_rigid_body_qualifies_finite_spin(self) -> None:
        result = qualify_spin_prediction("rigid_body", 1500.0)
        assert result["claim_qualified"] is True
        assert result["predicted_spin_rpm"] == 1500.0

    def test_supported_model_raises_on_none(self) -> None:
        with pytest.raises(ValueError):
            qualify_spin_prediction("rigid_body", None)

    def test_supported_model_raises_on_nan(self) -> None:
        with pytest.raises(ValueError):
            qualify_spin_prediction("rigid_body", math.nan)

    def test_supported_model_raises_on_inf(self) -> None:
        with pytest.raises(ValueError):
            qualify_spin_prediction("finite_time", math.inf)

    def test_unknown_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            qualify_spin_prediction("not_a_real_model", 100.0)


class TestVerifyImpactPassivity:
    def test_passes_for_dissipative_impact(self) -> None:
        assert verify_impact_passivity(ke_in_j=100.0, ke_out_j=80.0) is True

    def test_passes_within_atol_of_equal_energy(self) -> None:
        assert verify_impact_passivity(ke_in_j=100.0, ke_out_j=100.0 + 1e-10) is True

    def test_passes_with_external_work(self) -> None:
        assert (
            verify_impact_passivity(ke_in_j=100.0, ke_out_j=150.0, work_external_j=60.0)
            is True
        )

    def test_fails_when_energy_created(self) -> None:
        assert (
            verify_impact_passivity(ke_in_j=100.0, ke_out_j=200.0, atol_j=1e-9) is False
        )

    def test_raises_on_negative_ke_in(self) -> None:
        with pytest.raises(ValueError):
            verify_impact_passivity(ke_in_j=-1.0, ke_out_j=10.0)

    def test_raises_on_negative_ke_out(self) -> None:
        with pytest.raises(ValueError):
            verify_impact_passivity(ke_in_j=10.0, ke_out_j=-1.0)

    def test_raises_on_negative_atol(self) -> None:
        with pytest.raises(ValueError):
            verify_impact_passivity(ke_in_j=10.0, ke_out_j=5.0, atol_j=-1e-9)

    def test_raises_on_nan_ke_in(self) -> None:
        with pytest.raises(ValueError):
            verify_impact_passivity(ke_in_j=math.nan, ke_out_j=5.0)

    def test_raises_on_inf_ke_out(self) -> None:
        with pytest.raises(ValueError):
            verify_impact_passivity(ke_in_j=10.0, ke_out_j=math.inf)
