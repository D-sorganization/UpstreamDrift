"""Launcher capability profile contracts for the gradient taxonomy."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from src.api.routes.launcher import _build_engine_profiles

_GRADIENT_TAXONOMY_FIELDS = {
    "parameter_gradients",
    "state_control_gradients",
    "forward_sim",
    "contact_step",
    "trajectory_opt",
}


def _profiles() -> dict[str, dict[str, str]]:
    return {
        engine_id: capabilities.to_dict()
        for engine_id, capabilities in _build_engine_profiles().items()
    }


def test_launcher_profiles_include_jaxsim_gradient_taxonomy() -> None:
    profiles = _profiles()

    assert "jaxsim" in profiles
    jaxsim = profiles["jaxsim"]
    assert jaxsim["engine_name"] == "JaxSim"
    assert jaxsim["parameter_gradients"] == "full"
    assert jaxsim["state_control_gradients"] == "full"
    assert jaxsim["forward_sim"] == "full"
    assert jaxsim["contact_step"] == "partial"
    assert jaxsim["trajectory_opt"] == "partial"


def test_all_launcher_profiles_emit_gradient_taxonomy_fields() -> None:
    profiles = _profiles()

    for engine_id, profile in profiles.items():
        missing = _GRADIENT_TAXONOMY_FIELDS - set(profile)
        assert not missing, f"{engine_id} missing fields: {sorted(missing)}"
