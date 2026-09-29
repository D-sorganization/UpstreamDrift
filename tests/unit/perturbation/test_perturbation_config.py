"""Tests for src.shared.python.perturbation.config (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.perturbation.config import (
    PerturbationConfig,
    PerturbationSummary,
)


class TestPerturbationConfig:
    def test_perturbation_config_default_construction(self) -> None:
        cfg = PerturbationConfig()
        assert cfg.n_trials == 100
        assert cfg.noise_type == "white"
        assert cfg.noise_amplitude == pytest.approx(0.1)
        assert cfg.perturb_mode == "additive"
        assert cfg.seed is None

    def test_custom_construction(self) -> None:
        cfg = PerturbationConfig(
            n_trials=50,
            noise_type="pink",
            noise_amplitude=0.05,
            perturb_mode="multiplicative",
            seed=42,
        )
        assert cfg.n_trials == 50
        assert cfg.noise_type == "pink"
        assert cfg.noise_amplitude == pytest.approx(0.05)
        assert cfg.perturb_mode == "multiplicative"
        assert cfg.seed == 42

    def test_n_trials_zero_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            PerturbationConfig(n_trials=0)

    def test_n_trials_negative_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            PerturbationConfig(n_trials=-10)

    def test_noise_amplitude_zero_allowed(self) -> None:
        cfg = PerturbationConfig(noise_amplitude=0.0)
        assert cfg.noise_amplitude == pytest.approx(0.0)

    def test_noise_amplitude_negative_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            PerturbationConfig(noise_amplitude=-0.1)

    def test_noise_type_white(self) -> None:
        cfg = PerturbationConfig(noise_type="white")
        assert cfg.noise_type == "white"

    def test_noise_type_pink(self) -> None:
        cfg = PerturbationConfig(noise_type="pink")
        assert cfg.noise_type == "pink"

    def test_noise_type_brown(self) -> None:
        cfg = PerturbationConfig(noise_type="brown")
        assert cfg.noise_type == "brown"

    def test_noise_type_invalid_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            PerturbationConfig(noise_type="red")

    def test_perturb_mode_additive(self) -> None:
        cfg = PerturbationConfig(perturb_mode="additive")
        assert cfg.perturb_mode == "additive"

    def test_perturb_mode_multiplicative(self) -> None:
        cfg = PerturbationConfig(perturb_mode="multiplicative")
        assert cfg.perturb_mode == "multiplicative"

    def test_perturb_mode_both(self) -> None:
        cfg = PerturbationConfig(perturb_mode="both")
        assert cfg.perturb_mode == "both"

    def test_perturb_mode_invalid_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            PerturbationConfig(perturb_mode="subtractive")

    def test_seed_integer(self) -> None:
        cfg = PerturbationConfig(seed=123)
        assert cfg.seed == 123

    def test_seed_zero(self) -> None:
        cfg = PerturbationConfig(seed=0)
        assert cfg.seed == 0


class TestPerturbationSummary:
    def _make_summary(self, **kwargs) -> PerturbationSummary:
        defaults = {
            "engine_name": "test_engine",
            "config": PerturbationConfig(),
            "robustness_score": 0.85,
            "metrics": {"rmse": 0.12},
            "success_rate": 0.95,
            "execution_time_sec": 3.14,
        }
        defaults.update(kwargs)
        return PerturbationSummary(**defaults)

    def test_perturbation_config_construction(self) -> None:
        s = self._make_summary()
        assert s.engine_name == "test_engine"
        assert s.robustness_score == pytest.approx(0.85)
        assert s.success_rate == pytest.approx(0.95)
        assert s.execution_time_sec == pytest.approx(3.14)

    def test_perturbation_config_to_dict_returns_dict(self) -> None:
        s = self._make_summary()
        d = s.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_engine_name(self) -> None:
        s = self._make_summary(engine_name="my_engine")
        d = s.to_dict()
        assert d["engine_name"] == "my_engine"

    def test_to_dict_has_config(self) -> None:
        s = self._make_summary()
        d = s.to_dict()
        assert "config" in d
        assert isinstance(d["config"], dict)

    def test_to_dict_config_fields(self) -> None:
        cfg = PerturbationConfig(n_trials=50, noise_type="brown", seed=7)
        s = self._make_summary(config=cfg)
        d = s.to_dict()
        assert d["config"]["n_trials"] == 50
        assert d["config"]["noise_type"] == "brown"
        assert d["config"]["seed"] == 7

    def test_to_dict_has_robustness_score(self) -> None:
        s = self._make_summary(robustness_score=0.72)
        d = s.to_dict()
        assert d["robustness_score"] == pytest.approx(0.72)

    def test_to_dict_has_success_rate(self) -> None:
        s = self._make_summary(success_rate=0.9)
        d = s.to_dict()
        assert d["success_rate"] == pytest.approx(0.9)

    def test_to_dict_has_execution_time(self) -> None:
        s = self._make_summary(execution_time_sec=5.5)
        d = s.to_dict()
        assert d["execution_time_sec"] == pytest.approx(5.5)

    def test_to_dict_has_metrics(self) -> None:
        s = self._make_summary(metrics={"rmse": 0.05, "mae": 0.03})
        d = s.to_dict()
        assert "metrics" in d
        assert d["metrics"]["rmse"] == pytest.approx(0.05)

    def test_to_dict_json_serializable(self) -> None:
        import json

        s = self._make_summary()
        d = s.to_dict()
        # Should not raise
        json.dumps(d)
