"""Tests for surrogate training on 10k dataset (Issue #4075)."""

from __future__ import annotations

import pytest
import torch
from src.shared.python.motion_matching.dataset import (
    load_sweep_dataset,
    make_synthetic_sweep,
)
from src.shared.python.motion_matching.surrogate import (
    SurrogateConfig,
    SwingSurrogate,
    TrainConfig,
    train_surrogate,
)


class TestSurrogateArchitecture:
    """Tests for the FiLM-MLP surrogate architecture."""

    def test_config_creation(self) -> None:
        """Test SurrogateConfig creation and properties."""
        cfg = SurrogateConfig(n_joints=14, seq_len=300, hidden_dim=256, n_layers=3)
        assert cfg.n_joints == 14
        assert cfg.coeff_dim == 14 * 7
        assert cfg.seq_len == 300

    def test_model_creation(self) -> None:
        """Test SwingSurrogate instantiation."""
        cfg = SurrogateConfig(n_joints=14, seq_len=300, hidden_dim=256, n_layers=3)
        model = SwingSurrogate(cfg)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shapes."""
        cfg = SurrogateConfig(n_joints=14, seq_len=300, hidden_dim=256, n_layers=3)
        model = SwingSurrogate(cfg)
        batch_size = 16
        coeffs = torch.randn(batch_size, cfg.coeff_dim)

        output = model(coeffs)

        assert output.butt.shape == (batch_size, 300, 3)
        assert output.clubhead.shape == (batch_size, 300, 3)
        assert output.club_quat.shape == (batch_size, 300, 4)


class TestTraining:
    """Tests for the training loop."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a small synthetic dataset for testing."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            make_synthetic_sweep(tmpdir, n_trials=30, n_joints=14, n_timesteps=300)
            return load_sweep_dataset(tmpdir, lazy=False)

    def test_train_surrogate_basic(self, synthetic_dataset) -> None:
        """Test basic surrogate training on synthetic data."""
        cfg = TrainConfig(
            n_epochs=2,
            batch_size=8,
            lr=1e-3,
            device="cpu",
        )
        trained = train_surrogate(synthetic_dataset, cfg)

        assert trained.model is not None
        assert len(trained.curves.train_loss) == 2

    def test_main_synthetic(self, tmp_path) -> None:
        """Test the CLI entry point with synthetic data."""
        from src.shared.python.motion_matching.surrogate.train_10k import main
        import sys
        from unittest.mock import patch

        output_dir = tmp_path / "models"

        test_args = [
            "train_10k.py",
            "--use-synthetic",
            "--output-dir",
            str(output_dir),
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        assert (output_dir / "best_model.pt").exists()
        assert (output_dir / "config.pt").exists()
        assert (output_dir / "norm_stats.pt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
