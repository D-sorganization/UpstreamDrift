import torch
import numpy as np

from src.engines.physics_engines.myosuite.python.motion_matching.inverse_surrogate import (
    InverseSurrogateConfig,
    MyoSuiteInverseSurrogate,
)

def test_myosuite_inverse_surrogate_forward() -> None:
    """Test that the MyoSuite inverse surrogate predicts within [0, 1] bounds."""
    n_joints = 22
    n_muscles = 290
    seq_len = 300
    batch_size = 2
    
    cfg = InverseSurrogateConfig(
        n_joints=n_joints,
        n_muscles=n_muscles,
        seq_len=seq_len,
        hidden_dim=32,
        n_layers=2
    )
    model = MyoSuiteInverseSurrogate(cfg)
    
    # Mock inputs
    joint_q = torch.randn(batch_size, seq_len, n_joints)
    joint_v = torch.randn(batch_size, seq_len, n_joints)
    
    # Forward pass
    activations = model(joint_q, joint_v)
    
    # Shape check
    assert activations.shape == (batch_size, seq_len, n_muscles)
    
    # Bounds check (sigmoid enforces (0, 1))
    assert torch.all(activations > 0.0)
    assert torch.all(activations < 1.0)
