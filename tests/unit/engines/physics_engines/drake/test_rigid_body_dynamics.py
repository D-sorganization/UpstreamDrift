"""Tests for rigid_body_dynamics empty init."""


def test_rigid_body_dynamics_init():
    import src.engines.physics_engines.drake.python.src.rigid_body_dynamics as rbd

    assert hasattr(rbd, "__all__")
    assert isinstance(rbd.__all__, list)
