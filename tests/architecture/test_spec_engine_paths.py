import importlib

ENGINES = ["mujoco", "drake", "pinocchio", "opensim", "myosuite"]


def test_each_engine_has_a_resolvable_public_adapter():
    for name in ENGINES:
        module = importlib.import_module(f"src.engines.physics_engines.{name}")
        # The adapter must expose a get_engine() or similar entrypoint.
        assert hasattr(module, "get_engine") or hasattr(module, "Engine")
