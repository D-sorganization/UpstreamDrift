"""Test that engine adapters are resolvable as per SPEC.md."""

import importlib

ENGINES = ["mujoco", "drake", "pinocchio", "opensim", "myosuite"]


def test_each_engine_has_a_resolvable_public_adapter():
    """Each engine adapter must be importable and expose a get_engine() or Engine class."""
    for name in ENGINES:
        module = importlib.import_module(f"src.engines.physics_engines.{name}")
        # The adapter must expose a get_engine() or similar entrypoint.
        assert hasattr(module, "get_engine") or hasattr(module, "Engine"), (
            f"Engine module '{name}' must expose either 'get_engine' function or 'Engine' class"
        )
