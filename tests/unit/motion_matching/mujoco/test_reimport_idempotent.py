"""Pin: re-importing the MuJoCo motion-matching package must be a no-op.

Regression for issue #4712: importing the package twice (e.g. once via
the package, once via ``provider.py``) raised
``ValueError: engine_name 'mujoco' is already registered`` because
``importlib.reload`` rebuilt the provider class, defeating the
``type(existing) is type(provider)`` shortcut in ``register_provider``.
"""

from __future__ import annotations

import importlib

import pytest


def test_double_import_does_not_raise() -> None:
    """Importing the MuJoCo motion-matching package twice is a no-op."""
    pytest.importorskip("mujoco")
    import src.engines.physics_engines.mujoco.python.motion_matching as m

    original_id = id(m)
    importlib.reload(m)
    importlib.reload(m)
    assert id(m) == original_id, "Module identity should be preserved across reloads"


def test_provider_module_reload_does_not_raise() -> None:
    """Pin: reloading ``provider.py`` directly is also a no-op.

    The historical failure mode (#4712) reloaded ``provider.py`` itself,
    which rebuilt :class:`MujocoFitSwingProvider` as a *new* class object
    so the ``type(existing) is type(provider)`` shortcut in
    ``register_provider`` was bypassed.
    """
    pytest.importorskip("mujoco")
    import src.engines.physics_engines.mujoco.python.motion_matching  # noqa: F401
    import src.engines.physics_engines.mujoco.python.motion_matching.provider as p

    original_id = id(p)
    importlib.reload(p)
    importlib.reload(p)
    assert (
        id(p) == original_id
    ), "Provider module identity should be preserved across reloads"
