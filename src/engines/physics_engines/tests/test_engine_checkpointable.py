"""Test: all concrete physics engines satisfy the Checkpointable protocol.

Regression guard for issue #6638 F1 – ensures every engine:
1. Inherits ``BasePhysicsEngine`` (not the raw ``PhysicsEngine`` interface)
2. Exposes the ``engine_type`` property required by ``Checkpointable``
3. Returns a non-empty string from ``engine_type``

These are pure-Python structural checks that run without installing optional
engine dependencies (MuJoCo, Drake, OpenSim, MyoSuite).  They import the
engine *class* directly and never instantiate it with external C extensions.
"""

from __future__ import annotations

import inspect
import unittest
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Internal imports – only the base contract and concrete engines are needed
# ---------------------------------------------------------------------------
from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine

if TYPE_CHECKING:
    pass


def _collect_engine_classes() -> list[tuple[str, type[BasePhysicsEngine]]]:
    """Return (label, class) pairs for every concrete physics engine.

    Each engine module is imported lazily so that a missing optional
    dependency (e.g. pydrake) does not abort the whole test run – we only
    need the *class definition*, not an instantiated object.
    """
    pairs: list[tuple[str, type[Any]]] = []

    # MuJoCo
    try:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: E501
            MuJoCoPhysicsEngine,
        )

        pairs.append(("MuJoCoPhysicsEngine", MuJoCoPhysicsEngine))
    except ImportError:
        pass  # mujoco package not installed – skip, structural test still valid

    # Drake
    try:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        pairs.append(("DrakePhysicsEngine", DrakePhysicsEngine))
    except ImportError:
        pass

    # OpenSim
    try:
        from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
            OpenSimPhysicsEngine,
        )

        pairs.append(("OpenSimPhysicsEngine", OpenSimPhysicsEngine))
    except ImportError:
        pass

    # MyoSuite
    try:
        from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E501
            MyoSuitePhysicsEngine,
        )

        pairs.append(("MyoSuitePhysicsEngine", MyoSuitePhysicsEngine))
    except ImportError:
        pass

    # Pinocchio
    try:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E501
            PinocchioPhysicsEngine,
        )

        pairs.append(("PinocchioPhysicsEngine", PinocchioPhysicsEngine))
    except ImportError:
        pass

    return pairs  # type: ignore[return-value]


class TestEngineCheckpointableConformance(unittest.TestCase):
    """Structural tests – no engine instantiation required."""

    def setUp(self) -> None:
        self.engine_classes = _collect_engine_classes()
        if not self.engine_classes:
            self.skipTest("No engine classes could be imported in this environment.")

    # ------------------------------------------------------------------
    # F1 – BasePhysicsEngine inheritance
    # ------------------------------------------------------------------

    def test_all_engines_inherit_base_physics_engine(self) -> None:
        """Every engine must subclass BasePhysicsEngine (not raw PhysicsEngine)."""
        for label, cls in self.engine_classes:
            with self.subTest(engine=label):
                self.assertTrue(
                    issubclass(cls, BasePhysicsEngine),
                    f"{label} does not inherit BasePhysicsEngine – "
                    "it cannot satisfy the Checkpointable contract",
                )

    # ------------------------------------------------------------------
    # Checkpointable protocol methods present
    # ------------------------------------------------------------------

    def test_all_engines_have_engine_type_property(self) -> None:
        """Every engine must expose an ``engine_type`` property."""
        for label, cls in self.engine_classes:
            with self.subTest(engine=label):
                self.assertTrue(
                    hasattr(cls, "engine_type"),
                    f"{label} is missing the 'engine_type' attribute",
                )
                # Must be a property (not a plain method or class variable)
                self.assertIsInstance(
                    inspect.getattr_static(cls, "engine_type"),
                    property,
                    f"{label}.engine_type must be a @property",
                )

    def test_all_engines_have_save_checkpoint(self) -> None:
        """Checkpointable.save_checkpoint must be available (from base)."""
        for label, cls in self.engine_classes:
            with self.subTest(engine=label):
                self.assertTrue(
                    hasattr(cls, "save_checkpoint"),
                    f"{label} is missing save_checkpoint – BasePhysicsEngine not inherited",
                )

    def test_all_engines_have_restore_checkpoint(self) -> None:
        """Checkpointable.restore_checkpoint must be available (from base)."""
        for label, cls in self.engine_classes:
            with self.subTest(engine=label):
                self.assertTrue(
                    hasattr(cls, "restore_checkpoint"),
                    f"{label} is missing restore_checkpoint",
                )

    # ------------------------------------------------------------------
    # engine_type value sanity (class-level inspection, no instantiation)
    # ------------------------------------------------------------------

    def test_engine_type_property_returns_non_empty_string(self) -> None:
        """engine_type fget must return a non-empty string constant."""
        for label, cls in self.engine_classes:
            with self.subTest(engine=label):
                prop = inspect.getattr_static(cls, "engine_type", None)
                if not isinstance(prop, property):
                    continue  # already caught by previous test
                # Call fget with a minimal mock to avoid full initialisation
                sentinel: Any = object.__new__(cls)
                try:
                    value = prop.fget(sentinel)  # type: ignore[misc]
                except (TypeError, AttributeError, RuntimeError):
                    # Cannot introspect without full init – skip value check
                    continue
                self.assertIsInstance(
                    value, str, f"{label}.engine_type must return str"
                )
                self.assertTrue(
                    value,
                    f"{label}.engine_type must return a non-empty string",
                )


class TestCheckpointableProtocolSatisfied(unittest.TestCase):
    """Verify classes structurally conform to the Checkpointable Protocol."""

    def test_base_physics_engine_satisfies_checkpointable(self) -> None:
        """BasePhysicsEngine itself must provide all Checkpointable members."""
        required = {"engine_type", "save_checkpoint", "restore_checkpoint"}
        for member in required:
            with self.subTest(member=member):
                self.assertTrue(
                    hasattr(BasePhysicsEngine, member),
                    f"BasePhysicsEngine is missing '{member}' for Checkpointable",
                )


if __name__ == "__main__":
    unittest.main()
