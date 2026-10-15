"""Spec full-body model in the MyoSuite runtime, for same-input parity (#11612).

The MyoHub body in this package has an unrelated topology (11 of 44 spec
coordinates map), so torque-driven parity is only meaningful on the
specification model.  This module exports that model's MJCF and loads it
through MyoSuite's own simulation layer (``MujocoEnv``: ``MjSpec.from_file``
compiled to the ``mj_model`` / ``mj_data`` pair MyoSuite holds), then drives it
with the very same dynamics code as the MuJoCo adapter: mass matrix, bias and
Jacobians from the MyoSuite-held model, the shared contact law and the exact
weld KKT solve (``kkt_regularization = 0``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)

__all__ = ["MyoSuiteFullBodyModel", "build_parity_adapter", "load_myosuite_runtime"]


def load_myosuite_runtime(xml: str) -> Any:
    """Load MJCF text through the installed MyoSuite runtime (2.x or 3.x).

    Postcondition: ``env.mj_spec``, ``env.mj_model`` and ``env.mj_data`` are
    MyoSuite-held objects compiled from ``xml``.
    """
    from src.engines.physics_engines.myosuite.python.myosuite_runtime_compat import (
        load_runtime_from_xml,
    )

    return load_runtime_from_xml(xml)


class MyoSuiteFullBodyModel(NativeMujocoFullBodyModel):
    """Spec full-body adapter whose model and data are held by MyoSuite."""

    def __init__(self, model_bytes: bytes) -> None:
        self.myosuite_env: Any = None
        super().__init__(model_bytes)
        self.kkt_regularization = 0.0  # exact weld solve, as Drake and Pinocchio

    def _build_model(self, xml: str) -> tuple[Any, Any]:
        self.myosuite_env = load_myosuite_runtime(xml)
        return self.myosuite_env.mj_model, self.myosuite_env.mj_data

    def closure_residuals(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weld pose (6) and rate (6) residuals at ``(coordinates, rates)``.

        The rate residual is linear in ``rates``.  Same convention as the
        MuJoCo adapter, evaluated on the MyoSuite-held model and data.
        """
        self.accelerations(
            coordinates, rates, dict.fromkeys(self.coordinate_order, 0.0)
        )
        return self.closure_errors()


def build_parity_adapter(spec_bytes: bytes) -> MyoSuiteFullBodyModel:
    """Entry point used by ``same_input.plant.OPTIONAL_ENGINES['myosuite']``."""
    return MyoSuiteFullBodyModel(spec_bytes)
