"""Initialization handles native models whose rest world is not camera Y-up."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.motion_capture.reference.fit_seed import reference_initial_state
from src.motion_capture.reconstruct.model import ArticulatedModel, Joint, ModelSpec

pytestmark = pytest.mark.unit


def test_root_registration_recovers_rigid_motion_with_pre_rotation() -> None:
    model = ArticulatedModel(
        ModelSpec(
            "tripod",
            (
                Joint("root", None, axes="yxz", pre_rotvec=(0.1, 0.2, 0.3)),
                Joint("a", "root", (1, 0, 0), "length", axes=""),
                Joint("b", "root", (0, 0, 1), "length", axes=""),
            ),
            {"length": 1.0},
        )
    )
    rest = model.landmarks(np.zeros((2, model.n_dof)))
    observed = rest @ Rotation.from_euler("z", 0.7).as_matrix().T + [1, 2, 3]
    state = reference_initial_state(model, observed)
    np.testing.assert_allclose(model.landmarks(state), observed, atol=1e-10)


def test_underconstrained_seed_stays_finite() -> None:
    model = ArticulatedModel(ModelSpec("root", (Joint("root", None),)))
    observed = np.array([[[1.0, 2.0, 3.0]], [[np.nan] * 3]])
    state = reference_initial_state(model, observed)
    assert np.isfinite(state).all()
