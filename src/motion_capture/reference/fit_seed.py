"""Root-only rigid initialization; joint motion is still solved by the fitter."""

import numpy as np

from src.motion_capture.reconstruct.model import ArticulatedModel
from src.motion_capture.reconstruct.model.kinematics import decompose_primitives
from src.motion_capture.reconstruct.model.session import initial_state
from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation


def reference_initial_state(
    model: ArticulatedModel, observed: np.ndarray
) -> np.ndarray:
    """Seed root orientation from finite correspondences without changing data.

    Postcondition: finite (T, n_dof) states. Rank-deficient frames use the
    existing translation-only seed; internal joints retain their rest values.
    """
    if observed.ndim != 3 or observed.shape[1:] != (len(model.landmark_names), 3):
        raise ValueError("Observed landmarks must match the model")
    state = initial_state(model, observed)
    root = model.joints[0]
    if len(root.axes) != 3:
        return state
    rest = model.landmarks(np.zeros((1, model.n_dof)))[0]
    pre, _ = root.constant_frames()
    for frame, target in enumerate(observed):
        valid = np.isfinite(target).all(axis=1)
        if valid.sum() < 3:
            continue
        source_points, target_points = rest[valid], target[valid]
        source_center, target_center = (
            source_points.mean(axis=0),
            target_points.mean(axis=0),
        )
        source_zero, target_zero = (
            source_points - source_center,
            target_points - target_center,
        )
        if (
            min(np.linalg.matrix_rank(source_zero), np.linalg.matrix_rank(target_zero))
            < 2
        ):
            continue
        rotation = kabsch_rotation(source_zero, target_zero)
        angles, _ = decompose_primitives((pre.T @ rotation @ pre)[None], root.axes)
        state[frame, model.dof_slice(root.name)] = angles[0]
        state[frame, :3] = target_center - rotation @ source_center
    # Equivalent Euler branches should start near their temporal neighbours.
    state[:, 3:] = np.unwrap(state[:, 3:], axis=0)
    return state
