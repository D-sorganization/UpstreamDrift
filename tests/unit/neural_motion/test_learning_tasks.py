"""NM-01 (#10616): typed learning-task contracts and dimension validation."""

from __future__ import annotations

import math

import pytest

from src.shared.python.neural_motion.tasks import (
    ConditioningSpec,
    ForwardDynamicsTask,
    InverseDynamicsTask,
    InverseLabelPolicy,
    LearningTaskKind,
    MaskedTrajectoryTask,
    TaskDimensions,
    build_default_learning_tasks,
)
from src.shared.python.tour_baselines.registry import get_golf_model

pytestmark = pytest.mark.unit


def _dims_for(model_id: str) -> TaskDimensions:
    identity = get_golf_model(model_id)
    q = identity.dof
    return TaskDimensions(
        model_id=identity.model_id,
        backend=identity.backend.value,
        q_dim=q,
        v_dim=q,
        a_dim=q,
        u_dim=identity.independent_dof,
        reaction_dim=identity.constraint_count,
    )


def _conditioning(*, mask: tuple[bool, ...] | None = (True, False)) -> ConditioningSpec:
    return ConditioningSpec(
        geometry_id="geom.pilot.v1",
        q0=(0.0, 0.1),
        v0=(0.0, 0.0),
        horizon_s=0.85,
        time_step_s=0.01,
        constraint_profile="holonomic.v1",
        contact_profile="none.v1",
        observation_mask=mask if mask is not None else (),
    )


def test_default_tasks_cover_three_distinct_kinds() -> None:
    tasks = build_default_learning_tasks(model_id="driven_double_pendulum")
    kinds = {task.kind for task in tasks}
    assert kinds == {
        LearningTaskKind.FORWARD_DYNAMICS,
        LearningTaskKind.INVERSE_DYNAMICS,
        LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS,
    }


def test_task_dimensions_match_registered_model_not_hardcoded_27x7() -> None:
    dims = _dims_for("driven_double_pendulum")
    assert dims.q_dim == 2
    assert dims.u_dim == 2
    assert (dims.q_dim, dims.u_dim) != (27, 7)
    full = _dims_for("full_body_simscape")
    assert full.q_dim == get_golf_model("full_body_simscape").dof
    assert full.model_id == "full_body_simscape"


def test_task_dimensions_reject_model_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="backend"):
        TaskDimensions(
            model_id="driven_double_pendulum",
            backend="mujoco",
            q_dim=2,
            v_dim=2,
            a_dim=2,
            u_dim=2,
            reaction_dim=0,
        )


def test_conditioning_rejects_missing_geometry_time_or_mask() -> None:
    with pytest.raises(ValueError, match="geometry"):
        ConditioningSpec(
            geometry_id="",
            q0=(0.0,),
            v0=(0.0,),
            horizon_s=0.5,
            time_step_s=0.01,
            constraint_profile="none",
            contact_profile="none",
            observation_mask=(True,),
        )
    with pytest.raises(ValueError, match="horizon|time"):
        ConditioningSpec(
            geometry_id="g",
            q0=(0.0,),
            v0=(0.0,),
            horizon_s=0.0,
            time_step_s=0.01,
            constraint_profile="none",
            contact_profile="none",
            observation_mask=(True,),
        )
    with pytest.raises(ValueError, match="observation_mask"):
        ConditioningSpec(
            geometry_id="g",
            q0=(0.0,),
            v0=(0.0,),
            horizon_s=0.5,
            time_step_s=0.01,
            constraint_profile="none",
            contact_profile="none",
            observation_mask=(),
        )


def test_conditioning_rejects_nonfinite_initial_state() -> None:
    with pytest.raises(ValueError, match="finite"):
        ConditioningSpec(
            geometry_id="g",
            q0=(math.nan,),
            v0=(0.0,),
            horizon_s=0.5,
            time_step_s=0.01,
            constraint_profile="none",
            contact_profile="none",
            observation_mask=(True,),
        )


def test_forward_task_requires_matching_control_dim() -> None:
    dims = _dims_for("driven_double_pendulum")
    task = ForwardDynamicsTask(
        task_id="fwd.driven_double_pendulum",
        dimensions=dims,
        conditioning=_conditioning(mask=(True, True)),
        output_kind="acceleration",
    )
    assert task.kind is LearningTaskKind.FORWARD_DYNAMICS
    with pytest.raises(ValueError, match="u_dim|control"):
        ForwardDynamicsTask(
            task_id="fwd.bad",
            dimensions=TaskDimensions(
                model_id=dims.model_id,
                backend=dims.backend,
                q_dim=dims.q_dim,
                v_dim=dims.v_dim,
                a_dim=dims.a_dim,
                u_dim=0,
                reaction_dim=dims.reaction_dim,
            ),
            conditioning=_conditioning(mask=(True, True)),
            output_kind="acceleration",
        )


def test_inverse_task_requires_declared_nonunique_label_policy() -> None:
    dims = _dims_for("driven_double_pendulum")
    with pytest.raises(ValueError, match="label_policy|selection|multimodal"):
        InverseDynamicsTask(
            task_id="inv.bad",
            dimensions=dims,
            conditioning=_conditioning(mask=(True, True)),
            label_policy=None,  # type: ignore[arg-type]
            selection_objective=None,
            claims_physical_uniqueness=True,
        )
    task = InverseDynamicsTask(
        task_id="inv.driven_double_pendulum",
        dimensions=dims,
        conditioning=_conditioning(mask=(True, True)),
        label_policy=InverseLabelPolicy.SELECTION_OBJECTIVE,
        selection_objective="min_effort_with_contact_regularization",
        claims_physical_uniqueness=False,
    )
    assert task.kind is LearningTaskKind.INVERSE_DYNAMICS
    with pytest.raises(ValueError, match="uniqueness"):
        InverseDynamicsTask(
            task_id="inv.unique_claim",
            dimensions=dims,
            conditioning=_conditioning(mask=(True, True)),
            label_policy=InverseLabelPolicy.MULTIMODAL_DISTRIBUTION,
            selection_objective=None,
            claims_physical_uniqueness=True,
        )


def test_masked_trajectory_task_rejects_absent_mask_or_time() -> None:
    dims = _dims_for("driven_double_pendulum")
    with pytest.raises(ValueError, match="observation_mask"):
        MaskedTrajectoryTask(
            task_id="mask.bad",
            dimensions=dims,
            conditioning=_conditioning(mask=None),
            proposal_space="continuous_coefficients",
        )
    task = MaskedTrajectoryTask(
        task_id="mask.driven_double_pendulum",
        dimensions=dims,
        conditioning=_conditioning(mask=(True, False)),
        proposal_space="continuous_coefficients",
    )
    assert task.kind is LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS
