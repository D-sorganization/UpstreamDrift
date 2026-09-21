"""NM-01 (#10616): typed learning-task contracts and dimension validation.

Acceptance cases from the issue:
- distinct forward / inverse / masked-trajectory-to-controls contracts
- typed task and dimension validation against the TB-00 model roster
- model-versus-backend identity must agree
- missing mask / time / geometry is rejected
- inverse labels require a selection objective or multimodal target
- no physical uniqueness claim for inverse dynamics
- checkpoint dimensions come from the roster, not a hard-coded 27x7 shape
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.shared.python.neural_motion.roster import (
    CheckpointDimensions,
    dimensions_for_model,
)
from src.shared.python.neural_motion.tasks import (
    ForwardDynamicsTaskSpec,
    InverseDynamicsTaskSpec,
    InverseLabelMode,
    LearningTaskKind,
    MaskedTrajectoryTaskSpec,
    ObservationMask,
    TaskConditioning,
)
from src.shared.python.tour_baselines.models import BackendType
from src.shared.python.tour_baselines.registry import get_golf_model, list_golf_models

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mask(*channels: str) -> ObservationMask:
    return ObservationMask(channel_ids=channels)


def _conditioning(
    *,
    q0_dim: int = 2,
    v0_dim: int = 2,
    horizon_s: float = 0.8,
    n_steps: int = 192,
) -> TaskConditioning:
    return TaskConditioning(
        geometry_id="fixture.geometry.default",
        q0_dim=q0_dim,
        v0_dim=v0_dim,
        horizon_s=horizon_s,
        n_steps=n_steps,
        contact_regime="fixed_pivot",
        constraint_set_id="none",
        observation_mask=_mask("club_tip_xyz", "club_butt_xyz"),
    )


def _conditioning_for(model_id: str) -> TaskConditioning:
    model = get_golf_model(model_id)
    return _conditioning(q0_dim=model.dof, v0_dim=model.independent_dof)


def test_three_learning_tasks_are_distinct() -> None:
    kinds = {member.value for member in LearningTaskKind}
    assert kinds == {
        "forward_dynamics",
        "inverse_dynamics",
        "masked_trajectory_to_controls",
    }


def test_forward_task_requires_controls_and_dt_semantics() -> None:
    model = get_golf_model("driven_double_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=2)
    task = ForwardDynamicsTaskSpec(
        model_id=model.model_id,
        backend=model.backend,
        dimensions=dims,
        conditioning=_conditioning_for(model.model_id),
        output_kind="acceleration",
        include_dt=True,
    )
    assert task.kind is LearningTaskKind.FORWARD_DYNAMICS
    assert task.dimensions.n_q == model.dof
    assert task.dimensions.n_u == 2


def test_inverse_requires_selection_or_multimodal_not_uniqueness() -> None:
    model = get_golf_model("driven_triple_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=3)
    conditioning = _conditioning_for(model.model_id)
    selected = InverseDynamicsTaskSpec(
        model_id=model.model_id,
        backend=model.backend,
        dimensions=dims,
        conditioning=conditioning,
        label_mode=InverseLabelMode.SELECTION_OBJECTIVE,
        selection_objective="min_control_effort_l2",
    )
    multimodal = InverseDynamicsTaskSpec(
        model_id=model.model_id,
        backend=model.backend,
        dimensions=dims,
        conditioning=conditioning,
        label_mode=InverseLabelMode.MULTIMODAL_TARGET,
        selection_objective=None,
    )
    assert selected.claims_physical_uniqueness is False
    assert multimodal.claims_physical_uniqueness is False


def test_inverse_rejects_physical_uniqueness_claim() -> None:
    model = get_golf_model("driven_double_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=2)
    with pytest.raises(ValueError, match="physical uniqueness"):
        InverseDynamicsTaskSpec(
            model_id=model.model_id,
            backend=model.backend,
            dimensions=dims,
            conditioning=_conditioning_for(model.model_id),
            label_mode=InverseLabelMode.SELECTION_OBJECTIVE,
            selection_objective="min_control_effort_l2",
            claims_physical_uniqueness=True,
        )


def test_inverse_selection_mode_requires_objective() -> None:
    model = get_golf_model("driven_double_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=2)
    with pytest.raises(ValueError, match="selection_objective"):
        InverseDynamicsTaskSpec(
            model_id=model.model_id,
            backend=model.backend,
            dimensions=dims,
            conditioning=_conditioning_for(model.model_id),
            label_mode=InverseLabelMode.SELECTION_OBJECTIVE,
            selection_objective=None,
        )


def test_masked_trajectory_task_carries_mask_and_initial_state() -> None:
    model = get_golf_model("constrained_upper_body_golfer")
    dims = dimensions_for_model(model.model_id, n_u=5)
    task = MaskedTrajectoryTaskSpec(
        model_id=model.model_id,
        backend=model.backend,
        dimensions=dims,
        conditioning=TaskConditioning(
            geometry_id="fixture.geometry.upper_body",
            q0_dim=model.dof,
            v0_dim=model.independent_dof,
            horizon_s=1.0,
            n_steps=240,
            contact_regime="stance_contact",
            constraint_set_id="upper_body_closure",
            observation_mask=_mask("club_tip_xyz"),
        ),
        proposal_kind="control_coefficients",
    )
    assert task.kind is LearningTaskKind.MASKED_TRAJECTORY_TO_CONTROLS
    assert "club_tip_xyz" in task.conditioning.observation_mask.channel_ids


@pytest.mark.parametrize(
    "field_name,bad_value,match",
    [
        ("geometry_id", "", "geometry"),
        ("horizon_s", 0.0, "horizon"),
        ("n_steps", 0, "n_steps"),
        ("observation_mask", "empty", "mask"),
    ],
)
def test_missing_mask_time_or_geometry_is_rejected(
    field_name: str, bad_value: object, match: str
) -> None:
    base: dict[str, object] = {
        "geometry_id": "g",
        "q0_dim": 2,
        "v0_dim": 2,
        "horizon_s": 0.5,
        "n_steps": 10,
        "contact_regime": "fixed_pivot",
        "constraint_set_id": "none",
        "observation_mask": _mask("club_tip_xyz"),
    }
    if field_name == "observation_mask" and bad_value == "empty":
        # Constructing ObservationMask(()) already raises; also reject via
        # TaskConditioning if a bypassed empty mask were supplied.
        with pytest.raises(ValueError, match=match):
            ObservationMask(channel_ids=())
        return
    base[field_name] = bad_value
    with pytest.raises(ValueError, match=match):
        TaskConditioning(**base)  # type: ignore[arg-type]


def test_model_versus_backend_identity_mismatch_rejected() -> None:
    model = get_golf_model("driven_double_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=2)
    with pytest.raises(ValueError, match="backend"):
        ForwardDynamicsTaskSpec(
            model_id=model.model_id,
            backend=BackendType.MUJOCO,  # wrong for this roster entry
            dimensions=dims,
            conditioning=_conditioning_for(model.model_id),
            output_kind="next_state",
            include_dt=True,
        )


def test_checkpoint_dimensions_come_from_roster_not_hardcoded_27x7() -> None:
    """No hard-coded 27-DoF assumption; shapes follow TB-00 identities."""
    roster_dofs = {m.model_id: m.dof for m in list_golf_models()}
    assert 27 not in roster_dofs.values() or any(
        dof != 27 for dof in roster_dofs.values()
    )

    for model in list_golf_models():
        dims = dimensions_for_model(model.model_id, n_u=max(1, model.independent_dof))
        assert dims.n_q == model.dof
        assert dims.n_v == model.independent_dof
        assert dims.model_id == model.model_id
        # Explicitly reject a forged 27x7 shape for a non-27 model.
        if model.dof != 27:
            with pytest.raises(ValueError, match="n_q"):
                CheckpointDimensions(
                    model_id=model.model_id,
                    n_q=27,
                    n_v=27,
                    n_u=7,
                    n_contact=0,
                )


def test_dimension_mismatch_with_task_conditioning_rejected() -> None:
    model = get_golf_model("driven_triple_pendulum")
    dims = dimensions_for_model(model.model_id, n_u=3)
    with pytest.raises(ValueError, match="q0_dim"):
        ForwardDynamicsTaskSpec(
            model_id=model.model_id,
            backend=model.backend,
            dimensions=dims,
            conditioning=_conditioning(),  # q0_dim=2 but model dof=3
            output_kind="acceleration",
            include_dt=True,
        )


_OPT_SNIPPET = r"""
from src.shared.python.neural_motion.tasks import (
    InverseDynamicsTaskSpec,
    InverseLabelMode,
    ObservationMask,
    TaskConditioning,
)
from src.shared.python.neural_motion.roster import dimensions_for_model
from src.shared.python.tour_baselines.registry import get_golf_model

model = get_golf_model("driven_double_pendulum")
dims = dimensions_for_model(model.model_id, n_u=2)
try:
    InverseDynamicsTaskSpec(
        model_id=model.model_id,
        backend=model.backend,
        dimensions=dims,
        conditioning=TaskConditioning(
            geometry_id="g",
            q0_dim=2,
            v0_dim=2,
            horizon_s=0.5,
            n_steps=8,
            contact_regime="fixed_pivot",
            constraint_set_id="none",
            observation_mask=ObservationMask(channel_ids=("club_tip_xyz",)),
        ),
        label_mode=InverseLabelMode.SELECTION_OBJECTIVE,
        selection_objective="min_control_effort_l2",
        claims_physical_uniqueness=True,
    )
except ValueError as exc:
    assert "physical uniqueness" in str(exc).lower()
    print("OK")
else:
    raise SystemExit("expected ValueError")
"""


def test_inverse_uniqueness_rejection_survives_python_optimize() -> None:
    proc = subprocess.run(
        [sys.executable, "-O", "-c", _OPT_SNIPPET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
