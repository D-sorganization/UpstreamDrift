from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ConstraintType(Enum):
    NONE = "none"
    FIXED_IN_SPACE = "fixed_in_space"
    RELATIVE_TO_BODY = "relative_to_body"


@dataclass
class BodyConstraint:
    body_id: int
    constraint_type: ConstraintType
    target_position: np.ndarray | None = None
    target_orientation: np.ndarray | None = None
    reference_body_id: int | None = None
    relative_position: np.ndarray | None = None
    relative_orientation: np.ndarray | None = None
    active: bool = True


class ConstraintManagerMixin:
    constraints: dict[int, BodyConstraint]

    def add_constraint(
        self,
        body_id: int,
        constraint_type: ConstraintType,
        reference_body_id: int | None = None,
    ) -> None:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        if constraint_type == ConstraintType.FIXED_IN_SPACE:
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=constraint_type,
                target_position=self.data.xpos[body_id].copy(),
                target_orientation=self.data.xquat[body_id].copy(),
            )
        elif constraint_type == ConstraintType.RELATIVE_TO_BODY:
            if reference_body_id is None:
                msg = "Reference body required for relative constraint"
                raise ValueError(msg)

            rel_pos = self.data.xpos[body_id] - self.data.xpos[reference_body_id]
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=constraint_type,
                reference_body_id=reference_body_id,
                relative_position=rel_pos.copy(),
                relative_orientation=self.data.xquat[body_id].copy(),
            )
        else:
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=ConstraintType.NONE,
            )

        self.constraints[body_id] = constraint

    def remove_constraint(self, body_id: int) -> None:
        if body_id in self.constraints:
            del self.constraints[body_id]

    def toggle_constraint(self, body_id: int) -> bool:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        if body_id in self.constraints:
            self.constraints[body_id].active = not self.constraints[body_id].active
            return self.constraints[body_id].active
        return False

    def clear_constraints(self) -> None:
        self.constraints.clear()

    def enforce_constraints(self) -> None:
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        for body_id, constraint in self.constraints.items():
            if not constraint.active:
                continue

            if constraint.constraint_type == ConstraintType.FIXED_IN_SPACE:
                if constraint.target_position is not None:
                    self._solve_ik_for_body(
                        body_id,
                        constraint.target_position,
                        maintain_orientation=True,
                    )

            elif (
                constraint.constraint_type == ConstraintType.RELATIVE_TO_BODY
            ):  # noqa: SIM102
                if (
                    constraint.reference_body_id is not None
                    and constraint.relative_position is not None
                ):
                    ref_pos = self.data.xpos[constraint.reference_body_id]
                    target_pos = ref_pos + constraint.relative_position
                    self._solve_ik_for_body(
                        body_id,
                        target_pos,
                        maintain_orientation=False,
                    )

    def get_constrained_bodies(self) -> list[int]:
        return [
            body_id
            for body_id, constraint in self.constraints.items()
            if constraint.active
        ]
