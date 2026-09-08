"""Fittable models by name: the scapula golfer (default) and the pendulums (#9730).

Every entry is a :class:`ModelSpec` with the :class:`LandmarkMap` that ties
its landmarks to the reconstruct joints and the lengths the fit may learn.
The same continuous fit runs on each, so a double pendulum, a triple
pendulum and the full golfer are constrained, learned and compared on the
same data (#9731).

Pendulum family: the pivot is the golfer's upper trunk (the hub, base of the
neck) with three rotational DOFs that pick the swing plane; the segments
below are hinges about the plane normal (z of the pivot frame), which keeps
the pendulum planar in whatever plane the root chooses. The double pendulum
is pivot -> arm -> hands (the club is not observed by the detectors); the
triple pendulum adds the elbow. Hands are the mean of both wrists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.shared.python.core.contracts import require

from .golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from .kinematics import Joint, ModelSpec
from .session import LandmarkMap

DEFAULT_MODEL = "golfer"
HINGE_RANGE = ((-3.0, 3.0),)

# The pivot's x and y rotations choose the swing plane; its z rotation is the
# first link's angle in that plane. Each further hinge rotates the link below
# it about the same normal, so the chain stays planar with no redundant DOF.
DOUBLE_PENDULUM = ModelSpec(
    name="double-pendulum/1.0",
    joints=(
        Joint("pivot", None, axes="xyz"),
        Joint("hands", "pivot", (0.0, -1.0, 0.0), "arm", ""),
    ),
    lengths_m={"arm": 0.62},
)

TRIPLE_PENDULUM = ModelSpec(
    name="triple-pendulum/1.0",
    joints=(
        Joint("pivot", None, axes="xyz"),
        Joint("elbows", "pivot", (0.0, -1.0, 0.0), "upper_arm", "z", HINGE_RANGE),
        Joint("hands", "elbows", (0.0, -1.0, 0.0), "forearm", ""),
    ),
    lengths_m={"upper_arm": 0.32, "forearm": 0.30},
)

PENDULUM_MAP_DOUBLE = LandmarkMap(
    to_reconstruct={
        "pivot": ("left_shoulder", "right_shoulder"),
        "hands": ("left_wrist", "right_wrist"),
    }
)
PENDULUM_MAP_TRIPLE = LandmarkMap(
    to_reconstruct={
        "pivot": ("left_shoulder", "right_shoulder"),
        "elbows": ("left_elbow", "right_elbow"),
        "hands": ("left_wrist", "right_wrist"),
    }
)


@dataclass(frozen=True)
class RegisteredModel:
    """A fittable model: spec, landmark map, lengths the fit may learn, notes."""

    name: str
    spec: ModelSpec
    landmark_map: LandmarkMap
    learnable_lengths: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        unknown = [n for n in self.learnable_lengths if n not in self.spec.lengths_m]
        require(not unknown, "learnable lengths must exist in the spec", unknown)


MODELS: dict[str, RegisteredModel] = {
    "golfer": RegisteredModel(
        "golfer",
        GOLFER_SPEC,
        GOLFER_LANDMARK_MAP,
        ("hub_to_shoulder", "upper_torso", "lower_torso", "hip_half", "head"),
        "Scapula-capable golfer after the MATLAB 3-D model (default).",
    ),
    "double_pendulum": RegisteredModel(
        "double_pendulum",
        DOUBLE_PENDULUM,
        PENDULUM_MAP_DOUBLE,
        ("arm",),
        "Pivot at the shoulders, one rigid arm to the hands, planar hinge.",
    ),
    "triple_pendulum": RegisteredModel(
        "triple_pendulum",
        TRIPLE_PENDULUM,
        PENDULUM_MAP_TRIPLE,
        ("upper_arm", "forearm"),
        "Pivot at the shoulders, upper arm and forearm hinges to the hands.",
    ),
}


def get_model(name: str = DEFAULT_MODEL) -> RegisteredModel:
    """Precondition: ``name`` is registered (see :func:`model_names`)."""
    require(name in MODELS, "unknown model; see model_names()", name)
    return MODELS[name]


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def register_model(model: RegisteredModel) -> None:
    """Add a model (tests, plug-ins). Precondition: the name is free."""
    require(model.name not in MODELS, "model name already registered", model.name)
    MODELS[model.name] = model


def descriptions() -> Mapping[str, str]:
    return {name: m.description for name, m in MODELS.items()}
