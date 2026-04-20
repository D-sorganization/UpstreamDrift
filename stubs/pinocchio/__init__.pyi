"""Minimal type stubs for pinocchio robotics library.

These stubs cover the subset of the pinocchio API used in UpstreamDrift.
Generated for GH2034: reduce type:ignore comments systematically.

pinocchio is an optional dependency (upstream-drift[pinocchio]).
"""

from typing import Any

import numpy as np

# Core data structures
class Model:
    nq: int
    nv: int
    njoints: int
    nframes: int
    def __init__(self) -> None: ...

class Data:
    oMi: Any
    oMf: Any
    J: Any
    dJ: Any
    v: Any
    a: Any
    tau: Any
    def __init__(self, model: Model) -> None: ...

class GeometryModel:
    def __init__(self) -> None: ...

class GeometryData:
    def __init__(self, geom_model: GeometryModel) -> None: ...

class VisualModel(GeometryModel): ...
class CollisionModel(GeometryModel): ...

# SE3 / spatial algebra
class SE3:
    translation: np.ndarray
    rotation: np.ndarray
    @staticmethod
    def Identity() -> SE3: ...
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None: ...

class ReferenceFrame:
    LOCAL: int
    LOCAL_WORLD_ALIGNED: int
    WORLD: int

LOCAL_WORLD_ALIGNED: int

class GeometryType:
    VISUAL: int
    COLLISION: int

# Model builders
def buildModelFromUrdf(
    filename: str,
    root_joint: Any = ...,
) -> Model: ...
def buildModelsFromUrdf(
    filename: str,
    geom_types: Any = ...,
) -> tuple[Model, GeometryModel]: ...
def buildGeomFromUrdf(
    model: Model,
    filename: str,
    geom_type: Any,
) -> GeometryModel: ...

# Kinematics / dynamics
def neutral(model: Model) -> np.ndarray: ...
def integrate(model: Model, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
def forwardKinematics(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray | None = ...,
    a: np.ndarray | None = ...,
) -> None: ...
def updateFramePlacements(model: Model, data: Data) -> None: ...
def updateFramePlacement(model: Model, data: Data, frame_id: int) -> None: ...
def computeJointJacobians(
    model: Model,
    data: Data,
    q: np.ndarray,
) -> None: ...
def getFrameJacobian(
    model: Model,
    data: Data,
    frame_id: int,
    reference_frame: int,
) -> np.ndarray: ...
def getJointJacobian(
    model: Model,
    data: Data,
    joint_id: int,
    reference_frame: int,
) -> np.ndarray: ...
def getFrameVelocity(
    model: Model,
    data: Data,
    frame_id: int,
    reference_frame: int,
) -> Any: ...

# Inverse/forward dynamics
def rnea(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
) -> np.ndarray: ...
def aba(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray: ...
def crba(
    model: Model,
    data: Data,
    q: np.ndarray,
) -> np.ndarray: ...
def computeCoriolisMatrix(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray,
) -> np.ndarray: ...
def nle(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray,
) -> np.ndarray: ...
def computeGeneralizedGravity(
    model: Model,
    data: Data,
    q: np.ndarray,
) -> np.ndarray: ...
def computeKineticEnergy(
    model: Model,
    data: Data,
    q: np.ndarray,
    v: np.ndarray,
) -> float: ...
def computePotentialEnergy(
    model: Model,
    data: Data,
    q: np.ndarray,
) -> float: ...
