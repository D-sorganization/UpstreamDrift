"""Double and triple pendulum physics package.

Exports the core dynamics classes and parameter data-classes used to define
and simulate both double- and triple-pendulum systems.
"""

from .double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    JointTorques,
    LowerSegmentProperties,
    SegmentProperties,
)
from .triple_pendulum import (
    TripleJointTorques,
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
    TripleSegmentProperties,
)

__all__ = [
    "DoublePendulumDynamics",
    "DoublePendulumParameters",
    "DoublePendulumState",
    "JointTorques",
    "LowerSegmentProperties",
    "SegmentProperties",
    "TripleJointTorques",
    "TriplePendulumDynamics",
    "TriplePendulumParameters",
    "TriplePendulumState",
    "TripleSegmentProperties",
]
