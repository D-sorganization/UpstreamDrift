"""Articulated golfer model: typed joints, forward kinematics, continuous fit.

See epic #9709. :mod:`.kinematics` is the engine, :mod:`.fit` the
whole-trajectory solver, :mod:`.golfer` the model derived from the MATLAB
3-D golf model (scapula segments included).
"""

from .fit import FitOptions, ModelFit, fit_to_dict, fit_trajectory
from .kinematics import ArticulatedModel, Joint, ModelSpec, wrap_angles

__all__ = [
    "ArticulatedModel",
    "FitOptions",
    "Joint",
    "ModelFit",
    "ModelSpec",
    "fit_to_dict",
    "fit_trajectory",
    "wrap_angles",
]
