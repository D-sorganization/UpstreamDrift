"""Registered constants for double-pendulum identifiability evidence."""

from scripts.research.proximal_distal_energy.local_linear_diagnostics import (
    RankTolerance,
)

SCHEMA_VERSION = "proximal-distal-double-pendulum-identifiability/v2"
RANK_TOLERANCE = RankTolerance(absolute=1e-8, relative=1e-7)
COUNTEREXAMPLE_NAMES = (
    "gravity_plane_tradeoff",
    "lower_mass_com_coupling",
    "upper_mass_com_tradeoff",
)
TORQUE_NOISE_LEVELS_NM = (0.1, 0.5, 1.0, 2.0)
REFERENCE_NOISE_SD_NM = 1.0
COEFFICIENT_UNIT_CONVERSION_FIXTURE = (1e3, 1e3, 1e3, 1e2, 1e2, 60.0, 60.0)
INFERENCE_BOUNDARY = (
    "Exact invariance families establish non-uniqueness only for the declared "
    "eleven-entry reduced equation-of-motion parameterization. Full rank of "
    "the seven-column regressor is specific to one noiseless registered "
    "synthetic record and tolerance. The registered Gaussian uncertainty "
    "screen is an oracle-kinematics lower bound under assumed iid torque noise, "
    "not practical identifiability. Neither result establishes model adequacy, "
    "participant parameters, human mechanism, or coaching guidance."
)
