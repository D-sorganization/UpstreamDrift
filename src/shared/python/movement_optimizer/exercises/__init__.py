# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Exercise configuration factories.

Each ``make_*_config`` factory returns a tuple
``(dynamics, q_start, q_end, q_bounds, q_via)`` (or via-points list, for
gait and sit-to-stand) that can be fed directly into
:class:`movement_optimizer.trajectory.TrajectoryOptimizer`.

Public exports:
    make_clean_config: Olympic clean (floor to front rack).
    make_snatch_config: Olympic snatch (floor to overhead).
    make_jerk_config: Jerk drive (front rack to overhead lockout).
    make_gait_config: One full gait cycle (heel strike to heel strike).
    make_sit_to_stand_config: Seated to standing transition.
    GaitAnalyzer: Spatiotemporal and symmetry analysis for gait.

Squat, deadlift, and bench press factories live in
``movement_optimizer.models`` for historical reasons.
"""

from .clean import make_clean_config as make_clean_config
from .gait import GaitAnalyzer as GaitAnalyzer
from .gait import make_gait_config as make_gait_config
from .jerk import make_jerk_config as make_jerk_config
from .sit_to_stand import make_sit_to_stand_config as make_sit_to_stand_config
from .snatch import make_snatch_config as make_snatch_config
