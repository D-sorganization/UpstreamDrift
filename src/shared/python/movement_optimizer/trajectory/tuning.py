# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tuning parameters for the trajectory optimiser.

All weights below are **tuning parameters** chosen empirically to balance
competing objectives (torque minimisation, smoothness, balance, endpoint
accuracy).  They are not derived from first principles and may need
adjustment for new exercises or unusual body proportions.

Rationale for each weight:

- JERK_WEIGHT: Small (0.05) so jerk smoothing acts as a tie-breaker
  rather than dominating the torque objective.  Increasing it produces
  smoother but less torque-efficient trajectories.

- TORQUE_RATE_WEIGHT: Moderate (0.15) to penalise sudden torque spikes
  that would be physiologically unrealistic.  Too high damps out fast
  contractions needed for explosive lifts.

- ENDPOINT_WEIGHT: High (10.0) to strongly enforce near-zero velocity
  and acceleration at the start and end of the movement, preventing
  unrealistic "flying start" solutions.

- BALANCE_BARRIER_WEIGHT: Very high (8000) as a steep penalty to keep
  COM within the inner BOS.  Acts as a soft backup to the hard SLSQP
  constraint; prevents the optimizer from exploring infeasible regions.

- BALANCE_CENTER_WEIGHT: Low (12.0) soft preference for centering the
  COM over mid-foot.  Does not override the hard constraint but nudges
  the solution toward a more stable posture.
"""

# -- Objective function weights (tuning parameters, see module docstring) --
DEFAULT_JERK_WEIGHT: float = 0.05
DEFAULT_TORQUE_RATE_WEIGHT: float = 0.15
DEFAULT_ENDPOINT_WEIGHT: float = 10.0

# Endpoint damping shape parameters (used by the smoothness cost):
#  - ACCEL ratio: weight of the endpoint acceleration term relative to the
#    endpoint velocity term.
#  - SAMPLE fraction / MIN samples: how many evaluation samples at each end
#    get endpoint damping: n_damp = max(MIN, int(n_eval * FRACTION)).
ENDPOINT_ACCEL_WEIGHT_RATIO: float = 0.1
ENDPOINT_DAMP_SAMPLE_FRACTION: float = 0.125
ENDPOINT_DAMP_MIN_SAMPLES: int = 2

# Balance constraint weights
BALANCE_BARRIER_WEIGHT: float = 8000.0
BALANCE_CENTER_WEIGHT: float = 12.0

# Stall detection
STALL_WINDOW: int = 80
STALL_THRESHOLD: float = 1e-4

# Multi-start
DEFAULT_N_STARTS: int = 6
MAX_ITER_PER_START: int = 500
PERTURBATION_SCALE: float = 0.08
