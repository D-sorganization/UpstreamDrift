# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Anthropometric and equipment constants.

Sources:
    Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
    Segment mass and length fractions for a standard adult.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# ------------------------------------------------------------------
# Segment mass fractions  (fraction of total body mass)
# Bilateral segments are lumped into a single 2-D sagittal-plane value.
#
# Note: "trunk_head" includes head and neck mass -- there is no
# separate head/neck segment in the 2-D sagittal-plane model.
# The five fractions must sum to exactly 1.0.
# ------------------------------------------------------------------
MASS_FRAC: dict[str, float] = {
    "feet": 0.028,
    "lower_legs": 0.094,
    "upper_legs": 0.200,
    "trunk_head": 0.578,
    "arms": 0.100,
}
if abs(sum(MASS_FRAC.values()) - 1.0) >= 1e-9:
    raise ValueError(f"MASS_FRAC values must sum to 1.0, got {sum(MASS_FRAC.values())}")

# ------------------------------------------------------------------
# Segment length fractions  (fraction of body height)
# ------------------------------------------------------------------
LENGTH_FRAC: dict[str, float] = {
    "lower_leg": 0.246,
    "upper_leg": 0.245,
    "torso": 0.288,
    "arm": 0.387,
    "foot": 0.152,
    "neck": 0.075,  # C7 to base of skull, ~7.5% of height
}

# ------------------------------------------------------------------
# Neck joint angle limit (degrees)
#
# The neck can tilt up to 45 degrees in any direction relative to the
# torso.  The 2-D model does not currently have a separate neck DOF,
# but the constant is documented here so the rendering keeps the neck
# roughly aligned with the torso.
# ------------------------------------------------------------------
NECK_MAX_ANGLE_DEG: float = 45.0

# ------------------------------------------------------------------
# COM position as fraction of segment length from proximal joint
# ------------------------------------------------------------------
COM_FRAC: dict[str, float] = {
    "lower_leg": 0.433,
    "upper_leg": 0.433,
    "torso": 0.600,  # combined trunk+head per Winter (2009)
    "arm": 0.530,
    "foot": 0.500,
}

# ------------------------------------------------------------------
# Standard Olympic barbell
# ------------------------------------------------------------------
BAR_MASS_KG: float = 20.0
BAR_LENGTH_M: float = 2.20
PLATE_RADIUS_STD_M: float = 0.225
BAR_RADIUS_M: float = 0.025

# ------------------------------------------------------------------
# Anatomical joint angle limits (radians)
#
# These represent the physiological range of motion for each joint
# in the sagittal-plane model.  Convention: angles are measured from
# the vertical (0 = straight).
#
# Sources: Norkin & White (2016), Joint Range of Motion.
# ------------------------------------------------------------------
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # Ankle: dorsiflexion (-20°) to plantarflexion (+50°)
    "ankle": (np.radians(-20), np.radians(50)),
    # Knee: full extension (+5°) to full flexion (-140°)
    "knee": (np.radians(-140), np.radians(5)),
    # Hip: full flexion forward (+120°) to hyperextension (-10°)
    "hip": (np.radians(-10), np.radians(120)),
}

# Joint names for the 3-link chain (index → name)
JOINT_NAMES: tuple[str, ...] = ("ankle", "knee", "hip")

# ------------------------------------------------------------------
# Canonical exercise poses (degrees)
#
# These define the raw (pre-balance) joint angles for key positions.
# Factory functions in models.py feed these into balance_pose() to
# obtain the final COM-balanced configuration.
# ------------------------------------------------------------------

# Squat bottom: deep knee bend with moderate ankle & hip flexion.
SQUAT_BOTTOM_DEG: tuple[float, float, float] = (25.0, -90.0, 50.0)

# Standing: anatomically neutral upright (all joints at zero).
STANDING_DEG: tuple[float, float, float] = (0.0, 0.0, 0.0)

# ------------------------------------------------------------------
# Default maximum isometric joint torques (N·m)
#
# Representative values for an average adult male.  These are the
# τ_max parameter in the Hill-type torque-angle-velocity model.
#
# Sources: Anderson et al. (2007), Biomechanics and Motor Control.
# ------------------------------------------------------------------
DEFAULT_MAX_JOINT_TORQUES: dict[str, float] = {
    "ankle": 150.0,
    "knee": 250.0,
    "hip": 250.0,
}

# ------------------------------------------------------------------
# Hill-type torque model parameters
#
# The available torque at each joint is:
#   τ_avail = τ_max · f_angle(q) · f_velocity(qd)
#
# f_angle: Gaussian-shaped torque-angle curve centered at q_opt
#   f_angle = exp(-((q - q_opt) / angle_width)²)
#
# f_velocity: Hill's force-velocity relationship
#   concentric (shortening):  f_vel = (v_max - |qd|) / (v_max + |qd| / k_shape)
#   eccentric (lengthening):  f_vel = (1 + ecc_factor * |qd|) / (1 + |qd| / k_shape)
#   clamped to [0, max_eccentric_ratio]
#
# References:
#   Hill, A.V. (1938). The heat of shortening and the dynamic constants
#       of muscle. Proc. R. Soc. Lond. B, 126(843), 136-195.
#   Westing, S.H., Seger, J.Y., & Thorstensson, A. (1988). Effects of
#       electrical stimulation on eccentric and concentric torque-velocity
#       relationships during knee extension in man. Acta Physiol. Scand.
# ------------------------------------------------------------------
HILL_OPTIMAL_ANGLES: dict[str, float] = {
    "ankle": np.radians(15),
    "knee": np.radians(-45),
    "hip": np.radians(45),
}

HILL_ANGLE_WIDTH: float = np.radians(60)

# Per-joint maximum angular velocities (deg/s), converted to rad/s.
# Sources: Bobbert & van Ingen Schenau (1988), Westing et al. (1988).
HILL_MAX_ANGULAR_VELOCITY_PER_JOINT: dict[str, float] = {
    "ankle": np.radians(250),
    "knee": np.radians(600),
    "hip": np.radians(500),
}

# Scalar fallback for callers that expect a single value (uses the
# fastest joint -- knee -- to avoid under-estimating capacity).
HILL_MAX_ANGULAR_VELOCITY: float = np.radians(600)

HILL_K_SHAPE: float = 0.25

HILL_ECCENTRIC_FACTOR: float = 0.3

HILL_MAX_ECCENTRIC_RATIO: float = 1.4

# ------------------------------------------------------------------
# Bench press model constants
#
# The bench press is modelled as a supine press: the lifter lies on
# a bench with the bar at chest level and presses vertically.  In the
# sagittal plane this is a 2-link chain (upper arm + forearm) with the
# shoulder as the fixed pivot.  For the 3-link model we map:
#   q[0] → shoulder flexion/extension
#   q[1] → elbow flexion/extension
#   q[2] → wrist (held fixed, ≈ 0)
#
# Segment fractions for the arm chain (fraction of arm length):
# ------------------------------------------------------------------
# Wrist/hand segment length as a fraction of arm length.  Replaces the
# former fixed 0.01 m constant so the link scales allometrically with
# the lifter's body height (via arm_len).  ~1% of arm length yields
# ~7 mm for a 1.75 m person — effectively a grip-only link.
WRIST_SEGMENT_FRAC: float = 0.01

BENCH_UPPER_ARM_FRAC: float = 0.56  # shoulder to elbow (anatomical ~48% + shoulder width)
BENCH_FOREARM_FRAC: float = 0.44  # elbow to wrist (Winter 2009: ~44% of arm length)

BENCH_PRESS_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder": (np.radians(-5), np.radians(95)),  # main driver of the press
    "elbow": (np.radians(-110), np.radians(5)),  # tighter: lockout to ~110 deg flexion
    "wrist": (np.radians(-1), np.radians(1)),  # effectively locked straight
}

BENCH_PRESS_JOINT_NAMES: tuple[str, ...] = ("shoulder", "elbow", "wrist")

BENCH_PRESS_MAX_JOINT_TORQUES: dict[str, float] = {
    "shoulder": 120.0,
    "elbow": 80.0,
    "wrist": 15.0,
}

BENCH_PRESS_HILL_OPTIMAL_ANGLES: dict[str, float] = {
    "shoulder": np.radians(45),
    "elbow": np.radians(-70),
    "wrist": np.radians(0),
}

# ------------------------------------------------------------------
# Base-of-support constraint: fraction of foot that is "in bounds"
# The outer 20% on each end is excluded.
# ------------------------------------------------------------------
BOS_INNER_FRACTION: float = 0.60

# ------------------------------------------------------------------
# Post-solve feasibility tolerances
# ------------------------------------------------------------------
# After SLSQP reports ``success``, the returned spline is re-evaluated and
# its kinematics are checked against the hard constraints.  Numerical
# slack (spline overshoot between control points, solver tolerances) means
# an exactly-feasible solve can stray a hair outside the bounds, so the
# post-solve check allows these small margins before declaring the result
# infeasible.
#
# COM_FEASIBILITY_TOL_M: horizontal COM may sit this many metres outside
#   the inner base-of-support before the trajectory is rejected (5 mm).
# JOINT_FEASIBILITY_TOL_RAD: joint angles may exceed ``q_bounds`` by this
#   many radians (≈0.29°) before the trajectory is rejected.  This absorbs
#   benign cubic-spline overshoot while still catching gross violations.
COM_FEASIBILITY_TOL_M: float = 0.005
JOINT_FEASIBILITY_TOL_RAD: float = 0.005

# ------------------------------------------------------------------
# Trajectory optimisation tuning constants
# ------------------------------------------------------------------

# Bench press bar-path penalty weight: penalises horizontal deviation of
# the bar (hand position) from a vertical path during the press.
BENCH_BAR_PATH_WEIGHT: float = 500.0

# Total-variation weight ratio: TV regularisation is this fraction of the
# L2 torque-rate regularisation term.
TV_RATE_WEIGHT_RATIO: float = 0.1

# Minimum bar-to-knee clearance for pulling exercises (deadlift, clean,
# snatch).  The bar must stay at least this many metres in front of the
# knees throughout the lift.
BAR_KNEE_CLEARANCE_M: float = 0.05

# ------------------------------------------------------------------
# Radius of gyration as fraction of segment length (about COM)
#
# Used with parallel axis theorem: I_prox = I_com + m * d_com^2
# where I_com = m * (rho * L)^2.
#
# Source: Winter, D.A. (2009). Table 3.1.
# ------------------------------------------------------------------
RADIUS_OF_GYRATION_FRAC: dict[str, float] = {
    "lower_leg": 0.302,
    "upper_leg": 0.323,
    "trunk": 0.496,
}

# ------------------------------------------------------------------
# Dynamics simplification guards
# ------------------------------------------------------------------
# The analytic Coriolis vector keeps only centrifugal (qd_j^2) terms and omits
# the cross-velocity Coriolis terms (qd_i*qd_j, i != j). That is acceptable for
# slow barbell work but underestimates torque for fast lifts. When any joint
# speed exceeds this threshold the dynamics logs a one-time warning so callers
# know the omitted terms may be material (issue #491).
CORIOLIS_SLOW_LIMIT_RAD_S: float = 2.0

# Radius-of-gyration fractions for the arm chain (about each segment COM),
# used by BenchPressModel so its inertia convention matches BodyModel
# (centroidal I_com = m * (rho * L)^2). Winter (2009), Table 3.1.
ARM_RADIUS_OF_GYRATION_FRAC: dict[str, float] = {
    "upper_arm": 0.322,
    "forearm": 0.303,
    "hand": 0.297,
}

# ------------------------------------------------------------------
# GUI progress / status presentation constants
#
# These govern the optimisation sidebar's progress bar and status text.
# They are cosmetic (no effect on the physics) but are centralised here so
# the use sites carry names and rationale instead of bare literals.
# ------------------------------------------------------------------
# Progress bar saturates here (never shows 100% until the run actually
# finishes, to avoid implying completion mid-solve).
PROGRESS_MAX_PCT: int = 95
# Evaluation-count scale in the asymptotic progress curve
# pct = MAX * (1 - 1 / (1 + n_evals / SCALE)); larger => slower fill.
PROGRESS_EVAL_SCALE: float = 500.0
# Above this many evaluations the status label switches Exploring->Converging.
PROGRESS_PHASE_BOUNDARY_EVALS: int = 200
# Wall-clock seconds after which a non-stalled run shows a "taking longer
# than expected" hint.
STALL_HINT_ELAPSED_S: float = 120.0

# ------------------------------------------------------------------
# Exercise-tab plot layout (matplotlib GridSpec)
#
# 3 rows x 4 cols: a tall animation row on top of two analysis rows.
# Centralised so the layout is tunable in one place.
# ------------------------------------------------------------------
PLOT_GRID_ROWS: int = 3
PLOT_GRID_COLS: int = 4
PLOT_GRID_HEIGHT_RATIOS: tuple[int, int, int] = (3, 1, 1)
PLOT_GRID_HSPACE: float = 0.40
PLOT_GRID_WSPACE: float = 0.40
PLOT_GRID_MARGINS: dict[str, float] = {
    "left": 0.06,
    "right": 0.97,
    "top": 0.93,
    "bottom": 0.06,
}

# numpy compat shim (trapz renamed to trapezoid in numpy 2.0)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapz is None:
    raise ImportError("Neither np.trapezoid nor np.trapz found in numpy")
trapezoid: Callable[..., Any] = _trapz
