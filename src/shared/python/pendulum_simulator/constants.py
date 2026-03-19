"""
Shared physical constants for the pendulum simulator.

Single source of truth for all physical constants used across
the simulator's physics engines, GUI panels, and test fixtures.

Following the DRY principle (Don't Repeat Yourself):
these values were previously defined as magic numbers in 28+
locations across the codebase.

Reference values from NIST:
  https://physics.nist.gov/cgi-bin/cuu/Value?gn
"""

from __future__ import annotations

# ── Gravitational Constants ─────────────────────────────────────────────

#: Standard gravitational acceleration (m/s²).
#: Used as the default ``g`` parameter in physics engines and GUI panels.
#: This is the conventional standard value, not the precise value at any
#: particular latitude.
GRAVITY_MSS: float = 9.81

#: Standard acceleration of gravity (m/s²) — exact SI definition.
#: Used for unit conversions (e.g. kgf → N) where the precise value matters.
GRAVITY_STANDARD: float = 9.80665

# ── Conversion Factors ──────────────────────────────────────────────────

#: Newtons per kilogram-force (N/kgf).
NM_PER_KGFM: float = GRAVITY_STANDARD

#: Pounds-force per Newton.
LBF_PER_N: float = 0.224809

#: Inches per meter.
INCHES_PER_M: float = 39.3701

#: Meters per inch.
M_PER_INCH: float = 0.0254
