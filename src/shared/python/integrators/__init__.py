"""Shared integrator interfaces and implementations for physics engines.

This module provides standardized RK4 integrator definitions that ensure
parity across Drake, Pinocchio, OpenSim, and MuJoCo physics engines.
"""

from .rk4_standard import RK4StandardIntegrator

__all__ = ["RK4StandardIntegrator"]
