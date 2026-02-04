"""Sim-to-Real transfer techniques.

This module provides methods for transferring learned policies
from simulation to real robots:
- Domain Randomization
- System Identification
- Reality Gap Analysis
"""

from __future__ import annotations

from src.learning.sim2real.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)
from src.learning.sim2real.system_identification import SystemIdentifier

__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "SystemIdentifier",
]
