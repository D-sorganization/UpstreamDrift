"""Contact dynamics module.

This module provides contact detection, management, and analysis
capabilities for multi-contact scenarios in robotics.

Components:
    - ContactManager: Manages multi-contact scenarios
    - FrictionCone: Friction cone utilities and linearization
    - GraspAnalysis: Grasp quality metrics

Example:
    >>> from src.robotics.contact import ContactManager
    >>> from src.robotics.core.types import ContactState
    >>>
    >>> manager = ContactManager(engine)
    >>> contacts = manager.detect_contacts()
    >>> for contact in contacts:
    ...     print(f"Contact {contact.contact_id}: {contact.normal_force:.2f} N")
"""

from __future__ import annotations

from src.robotics.contact.contact_manager import ContactManager
from src.robotics.contact.friction_cone import (
    FrictionCone,
    compute_friction_cone_constraint,
    linearize_friction_cone,
)
from src.robotics.contact.grasp_analysis import (
    check_force_closure,
    compute_grasp_matrix,
    compute_grasp_quality,
)

__all__ = [
    "ContactManager",
    "FrictionCone",
    "linearize_friction_cone",
    "compute_friction_cone_constraint",
    "compute_grasp_matrix",
    "check_force_closure",
    "compute_grasp_quality",
]
