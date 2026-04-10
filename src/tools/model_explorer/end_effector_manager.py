"""End Effector Manager - Tools for swapping and managing end effectors.

Provides visual interface for easily swapping end effectors between URDFs,
changing attachment points, and managing end effector configurations.
"""

from __future__ import annotations

from src.tools.model_explorer._attachment_dialog import AttachmentPointSelector
from src.tools.model_explorer._ee_library import EndEffectorLibrary
from src.tools.model_explorer._ee_model import EndEffector
from src.tools.model_explorer._ee_widget import EndEffectorManagerWidget

__all__ = [
    "AttachmentPointSelector",
    "EndEffector",
    "EndEffectorLibrary",
    "EndEffectorManagerWidget",
]
