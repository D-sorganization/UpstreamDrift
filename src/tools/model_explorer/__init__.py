"""Interactive URDF Generator for Golf Modeling Suite.

This module provides an interactive GUI tool for creating URDF files
with support for parallel kinematic configurations commonly used in
golf swing modeling.

New in v2.0.0:
- Component library with read-only protection and copy-to-edit
- URDF code editor with syntax highlighting and validation
- Frankenstein mode for combining URDFs
- Chain manipulation tools for inserting/editing branches
- End effector swap system with visual interface
- Joint auto-loader and manipulation panel
- Mesh/STL browser with copy functionality
"""

__version__ = "2.0.0"
__author__ = "Golf Modeling Suite Team"

from .segment_manager import SegmentManager
from .urdf_builder import Handedness, URDFBuilder

__all__ = [
    # Main windows
    "URDFGeneratorWindow",
    "URDFEditorWindow",
    # Core classes
    "URDFBuilder",
    "SegmentManager",
    "Handedness",
    # Component library
    "ComponentLibrary",
    "ComponentLibraryWidget",
    # Code editor
    "URDFCodeEditor",
    "URDFCodeEditorWidget",
    # Frankenstein mode
    "FrankensteinEditor",
    # Chain manipulation
    "ChainManipulationWidget",
    "KinematicTree",
    # End effector manager
    "EndEffectorManagerWidget",
    "EndEffectorLibrary",
    # Joint manipulator
    "JointManipulatorWidget",
    # Mesh browser
    "MeshBrowserWidget",
]


def __getattr__(name: str) -> type:
    """Lazy import for GUI components that require PyQt6."""
    if name == "URDFGeneratorWindow":
        from .main_window import URDFGeneratorWindow

        return URDFGeneratorWindow
    if name == "URDFEditorWindow":
        from .urdf_editor_window import URDFEditorWindow

        return URDFEditorWindow
    if name == "ComponentLibrary":
        from .component_library import ComponentLibrary

        return ComponentLibrary
    if name == "ComponentLibraryWidget":
        from .component_library import ComponentLibraryWidget

        return ComponentLibraryWidget
    if name == "URDFCodeEditor":
        from .urdf_code_editor import URDFCodeEditor

        return URDFCodeEditor
    if name == "URDFCodeEditorWidget":
        from .urdf_code_editor import URDFCodeEditorWidget

        return URDFCodeEditorWidget
    if name == "FrankensteinEditor":
        from .frankenstein_editor import FrankensteinEditor

        return FrankensteinEditor
    if name == "ChainManipulationWidget":
        from .chain_manipulation import ChainManipulationWidget

        return ChainManipulationWidget
    if name == "KinematicTree":
        from .chain_manipulation import KinematicTree

        return KinematicTree
    if name == "EndEffectorManagerWidget":
        from .end_effector_manager import EndEffectorManagerWidget

        return EndEffectorManagerWidget
    if name == "EndEffectorLibrary":
        from .end_effector_manager import EndEffectorLibrary

        return EndEffectorLibrary
    if name == "JointManipulatorWidget":
        from .joint_manipulator import JointManipulatorWidget

        return JointManipulatorWidget
    if name == "MeshBrowserWidget":
        from .mesh_browser import MeshBrowserWidget

        return MeshBrowserWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
