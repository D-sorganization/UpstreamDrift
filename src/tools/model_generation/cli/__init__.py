"""
Command-line interface for model_generation.

Provides the `model-gen` command for URDF generation, conversion,
editing, and library operations.

Usage:
    model-gen generate my_robot --humanoid --output robot.urdf
    model-gen convert robot.slx -o robot.urdf
    model-gen validate robot.urdf
    model-gen library list
"""

from model_generation.cli.main import main, create_parser

__all__ = ["main", "create_parser"]
