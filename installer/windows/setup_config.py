"""Pure setup configuration helpers for installer profile builds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from installer.windows.packaging_profiles import PackagingProfile, get_packaging_profile

PHYSICS_ENGINES: dict[str, dict[str, Any]] = {
    "mujoco": {
        "name": "MuJoCo Physics Engine",
        "description": "High-performance physics simulation with contact dynamics",
        "modules": ("mujoco", "engines.physics_engines.mujoco"),
        "required": True,
    },
    "drake": {
        "name": "Drake Manipulation Planning",
        "description": "Trajectory optimization and system analysis",
        "modules": ("pydrake", "engines.physics_engines.drake"),
        "required": False,
    },
    "pinocchio": {
        "name": "Pinocchio Rigid Body Dynamics",
        "description": "Fast rigid body dynamics and derivatives",
        "modules": ("pinocchio", "engines.physics_engines.pinocchio"),
        "required": False,
    },
    "myosuite": {
        "name": "MyoSuite Muscle Simulation",
        "description": "Realistic muscle dynamics and neural control",
        "modules": ("myosuite", "engines.physics_engines.myosuite"),
        "required": False,
    },
    "opensim": {
        "name": "OpenSim Biomechanics",
        "description": "Biomechanical modeling and analysis",
        "modules": ("opensim", "engines.physics_engines.opensim"),
        "required": False,
    },
}

BASE_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "PyQt6",
    "yaml",
    "structlog",
    "ezc3d",
    "sympy",
    "defusedxml",
    "shared",
    "launchers",
    "api",
    "tools",
]

EXCLUDES = [
    "tkinter",
    "unittest",
    "pydoc",
    "difflib",
    "calendar",
    "doctest",
    "inspect",
    "pickle",
    "pdb",
    "profile",
    "pstats",
    "timeit",
    "trace",
]


@dataclass(frozen=True)
class ExecutableSpec:
    """Small serializable executable description used by tests and setup.py."""

    script: str
    base: str
    target_name: str
    icon: str
    shortcut_name: str | None = None
    shortcut_dir: str | None = None


@dataclass(frozen=True)
class SetupProfileConfiguration:
    """Resolved setup configuration for a selected packaging profile."""

    profile: PackagingProfile
    available_engines: tuple[str, ...]
    build_exe_options: dict[str, Any]
    bdist_msi_options: dict[str, Any]
    executables: tuple[ExecutableSpec, ...]


def detect_available_engines(
    profile: PackagingProfile,
    *,
    importer: Callable[[str], Any] = __import__,
) -> tuple[str, ...]:
    """Detect engine modules eligible for the selected packaging profile."""
    available: list[str] = []
    for engine_id, engine_info in PHYSICS_ENGINES.items():
        include_engine = profile.bundle_optional_engines or engine_info["required"]
        if not include_engine:
            continue
        module_name = engine_info["modules"][0]
        try:
            importer(module_name)
        except ImportError:
            if engine_info["required"]:
                raise
            continue
        available.append(engine_id)
    return tuple(available)


def build_executable_specs(
    project_root: Path,
    profile: PackagingProfile,
) -> tuple[ExecutableSpec, ...]:
    """Build executable metadata for the selected packaging profile."""
    icon_path = str(project_root / "src" / "launchers" / "assets" / "golf_robot_icon.ico")
    executables = [
        ExecutableSpec(
            script=str(project_root / "src" / "launchers" / "upstream_drift_launcher.py"),
            base="Win32GUI",
            target_name="GolfModelingSuite.exe",
            icon=icon_path,
            shortcut_name="UpstreamDrift",
            shortcut_dir="DesktopFolder",
        )
    ]
    if profile.include_api_executable:
        executables.append(
            ExecutableSpec(
                script=str(project_root / "src" / "api" / "server.py"),
                base="Console",
                target_name="GolfAPI.exe",
                icon=icon_path,
            )
        )
    return tuple(executables)


def build_setup_configuration(
    project_root: Path,
    profile_name: str | None,
    *,
    importer: Callable[[str], Any] = __import__,
) -> SetupProfileConfiguration:
    """Resolve the full cx_Freeze setup configuration for the selected profile."""
    profile = get_packaging_profile(profile_name)
    available_engines = detect_available_engines(profile, importer=importer)
    packages = list(BASE_PACKAGES)
    for engine_id in available_engines:
        packages.extend(PHYSICS_ENGINES[engine_id]["modules"])

    build_exe_options = {
        "packages": packages,
        "excludes": list(EXCLUDES),
        "include_files": [
            (str(project_root / "src" / "shared" / "urdf"), "shared/urdf"),
            (str(project_root / "src" / "shared" / "models"), "shared/models"),
            (str(project_root / "config"), "config"),
            (str(project_root / "docs"), "docs"),
            (str(project_root / "README.md"), "README.md"),
            (str(project_root / "LICENSE"), "LICENSE"),
        ],
        "include_msvcrt": True,
        "optimize": 2,
        "build_exe": f"build/upstream_drift_{profile.profile_id}",
    }

    bdist_msi_options = {
        "upgrade_code": "{12345678-1234-5678-9012-123456789012}",
        "add_to_path": True,
        "initial_target_dir": rf"[ProgramFilesFolder]\UpstreamDrift\{profile.profile_id}",
        "install_icon": str(project_root / "src" / "launchers" / "assets" / "golf_robot_icon.ico"),
        "summary_data": {
            "author": "UpstreamDrift Team",
            "comments": profile.description,
            "keywords": "golf, biomechanics, physics, simulation",
        },
    }

    return SetupProfileConfiguration(
        profile=profile,
        available_engines=available_engines,
        build_exe_options=build_exe_options,
        bdist_msi_options=bdist_msi_options,
        executables=build_executable_specs(project_root, profile),
    )
