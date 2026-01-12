"""Windows MSI installer setup for Golf Modeling Suite.

This script creates a professional Windows MSI installer with:
- Modular physics engine selection
- Desktop shortcuts and Start Menu entries
- Automatic dependency management
- Uninstaller support
"""

import sys
from pathlib import Path

from cx_Freeze import Executable, setup

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import version and metadata
try:
    from shared.python.version import __description__, __version__
except ImportError:
    __version__ = "1.0.0"
    __description__ = "Golf Modeling Suite - Professional Biomechanical Analysis"

# Define physics engine modules
PHYSICS_ENGINES = {
    "mujoco": {
        "name": "MuJoCo Physics Engine",
        "description": "High-performance physics simulation with contact dynamics",
        "modules": ["mujoco", "engines.physics_engines.mujoco"],
        "required": True,  # Core engine
    },
    "drake": {
        "name": "Drake Manipulation Planning",
        "description": "Trajectory optimization and system analysis",
        "modules": ["pydrake", "engines.physics_engines.drake"],
        "required": False,
    },
    "pinocchio": {
        "name": "Pinocchio Rigid Body Dynamics",
        "description": "Fast rigid body dynamics and derivatives",
        "modules": ["pinocchio", "engines.physics_engines.pinocchio"],
        "required": False,
    },
    "myosuite": {
        "name": "MyoSuite Muscle Simulation",
        "description": "Realistic muscle dynamics and neural control",
        "modules": ["myosuite", "engines.physics_engines.myosuite"],
        "required": False,
    },
    "opensim": {
        "name": "OpenSim Biomechanics",
        "description": "Biomechanical modeling and analysis",
        "modules": ["opensim", "engines.physics_engines.opensim"],
        "required": False,
    },
}

# Base packages always included
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

# Build options
build_exe_options = {
    "packages": BASE_PACKAGES,
    "excludes": [
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
    ],
    "include_files": [
        (str(project_root / "shared" / "urdf"), "shared/urdf"),
        (str(project_root / "shared" / "meshes"), "shared/meshes"),
        (str(project_root / "config"), "config"),
        (str(project_root / "docs"), "docs"),
        (str(project_root / "README.md"), "README.md"),
        (str(project_root / "LICENSE"), "LICENSE"),
    ],
    "include_msvcrt": True,
    "optimize": 2,
    "build_exe": "build/golf_modeling_suite",
}

# MSI options
bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-9012-123456789012}",
    "add_to_path": True,
    "initial_target_dir": r"[ProgramFilesFolder]\Golf Modeling Suite",
    "install_icon": str(project_root / "shared" / "icons" / "golf_robot.ico"),
    "summary_data": {
        "author": "Golf Modeling Suite Team",
        "comments": "Professional biomechanical analysis software",
        "keywords": "golf, biomechanics, physics, simulation",
    },
}

# Executables
executables = [
    Executable(
        script=str(project_root / "launchers" / "golf_launcher.py"),
        base="Win32GUI",
        target_name="GolfModelingSuite.exe",
        icon=str(project_root / "shared" / "icons" / "golf_robot.ico"),
        shortcut_name="Golf Modeling Suite",
        shortcut_dir="DesktopFolder",
    ),
    Executable(
        script=str(project_root / "api" / "server.py"),
        base="Console",
        target_name="GolfAPI.exe",
        icon=str(project_root / "shared" / "icons" / "golf_robot.ico"),
    ),
]


# Dynamic package inclusion based on available engines
def get_available_engines():
    """Detect which physics engines are available for inclusion."""
    available = []

    for engine_id, engine_info in PHYSICS_ENGINES.items():
        try:
            # Try to import the main module
            main_module = engine_info["modules"][0]
            __import__(main_module)
            available.append(engine_id)
        except ImportError:
            if engine_info["required"]:
                sys.exit(1)
            else:
                pass

    return available


# Add available engine packages
available_engines = get_available_engines()
for engine_id in available_engines:
    engine_modules = PHYSICS_ENGINES[engine_id]["modules"]
    build_exe_options["packages"].extend(engine_modules)

for _engine_id in available_engines:
    pass

# Setup configuration
setup(
    name="Golf Modeling Suite",
    version=__version__,
    description=__description__,
    long_description="""
Golf Modeling Suite is a professional-grade biomechanical analysis platform
for golf swing modeling and simulation. It provides:

• Multiple physics engines (MuJoCo, Drake, Pinocchio, MyoSuite, OpenSim)
• Video-based pose estimation with MediaPipe
• Ball flight physics with Magnus effect
• Cross-engine validation and comparison
• Professional visualization and analysis tools
• REST API for cloud integration

Perfect for researchers, coaches, and equipment manufacturers seeking
research-grade biomechanical insights.
    """.strip(),
    author="Golf Modeling Suite Team",
    author_email="support@golfmodelingsuite.com",
    url="https://github.com/D-sorganization/Golf_Modeling_Suite",
    license="MIT",
    executables=executables,
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: Microsoft :: Windows",
    ],
)
