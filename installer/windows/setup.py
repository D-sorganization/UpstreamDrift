"""Windows MSI installer setup for UpstreamDrift."""

import os
from pathlib import Path

from cx_Freeze import Executable, setup  # type: ignore[import-not-found]

from installer.windows.setup_config import build_setup_configuration

_this_file = Path(__file__)
_installer_dir = _this_file.parent
project_root = _installer_dir.parent.parent
# Import version and metadata
try:
    from shared.python.version import (  # type: ignore[import-not-found]
        __description__,
        __version__,
    )
except ImportError:
    __version__ = "1.0.0"
    __description__ = "UpstreamDrift - Professional Biomechanical Analysis"

setup_configuration = build_setup_configuration(
    project_root,
    os.environ.get("UPSTREAM_DRIFT_INSTALL_PROFILE"),
)
executables = [
    Executable(
        script=executable.script,
        base=executable.base,
        target_name=executable.target_name,
        icon=executable.icon,
        shortcut_name=executable.shortcut_name,
        shortcut_dir=executable.shortcut_dir,
    )
    for executable in setup_configuration.executables
]

# Setup configuration
setup(
    name="UpstreamDrift",
    version=__version__,
    description=__description__,
    long_description="""
UpstreamDrift is a professional-grade biomechanical analysis platform
for physics-based modeling and simulation. It provides:

• Multiple physics engines (MuJoCo, Drake, Pinocchio, MyoSuite, OpenSim)
• Video-based pose estimation with MediaPipe
• Ball flight physics with Magnus effect
• Cross-engine validation and comparison
• Professional visualization and analysis tools
• REST API for cloud integration

Perfect for researchers and engineers seeking
research-grade biomechanical insights.
    """.strip(),
    author="UpstreamDrift Team",
    author_email="support@upstreamdrift.dev",
    url="https://github.com/D-sorganization/UpstreamDrift",
    license="MIT",
    executables=executables,
    options={
        "build_exe": setup_configuration.build_exe_options,
        "bdist_msi": setup_configuration.bdist_msi_options,
    },
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
