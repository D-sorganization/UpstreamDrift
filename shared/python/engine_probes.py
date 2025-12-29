"""Engine readiness probe system.

This module provides infrastructure for checking if physics engines
are properly installed and ready to use, with actionable diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ProbeStatus(Enum):
    """Status of an engine probe."""

    AVAILABLE = "available"
    MISSING_BINARY = "missing_binary"
    MISSING_ASSETS = "missing_assets"
    VERSION_MISMATCH = "version_mismatch"
    NOT_INSTALLED = "not_installed"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class EngineProbeResult:
    """Result of an engine readiness probe."""

    engine_name: str
    status: ProbeStatus
    version: str | None
    missing_dependencies: list[str]
    diagnostic_message: str
    details: dict[str, Any] | None = None

    def is_available(self) -> bool:
        """Check if engine is available for use."""
        return self.status == ProbeStatus.AVAILABLE

    def get_fix_instructions(self) -> str:
        """Get instructions for fixing issues."""
        if self.status == ProbeStatus.NOT_INSTALLED:
            return f"Install {self.engine_name} dependencies"
        elif self.status == ProbeStatus.MISSING_BINARY:
            return f"Install {self.engine_name} binaries"
        elif self.status == ProbeStatus.MISSING_ASSETS:
            return f"Install {self.engine_name} assets/models"
        elif self.status == ProbeStatus.VERSION_MISMATCH:
            return f"Update {self.engine_name} to compatible version"
        elif self.status == ProbeStatus.CONFIGURATION_ERROR:
            return f"Fix {self.engine_name} configuration"
        return "Engine is available"


class EngineProbe:
    """Base class for engine readiness probes."""

    def __init__(self, engine_name: str, suite_root: Path) -> None:
        """Initialize engine probe.

        Args:
            engine_name: Name of the engine
            suite_root: Root directory of the suite
        """
        self.engine_name = engine_name
        self.suite_root = suite_root

    def is_available(self) -> bool:
        """Check availability via probe."""
        try:
            return self.probe().is_available()
        except Exception:
            return False

    def probe(self) -> EngineProbeResult:
        """Check if engine is ready to use.

        Returns:
            Probe result with status and diagnostics
        """
        raise NotImplementedError


class MuJoCoProbe(EngineProbe):
    """Probe for MuJoCo physics engine."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize MuJoCo probe."""
        super().__init__("MuJoCo", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check MuJoCo readiness."""
        missing = []

        # Check for mujoco package
        try:
            import mujoco

            version = getattr(mujoco, "__version__", "unknown")
        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["mujoco"],
                diagnostic_message="MuJoCo Python package not installed. "
                "Install with: pip install mujoco",
            )
        except OSError as e:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_BINARY,
                version=None,
                missing_dependencies=["MuJoCo DLLs"],
                diagnostic_message=f"DLL error: {e}. "
                "MuJoCo binaries may be missing or incompatible. "
                "This feature works in Docker.",
            )

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "physics_engines" / "mujoco"
        if not engine_dir.exists():
            missing.append("engine directory")

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            # Check for key modules
            key_modules = ["humanoid_launcher.py", "mujoco_humanoid_golf"]
            for module in key_modules:
                if not (python_dir / module).exists():
                    missing.append(f"module: {module}")
        else:
            missing.append("python directory")

        # Check for assets
        assets_dir = engine_dir / "assets"
        myo_sim_dir = engine_dir / "myo_sim"

        valid_assets_dir = None
        if assets_dir.exists():
            valid_assets_dir = assets_dir
        elif myo_sim_dir.exists():
            valid_assets_dir = myo_sim_dir

        if valid_assets_dir:
            # Check for model files
            models = list(valid_assets_dir.glob("**/*.xml"))
            if not models:
                missing.append("model XML files")
        else:
            missing.append("assets or myo_sim directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"MuJoCo {version} installed but missing: "
                f"{', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"MuJoCo {version} ready",
            details={
                "engine_dir": str(engine_dir),
                "assets_dir": str(valid_assets_dir),
            },
        )


class DrakeProbe(EngineProbe):
    """Probe for Drake physics engine."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize Drake probe."""
        super().__init__("Drake", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check Drake readiness."""
        missing = []

        # Check for pydrake package
        try:
            import pydrake

            version = getattr(pydrake, "__version__", "unknown")

            # Verify core modules
            try:
                import pydrake.multibody
            except ImportError:
                return EngineProbeResult(
                    engine_name=self.engine_name,
                    status=ProbeStatus.MISSING_BINARY,
                    version=version,
                    missing_dependencies=["pydrake.multibody"],
                    diagnostic_message="Drake installed but 'pydrake.multibody' missing. "
                    "Installation might be corrupted.",
                )

        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["drake"],
                diagnostic_message="Drake Python package not installed. "
                "Install with: pip install drake",
            )

        # Check meshcat port availability
        import socket

        meshcat_available = False
        available_port = None
        for port in range(7000, 7011):
            try:
                sock = socket.socket()
                sock.bind(("localhost", port))
                sock.close()
                meshcat_available = True
                available_port = port
                break
            except OSError:
                continue

        if not meshcat_available:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.CONFIGURATION_ERROR,
                version=version,
                missing_dependencies=["meshcat ports 7000-7010"],
                diagnostic_message=f"Drake {version} installed but meshcat ports "
                "7000-7010 are all blocked. Close other instances or use Docker.",
            )

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "physics_engines" / "drake"
        if not engine_dir.exists():
            missing.append("engine directory")

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            src_dir = python_dir / "src"
            if src_dir.exists():
                key_files = ["golf_gui.py"]
                for file in key_files:
                    if not (src_dir / file).exists():
                        missing.append(f"module: {file}")
            else:
                missing.append("src directory")
        else:
            missing.append("python directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"Drake {version} installed but missing: "
                f"{', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=(
                f"Drake {version} ready, meshcat port {available_port} available"
            ),
            details={
                "engine_dir": str(engine_dir),
                "meshcat_port": available_port,
            },
        )


class PinocchioProbe(EngineProbe):
    """Probe for Pinocchio physics engine."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize Pinocchio probe."""
        super().__init__("Pinocchio", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check Pinocchio readiness."""
        missing = []

        # Check for pinocchio package
        try:
            import pinocchio

            version = getattr(pinocchio, "__version__", "unknown")
        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["pinocchio"],
                diagnostic_message="Pinocchio Python package not installed. "
                "Install with: pip install pin",
            )

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "physics_engines" / "pinocchio"
        if not engine_dir.exists():
            missing.append("engine directory")

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            key_dirs = ["pinocchio_golf"]
            for dir_name in key_dirs:
                if not (python_dir / dir_name).exists():
                    missing.append(f"module: {dir_name}")
        else:
            missing.append("python directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"Pinocchio {version} installed but missing: "
                f"{', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"Pinocchio {version} ready",
            details={"engine_dir": str(engine_dir)},
        )


class PendulumProbe(EngineProbe):
    """Probe for Pendulum models."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize Pendulum probe."""
        super().__init__("Pendulum", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check Pendulum models readiness."""
        missing = []

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "pendulum_models"
        if not engine_dir.exists():
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=None,
                missing_dependencies=["engine directory"],
                diagnostic_message="Pendulum models directory not found",
            )

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            src_dir = python_dir / "src"
            if src_dir.exists():
                key_files = ["constants.py", "pendulum_solver.py"]
                for file in key_files:
                    if not (src_dir / file).exists():
                        missing.append(f"module: {file}")
            else:
                missing.append("src directory")
        else:
            missing.append("python directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version="local",
                missing_dependencies=missing,
                diagnostic_message=f"Pendulum models missing: {', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version="local",
            missing_dependencies=[],
            diagnostic_message="Pendulum models ready",
            details={"engine_dir": str(engine_dir)},
        )


class MatlabProbe(EngineProbe):
    """Probe for MATLAB engine."""

    def __init__(self, suite_root: Path, is_3d: bool = False) -> None:
        """Initialize MATLAB probe.

        Args:
            suite_root: Root directory of the suite
            is_3d: Whether to probe for 3D model (default: 2D)
        """
        name = "MATLAB 3D" if is_3d else "MATLAB 2D"
        super().__init__(name, suite_root)
        self.is_3d = is_3d

    def probe(self) -> EngineProbeResult:
        """Check MATLAB readiness."""
        missing = []
        version = None

        # Check for MATLAB engine API
        try:
            import matlab.engine  # noqa: F401

            # We can't easily check version without starting the engine,
            # which is too slow for a probe. Just assume it's there if import works.
            version = "installed (version check skipped)"
        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["matlab.engine"],
                diagnostic_message="MATLAB Engine for Python not installed. "
                "See README for installation instructions.",
            )

        # Check for model directory
        model_type = "3D_Golf_Model" if self.is_3d else "2D_Golf_Model"
        engine_dir = (
            self.suite_root / "engines" / "Simscape_Multibody_Models" / model_type
        )

        if not engine_dir.exists():
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=["model directory"],
                diagnostic_message=f"MATLAB model directory not found at {engine_dir}",
            )

        # Basic check for contents
        if not any(engine_dir.glob("*.slx")) and not any(engine_dir.glob("*.m")):
            missing.append("Simulink/MATLAB files")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"MATLAB model files missing in {engine_dir}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"{self.engine_name} ready",
            details={"engine_dir": str(engine_dir)},
        )


class OpenSimProbe(EngineProbe):
    """Probe for OpenSim physics engine."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize OpenSim probe."""
        super().__init__("OpenSim", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check OpenSim readiness."""
        missing = []

        # Check for opensim package
        try:
            import opensim

            version = getattr(opensim, "__version__", "unknown")
            if version == "unknown":
                # Try getting version from build info if available
                try:
                    version = opensim.GetVersion()
                except AttributeError:
                    pass

        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["opensim"],
                diagnostic_message="OpenSim Python package not installed. "
                "See OpenSim documentation for installation.",
            )

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "physics_engines" / "opensim"
        if not engine_dir.exists():
            missing.append("engine directory")

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            key_dirs = ["opensim_physics_engine.py"]
            for dir_name in key_dirs:
                if not (python_dir / dir_name).exists():
                    missing.append(f"module: {dir_name}")
        else:
            missing.append("python directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"OpenSim {version} installed but missing: "
                f"{', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"OpenSim {version} ready",
            details={"engine_dir": str(engine_dir)},
        )


class MyoSimProbe(EngineProbe):
    """Probe for MyoSim physics engine."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize MyoSim probe."""
        super().__init__("MyoSim", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check MyoSim readiness."""
        missing = []

        # Check for mujoco package (MyoSim depends on MuJoCo)
        try:
            import mujoco

            version = getattr(mujoco, "__version__", "unknown")
        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["mujoco"],
                diagnostic_message="MuJoCo Python package not installed (required for MyoSim). "
                "Install with: pip install mujoco",
            )

        # Check for engine directory
        engine_dir = self.suite_root / "engines" / "physics_engines" / "myosim"
        if not engine_dir.exists():
            missing.append("engine directory")

        # Check for Python modules
        python_dir = engine_dir / "python"
        if python_dir.exists():
            if not (python_dir / "myosim_physics_engine.py").exists():
                 missing.append("module: myosim_physics_engine.py")
        else:
            missing.append("python directory")

        if missing:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=missing,
                diagnostic_message=f"MyoSim installed but missing: "
                f"{', '.join(missing)}",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"MyoSim ready (via MuJoCo {version})",
            details={"engine_dir": str(engine_dir)},
        )
