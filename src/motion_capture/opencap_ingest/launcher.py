"""OpenCap sidecar runner and launcher (#11406).

Spawns a separately installed ``opencap-core`` in an isolated environment
(via subprocess using ``managed_popen`` or Docker) so that UpstreamDrift's
product core never imports OpenCap or TensorFlow at runtime, adhering strictly
to ADR-0053.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.process_safety import managed_popen
from src.shared.python.motion_pipeline.sources.opencap_session import (
    OpenCapSession,
    load_opencap_session,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenCapLaunchConfig",
    "OpenCapLaunchResult",
    "OpenCapLauncher",
    "OpenCapSidecarNotFoundError",
    "run_opencap_sidecar",
]

_INSTALL_HINT = (
    "opencap-core sidecar is not installed or accessible. "
    "To run OpenCap locally:\n"
    "1. Conda / venv: create a Python 3.9/3.10 environment named 'opencap-core' "
    "(e.g. `conda create -n opencap-core python=3.10` and install opencap-core "
    "per https://github.com/stanfordnmbl/opencap-core), or set OPENCAP_PYTHON.\n"
    "2. Docker: install Docker and run `docker pull opencap/core:latest`."
)

_DRY_RUN_METADATA_YAML = """\
calibrationSettings:
  overwriteDeployedIntrinsics: false
  saveSessionIntrinsics: false
gender_mf: m
height_m: 1.80
iphoneModel:
  Cam0: iphone13,3
  Cam1: iphone13,3
markerAugmentationSettings:
  markerAugmenterModel: LSTM
mass_kg: 75.0
openSimModel: LaiUhlrich2022
subjectID: subject-stub
"""

_DRY_RUN_MODEL_OSIM = """\
<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
<Model name="LaiUhlrich2022_scaled">
<JointSet><objects>
<CustomJoint name="ground_pelvis">
<coordinates><Coordinate name="pelvis_tilt" /><Coordinate name="pelvis_tx" /></coordinates>
<SpatialTransform>
<TransformAxis name="rotation1"><coordinates>pelvis_tilt</coordinates><axis>1 0 0</axis></TransformAxis>
<TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis></TransformAxis>
<TransformAxis name="rotation3"><coordinates></coordinates><axis>0 0 1</axis></TransformAxis>
<TransformAxis name="translation1"><coordinates>pelvis_tx</coordinates><axis>1 0 0</axis></TransformAxis>
<TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis></TransformAxis>
<TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis></TransformAxis>
</SpatialTransform>
</CustomJoint>
</objects></JointSet>
</Model>
</OpenSimDocument>
"""


class OpenCapSidecarNotFoundError(RuntimeError):
    """Raised when opencap-core cannot be found in environment or Docker."""


@dataclass
class OpenCapLaunchConfig:
    """Configuration for launching the OpenCap sidecar pipeline."""

    session_dir: Path
    video_dir: Path | None = None
    calibration_dir: Path | None = None
    opencap_env: Path | None = None
    docker_image: str = "opencap/core:latest"
    runner_type: str = "auto"  # "auto", "subprocess", "docker"
    detector: str = "hrnet"  # ADR-0053 §2: HRNet is commercial default
    allow_non_commercial: bool = False  # Required if detector == "openpose"
    timeout_seconds: int = 3600
    trial_name: str | None = None
    extra_args: list[str] | None = None
    dry_run: bool = False


@dataclass
class OpenCapLaunchResult:
    """Result of an OpenCap sidecar invocation."""

    success: bool
    return_code: int
    session_dir: Path | None = None
    session: OpenCapSession | None = None
    log_file: Path | None = None
    error_message: str | None = None
    used_docker: bool = False
    used_real_opencap: bool = False


class OpenCapLauncher:
    """Launcher for OpenCap sidecar pipeline in subprocess or container."""

    DEFAULT_OPENCAP_ENV_NAME = "opencap-core"
    DEFAULT_DOCKER_IMAGE = "opencap/core:latest"

    def __init__(self, log_level: int = logging.INFO) -> None:
        self.log_level = log_level

    def find_opencap_python(self, env_path: Path | None = None) -> str | None:
        """Find Python interpreter in the opencap-core environment."""
        if env_path is not None:
            resolved_env = Path(env_path).expanduser().resolve()
            for candidate in (
                resolved_env / "bin" / "python",
                resolved_env / "Scripts" / "python.exe",
            ):
                if candidate.exists():
                    return str(candidate)

        for env_var in ("OPENCAP_PYTHON", "OPENCAP_CORE_PATH"):
            val = os.environ.get(env_var)
            if val and Path(val).exists():
                return str(Path(val).resolve())

        home = Path.home()
        candidates = [
            home / "miniconda3" / "envs" / self.DEFAULT_OPENCAP_ENV_NAME,
            home / "anaconda3" / "envs" / self.DEFAULT_OPENCAP_ENV_NAME,
            home / ".venvs" / self.DEFAULT_OPENCAP_ENV_NAME,
            home / self.DEFAULT_OPENCAP_ENV_NAME,
            Path("/opt/opencap-core"),
        ]
        for base in candidates:
            for py in (base / "bin" / "python", base / "Scripts" / "python.exe"):
                if py.exists():
                    return str(py)
        return None

    def is_docker_available(self) -> bool:
        """Check if Docker CLI is installed and available on PATH."""
        return shutil.which("docker") is not None

    def is_available(self) -> bool:
        """Check if either local Python environment or Docker is available."""
        return self.find_opencap_python() is not None or self.is_docker_available()

    def get_install_hint(self) -> str:
        """Return human-readable installation instructions."""
        return _INSTALL_HINT

    def build_command(self, config: OpenCapLaunchConfig, python_exe: str) -> list[str]:
        """Build subprocess command list for launching opencap-core."""
        cmd = [
            python_exe,
            "-m",
            "opencap_core",
            "--session_dir",
            str(config.session_dir),
            "--detector",
            config.detector,
        ]
        if config.video_dir is not None:
            cmd.extend(["--video_dir", str(config.video_dir)])
        if config.calibration_dir is not None:
            cmd.extend(["--calibration_dir", str(config.calibration_dir)])
        if config.trial_name is not None:
            cmd.extend(["--trial_name", config.trial_name])
        if config.extra_args:
            cmd.extend(config.extra_args)
        return cmd

    def build_docker_command(self, config: OpenCapLaunchConfig) -> list[str]:
        """Build Docker run command list for launching opencap-core container."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{config.session_dir}:/workspace/session",
        ]
        if config.video_dir is not None:
            cmd.extend(["-v", f"{config.video_dir}:/workspace/videos"])
        if config.calibration_dir is not None:
            cmd.extend(["-v", f"{config.calibration_dir}:/workspace/calibration"])
        cmd.append(config.docker_image)
        cmd.extend(
            ["--session_dir", "/workspace/session", "--detector", config.detector]
        )
        if config.video_dir is not None:
            cmd.extend(["--video_dir", "/workspace/videos"])
        if config.calibration_dir is not None:
            cmd.extend(["--calibration_dir", "/workspace/calibration"])
        if config.trial_name is not None:
            cmd.extend(["--trial_name", config.trial_name])
        if config.extra_args:
            cmd.extend(config.extra_args)
        return cmd

    def _setup_logging(self, session_dir: Path) -> Path:
        log_dir = session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return log_dir / f"opencap_{timestamp}.log"

    def _write_dry_run_artifacts(self, session_dir: Path, trial_name: str) -> None:
        """Write synthetic valid OpenCap session layout for dry-run/testing."""
        marker_dir = session_dir / "MarkerData"
        marker_dir.mkdir(parents=True, exist_ok=True)
        trc_file = marker_dir / f"{trial_name}.trc"
        headers = [
            f"PathFileType\t4\t(X/Y/Z)\t{trc_file.name}",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
            "60.0\t60.0\t2\t2\tm\t60.0\t1\t2",
            "Frame#\tTime\tr.ASIS_study\t\t\tL.ASIS_study\t\t\t",
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
            "1\t0.000000\t0.1000\t1.1000\t2.1000\t0.2000\t1.2000\t2.2000",
            "2\t0.016667\t0.1100\t1.1000\t2.1000\t0.2100\t1.2000\t2.2000",
        ]
        trc_file.write_text("\n".join(headers) + "\n", encoding="utf-8")

        (session_dir / "sessionMetadata.yaml").write_text(
            _DRY_RUN_METADATA_YAML, encoding="utf-8"
        )

        model_dir = session_dir / "OpenSimData" / "Model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "LaiUhlrich2022_scaled.osim").write_text(
            _DRY_RUN_MODEL_OSIM, encoding="utf-8"
        )

    def _resolve_runner_command(
        self, config: OpenCapLaunchConfig
    ) -> tuple[list[str], bool]:
        """Resolve command list and whether Docker is being used."""
        runner_type = config.runner_type.lower()
        if runner_type in ("subprocess", "auto"):
            python_exe = self.find_opencap_python(config.opencap_env)
            if python_exe is not None:
                return self.build_command(config, python_exe), False
            if runner_type == "subprocess":
                raise OpenCapSidecarNotFoundError(self.get_install_hint())

        if runner_type in ("docker", "auto"):
            if self.is_docker_available():
                return self.build_docker_command(config), True
            if runner_type == "docker":
                raise OpenCapSidecarNotFoundError(self.get_install_hint())

        raise OpenCapSidecarNotFoundError(self.get_install_hint())

    def _execute_subprocess(
        self, cmd: list[str], config: OpenCapLaunchConfig, log_file: Path
    ) -> int:
        """Run command safely using managed_popen and log output."""
        with open(log_file, "a", encoding="utf-8") as log_fh:
            with managed_popen(
                cmd,
                timeout=config.timeout_seconds,
                cwd=str(config.session_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            ) as proc:
                pass
            return proc.returncode if proc.returncode is not None else -1

    def launch(
        self, config: OpenCapLaunchConfig, raise_on_error: bool = False
    ) -> OpenCapLaunchResult:
        """Launch OpenCap sidecar pipeline with guaranteed resource cleanup.

        Args:
            config: OpenCap launch parameters and paths.
            raise_on_error: If True, raises exceptions instead of returning
                failed LaunchResult.

        Returns:
            OpenCapLaunchResult with status, log file, and loaded session.
        """
        # ADR-0053 §2: Commercial default detector check
        if config.detector.lower() == "openpose" and not config.allow_non_commercial:
            raise ValueError(
                "OpenPose has a non-commercial licence (ADR-0053). "
                "Set allow_non_commercial=True to use OpenPose."
            )

        session_dir = Path(config.session_dir).expanduser().resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        trial = config.trial_name or "trial1"

        if config.dry_run:
            self._write_dry_run_artifacts(session_dir, trial)
            session = load_opencap_session(session_dir, trial=trial)
            return OpenCapLaunchResult(
                success=True,
                return_code=0,
                session_dir=session_dir,
                session=session,
            )

        try:
            cmd, used_docker = self._resolve_runner_command(config)
        except OpenCapSidecarNotFoundError as exc:
            if raise_on_error:
                raise
            return OpenCapLaunchResult(
                success=False,
                return_code=-1,
                session_dir=session_dir,
                error_message=str(exc),
            )

        log_file = self._setup_logging(session_dir)
        try:
            return_code = self._execute_subprocess(cmd, config, log_file)
            if return_code == 0:
                session = load_opencap_session(session_dir, trial=config.trial_name)
                return OpenCapLaunchResult(
                    success=True,
                    return_code=0,
                    session_dir=session_dir,
                    session=session,
                    log_file=log_file,
                    used_docker=used_docker,
                    used_real_opencap=True,
                )

            err_msg = f"opencap-core exited with code {return_code}"
            if raise_on_error:
                raise RuntimeError(err_msg)
            return OpenCapLaunchResult(
                success=False,
                return_code=return_code,
                session_dir=session_dir,
                log_file=log_file,
                error_message=err_msg,
                used_docker=used_docker,
                used_real_opencap=True,
            )

        except Exception as exc:
            if raise_on_error:
                raise
            return OpenCapLaunchResult(
                success=False,
                return_code=-1,
                session_dir=session_dir,
                log_file=log_file,
                error_message=str(exc),
                used_docker=used_docker,
            )


def run_opencap_sidecar(
    session_dir: str | Path,
    video_dir: str | Path | None = None,
    calibration_dir: str | Path | None = None,
    opencap_env: str | Path | None = None,
    detector: str = "hrnet",
    allow_non_commercial: bool = False,
    dry_run: bool = False,
    timeout_seconds: int = 3600,
) -> OpenCapLaunchResult:
    """Convenience helper to launch OpenCap sidecar runner."""
    cfg = OpenCapLaunchConfig(
        session_dir=Path(session_dir),
        video_dir=Path(video_dir) if video_dir is not None else None,
        calibration_dir=Path(calibration_dir) if calibration_dir is not None else None,
        opencap_env=Path(opencap_env) if opencap_env is not None else None,
        detector=detector,
        allow_non_commercial=allow_non_commercial,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
    )
    launcher = OpenCapLauncher()
    return launcher.launch(cfg)
