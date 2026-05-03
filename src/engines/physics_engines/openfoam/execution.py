"""OpenFOAM case execution and MPI command plumbing."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jinja2

OPENFOAM_DECOMPOSITION_METHODS = frozenset(
    {"scotch", "simple", "hierarchical", "manual", "metis", "kahip", "structured"}
)
OPENFOAM_DEFAULT_SOLVER = "buoyantPimpleFoam"
OPENFOAM_DEFAULT_MPI_EXECUTABLE = "mpirun"
OPENFOAM_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+-"
)

OpenFoamRunner = Callable[
    [list[str], Path | str | None, float], subprocess.CompletedProcess[str]
]
_ASSETS_DIR = Path(__file__).with_name("assets")


def _template_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(_ASSETS_DIR),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_command_token(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    token = value.strip()
    if not token:
        raise ValueError(f"{field_name} must not be empty")
    if any(separator in token for separator in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must be an executable name, not a path")
    if any(character.isspace() for character in token):
        raise ValueError(f"{field_name} must not contain whitespace")
    if any(character not in OPENFOAM_TOKEN_CHARS for character in token):
        raise ValueError(f"{field_name} contains unsupported characters")
    return token


def _atomic_write_text(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_openfoam_command(
    cmd: list[str], cwd: Path | str | None, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        timeout=timeout,
        check=True,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class OpenFoamDecompositionConfig:
    """Validated OpenFOAM domain decomposition settings.

    Postcondition: instances render to a complete `decomposeParDict` body.
    """

    number_of_subdomains: int
    method: str = "scotch"

    def __post_init__(self) -> None:
        if not isinstance(self.number_of_subdomains, int):
            raise TypeError("number_of_subdomains must be an integer")
        if self.number_of_subdomains < 1:
            raise ValueError("number_of_subdomains must be positive")
        method = _validate_command_token(self.method, "method")
        if method not in OPENFOAM_DECOMPOSITION_METHODS:
            raise ValueError(f"Unsupported OpenFOAM decomposition method: {method}")
        object.__setattr__(self, "method", method)

    def render_decompose_par_dict(self) -> str:
        """Return a deterministic OpenFOAM `decomposeParDict` document."""
        return (
            _template_env()
            .get_template("decompose_par_dict.j2")
            .render(
                number_of_subdomains=self.number_of_subdomains,
                method=self.method,
            )
        )


@dataclass(frozen=True)
class OpenFoamExecutionConfig:
    """Validated OpenFOAM execution settings.

    Postcondition: `build_solver_command` can produce a shell-free argv list for
    either sequential OpenFOAM execution or MPI-backed parallel execution.
    """

    case_dir: Path
    solver: str = OPENFOAM_DEFAULT_SOLVER
    processes: int = 1
    decomposition: OpenFoamDecompositionConfig | None = None
    mpi_executable: str = OPENFOAM_DEFAULT_MPI_EXECUTABLE
    decompose_before_run: bool = True
    timeout_seconds: float = 30.0
    artifact_dir: Path | None = None
    case_lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        case_dir = Path(self.case_dir)
        if not isinstance(self.processes, int):
            raise TypeError("processes must be an integer")
        if self.processes < 1:
            raise ValueError("processes must be positive")
        if not isinstance(self.decompose_before_run, bool):
            raise TypeError("decompose_before_run must be a boolean")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.case_lock_timeout_seconds <= 0:
            raise ValueError("case_lock_timeout_seconds must be positive")

        solver = _validate_command_token(self.solver, "solver")
        mpi_executable = _validate_command_token(self.mpi_executable, "mpi_executable")
        decomposition = self.decomposition
        if decomposition is None and self.processes > 1:
            decomposition = OpenFoamDecompositionConfig(
                number_of_subdomains=self.processes
            )
        if decomposition is not None:
            self._validate_decomposition(decomposition)

        object.__setattr__(self, "case_dir", case_dir)
        if self.artifact_dir is not None:
            object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "mpi_executable", mpi_executable)
        object.__setattr__(self, "decomposition", decomposition)

    def _validate_decomposition(
        self, decomposition: OpenFoamDecompositionConfig
    ) -> None:
        if not isinstance(decomposition, OpenFoamDecompositionConfig):
            raise TypeError("decomposition must be an OpenFoamDecompositionConfig")
        if decomposition.number_of_subdomains != self.processes:
            raise ValueError("decomposition number_of_subdomains must match processes")

    def build_solver_command(self) -> list[str]:
        """Build the solver command argv without invoking a shell."""
        if self.processes == 1:
            return [self.solver, "-case", str(self.case_dir)]
        return [
            self.mpi_executable,
            "-np",
            str(self.processes),
            self.solver,
            "-parallel",
            "-case",
            str(self.case_dir),
        ]


@dataclass(frozen=True)
class OpenFoamRunContext:
    """Durable OpenFOAM run provenance.

    Postcondition: `artifact_dir` is unique to `run_id` and ready for command
    metadata and output artifacts.
    """

    run_id: str
    artifact_dir: Path

    @classmethod
    def create(cls, config: OpenFoamExecutionConfig) -> OpenFoamRunContext:
        run_id = uuid.uuid4().hex
        artifact_root = config.artifact_dir
        if artifact_root is None:
            artifact_root = (
                config.case_dir / ".upstreamdrift" / "openfoam-runs" / run_id
            )
        else:
            artifact_root = artifact_root / run_id
        artifact_root.mkdir(parents=True, exist_ok=False)
        return cls(run_id=run_id, artifact_dir=artifact_root)


class _OpenFoamCaseLock:
    def __init__(self, config: OpenFoamExecutionConfig, run_id: str) -> None:
        self._config = config
        self._run_id = run_id
        self._lock_path = (
            config.case_dir / ".upstreamdrift" / "locks" / "openfoam-case.lock"
        )
        self._fd: int | None = None

    def __enter__(self) -> _OpenFoamCaseLock:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._config.case_lock_timeout_seconds
        while True:
            try:
                self._fd = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(self._fd, self._run_id.encode("utf-8"))
                return self
            except FileExistsError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring OpenFOAM case lock: {self._lock_path}"
                    ) from error
                time.sleep(0.02)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._lock_path.unlink(missing_ok=True)


class OpenFoamExecutionEngine:
    """Execute OpenFOAM cases sequentially or through MPI.

    Postcondition: `run` returns the solver `CompletedProcess`; MPI runs create
    `system/decomposeParDict` and invoke `decomposePar` before the solver when
    `decompose_before_run` is enabled.
    """

    def __init__(
        self,
        config: OpenFoamExecutionConfig,
        runner: OpenFoamRunner | None = None,
    ) -> None:
        if not isinstance(config, OpenFoamExecutionConfig):
            raise TypeError("config must be an OpenFoamExecutionConfig")
        self._config = config
        self._runner = runner or _run_openfoam_command
        self._run_context = OpenFoamRunContext.create(config)

    @property
    def run_context(self) -> OpenFoamRunContext:
        """Return the durable context for this engine instance."""
        return self._run_context

    def write_decompose_par_dict(self) -> Path:
        """Write `system/decomposeParDict` for the configured case."""
        decomposition = self._config.decomposition
        if decomposition is None:
            raise ValueError("decomposition is required to write decomposeParDict")
        system_dir = self._config.case_dir / "system"
        output_path = system_dir / "decomposeParDict"
        _atomic_write_text(
            output_path,
            decomposition.render_decompose_par_dict(),
        )
        return output_path

    def _run_logged_command(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        sequence = len(list(self._run_context.artifact_dir.glob("command-*.json"))) + 1
        started_at = _utc_now_iso()
        stdout_path = self._run_context.artifact_dir / f"command-{sequence:02d}.stdout"
        stderr_path = self._run_context.artifact_dir / f"command-{sequence:02d}.stderr"
        metadata_path = self._run_context.artifact_dir / f"command-{sequence:02d}.json"
        metadata: dict[str, object] = {
            "run_id": self._run_context.run_id,
            "argv": cmd,
            "cwd": str(self._config.case_dir),
            "timeout_seconds": self._config.timeout_seconds,
            "started_at": started_at,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        try:
            result = self._runner(
                cmd,
                self._config.case_dir,
                self._config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            metadata.update(
                {
                    "completed_at": _utc_now_iso(),
                    "status": "timeout",
                    "returncode": None,
                    "exception_type": type(error).__name__,
                }
            )
            stdout_path.write_text(str(error.output or ""), encoding="utf-8")
            stderr_path.write_text(str(error.stderr or ""), encoding="utf-8")
            _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
            raise
        except subprocess.CalledProcessError as error:
            metadata.update(
                {
                    "completed_at": _utc_now_iso(),
                    "status": "failed",
                    "returncode": error.returncode,
                    "exception_type": type(error).__name__,
                }
            )
            stdout_path.write_text(str(error.output or ""), encoding="utf-8")
            stderr_path.write_text(str(error.stderr or ""), encoding="utf-8")
            _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
            raise
        metadata.update(
            {
                "completed_at": _utc_now_iso(),
                "status": "completed",
                "returncode": result.returncode,
            }
        )
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))
        return result

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the configured OpenFOAM case and return the solver result."""
        with _OpenFoamCaseLock(self._config, self._run_context.run_id):
            if self._config.processes > 1 and self._config.decompose_before_run:
                self.write_decompose_par_dict()
                self._run_logged_command(
                    ["decomposePar", "-force", "-case", str(self._config.case_dir)]
                )
            return self._run_logged_command(self._config.build_solver_command())
