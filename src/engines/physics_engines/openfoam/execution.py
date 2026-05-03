"""OpenFOAM case execution and MPI command plumbing."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    object      decomposeParDict;\n"
            "}\n"
            "\n"
            f"numberOfSubdomains {self.number_of_subdomains};\n"
            f"method          {self.method};\n"
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

    def write_decompose_par_dict(self) -> Path:
        """Write `system/decomposeParDict` for the configured case."""
        decomposition = self._config.decomposition
        if decomposition is None:
            raise ValueError("decomposition is required to write decomposeParDict")
        system_dir = self._config.case_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        output_path = system_dir / "decomposeParDict"
        output_path.write_text(
            decomposition.render_decompose_par_dict(),
            encoding="utf-8",
        )
        return output_path

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the configured OpenFOAM case and return the solver result."""
        if self._config.processes > 1 and self._config.decompose_before_run:
            self.write_decompose_par_dict()
            self._runner(
                ["decomposePar", "-force", "-case", str(self._config.case_dir)],
                self._config.case_dir,
                self._config.timeout_seconds,
            )
        return self._runner(
            self._config.build_solver_command(),
            self._config.case_dir,
            self._config.timeout_seconds,
        )
