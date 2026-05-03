from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from src.engines.physics_engines.openfoam import (
    OpenFoamDecompositionConfig,
    OpenFoamExecutionConfig,
    OpenFoamExecutionEngine,
)


def _completed(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_sequential_openfoam_run_uses_solver_without_mpi(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    case_dir = tmp_path / "case"

    def runner(
        cmd: list[str], cwd: Path | str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert cwd == case_dir
        assert timeout == 30.0
        return _completed(cmd)

    config = OpenFoamExecutionConfig(case_dir=case_dir)
    engine = OpenFoamExecutionEngine(config, runner=runner)

    result = engine.run()

    assert result.args == ["buoyantPimpleFoam", "-case", str(case_dir)]
    assert calls == [["buoyantPimpleFoam", "-case", str(case_dir)]]
    assert not (case_dir / "system" / "decomposeParDict").exists()


def test_mpi_openfoam_run_writes_decompose_dict_and_uses_mpirun(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    case_dir = tmp_path / "case"

    def runner(
        cmd: list[str], cwd: Path | str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert cwd == case_dir
        assert timeout == 120.0
        return _completed(cmd)

    config = OpenFoamExecutionConfig(
        case_dir=case_dir,
        processes=4,
        timeout_seconds=120.0,
        decomposition=OpenFoamDecompositionConfig(
            number_of_subdomains=4,
            method="scotch",
        ),
    )
    engine = OpenFoamExecutionEngine(config, runner=runner)

    result = engine.run()

    decompose_dict = case_dir / "system" / "decomposeParDict"
    assert "numberOfSubdomains 4;" in decompose_dict.read_text(encoding="utf-8")
    assert "method          scotch;" in decompose_dict.read_text(encoding="utf-8")
    assert calls == [
        ["decomposePar", "-force", "-case", str(case_dir)],
        [
            "mpirun",
            "-np",
            "4",
            "buoyantPimpleFoam",
            "-parallel",
            "-case",
            str(case_dir),
        ],
    ]
    assert result.args == calls[-1]


@pytest.mark.parametrize("processes", [0, -1])
def test_execution_config_rejects_non_positive_processes(
    tmp_path: Path, processes: int
) -> None:
    with pytest.raises(ValueError, match="processes must be positive"):
        OpenFoamExecutionConfig(case_dir=tmp_path / "case", processes=processes)


def test_execution_config_rejects_mismatched_decomposition(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must match processes"):
        OpenFoamExecutionConfig(
            case_dir=tmp_path / "case",
            processes=4,
            decomposition=OpenFoamDecompositionConfig(number_of_subdomains=2),
        )


def test_decomposition_config_rejects_unsafe_method() -> None:
    with pytest.raises(ValueError, match="method contains unsupported characters"):
        OpenFoamDecompositionConfig(number_of_subdomains=2, method="scotch;rm")
