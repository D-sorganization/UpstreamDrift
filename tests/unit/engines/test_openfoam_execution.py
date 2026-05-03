from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
from src.engines.physics_engines.openfoam import (
    OpenFoamDecompositionConfig,
    OpenFoamExecutionConfig,
    OpenFoamExecutionEngine,
)
from src.engines.physics_engines.openfoam import execution as openfoam_execution


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


def test_decompose_dict_write_preserves_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case"
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True)
    decompose_dict = system_dir / "decomposeParDict"
    decompose_dict.write_text("existing valid dictionary\n", encoding="utf-8")

    def fail_replace(source: Path | str, target: Path | str) -> None:
        assert Path(source).exists()
        assert Path(target) == decompose_dict
        raise OSError("simulated replace failure")

    monkeypatch.setattr(openfoam_execution.os, "replace", fail_replace)
    config = OpenFoamExecutionConfig(case_dir=case_dir, processes=2)
    engine = OpenFoamExecutionEngine(config)

    with pytest.raises(OSError, match="simulated replace failure"):
        engine.write_decompose_par_dict()

    assert decompose_dict.read_text(encoding="utf-8") == "existing valid dictionary\n"
    assert list(system_dir.glob(".decomposeParDict.*.tmp")) == []


def test_mpi_run_persists_run_id_and_command_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    calls: list[list[str]] = []

    def runner(
        cmd: list[str], cwd: Path | str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"stdout for {cmd[0]}",
            stderr=f"stderr for {cmd[0]}",
        )

    config = OpenFoamExecutionConfig(
        case_dir=case_dir,
        processes=2,
        timeout_seconds=45.0,
    )
    engine = OpenFoamExecutionEngine(config, runner=runner)

    engine.run()

    context = engine.run_context
    assert context.run_id
    assert context.artifact_dir.is_dir()
    command_logs = sorted(context.artifact_dir.glob("command-*.json"))
    assert len(command_logs) == 2
    first_log = json.loads(command_logs[0].read_text(encoding="utf-8"))
    assert first_log["run_id"] == context.run_id
    assert first_log["argv"] == calls[0]
    assert first_log["cwd"] == str(case_dir)
    assert first_log["timeout_seconds"] == 45.0
    assert first_log["returncode"] == 0
    assert Path(first_log["stdout_path"]).read_text(encoding="utf-8") == (
        "stdout for decomposePar"
    )
    assert Path(first_log["stderr_path"]).read_text(encoding="utf-8") == (
        "stderr for decomposePar"
    )


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_returncode"),
    [
        (
            subprocess.TimeoutExpired(["decomposePar"], timeout=1.5),
            "timeout",
            None,
        ),
        (
            subprocess.CalledProcessError(
                returncode=3,
                cmd=["decomposePar"],
                output="bad stdout",
                stderr="bad stderr",
            ),
            "failed",
            3,
        ),
    ],
)
def test_run_persists_failure_and_timeout_metadata(
    tmp_path: Path,
    failure: Exception,
    expected_status: str,
    expected_returncode: int | None,
) -> None:
    case_dir = tmp_path / "case"

    def runner(
        cmd: list[str], cwd: Path | str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        raise failure

    config = OpenFoamExecutionConfig(
        case_dir=case_dir,
        processes=2,
        timeout_seconds=1.5,
    )
    engine = OpenFoamExecutionEngine(config, runner=runner)

    with pytest.raises(type(failure)):
        engine.run()

    command_logs = list(engine.run_context.artifact_dir.glob("command-*.json"))
    assert len(command_logs) == 1
    metadata = json.loads(command_logs[0].read_text(encoding="utf-8"))
    assert metadata["status"] == expected_status
    assert metadata["timeout_seconds"] == 1.5
    assert metadata["returncode"] == expected_returncode
    assert metadata["exception_type"] == type(failure).__name__


def test_same_case_runs_are_serialized_by_case_lock(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    release_first = threading.Event()
    first_started = threading.Event()
    second_seen_first_active = threading.Event()
    active_count = 0
    max_active_count = 0
    state_lock = threading.Lock()

    def runner(
        cmd: list[str], cwd: Path | str | None = None, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        nonlocal active_count, max_active_count
        with state_lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
            current_active = active_count
        if cmd[0] == "decomposePar" and current_active == 1:
            first_started.set()
            release_first.wait(timeout=5.0)
        elif cmd[0] == "decomposePar":
            second_seen_first_active.set()
        with state_lock:
            active_count -= 1
        return _completed(cmd)

    config = OpenFoamExecutionConfig(case_dir=case_dir, processes=2)
    first = OpenFoamExecutionEngine(config, runner=runner)
    second = OpenFoamExecutionEngine(config, runner=runner)
    first_thread = threading.Thread(target=first.run)
    second_thread = threading.Thread(target=second.run)

    first_thread.start()
    assert first_started.wait(timeout=5.0)
    second_thread.start()
    time.sleep(0.1)
    assert not second_seen_first_active.is_set()
    release_first.set()
    first_thread.join(timeout=5.0)
    second_thread.join(timeout=5.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert max_active_count == 1
