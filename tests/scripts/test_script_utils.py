import pytest

from scripts.script_utils import run_command, run_pytest


def test_run_command_invalid_args():
    # Test cmd validation
    with pytest.raises(ValueError, match="cmd must be provided"):
        run_command([])

    with pytest.raises(TypeError, match="cmd must be a list"):
        run_command("echo hello")  # type: ignore

    # Test cwd validation
    with pytest.raises(TypeError, match="cwd must be a Path or str"):
        run_command(["echo", "hello"], cwd=123)  # type: ignore


def test_run_pytest_invalid_args():
    # Test path validation
    with pytest.raises(ValueError, match="path must be provided"):
        run_pytest(path=None)  # type: ignore

    # Test verbose validation
    with pytest.raises(TypeError, match="verbose must be a bool"):
        run_pytest(verbose="yes")  # type: ignore

    # Test markers validation
    with pytest.raises(TypeError, match="markers must be a str"):
        run_pytest(markers=123)  # type: ignore


def test_run_command_valid(mocker):
    # Just to ensure valid calls still work
    mocker.patch("subprocess.run")
    run_command(["echo", "hello"], cwd=".")


def test_run_pytest_valid(mocker):
    mocker.patch("subprocess.run", return_value=mocker.MagicMock(returncode=0))
    assert run_pytest(path="tests", verbose=True, markers="not slow") is True
