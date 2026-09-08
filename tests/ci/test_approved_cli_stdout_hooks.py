"""Keep the print hook consistent with explicitly approved CLI stdout policy."""

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _print_hook() -> dict[str, str]:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    return next(
        hook
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "no-print-in-src"
    )


def test_declared_runnable_example_stdout_survives_normal_print_hook() -> None:
    path = "src/shared/python/optimization/examples/bioptim_tracking_example.py"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    ignores = project["tool"]["ruff"]["lint"]["per-file-ignores"]
    assert any(
        fnmatch.fnmatchcase(path, key) and "T201" in rules
        for key, rules in ignores.items()
    )
    source = (ROOT / path).read_text()
    assert 'if __name__ == "__main__":' in source
    assert re.search(_print_hook()["exclude"], path), "approved CLI stdout must pass"


@pytest.mark.parametrize(
    "path",
    [
        "src/shared/python/optimization/ocp/swing_ocp.py",
        "src/shared/python/optimization/examples_extra/library.py",
        "src/shared/python/engine_core/library.py",
        "src/unapproved/examples/library.py",
    ],
)
def test_ordinary_libraries_remain_subject_to_the_print_prohibition(path: str) -> None:
    hook = _print_hook()
    assert re.search(hook["files"], path)
    assert not re.search(hook["exclude"], path)
    assert re.search(hook["entry"], '    print("diagnostic")')
