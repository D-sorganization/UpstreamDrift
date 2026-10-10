"""The pinned-Tools package loader stays outside the shared namespaces (#11994)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.shared.python._seam_redirect import (
    SeamResolutionError,
    load_pinned_tools_package,
)

pytestmark = pytest.mark.unit


def test_loads_the_pinned_package_under_a_private_name() -> None:
    from sidekick import lab

    before = list(lab.__path__)
    public_spec_before = importlib.util.find_spec("sidekick.lab.mocap")
    public_module_before = sys.modules.get("sidekick.lab.mocap")
    mocap = load_pinned_tools_package("sidekick.lab.mocap")

    assert mocap.__name__.startswith("_pinned_tools__")
    assert hasattr(mocap, "ExperimentReplayBundle")
    assert list(lab.__path__) == before
    assert sys.modules.get("sidekick.lab.mocap") is public_module_before
    public_spec_after = importlib.util.find_spec("sidekick.lab.mocap")
    assert (public_spec_after is None) == (public_spec_before is None)
    if public_spec_before is not None:
        assert public_spec_after is not None
        assert public_spec_after.origin == public_spec_before.origin
        assert type(public_spec_after.loader) is type(public_spec_before.loader)


@pytest.mark.parametrize("expose_tools", [False, True])
def test_private_loading_preserves_fresh_process_public_resolution(
    expose_tools: bool, tmp_path: Path
) -> None:
    """Explicit Tools roots can expose aliases before the private loader runs."""
    root = Path(__file__).resolve().parents[3]
    paths = [str(root), str(root / "src")]
    if expose_tools:
        paths.extend(
            str(root / "vendor" / "ud-tools" / suffix)
            for suffix in ("src/shared/python", "src")
        )
    else:
        paths.append(str(root / "src/shared/python"))
    script = """
import importlib.util
import sys
from sidekick import lab
from src.shared.python._seam_redirect import load_pinned_tools_package

def resolution():
    spec = importlib.util.find_spec("sidekick.lab.mocap")
    if spec is None:
        return None
    return (spec.origin, tuple(spec.submodule_search_locations or ()),
            type(spec.loader).__module__, type(spec.loader).__qualname__)

before = resolution()
if sys.argv[1] == "True":
    assert before is not None, "explicit Tools roots should expose the Tools alias"
paths = tuple(lab.__path__)
modules = {key: value for key, value in sys.modules.items()
           if key == "sidekick" or key.startswith("sidekick.")}
private = load_pinned_tools_package("sidekick.lab.mocap")
assert load_pinned_tools_package("sidekick.lab.mocap") is private
assert private.ExperimentReplayBundle.__module__.startswith("_pinned_tools__")
assert resolution() == before
assert tuple(lab.__path__) == paths
assert {key: value for key, value in sys.modules.items()
        if key == "sidekick" or key.startswith("sidekick.")} == modules
"""
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(expose_tools)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_repeated_loads_return_one_module_object() -> None:
    first = load_pinned_tools_package("sidekick.lab.mocap")
    assert load_pinned_tools_package("sidekick.lab.mocap") is first
    assert sys.modules[first.__name__] is first


@pytest.mark.parametrize("relative", ["", "sidekick..lab", "sidekick/lab", "1abc"])
def test_rejects_a_malformed_package_path(relative: str) -> None:
    with pytest.raises(ValueError, match="dotted package path"):
        load_pinned_tools_package(relative)


def test_a_package_missing_from_the_pinned_tree_fails_loudly() -> None:
    with pytest.raises(SeamResolutionError, match="not in the pinned Tools tree"):
        load_pinned_tools_package("sidekick.lab.no_such_package")
