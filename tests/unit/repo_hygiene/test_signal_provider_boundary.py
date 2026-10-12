"""Signal Consumers Share the Pinned Provider in Every Import Order."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "first",
    [
        "signal_toolkit",
        "shared.python.signal_toolkit",
        "src.shared.python.signal_toolkit",
    ],
)
def test_signal_provider_identity_in_fresh_process(first):
    script = r"""
import importlib
import inspect
import sys
from pathlib import Path
import numpy as np

root = Path.cwd()
sys.path[:0] = [
    str(root / "vendor/ud-tools/src/shared/python"),
    str(root / "vendor/ud-tools/src/python/src"),
    str(root / "vendor/ud-tools/src"),
    str(root), str(root / "src"), str(root / "src/shared/python"),
]
first = importlib.import_module(sys.argv[1])
first_signal = first.Signal
prefixes = ("signal_toolkit", "shared.python.signal_toolkit", "src.shared.python.signal_toolkit")
packages = [importlib.import_module(prefix) for prefix in prefixes]
cores = [importlib.import_module(prefix + ".core") for prefix in prefixes]
fittings = [importlib.import_module(prefix + ".fitting") for prefix in prefixes]
ios = [importlib.import_module(prefix + ".io") for prefix in prefixes]
provider = root / "vendor/ud-tools/src/shared/python/signal_toolkit"
assert all(package is first for package in packages), "duplicate package identity"
assert all(core is cores[0] for core in cores), "duplicate core identity"
assert all(package.Signal is first_signal for package in packages), "duplicate public Signal"
assert all(core.Signal is first_signal for core in cores), "duplicate core Signal"
assert all(fitting.Signal is first_signal for fitting in fittings), "duplicate fitting Signal"
assert all(io.Signal is first_signal for io in ios), "duplicate I/O Signal"
for name, module in tuple(sys.modules.items()):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
        held_signal = vars(module).get("Signal") if module else None
        if held_signal is not None:
            assert held_signal is first_signal, (name, "retained duplicate Signal")
assert Path(inspect.getfile(first_signal)).resolve().parent == provider
signal = first_signal(time=np.arange(3.0), values=np.array([1.0, 2.0, 3.0]))
assert isinstance(signal, ios[0].Signal)
extension = importlib.import_module("src.shared.python.signal_toolkit.signal_processing")
assert Path(extension.__file__).resolve() == root / "src/shared/python/signal_toolkit/signal_processing.py"
assert extension is importlib.import_module("shared.python.signal_toolkit.signal_processing")
"""
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "MPLBACKEND": "Agg",
        "MUJOCO_GL": "egl",
        "SDL_VIDEODRIVER": "dummy",
        "OPENBLAS_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, first],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
