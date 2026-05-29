import sys
import contextlib
from pathlib import Path

repo_root = Path(r"C:\Users\diete\Repositories\UpstreamDrift")
engine_python = (
    repo_root
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
)
engine_src = engine_python / "src"

# Simulate tests/conftest.py setup
import types

_data_io_name = "src.shared.python.data_io"
_data_io_mod = types.ModuleType(_data_io_name)
_data_io_mod.__path__ = ["src/shared/python/data_io"]
_data_io_mod.__package__ = _data_io_name
sys.modules[_data_io_name] = _data_io_mod


# Now run pivot
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
import importlib

for qual in (
    "sidekick.lab.bio.c3d_reader",
    "src.shared.python.qt_utils.wheel_event_filter",
    "src.shared.python.motion_matching.body_skeleton",
):
    with contextlib.suppress(ImportError):
        sys.modules[qual] = importlib.import_module(qual)

keep_prefix = "src.shared."
for modname in list(sys.modules):
    if modname == "src" or modname.startswith("src."):
        if modname.startswith(keep_prefix):
            continue
        del sys.modules[modname]

import importlib.util as _util

spec = _util.spec_from_file_location(
    "src",
    str(engine_src / "__init__.py"),
    submodule_search_locations=[str(engine_src)],
)
src_mod = _util.module_from_spec(spec)
sys.modules["src"] = src_mod
spec.loader.exec_module(src_mod)
repo_src = repo_root / "src"
src_mod.__path__ = [str(engine_src), str(repo_src)]

# Try to import export module
try:
    pass

except Exception:  # noqa: BLE001 - debug script reports any failure
    import traceback

    traceback.print_exc()

# Remove pre-cached namespaces
for name in list(sys.modules):
    if name.startswith("src.shared"):
        del sys.modules[name]

try:
    # Python caches module lookup failures in some contexts. Let's import src.shared first.

    pass

except Exception:  # noqa: BLE001 - debug script reports any failure
    import traceback

    traceback.print_exc()
