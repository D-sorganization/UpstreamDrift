import sys
from pathlib import Path

here = Path(__file__).resolve()
repo_root = here.parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# 1. Import src.shared first
import src.shared.python.logging_pkg.logger_utils as logger_utils
print("Imported src.shared initially")

engine_python = (
    repo_root
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
)
engine_src = engine_python / "src"

# 2. Run the pivot code from conftest.py
# Drop the repo's ``src`` package so we can rebind it to the engine's.
keep_prefix = "src.shared."
for modname in list(sys.modules):
    if modname == "src" or modname.startswith("src."):
        if modname.startswith(keep_prefix):
            continue
        del sys.modules[modname]

# Bind ``src`` directly to the engine's package via importlib.util
import importlib.util as _util

spec = _util.spec_from_file_location(
    "src",
    str(engine_src / "__init__.py"),
    submodule_search_locations=[str(engine_src)],
)
if spec is not None and spec.loader is not None:
    src_mod = _util.module_from_spec(spec)
    sys.modules["src"] = src_mod
    spec.loader.exec_module(src_mod)
    repo_src = repo_root / "src"
    src_mod.__path__ = [str(engine_src), str(repo_src)]

# Add ``<repo>/src`` for bare ``shared.python.*`` imports
repo_src_str = str(repo_src)
if repo_src_str not in sys.path:
    sys.path.append(repo_src_str)

# Try importing a different src.shared submodule
try:
    import src.shared.python.data_io.export
    print("Success importing src.shared submodule after pivot!")
except Exception as e:
    import traceback
    traceback.print_exc()
