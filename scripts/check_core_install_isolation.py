"""Verify a core-only install does not expose optional engine packages."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_ROOT]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

FORBIDDEN_OPTIONAL_MODULES = ("drake", "pinocchio", "opensim", "myosuite", "jaxsim")
CORE_IMPORT_MODULES = (
    "src.api",
    "src.engines.physics_engines.mujoco",
    "src.shared.python.physics",
    "src.shared.python.spatial_algebra",
)


def _module_is_loaded(module_name: str) -> bool:
    return any(
        loaded_name == module_name or loaded_name.startswith(f"{module_name}.")
        for loaded_name in sys.modules
    )


def _import_core_modules() -> None:
    for module_name in CORE_IMPORT_MODULES:
        importlib.import_module(module_name)


def find_import_isolation_violations(*, import_core_modules: bool = True) -> list[str]:
    """Return core-only install/import isolation violations.

    Postcondition: an empty list means optional engine packages were neither
    importable nor loaded after importing the core public surface.
    """
    violations: list[str] = []

    for module_name in FORBIDDEN_OPTIONAL_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            violations.append(f"{module_name} is importable in a core-only environment")

    if import_core_modules:
        _import_core_modules()

    for module_name in FORBIDDEN_OPTIONAL_MODULES:
        if _module_is_loaded(module_name):
            violations.append(f"{module_name} is loaded after core imports")

    return violations


def main() -> int:
    violations = find_import_isolation_violations()
    if violations:
        for violation in violations:
            print(f"ERROR: {violation}")
        return 1

    print("Core-only install/import isolation is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
