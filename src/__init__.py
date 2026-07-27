"""Golf Modeling Suite source package."""

import sys
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

_CANONICAL_ALIAS_MODULES = frozenset(
    {
        "shared",
        "shared.python",
        "shared.python.import_aliases",
    }
)


def _restore_import_state(
    previous_modules: Mapping[str, ModuleType],
    previous_meta_path: Sequence[Any],
) -> None:
    """Restore the interpreter state captured before an alias attempt."""
    for name in tuple(sys.modules):
        if name not in previous_modules:
            sys.modules.pop(name, None)
    sys.modules.update(previous_modules)
    sys.meta_path[:] = previous_meta_path


def _install_parent_shared_aliases() -> bool:
    """Atomically install Tools-owned aliases when their module is available."""
    previous_modules = dict(sys.modules)
    previous_meta_path = list(sys.meta_path)
    try:
        from shared.python.import_aliases import install_shared_import_aliases
    except ModuleNotFoundError as exc:
        _restore_import_state(previous_modules, previous_meta_path)
        if exc.name not in _CANONICAL_ALIAS_MODULES:
            raise
        return False
    except Exception:
        _restore_import_state(previous_modules, previous_meta_path)
        raise

    try:
        install_shared_import_aliases()
    except Exception:
        _restore_import_state(previous_modules, previous_meta_path)
        raise
    return True


_PARENT_SHARED_ALIASES_INSTALLED = _install_parent_shared_aliases()
