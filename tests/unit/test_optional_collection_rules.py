"""Meta-tests for the conftest optional-collection allowlist.

``tests/conftest.py`` can silently drop whole test files via
``pytest_ignore_collect`` when a named module is unavailable. That is legitimate
for genuinely optional stacks, but three of the rules named modules that do not
exist in *any* supported configuration, so 20 tests had never run and could
never run — with no skip entry, no summary count and no CI signal (#8006).

These tests make an unsatisfiable rule fail the build instead of deleting tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.conftest import _OPTIONAL_COLLECTION_RULES

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def _module_is_resolvable(name: str) -> bool:
    """True if ``name`` can be located on the current import path."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A parent package that itself fails to import still means the guarded
        # stack is genuinely absent rather than misnamed.
        return True


def _rule_targets() -> list[tuple[str, str]]:
    """(rule path suffix, module name) for every module named by a rule."""
    targets: list[tuple[str, str]] = []
    for rule in _OPTIONAL_COLLECTION_RULES:
        label = rule.path_suffixes[0]
        targets += [(label, module) for module in rule.modules]
        targets += [(label, module) for module, _symbol in rule.symbols]
    return targets


def test_every_guarded_module_is_resolvable() -> None:
    """A guard must protect an *optional* stack, not a misspelled or dead name."""
    unresolvable = [
        (label, module)
        for label, module in _rule_targets()
        if not _module_is_resolvable(module)
    ]

    assert not unresolvable, (
        "Optional-collection rules name modules that cannot be located in this "
        "configuration. A rule like this does not tolerate a missing extra -- it "
        "permanently deletes the guarded tests with no skip entry (#8006). Fix "
        "the import or delete the rule:\n"
        + "\n".join(f"  {label}: {module}" for label, module in unresolvable)
    )


def test_every_guarded_path_exists() -> None:
    """A rule pointing at a deleted path is dead configuration."""
    missing = [
        suffix
        for rule in _OPTIONAL_COLLECTION_RULES
        for suffix in rule.path_suffixes
        if not (REPO_ROOT / suffix).exists()
    ]
    assert (
        not missing
    ), f"Optional-collection rules reference paths that no longer exist: {missing}"


@pytest.mark.parametrize(
    "path",
    [
        "tests/unit/test_start_api_server.py",
        "tests/unit/test_c3d_export_features.py",
    ],
)
def test_previously_hidden_files_are_collectable(path: str) -> None:
    """The files un-hidden by #8006 must stay importable."""
    target = REPO_ROOT / path
    assert target.is_file(), f"{path} is missing"

    suffixes = {
        suffix for rule in _OPTIONAL_COLLECTION_RULES for suffix in rule.path_suffixes
    }
    assert path not in suffixes, (
        f"{path} was re-added to the optional-collection allowlist; it imports "
        "modules that are present in the default configuration and must be "
        "collected (#8006)."
    )
