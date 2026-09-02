"""Regression tests for issue #8834.

``scripts/validate_suite.py`` validated a pre-migration directory layout
and could never pass: several validators resolved repo-root paths
(``.git``, ``.gitignore``, ``README.md``, ``docs/``) against
``get_src_root()`` (i.e. ``src/``) instead of the project root, the shared
components validator imported ``shared.python...`` instead of
``src.shared.python...``, and the launcher validator never put ``src/`` on
``sys.path`` -- so any bare ``import bunkershot3d`` transitively pulled in
by the launcher import chain failed with ``ModuleNotFoundError``.

These tests exercise the validators against the actual checked-out repo
(the script's whole purpose is a live layout/import check, so there is no
meaningful fake to substitute).
"""

from __future__ import annotations

import pytest

from scripts import validate_suite

pytestmark = pytest.mark.unit


def test_directory_structure_passes_against_current_layout() -> None:
    assert validate_suite.validate_directory_structure() is True


def test_launchers_import_successfully() -> None:
    assert validate_suite.validate_launchers() is True


def test_shared_components_import_with_src_prefix() -> None:
    assert validate_suite.validate_shared_components() is True


def test_engine_structure_passes() -> None:
    assert validate_suite.validate_engine_structure() is True


def test_git_repository_check_uses_the_project_root() -> None:
    assert validate_suite.validate_git_repository() is True


def test_configuration_files_check_uses_the_project_root() -> None:
    assert validate_suite.validate_configuration_files() is True


def test_comprehensive_validation_passes_end_to_end() -> None:
    assert validate_suite.run_comprehensive_validation() is True
