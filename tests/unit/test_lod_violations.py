"""TDD tests for LOD (Law of Demeter) violations fixed in issue #2507.

Three violations:
1. AntagonistPair consumers call muscle_system.agonist.muscles.keys() (3-level chain).
   Fix: add AntagonistPair.muscle_names property.
2. help_system.py uses Path(__file__).parent.parent.parent.parent (4-level chain).
   Fix: import SUITE_ROOT from src.shared.python.
3. launcher_utils.py duplicates the same 4-level chain.
   Fix: import SUITE_ROOT from src.shared.python.
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestAntagonistPairMuscleNames:
    """AntagonistPair must expose muscle_names to avoid 3-level chain traversal."""

    def test_muscle_names_property_exists(self) -> None:
        """AntagonistPair must have a muscle_names property."""
        source = Path("src/shared/python/biomechanics/multi_muscle.py").read_text()
        tree = ast.parse(source)
        pair_class = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "AntagonistPair"
        )
        method_names = [
            n.name
            for n in ast.walk(pair_class)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "muscle_names" in method_names, (
            "AntagonistPair must have a muscle_names property to avoid "
            "3-level chain (muscle_system.agonist.muscles.keys())"
        )

    def test_myosuite_adapter_uses_muscle_names(self) -> None:
        """_get_muscle_names in myosuite_adapter.py must not use .agonist.muscles chain."""
        source = Path("src/shared/python/biomechanics/myosuite_adapter.py").read_text()
        lines = source.splitlines()
        violations = [
            line
            for line in lines
            if ".agonist.muscles.keys()" in line and not line.strip().startswith("#")
        ]
        assert not violations, (
            "myosuite_adapter.py still uses 3-level LOD chain .agonist.muscles.keys(). "
            "Violations:\n" + "\n".join(violations)
        )


class TestHelpSystemPathLOD:
    """help_system.py must not use 4-level Path.parent chain."""

    def test_help_system_no_four_level_parent_chain(self) -> None:
        """help_system.py must import SUITE_ROOT instead of chaining 4x .parent."""
        source = Path("src/shared/python/gui_pkg/help_system.py").read_text()
        lines = source.splitlines()
        violations = [
            line
            for line in lines
            if ".parent.parent.parent.parent" in line
            and not line.strip().startswith("#")
        ]
        assert not violations, (
            "help_system.py still uses 4-level Path.parent chain. "
            "Violations:\n" + "\n".join(violations)
        )

    def test_help_system_imports_suite_root(self) -> None:
        """help_system.py must import SUITE_ROOT from src.shared.python."""
        source = Path("src/shared/python/gui_pkg/help_system.py").read_text()
        assert "SUITE_ROOT" in source, (
            "help_system.py must import and use SUITE_ROOT from src.shared.python "
            "instead of re-computing 4-level parent chain."
        )


class TestLauncherUtilsPathLOD:
    """launcher_utils.py must not use 4-level Path.parent chain."""

    def test_launcher_utils_no_four_level_parent_chain(self) -> None:
        """launcher_utils.py must not chain 4x .parent to find repo root."""
        source = Path("src/shared/python/gui_pkg/launcher_utils.py").read_text()
        lines = source.splitlines()
        violations = [
            line
            for line in lines
            if ".parent.parent.parent.parent" in line
            and not line.strip().startswith("#")
        ]
        assert not violations, (
            "launcher_utils.py still uses 4-level Path.parent chain. "
            "Violations:\n" + "\n".join(violations)
        )

    def test_launcher_utils_imports_suite_root(self) -> None:
        """launcher_utils.py must import SUITE_ROOT from src.shared.python."""
        source = Path("src/shared/python/gui_pkg/launcher_utils.py").read_text()
        assert "SUITE_ROOT" in source, (
            "launcher_utils.py must import and use SUITE_ROOT from src.shared.python "
            "instead of re-computing 4-level parent chain."
        )
