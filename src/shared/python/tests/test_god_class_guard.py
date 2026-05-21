"""God class guard — CI enforcement for GH1692.

Verifies that known large classes in src/shared/python/ do not exceed the 35-method
ceiling. Uses AST inspection (no class instantiation), so no PyQt6 display is required.

Any class that previously exceeded the 25-method threshold is tracked here. The guard
fires if:
  - A tracked class exceeds the per-class limit listed below, OR
  - A new untracked class in a monitored file exceeds METHOD_CEILING.

Adding a new class above the ceiling requires an explicit exception in KNOWN_CLASSES
with a documented justification — this makes intentional growth visible in code review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hard ceiling for any single class in the monitored files (GH1692).
METHOD_CEILING = 35

# Repo root — two levels up from this file (tests/ -> shared/python/ -> src/ -> repo)
_SHARED_PYTHON = Path(__file__).resolve().parent.parent
_SRC = _SHARED_PYTHON.parent.parent  # src/


def _count_direct_methods(cls_node: ast.ClassDef) -> list[str]:
    """Return names of direct (non-nested) methods on a class node."""
    return [
        n.name
        for n in cls_node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _get_class_method_counts(filepath: Path) -> dict[str, int]:
    """Parse *filepath* and return {class_name: direct_method_count}."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = _count_direct_methods(node)
            counts[node.name] = len(methods)
    return counts


# ---------------------------------------------------------------------------
# Files and per-class limits
# ---------------------------------------------------------------------------

# Relative paths from _SHARED_PYTHON
_MONITORED_FILES: list[str] = [
    "upstream_drift_tools/ui/mixins/calculator_state_mixin.py",
    "upstream_drift_tools/ui/widgets/data_processor_widget.py",
    "theme/theme_manager.py",
    "signal_toolkit/widget_processing.py",
    "model_generation/editor/frankenstein_editor.py",
    "data_processing/processor.py",
    "upstream_drift_tools/data_processing/core.py",
]

# Classes with documented per-class limits (may be higher than the default ceiling
# only when justified). All other classes must stay under METHOD_CEILING.
KNOWN_CLASSES: dict[str, int] = {
    # CalculatorStateMixin refactored (GH1692): 16 direct methods after sub-mixin
    # extraction. Ceiling kept at 20 to allow minor growth without re-review.
    "CalculatorStateMixin": 20,
    # _SplitterStateMixin — private sub-mixin, tightly scoped (5 methods).
    "_SplitterStateMixin": 10,
    # _ClipboardMixin — private sub-mixin for copy/paste (14 methods).
    "_ClipboardMixin": 18,
    # DataProcessorWidget: 29 methods (GUI class — PyQt6 display required for refactor,
    # deferred). Current count documented; ceiling set at 30 to flag any growth.
    "DataProcessorWidget": 30,
    # ThemeManager: 27 methods, well-structured with clear sections. Ceiling at 30.
    "ThemeManager": 30,
    # ProcessingMixin: 26 methods (signal-processing mixin, GUI display required).
    "ProcessingMixin": 30,
    # FrankensteinEditor: 25 methods (model-edit operations, bounded domain).
    "FrankensteinEditor": 30,
    # DataProcessor: 25 methods (facade class, 5 clear operation groups).
    "DataProcessor": 30,
    # DataProcessorEngine: 25 methods (core engine, 5 clear operation groups).
    "DataProcessorEngine": 30,
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.headless_safe
def test_no_god_classes_in_monitored_files() -> None:
    """Assert that no monitored class exceeds its allowed method ceiling.

    Any class exceeding the ceiling must be refactored or explicitly added to
    KNOWN_CLASSES with a justification comment (GH1692 policy).
    """
    violations: list[str] = []

    for rel_path in _MONITORED_FILES:
        filepath = _SHARED_PYTHON / rel_path
        if not filepath.exists():
            # File may have been removed in a future refactor — that's fine.
            continue

        counts = _get_class_method_counts(filepath)
        for class_name, method_count in counts.items():
            limit = KNOWN_CLASSES.get(class_name, METHOD_CEILING)
            if method_count > limit:
                violations.append(
                    f"{rel_path}::{class_name} — {method_count} methods "
                    f"(limit {limit}). "
                    "Refactor or add to KNOWN_CLASSES with justification (GH1692)."
                )

    assert not violations, (
        "God class ceiling exceeded in monitored files:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


@pytest.mark.unit
@pytest.mark.headless_safe
def test_calculator_state_mixin_reduced() -> None:
    """CalculatorStateMixin must have <= 20 direct methods after GH1692 refactor.

    This is a regression guard: the class was reduced from 35 to 16 methods by
    extracting _SplitterStateMixin and _ClipboardMixin sub-mixins.
    """
    filepath = (
        _SHARED_PYTHON / "upstream_drift_tools/ui/mixins/calculator_state_mixin.py"
    )
    counts = _get_class_method_counts(filepath)

    mixin_count = counts.get("CalculatorStateMixin", 0)
    assert mixin_count <= 20, (
        f"CalculatorStateMixin has {mixin_count} methods (max 20). "
        "Sub-mixin extraction was reverted or new methods were added without review."
    )

    # Also verify sub-mixins exist and are bounded
    assert "_SplitterStateMixin" in counts, (
        "_SplitterStateMixin sub-mixin missing from calculator_state_mixin.py"
    )
    assert "_ClipboardMixin" in counts, (
        "_ClipboardMixin sub-mixin missing from calculator_state_mixin.py"
    )
