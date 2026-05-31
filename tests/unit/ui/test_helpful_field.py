"""Unit tests for the HelpfulField PyQt6 wrapper (epic #5968).

TDD-first.  The wrapper consumes the *existing* ``FieldMetadata``
registry (DRY — no metadata logic is duplicated here) and configures a
plain numeric/enum input widget with tooltip, whatsThis and a validator
derived from the field's id.  A ``field_violated`` signal fires when the
current value breaches the declared range.

Qt safety: every test uses a *single* widget under the shared offscreen
``QApplication`` to avoid the known multi-widget Sidekick segfault.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from src.shared.python.ux.field_metadata import load_registry  # noqa: E402

pytestmark = pytest.mark.unit

_YAML = "configs/ux/field_metadata.yaml"


@pytest.fixture(scope="module")
def registry():
    return load_registry(_YAML)


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as e:  # noqa: F841
        pytest.skip(f"PyQt6 runtime unavailable: {e}")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def test_numeric_field_sets_tooltip_and_whatsthis(qt_app, registry) -> None:
    from src.shared.python.ui.helpful_field import HelpfulField

    fm = registry.get("simulation.duration")
    field = HelpfulField("simulation.duration", registry=registry)

    assert field.toolTip() == fm.short_help
    assert fm.long_help.splitlines()[0] in field.whatsThis()
    assert fm.default_source in field.whatsThis()


def test_free_form_field_uses_editable_line_edit(qt_app, registry) -> None:
    from PyQt6.QtWidgets import QLineEdit

    from src.shared.python.ui.helpful_field import HelpfulField

    field = HelpfulField("simulation.model", registry=registry)

    assert isinstance(field.editor(), QLineEdit)
    field.editor().setText("custom_golf_model")
    assert field.value() == "custom_golf_model"


def test_numeric_field_clamps_validator_to_range(qt_app, registry) -> None:
    from src.shared.python.ui.helpful_field import HelpfulField

    field = HelpfulField("simulation.duration", registry=registry)
    # valid_range for duration is [0.1, 60.0]; widget value defaults to
    # the field default and stays within range.
    assert 0.1 <= field.value() <= 60.0


def test_field_violated_fires_on_range_breach(qt_app, registry) -> None:
    from src.shared.python.ui.helpful_field import HelpfulField

    field = HelpfulField("simulation.duration", registry=registry)
    seen: list[tuple[str, float]] = []
    field.field_violated.connect(lambda fid, val: seen.append((fid, val)))

    # 1000 is well outside [0.1, 60.0].
    field.check_value(1000.0)

    assert seen == [("simulation.duration", 1000.0)]


def test_in_range_value_does_not_fire(qt_app, registry) -> None:
    from src.shared.python.ui.helpful_field import HelpfulField

    field = HelpfulField("simulation.duration", registry=registry)
    seen: list[tuple[str, float]] = []
    field.field_violated.connect(lambda fid, val: seen.append((fid, val)))

    field.check_value(3.0)

    assert seen == []


def test_unknown_field_id_raises(qt_app, registry) -> None:
    from src.shared.python.ui.helpful_field import HelpfulField

    with pytest.raises(KeyError):
        HelpfulField("does.not.exist", registry=registry)
