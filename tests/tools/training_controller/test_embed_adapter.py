"""Smoke tests for the training-controller embeddable adapter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.shared.python.launcher_embed import (
    get_embeddable_tool,
    unregister_embeddable_tool,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_adapter_imports_without_shared_python_on_path() -> None:
    """Regression for #6560.

    At launcher runtime the bootstrap does NOT add ``src/shared/python`` to
    ``sys.path`` (only ``vendor/ud-tools/src/shared/python``). The test suite
    masked the bug because ``tests/conftest.py`` injects ``src/shared/python``.
    Import the adapter in a subprocess whose path contains only the repo root
    so the backend must resolve as ``src.shared.python.training`` — a bare
    ``from training import`` would raise ``ModuleNotFoundError`` here.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import src.tools.training_controller._embed_adapter"],
        cwd=str(_REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(_REPO_ROOT),
            "QT_QPA_PLATFORM": "offscreen",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "training_controller adapter failed to import at simulated runtime:\n"
        f"{result.stderr}"
    )


def test_embed_adapter_tool_id_and_capabilities() -> None:
    from src.tools.training_controller._embed_adapter import (
        _TrainingControllerEmbedAdapter,
    )

    adapter = _TrainingControllerEmbedAdapter()
    assert adapter.tool_id == "training_controller"
    caps = adapter.embed_capabilities()
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.requires_separate_qapplication is False
    assert caps.min_size == (1024, 720)


def test_embed_adapter_create_main_widget_lazy_imports() -> None:
    from src.tools.training_controller._embed_adapter import (
        _TrainingControllerEmbedAdapter,
    )

    adapter = _TrainingControllerEmbedAdapter()
    fake_widget = MagicMock(name="training-controller-widget")
    fake_module = MagicMock()
    fake_module.build_default_controller.return_value = "controller"
    fake_module.MainWidget.return_value = fake_widget

    with patch.dict(
        sys.modules,
        {"src.tools.training_controller.gui": fake_module},
    ):
        widget = adapter.create_main_widget(parent="parent")

    assert widget is fake_widget
    fake_module.build_default_controller.assert_called_once()
    fake_module.MainWidget.assert_called_once_with("controller", parent="parent")


def test_embed_adapter_cleanup_calls_widget_cleanup() -> None:
    from src.tools.training_controller._embed_adapter import (
        _TrainingControllerEmbedAdapter,
    )

    adapter = _TrainingControllerEmbedAdapter()
    widget = MagicMock()
    adapter._widgets.append(widget)
    adapter.cleanup()
    widget.cleanup.assert_called_once()
    assert adapter._widgets == []


def test_package_registers_embed_adapter() -> None:
    import src.tools.training_controller as package

    tool = get_embeddable_tool("training_controller")
    assert tool is not None
    assert tool.tool_id == "training_controller"

    unregister_embeddable_tool("training_controller")
    package._register_embed_adapter()
    assert get_embeddable_tool("training_controller") is not None
