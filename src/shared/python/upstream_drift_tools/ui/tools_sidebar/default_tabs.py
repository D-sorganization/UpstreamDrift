"""Default Sidekick tab builders for UpstreamDrift.

This module is the canonical ``default_tabs`` surface under the
``upstream_drift_tools`` namespace.  It exposes the same builder API as
``sidekick.ui.tools_sidebar.default_tabs`` but keeps
``build_workspace_tab`` as a thin function whose body is inspectable by
the regression guard in ``tests/unit/sidekick/test_python_repl_tab_wiring.py``
(UD #5649).

Design contract:
    - ``build_workspace_tab`` **must** instantiate ``PythonReplWidget``
      (not a ``QLabel`` placeholder).  The test uses AST inspection to
      enforce this, so the call must appear in the function body here and
      not be hidden behind a helper call.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Re-export the full default-tab suite from the canonical sidekick package so
# callers that do ``from upstream_drift_tools.ui.tools_sidebar import default_tabs``
# and access ``default_tabs.build_default_tab_definitions`` (etc.) still work.
from src.shared.python.sidekick.ui.tools_sidebar.default_tabs import (  # noqa: F401
    build_calculator_plot_tab,
    build_default_tab_definitions,
    build_file_explorer_tab,
    build_function_generator_tab,
    build_jupyter_tab,
    build_notes_tab,
    build_rotation_converter_tab,
    build_terminal_tab,
    build_unit_converter_tab,
    placeholder,
    refresh_workspace_list,
    set_project_explorer_root,
)
from src.shared.python.sidekick.ui.tools_sidebar.runtime_tabs import (  # noqa: F401
    PythonReplWidget,
    build_chat_tab,
    build_calculator_tab,
    build_python_repl_tab,
)


def build_workspace_tab(sidebar: Any) -> Any:
    """Build the workspace Python REPL tab for the UpstreamDrift sidebar.

    UD #5649 — regression guard ensures this function instantiates
    ``PythonReplWidget`` rather than returning a ``QLabel`` placeholder.
    The AST-level test in ``test_python_repl_tab_wiring.py`` inspects the
    source of *this specific function*, so the ``PythonReplWidget(...)`` call
    must appear in this function body and not be delegated to a helper.

    Args:
        sidebar: The parent Sidekick sidebar widget.  Must expose
            ``registry`` and ``set_context_variable`` attributes.

    Returns:
        A :class:`PythonReplWidget` bound to the sidebar workspace registry.
    """
    from src.shared.python.sidekick.ui.tools_sidebar import design_tokens as theme

    widget = PythonReplWidget(
        registry=sidebar.registry,
        set_variable=sidebar.set_context_variable,
        parent=sidebar,
    )
    widget.setToolTip("Python REPL sharing variables with the Workspace tab.")
    try:
        widget.apply_theme(
            theme.SidekickTerminalTheme.inherited(
                getattr(sidebar, "_design_tokens", None)
            )
        )
    except Exception as exc:  # noqa: BLE001 - theme is optional
        logger.debug("Could not apply terminal theme to workspace REPL: %s", exc)
    return widget
