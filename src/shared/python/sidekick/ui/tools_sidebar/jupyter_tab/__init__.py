"""Sidekick Jupyter notebook tab — Phase 1 + Phase 2 + Phase 3 (Tools #2875–#2877).

Phase 1 (Tools #2875): render an ``.ipynb`` notebook inside the Sidekick tab
as a read-only document with markdown, code, and raw cells.

Phase 2 (Tools #2876): session model with path validation and secure
persistence via :class:`NotebookSessionModel` and
:class:`NotebookSessionManager`.  :class:`SidekickNotebookWidget` is the
session-aware widget wrapper introduced in Phase 2.

Phase 3 (Tools #2877): workspace bridge that exports selected Sidekick
workspace variables into the kernel environment via
:class:`WorkspaceBridge`.  :meth:`SidekickNotebookWidget.update_workspace`
is the entry point for pushing variables into the kernel.
"""

from .availability import JupyterTabAvailability
from .notebook_loader import NotebookLoadError, load_notebook
from .notebook_model import (
    CellOutput,
    CodeCell,
    MarkdownCell,
    NotebookCell,
    NotebookDocument,
    RawCell,
)
from .notebook_session import NotebookSessionManager, NotebookSessionModel
from .sidekick_notebook_widget import SidekickNotebookWidget
from .unavailable_widget import JupyterUnavailableWidget
from .widget import JupyterNotebookWidget
from .workspace_bridge import WorkspaceBridge

JUPYTER_TAB_ID = "jupyter"

__all__ = [
    "JUPYTER_TAB_ID",
    "CellOutput",
    "CodeCell",
    "JupyterNotebookWidget",
    "JupyterTabAvailability",
    "JupyterUnavailableWidget",
    "MarkdownCell",
    "NotebookCell",
    "NotebookDocument",
    "NotebookLoadError",
    "NotebookSessionManager",
    "NotebookSessionModel",
    "RawCell",
    "SidekickNotebookWidget",
    "WorkspaceBridge",
    "load_notebook",
]
