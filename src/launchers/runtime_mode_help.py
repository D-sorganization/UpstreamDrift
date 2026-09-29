"""Shared help text and ``?`` button factory for the engine-runtime selector.

The launcher exposes the engine-runtime selector in two places:

  * The compact ``Runtime:`` label in the top bar.
  * The ``Engine Runtime`` group in Settings → Configuration.

Both surfaces should describe the runtimes with the same words. This
module centralises the help text plus a tiny factory that builds a small
``?`` button which pops a single information dialog. Importers can place
the button next to a label, beside a checkbox, or on a group header
without needing to repeat the explanatory copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QToolButton

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


# Authoritative, source-of-truth explanation. Rendered as Qt rich text
# (a constrained HTML subset). Keep paragraphs short — Qt's default
# QMessageBox sizing wraps unhelpfully on long lines.
RUNTIME_MODE_HELP_HTML = """
<h3>Where do physics engines run?</h3>

<p>The launcher can execute engines (MuJoCo, Drake, Pinocchio, OpenSim,
MyoSuite) in three different runtimes. Pick one based on what your
machine has installed and what kind of isolation you want.</p>

<h4>Native Windows <i>(default)</i></h4>
<p>Engines run directly in the launcher's own Python interpreter on
Windows. Fastest start-up and easiest debugging, but only the engines
with native Windows wheels work end-to-end (MuJoCo, MyoSuite, the
matlab-driven Simscape models). Drake and Pinocchio do not ship Windows
wheels and will fall back to limited or stub modes.</p>

<h4>Docker container <i>(Linux, sandboxed)</i></h4>
<p>Engines run inside a Linux container built from
<code>src/engines/physics_engines/mujoco/Dockerfile</code>. Full Drake
and Pinocchio support, identical environment across machines, and
your host Python stays untouched. Requires Docker (Engine + CLI), and
the <code>upstream-drift:engine</code> image to be built first via
<i>Settings → Configuration → Docker Image → Build Image</i>.</p>

<h4>WSL2 Ubuntu <i>(Linux, native filesystem)</i></h4>
<p>Engines run in your WSL2 Ubuntu user environment — same Linux wheels
as the Docker mode but without the container layer, so you get faster
file I/O against your repo and easier interactive debugging from a WSL
shell. Requires WSL2 enabled and the engine Python deps installed in
your WSL distro.</p>

<h4>How to switch</h4>
<p>Tick <b>Docker</b> or <b>WSL2</b> in the <i>Engine Runtime</i> group
in Settings → Configuration (or the matching checkboxes wired into the
top bar). Unticking both falls back to Native Windows. The current
choice is always shown in the top-bar <code>Runtime:</code> label.</p>

<p><i>Note:</i> the runtime choice is independent of building the
Docker image. You can build the image without ever switching to Docker
mode (the image just sits in your local image store, ready to use).</p>
"""


def show_runtime_mode_help(parent: QWidget | None = None) -> None:
    """Open the runtime-mode help dialog.

    Args:
        parent: Optional parent widget so the dialog inherits modality
            and centres over the calling window.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Engine Runtime — what does this control?")
    box.setIcon(QMessageBox.Icon.Information)
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setText(RUNTIME_MODE_HELP_HTML)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def make_runtime_mode_help_button(parent: QWidget | None = None) -> QToolButton:
    """Create a small inline ``?`` button that opens the runtime-mode help.

    Styled as a flat tool button so it can sit next to a label or
    checkbox without dominating the layout. Caller is responsible for
    adding it to its parent's layout.

    Args:
        parent: Optional parent widget.

    Returns:
        Configured QToolButton; click it to show :func:`show_runtime_mode_help`.
    """
    from src.shared.python.ui.info_button import make_info_button

    return make_info_button(
        parent,
        tooltip=("What's the difference between Native, Docker, and WSL2 runtimes?"),
        accessible_name="Engine runtime help",
        on_click=lambda: show_runtime_mode_help(parent),
    )


__all__ = [
    "RUNTIME_MODE_HELP_HTML",
    "make_runtime_mode_help_button",
    "show_runtime_mode_help",
]
