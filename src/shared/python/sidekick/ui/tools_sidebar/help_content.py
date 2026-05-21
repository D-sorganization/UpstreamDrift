"""Pure-Python Sidekick help metadata and action descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from .calculator_assist import CALCULATOR_HELP


@dataclass(frozen=True)
class SidebarActionMetadata:
    """Tooltip and status metadata for one reusable sidebar action."""

    label: str
    tooltip: str
    status_tip: str


def _tab_help(
    title: str,
    summary: str,
    *,
    tips: tuple[str, ...] = (),
    examples: tuple[str, ...] = (),
    source: str = "upstream_drift_tools.ui.tools_sidebar.help_content",
) -> dict[str, str]:
    metadata = {
        "title": title,
        "summary": summary,
        "source": source,
    }
    if tips:
        metadata["tips"] = "\n".join(tips)
    if examples:
        metadata["examples"] = "\n".join(examples)
    return metadata


DEFAULT_SIDEBAR_TAB_HELP: dict[str, dict[str, str]] = {
    "files": _tab_help(
        "Files",
        "Browse the current project tree and hand files back to the host app.",
        tips=(
            "Use this tab to inspect project-scoped files without leaving Sidekick.",
            "Double-clicking a file emits file_open_requested for the host.",
        ),
    ),
    "workspace": _tab_help(
        "Workspace",
        "Review the shared variables currently available to Sidekick tabs.",
        tips=(
            "Workspace values come from the host app and Sidekick runtime tabs.",
            "Use this list to confirm names, shapes, and previews before reuse.",
        ),
    ),
    "chat": _tab_help(
        "Chat",
        "Open the shared chat runtime when available for project-scoped assistance.",
        tips=(
            "PyQt hosts can embed the full shared chat dock inside this tab.",
            "Headless or reduced installs fall back to a status panel instead.",
        ),
    ),
    "terminal": _tab_help(
        "Terminal",
        "Launch a real interactive OS shell (bash, zsh, pwsh, powershell, cmd, "
        "or a WSL distro) backed by a PTY.",
        tips=(
            "The shell selector switches between every discovered shell.",
            "Install the optional ``terminal`` extra (pywinpty on Windows, "
            "ptyprocess elsewhere) for full interactive features.",
            "The live cwd label tracks the shell via OSC 7 escapes.",
        ),
        examples=(
            "ls",
            "git status",
            "python --version",
        ),
    ),
    "python_repl": _tab_help(
        "Python REPL",
        "Run bounded Python snippets against the shared workspace registry.",
        tips=(
            "Assignments export new values back into the shared workspace.",
            "Scientific helpers are preloaded when available.",
        ),
        examples=(
            "answer = 6 * 7",
            "matrix = np.array([[1, 2], [3, 4]])",
        ),
    ),
    "calculator": CALCULATOR_HELP.to_metadata(),
    "calculator_plot": _tab_help(
        "Calculator Plot",
        "Build validated plot requests from calculator expressions and "
        "workspace variables.",
        tips=(
            "Plot requests stay explicit through the shared PlotSpec contract.",
            "Use calculator or workspace values as the source for plotted series.",
        ),
        source="upstream_drift_tools.ui.tools_sidebar.calculator_plotting",
    ),
    "data_explorer": _tab_help(
        "Data Explorer",
        "Preview project data files with bounded schema, null-count, and "
        "sample-row summaries.",
        tips=(
            "Use this tab for lightweight inspection before opening heavier "
            "processing tools.",
            "Exports stay bounded to preview-safe rows and validated workspace "
            "variables.",
        ),
        examples=(
            "data/example.csv",
            "results/run_001.json",
        ),
        source="upstream_drift_tools.ui.tools_sidebar.data_explorer_service",
    ),
    "data_processor": _tab_help(
        "Data Processor",
        "Open the heavier Data Processor surface on demand and export selected "
        "results into the shared workspace.",
        tips=(
            "The tab stays optional so Sidekick startup does not depend on the "
            "full Data Processor UI stack.",
            "Workspace exports validate variable names and selected columns "
            "before mutating shared state.",
        ),
        examples=(
            "temperature",
            "temperature, pressure",
        ),
        source="upstream_drift_tools.ui.tools_sidebar.data_processor_tab",
    ),
    "units": _tab_help(
        "Units",
        "Convert values between supported engineering unit systems.",
        tips=(
            "This tab reuses the shared unit converter widget when its UI "
            "surface is available.",
        ),
    ),
    "rotation_converter": _tab_help(
        "Rotation Converter",
        "Convert between quaternions, Euler angles, matrices, and related "
        "rigid-body frames.",
        tips=("The tab stays hidden by default until the host enables it.",),
        source="rotation_converter.gui_registration",
    ),
    "function_generator": _tab_help(
        "Function Generator",
        "Generate and visualize common waveform signals inside Sidekick when "
        "the optional PyQt Function Generator stack is installed.",
        tips=(
            "The tab stays hidden by default so Sidekick startup does not import "
            "PyQt, matplotlib, numpy, or signal generation dependencies.",
            "Enable it when a workflow needs sine, square, triangle, chirp, or "
            "other generated signal previews.",
        ),
        examples=(
            "Sinusoid, 5 Hz, amplitude 2",
            "Square Wave, duty cycle 0.5",
        ),
        source="function_generator.gui_registration",
    ),
    "notes": _tab_help(
        "Notes",
        "Capture project notes with explicit save, clear, and restore controls.",
        tips=(
            "Notes persist per project root through the shared notes storage contract.",
            "Autosave keeps the latest text synchronized while preserving "
            "restore points.",
        ),
    ),
    "reporting": _tab_help(
        "Reporting",
        "Agentic reporting and summarization.",
        tips=(
            "Generate comprehensive session reports with workspace context",
            "and chat history.",
        ),
        source="upstream_drift_tools.ui.tools_sidebar.reporting_tab",
    ),
    "jupyter": _tab_help(
        "Jupyter",
        "Read-only viewer for Jupyter notebooks (.ipynb) — Phase 1 of the "
        "Sidekick Jupyter integration.",
        tips=(
            "Markdown, code, and text outputs render in-place; rich outputs "
            "(images, HTML) appear as Phase 2 placeholders.",
            "When nbformat is not installed the tab shows an actionable "
            "install hint with a copy-to-clipboard button.",
        ),
        examples=("pip install '.[jupyter]'",),
        source="upstream_drift_tools.ui.tools_sidebar.jupyter_tab",
    ),
}


SIDEBAR_CONTEXT_ACTIONS: dict[str, SidebarActionMetadata] = {
    "move_left": SidebarActionMetadata(
        label="Left",
        tooltip="Dock the entire Sidekick sidebar on the left edge of the host window.",
        status_tip="Move Sidebar to the left side of the host window.",
    ),
    "move_right": SidebarActionMetadata(
        label="Right",
        tooltip="Dock the entire Sidekick sidebar on the right edge of the "
        "host window.",
        status_tip="Move Sidebar to the right side of the host window.",
    ),
    "pop_out": SidebarActionMetadata(
        label="Pop Out",
        tooltip="Open this tab in its own window without losing its persistent tab id.",
        status_tip="Pop the active tab into a separate window.",
    ),
    "duplicate": SidebarActionMetadata(
        label="Duplicate",
        tooltip="Create another instance of this tab when the tab contract allows it.",
        status_tip="Duplicate the current tab instance.",
    ),
    "rename": SidebarActionMetadata(
        label="Rename",
        tooltip="Assign a custom display name while keeping the stable tab id "
        "unchanged.",
        status_tip="Rename the current tab display name.",
    ),
    "reset_name": SidebarActionMetadata(
        label="Reset Name",
        tooltip="Restore the original tab title and remove the custom "
        "display-name override.",
        status_tip="Reset the tab display name to its default title.",
    ),
    "help": SidebarActionMetadata(
        label="Help",
        tooltip="Open the tab-specific help summary, tips, and examples for "
        "this Sidekick surface.",
        status_tip="Show help for the current tab.",
    ),
    "close": SidebarActionMetadata(
        label="Close",
        tooltip="Hide this tab from the dock while keeping it available in "
        "sidebar state.",
        status_tip="Hide the current tab from the sidebar.",
    ),
    "minimize": SidebarActionMetadata(
        label="Minimize Sidebar",
        tooltip="Collapse the sidebar without discarding tab order or pop-out state.",
        status_tip="Minimize the sidebar while preserving its state.",
    ),
}


def render_help_markdown(metadata: dict[str, str]) -> str:
    """Render tab help metadata into a compact markdown document."""
    lines = [f"# {metadata['title']}", "", metadata["summary"]]
    tips = metadata.get("tips", "").strip()
    if tips:
        lines.extend(["", "## Tips", ""])
        lines.extend(f"- {tip}" for tip in tips.splitlines() if tip.strip())
    examples = metadata.get("examples", "").strip()
    if examples:
        lines.extend(["", "## Examples", ""])
        lines.extend(
            f"- `{example}`" for example in examples.splitlines() if example.strip()
        )
    source = metadata.get("source", "").strip()
    if source:
        lines.extend(["", f"_Source: `{source}`_"])
    return "\n".join(lines)
