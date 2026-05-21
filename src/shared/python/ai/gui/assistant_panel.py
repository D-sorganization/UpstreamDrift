"""Reusable AI assistant conversation panel.

The panel is a thin coordinator that wires together four specialised
controllers:

* :class:`PanelHeaderController` — header strip (combos + buttons).
* :class:`MessageDisplayController` — scrollable message log.
* :class:`AdapterLifecycleManager` — provider/key resolution and adapter creation.
* :class:`IndexingController` — RAG codebase indexing lifecycle.

Public attributes used by tests (``_provider_combo``, ``_model_combo``,
``_mode_combo``, ``_message_layout``, etc.) remain exposed as forwarding
references for back-compatibility.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.access_policy import (
    ChatAccessMode,
    coerce_access_mode,
    tool_declarations_for_access_mode,
)
from src.shared.python.ai.gui._adapter_lifecycle import AdapterLifecycleManager
from src.shared.python.ai.gui._indexing import IndexingController
from src.shared.python.ai.gui._input_area import InputArea
from src.shared.python.ai.gui._message_display import MessageDisplayController
from src.shared.python.ai.gui._panel_header import PanelHeaderController
from src.shared.python.ai.gui._panel_tools import register_panel_tools
from src.shared.python.ai.gui.assistant_widgets import (
    ChatInput,
    MessageWidget,
    StreamWorker,
)
from src.shared.python.ai.gui.chat_export import (
    copy_thread_to_clipboard,
    save_thread_as_markdown,
)
from src.shared.python.ai.gui.history_sidebar import ChatHistorySidebar
from src.shared.python.ai.gui.session_manager import ChatSessionManager

if TYPE_CHECKING:
    from src.shared.python.ai._settings_model import AISettings
from src.shared.python.ai.gui._provider_registry_data import (
    AIProvider,
    provider_display_name,
)
from src.shared.python.ai.mcp.gui import McpStatusIndicator
from src.shared.python.ai.memory_manager import MemoryManager
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.ai.thread_condensation import (
    condense_thread,
    estimate_token_count,
)
from src.shared.python.ai.tool_registry import get_global_registry
from src.shared.python.ai.tools.codemap_tools import register_codemap_tools
from src.shared.python.ai.tools.file_ops import register_file_tools
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import BaseAgentAdapter

from src.shared.python.ai.types import ConversationContext, ExpertiseLevel

logger = get_logger(__name__)


def _discover_project_root(start: Path) -> Path:
    """Find a nearby project root for repository instruction loading."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "AGENTS.md").is_file():
            return candidate
    return current


class AIAssistantPanel(QWidget):
    """Main AI Assistant conversation panel (coordinator)."""

    message_sent = pyqtSignal(str)
    settings_requested = pyqtSignal()
    close_requested = pyqtSignal()

    _CHAT_MODES = (
        ("Ask", "ask"),
        ("Diagnose (read-only)", "diagnose"),
        ("Agent", "agent"),
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(parent)

        # --- Domain state -------------------------------------------------
        self._context = ConversationContext()
        self._adapter: BaseAgentAdapter | None = None
        self._current_worker: StreamWorker | None = None
        self._current_assistant_message: MessageWidget | None = None
        from src.shared.python.ai._settings_model import AISettings

        self._current_settings = AISettings.load()
        self._access_mode = ChatAccessMode.NO_REPO_ACCESS
        self._rag_enabled = True
        self._auto_index_on_open = False
        self._project_root = project_root or _discover_project_root(Path.cwd())

        # --- Thread condensation state ------------------------------------
        # Raw history is kept separately so the user can undo condensation.
        self._raw_history: list[Any] = []  # original Message objects before condense
        self._is_condensed: bool = False

        # --- Tools / RAG / memory ----------------------------------------
        self._tools_registry = get_global_registry()
        self._rag_store = SimpleRAGStore()
        self._memory_manager = MemoryManager()
        self._refresh_prompt_memory()
        self._mcp_pool: Any = None  # McpClientPool — wired in after setup

        # --- Session manager + history ----------------------------------
        self._session_manager = ChatSessionManager()
        self._session_manager.session_loaded.connect(self._on_session_loaded)
        self._load_history()

        self._init_tools()

        # --- Controllers -------------------------------------------------
        self._header = PanelHeaderController(self._current_settings)
        self._messages = MessageDisplayController()
        self._adapter_mgr = AdapterLifecycleManager(self)
        self._indexer = IndexingController(self._rag_store, self)
        self._input_container = InputArea()

        self._wire_controllers()
        self._setup_ui()
        self._messages.restore_from_context(self._context)

    # ------------------------------------------------------------------
    # Back-compat attribute proxies (read by external tests & code)
    # ------------------------------------------------------------------
    @property
    def _provider_combo(self) -> Any:
        return self._header.provider_combo

    @property
    def _model_combo(self) -> Any:
        return self._header.model_combo

    @property
    def _mode_combo(self) -> Any:
        return self._header.mode_combo

    @property
    def _access_mode_combo(self) -> Any:
        return self._header.access_mode_combo

    @property
    def _provider_icon(self) -> Any:
        return self._header.provider_icon

    @property
    def _model_label(self) -> Any:
        return self._header.model_label

    @property
    def _status_label(self) -> Any:
        return self._header.status_label

    @property
    def chk_auto_index(self) -> Any:
        return self._header.auto_index_checkbox

    @property
    def _message_layout(self) -> Any:
        return self._messages.message_layout

    @property
    def _message_container(self) -> Any:
        return self._messages.message_container

    @property
    def _message_area(self) -> Any:
        return self._messages.scroll_area

    @property
    def _input_edit(self) -> Any:
        return self._input_container.input_edit

    @property
    def _send_btn(self) -> Any:
        return self._input_container.send_btn

    @property
    def _expertise_label(self) -> Any:
        return self._input_container.expertise_label

    @property
    def _indexer_worker(self) -> Any:
        return self._indexer.worker

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _wire_controllers(self) -> None:
        """Connect controller signals to coordination slots on the panel."""
        h = self._header
        h.provider_changed.connect(self._on_header_provider_changed)
        h.model_changed.connect(self._on_header_model_changed)
        h.mode_changed.connect(self._on_header_mode_changed)
        h.access_mode_changed.connect(self._on_access_mode_changed)
        h.new_chat_requested.connect(self._on_new_chat)
        h.peer_review_requested.connect(self._on_peer_review_requested)
        h.condense_requested.connect(self._on_condense_thread)
        h.show_full_history_requested.connect(self._on_show_full_history)
        h.settings_requested.connect(self._show_settings)
        h.close_requested.connect(self.close_requested.emit)
        h.copy_thread_requested.connect(self._on_copy_thread)
        h.save_thread_requested.connect(self._on_save_thread)

        self._adapter_mgr.adapter_changed.connect(self._on_adapter_changed)
        self._adapter_mgr.system_message.connect(self._add_system_message)

        self._indexer.status_changed.connect(self._set_status)
        self._indexer.system_message.connect(self._add_system_message)

        self._input_container.send_requested.connect(self._on_send)

    # ------------------------------------------------------------------
    # History / session
    # ------------------------------------------------------------------
    def _load_history(self) -> None:
        sessions = self._session_manager.list_sessions()
        active = [s for s in sessions if not s.get("archived", False)]
        if active:
            latest_id = active[0]["id"]
            loaded = self._session_manager.load_session(latest_id)
            if loaded:
                self._context = loaded
                self._refresh_prompt_memory()
                logger.info(f"Loaded chat session {latest_id}")

    def _save_history(self) -> None:
        try:
            self._refresh_prompt_memory()
            self._session_manager.save_session(self._context)
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            logger.warning(f"Failed to save chat session: {exc}")

    def _on_session_loaded(self, context: ConversationContext) -> None:
        self._context = context
        self._refresh_prompt_memory()
        self._messages.clear_messages()
        self._messages.restore_from_context(self._context)

    # ------------------------------------------------------------------
    # Tools registration
    # ------------------------------------------------------------------
    def _init_tools(self) -> None:
        register_file_tools(self._tools_registry)
        register_codemap_tools(self._tools_registry)
        register_panel_tools(self._tools_registry, self._rag_store)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #3c3c3c; }")

        self._sidebar = ChatHistorySidebar(self._session_manager)
        self._sidebar.session_selected.connect(self._session_manager.load_session)
        self._sidebar.new_chat_requested.connect(self._on_new_chat)
        self._sidebar.memory_sync_requested.connect(self._on_memory_sync_requested)
        main_splitter.addWidget(self._sidebar)

        msg_splitter = QSplitter(Qt.Orientation.Vertical)
        msg_splitter.setHandleWidth(1)
        msg_splitter.setStyleSheet("QSplitter::handle { background-color: #3c3c3c; }")
        msg_splitter.addWidget(self._messages)
        msg_splitter.addWidget(self._input_container)
        msg_splitter.setSizes([400, 100])
        msg_splitter.setStretchFactor(0, 4)
        msg_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(msg_splitter)
        main_splitter.setSizes([250, 550])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter)

        # MCP connection status indicator — lives at the bottom of the panel.
        self._mcp_indicator = McpStatusIndicator()
        layout.addWidget(self._mcp_indicator)

        self._add_system_message(
            "👋 Welcome to the shared Tools AI Assistant!\n\n"
            "I can help you:\n"
            "- Load and analyze C3D motion capture files\n"
            "- Run inverse dynamics simulations\n"
            "- Interpret joint torques and forces\n"
            "- Explain biomechanics concepts\n\n"
            "How can I help you today?"
        )

        QtCore.QTimer.singleShot(100, self._auto_load_settings)
        self.refresh_theme()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def refresh_theme(self) -> None:
        try:
            from src.shared.python.theme.theme_manager import get_theme_manager

            color_source: object = get_theme_manager().get_current_colors()

            def _get(key: str, fallback: Any) -> Any:
                if isinstance(color_source, dict):
                    return color_source.get(key, fallback)
                return getattr(color_source, key, fallback)

            colors = {
                "bg_primary": _get("bg", "#1e1e1e"),
                "bg_alt": _get("bg_elevated", _get("group_bg", "#2d2d2d")),
                "text_primary": _get("text_primary", _get("text", "#e0e0e0")),
                "text_muted": _get("text_secondary", "#888888"),
                "border": _get("border_default", _get("border", "#444444")),
                "accent": _get("primary", _get("accent", "#FF8800")),
                "button_hover": _get("bg_highlight", "#cc6d00"),
            }
        except ImportError:
            return

        self.setStyleSheet(
            f"background-color: {colors['bg_primary']}; "
            f"color: {colors['text_primary']};"
        )
        if hasattr(self, "_sidebar"):
            self._sidebar.refresh_theme()
        self._header.apply_theme(colors)
        if hasattr(self, "_input_container"):
            self._input_container.apply_theme(colors)
        self._messages.apply_theme(colors)

    # ------------------------------------------------------------------
    # showEvent / lifecycle
    # ------------------------------------------------------------------
    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._refresh_models()
        if self._auto_index_enabled():
            self._start_indexing()

    def _refresh_models(self) -> None:
        if self._adapter is not None:
            try:
                self._auto_load_settings()
            except Exception as exc:  # noqa: BLE001 - best-effort refresh
                logger.warning(f"Failed to refresh models: {exc}")

    def _auto_load_settings(self) -> None:
        try:
            from src.shared.python.ai._settings_model import AISettings

            settings = AISettings.load()
        except ImportError as exc:
            logger.warning("Failed to auto-load AI settings: %s", exc)
            return
        self.apply_settings(settings)

    # ------------------------------------------------------------------
    # Header back-compat helpers (tests touch these names)
    # ------------------------------------------------------------------
    def _sync_header_controls(self, settings: AISettings) -> None:
        self._header.sync_controls(settings)

    def _on_chat_models_refreshed(self, models: list[Any]) -> None:
        names: list[str] = []
        for entry in models:
            if isinstance(entry, str) and entry:
                names.append(entry)
            elif isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        self._header.update_models(names)

    def bind_to_chat_dock(self, chat_dock: Any) -> None:
        signal = getattr(chat_dock, "models_refreshed", None)
        if signal is None or not hasattr(signal, "connect"):
            return
        with contextlib.suppress(TypeError):
            signal.disconnect(self._on_chat_models_refreshed)
        signal.connect(self._on_chat_models_refreshed)

    # ------------------------------------------------------------------
    # Header signal handlers
    # ------------------------------------------------------------------
    def _persist_header_selection(self, *, reconnect: bool) -> None:
        provider = self._header.provider_combo.currentData()
        if not isinstance(provider, AIProvider):
            return
        self._current_settings.provider = provider
        self._current_settings.model = self._header.model_combo.currentText()
        mode = self._header.mode_combo.currentData()
        self._current_settings.chat_mode = mode if isinstance(mode, str) else "ask"
        self._current_settings.save()
        if reconnect:
            self.apply_settings(self._current_settings)

    def _on_header_provider_changed(self, _provider: AIProvider) -> None:
        self._persist_header_selection(reconnect=True)

    def _on_header_model_changed(self, _model: str) -> None:
        self._persist_header_selection(reconnect=True)

    def _on_header_mode_changed(self, _mode: str) -> None:
        self._persist_header_selection(reconnect=False)

    def _on_access_mode_changed(self, mode: ChatAccessMode) -> None:
        self._access_mode = coerce_access_mode(mode)
        try:
            settings = AISettings.load()
            settings.access_mode = self._access_mode
            settings.save()
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            logger.warning("Failed to persist chat access mode: %s", exc)

    # ------------------------------------------------------------------
    # Send / streaming
    # ------------------------------------------------------------------
    def _on_send(self, message: str | None = None) -> None:
        if message is None:
            message = self._input_edit.toPlainText().strip()
            if not message:
                return
            self._input_edit.clear()
        self._add_message("user", message)
        self._context.add_user_message(message)
        self._save_history()
        self.message_sent.emit(message)
        if self._adapter:
            self._process_message(message)

    def _process_message(self, message: str) -> None:
        if not self._adapter:
            self._add_system_message(
                "⚠️ No AI provider configured. Click ⚙️ to set up a provider."
            )
            return
        self._refresh_prompt_memory()
        # Refresh MCP tool list before each prompt so newly connected servers
        # are picked up without restarting the panel.
        self._refresh_mcp_status()
        self._set_status("Thinking...")
        self._input_container.set_busy(True)
        tools = self._build_tool_declarations()
        self._current_worker = StreamWorker(
            self._adapter, message, self._context, tools
        )
        self._current_worker.chunk_received.connect(self._on_stream_chunk)
        self._current_worker.finished.connect(self._on_stream_finished)
        self._current_worker.error.connect(self._on_stream_error)
        self._current_assistant_message = self._add_message(
            "assistant", "*Thinking...*"
        )
        self._is_first_chunk = True
        self._current_worker.start()

    def _on_stream_chunk(self, content: str) -> None:
        if self._current_assistant_message:
            if self._is_first_chunk:
                self._current_assistant_message.set_content(content)
                self._is_first_chunk = False
            else:
                self._current_assistant_message.append_content(content)
            self._messages.scroll_to_bottom()

    def _on_stream_finished(self) -> None:
        self._set_status("Ready")
        self._input_container.set_busy(False)
        if self._current_assistant_message:
            self._context.add_assistant_message(
                self._current_assistant_message.get_content()
            )
            self._save_history()
        self._update_token_count_display()
        self._current_assistant_message = None
        self._disconnect_worker()

    def _on_stream_error(self, error: str) -> None:
        self._set_status("Error")
        self._input_container.set_busy(False)
        if self._current_assistant_message:
            self._current_assistant_message.append_content(f"\n\n⚠️ **Error:** {error}")
        self._current_assistant_message = None
        self._disconnect_worker()

    def _disconnect_worker(self) -> None:
        if self._current_worker is None:
            return
        try:
            self._current_worker.chunk_received.disconnect(self._on_stream_chunk)
            self._current_worker.finished.disconnect(self._on_stream_finished)
            self._current_worker.error.disconnect(self._on_stream_error)
        except (TypeError, RuntimeError):
            pass
        self._current_worker = None

    # ------------------------------------------------------------------
    # Message helpers (delegate to MessageDisplayController)
    # ------------------------------------------------------------------
    def _add_message_to_ui(
        self, role: str, content: str, timestamp: datetime | None = None
    ) -> MessageWidget:
        return self._messages.add_message(role, content, timestamp)

    def _add_message(
        self, role: str, content: str, timestamp: datetime | None = None
    ) -> MessageWidget:
        return self._messages.add_message(role, content, timestamp)

    def _add_system_message(self, content: str) -> MessageWidget:
        return self._messages.add_system_message(content)

    def _restore_ui_messages(self) -> None:
        self._messages.restore_from_context(self._context)

    def _scroll_to_bottom(self) -> None:
        self._messages.scroll_to_bottom()

    def _set_status(self, status: str) -> None:
        self._header.set_status(status)

    # ------------------------------------------------------------------
    # Indexing / memory
    # ------------------------------------------------------------------
    def _auto_index_enabled(self) -> bool:
        if hasattr(self._header, "auto_index_checkbox"):
            return self._header.auto_index_enabled()
        return self._auto_index_on_open

    def _start_indexing(self) -> None:
        self._indexer.start()

    def _on_indexing_finished(self, docs_indexed: int) -> None:
        self._set_status(f"Index ready ({docs_indexed} docs)")

    def _on_indexing_error(self, error: str) -> None:
        self._set_status(f"Index error: {error}")

    def _refresh_prompt_memory(self) -> None:
        self._context.metadata["prompt_memory"] = (
            self._memory_manager.build_prompt_memory()
        )
        self._context.metadata["project_root"] = str(self._project_root)

    def _on_memory_sync_requested(self) -> None:
        archived_contexts: list[ConversationContext] = []
        for session in self._session_manager.list_sessions():
            if not session.get("archived", False):
                continue
            context = self._session_manager.load_session(str(session["id"]), emit=False)
            if context is not None:
                archived_contexts.append(context)

        inserted = self._memory_manager.digest_archived_contexts(archived_contexts)
        self._refresh_prompt_memory()
        self._add_system_message(
            f"Memory sync complete. Added {inserted} archived preference(s)."
        )

    # ------------------------------------------------------------------
    # Peer review
    # ------------------------------------------------------------------
    def _on_peer_review_requested(self) -> None:
        """Open the peer-review config dialog and launch a reviewer chat tab."""
        from src.shared.python.ai.peer_review.gui import PeerReviewConfigDialog
        from src.shared.python.ai.peer_review.prompts import PEER_REVIEW_SYSTEM_PROMPT
        from src.shared.python.ai.peer_review.transcript import format_transcript

        dialog = PeerReviewConfigDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        provider_name, model = dialog.get_config()
        transcript = format_transcript(
            [{"role": m.role, "content": m.content} for m in self._context.messages]
        )

        reviewer_panel = AIAssistantPanel(parent=None)
        reviewer_panel.setWindowTitle(f"Peer Review — {provider_name} / {model}")

        injected_prompt = (
            f"{PEER_REVIEW_SYSTEM_PROMPT}\n\n"
            f"The conversation to review follows.\n\n{transcript}"
        )
        reviewer_panel._add_system_message(injected_prompt)
        reviewer_panel.show()
        logger.info(
            "Peer review panel opened with provider=%s model=%s",
            provider_name,
            model,
        )

    # ------------------------------------------------------------------
    # Thread condensation
    # ------------------------------------------------------------------
    def _on_condense_thread(self) -> None:
        """Condense the active conversation into a summary block."""
        if not self._adapter:
            self._add_system_message("Cannot condense: no AI provider configured.")
            return

        messages = list(self._context.messages)
        if not messages:
            self._add_system_message("Nothing to condense yet.")
            return

        self._set_status("Condensing thread...")
        try:
            summary, active = condense_thread(messages, self._adapter)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Thread condensation failed: %s", exc)
            self._set_status("Condense failed")
            self._add_system_message(f"Condensation failed: {exc}")
            return

        # Preserve raw history for undo
        self._raw_history = messages
        self._is_condensed = True

        # Replace active context with [summary_message, *recent_tail]
        self._context.messages.clear()
        self._context.messages.append(summary.to_message())
        for msg in active[1:]:
            self._context.messages.append(msg)

        self._save_history()

        # Refresh display
        self._messages.clear_messages()
        self._add_system_message(
            f"Thread condensed. {len(self._raw_history)} message(s) "
            f"summarised into a summary block. Use 'Full History' to undo."
        )
        self._messages.restore_from_context(self._context)

        token_count = estimate_token_count(self._context.messages)
        self._header.set_token_count(token_count)
        self._header.set_condensed_mode(True)
        self._set_status("Ready")

    def _on_show_full_history(self) -> None:
        """Restore the full raw conversation history (undo condensation)."""
        if not self._raw_history:
            return

        self._context.messages.clear()
        for msg in self._raw_history:
            self._context.messages.append(msg)

        self._raw_history = []
        self._is_condensed = False

        self._save_history()
        self._messages.clear_messages()
        self._messages.restore_from_context(self._context)

        token_count = estimate_token_count(self._context.messages)
        self._header.set_token_count(token_count)
        self._header.set_condensed_mode(False)
        self._set_status("Full history restored")

    def _update_token_count_display(self) -> None:
        """Refresh the token count label in the toolbar."""
        count = estimate_token_count(self._context.messages)
        self._header.set_token_count(count)

    # ------------------------------------------------------------------
    # New chat
    # ------------------------------------------------------------------
    def _on_new_chat(self) -> None:
        self._messages.clear_messages()
        self._context = ConversationContext()
        self._raw_history = []
        self._is_condensed = False
        self._refresh_prompt_memory()
        self._save_history()
        self._header.set_token_count(0)
        self._header.set_condensed_mode(False)
        self._add_system_message("🔄 New chat started. How can I help you?")

    # ------------------------------------------------------------------
    # Export / copy handlers
    # ------------------------------------------------------------------
    def _on_copy_thread(self) -> None:
        """Copy the full conversation thread to the system clipboard."""
        copy_thread_to_clipboard(self._context.messages)
        n = len(self._context.messages)
        logger.info("Thread copied to clipboard (%d messages)", n)

    def _on_save_thread(self) -> None:
        """Prompt the user for a file path and save the thread as markdown."""
        save_thread_as_markdown(self._context.messages, parent=self)

    # ------------------------------------------------------------------
    # Tool declarations
    # ------------------------------------------------------------------
    def _provider_tool_format(self) -> str:
        if self._adapter is None:
            return "openai"
        adapter_name = type(self._adapter).__name__.lower()
        if "anthropic" in adapter_name:
            return "anthropic"
        return "openai"

    def _build_tool_declarations(self) -> list[dict[str, Any]]:
        return tool_declarations_for_access_mode(
            self._tools_registry,
            self._access_mode,
            provider_format=self._provider_tool_format(),
            rag_enabled=self._rag_enabled,
            max_expertise=self._context.user_expertise.value,
        )

    # ------------------------------------------------------------------
    # Adapter / settings
    # ------------------------------------------------------------------
    def set_mcp_pool(self, pool: Any) -> None:
        """Attach an ``McpClientPool`` and refresh the MCP status indicator.

        The panel does not own the pool lifecycle (start/stop). Callers start
        the pool before calling this and stop it on shutdown.

        Args:
            pool: A started ``McpClientPool`` instance.
        """
        self._mcp_pool = pool
        self._refresh_mcp_status()

    def _refresh_mcp_status(self) -> None:
        """Query the pool for connected-server count and update the indicator."""
        if self._mcp_pool is None:
            self._mcp_indicator.update_status(connected_count=0, total_count=0)
            return
        server_names = getattr(self._mcp_pool, "server_names", [])
        total = len(server_names)
        clients = getattr(self._mcp_pool, "_clients", {})
        connected = sum(
            1 for c in clients.values() if getattr(c, "is_connected", False)
        )
        self._mcp_indicator.update_status(connected_count=connected, total_count=total)

    def set_adapter(self, adapter: BaseAgentAdapter) -> None:
        if adapter is None:
            raise ValueError("adapter must be provided")
        self._adapter = adapter
        self._set_status("Ready")

    def _on_adapter_changed(self, adapter: Any, _adapter_id: str) -> None:
        if adapter is not None:
            self._adapter = adapter
            self._set_status("Ready")

    def set_expertise_level(self, level: ExpertiseLevel) -> None:
        if level is None:
            raise ValueError("level must be provided")
        self._context.user_expertise = level
        level_names = {
            ExpertiseLevel.BEGINNER: "Verbose",
            ExpertiseLevel.INTERMEDIATE: "Normal",
            ExpertiseLevel.ADVANCED: "Concise",
            ExpertiseLevel.EXPERT: "Minimal",
        }
        self._input_container.set_expertise_text(f"Verbosity: {level_names[level]}")

    def apply_settings(self, settings: AISettings) -> None:
        """Apply settings: sync header, update labels, build adapter."""
        if settings is None:
            raise ValueError("settings must be provided")

        self._current_settings = settings
        self._header.sync_controls(settings)

        provider_icons = {
            AIProvider.OLLAMA: "🦙",
            AIProvider.OPENAI: "🧠",
            AIProvider.ANTHROPIC: "🤖",
            AIProvider.GEMINI: "✨",
        }
        self._header.set_provider_icon(provider_icons.get(settings.provider, "🤖"))
        self._header.set_model_label(
            f"{provider_display_name(settings.provider)} ({settings.model})",
            tooltip=f"Provider: {settings.provider.name}\nModel: {settings.model}",
        )

        level_map = {
            1: ExpertiseLevel.BEGINNER,
            2: ExpertiseLevel.INTERMEDIATE,
            3: ExpertiseLevel.ADVANCED,
            4: ExpertiseLevel.EXPERT,
        }
        self.set_expertise_level(
            level_map.get(settings.expertise_level, ExpertiseLevel.BEGINNER)
        )

        self._rag_enabled = settings.rag_enabled
        self._auto_index_on_open = settings.auto_index_on_open
        self._access_mode = coerce_access_mode(settings.access_mode)
        self._header.set_auto_index_checked(settings.auto_index_on_open)
        self._header.sync_access_mode(self._access_mode)

        self._adapter_mgr.build(settings)

    def _show_settings(self) -> None:
        from src.shared.python.ai.gui.settings_dialog import AISettingsDialog

        dialog = AISettingsDialog(self)
        dialog.settings_changed.connect(self.apply_settings)
        if hasattr(dialog, "rebuild_index_requested"):
            dialog.rebuild_index_requested.connect(self._start_indexing)
        dialog.open()
