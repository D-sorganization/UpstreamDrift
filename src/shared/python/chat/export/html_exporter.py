"""Single-file HTML exporter (Tools issue #2735).

The output contains an inline ``<style>`` block using shared Sidekick
theme tokens; no external ``<link>`` or ``<script src=>`` is emitted so
the document is fully portable.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from chat.service_base import ChatMessage, ChatSession

from .contracts import ChatExportRequest, ChatExportResult
from .secret_redactor import SecretRedactor

# Theme tokens kept in one place. These mirror the Sidekick default-dark
# palette so the document looks native when previewed in a browser.
_THEME = {
    "bg": "#1e1e1e",
    "bg_alt": "#2d2d2d",
    "text": "#e0e0e0",
    "muted": "#8b949e",
    "accent": "#58a6ff",
    "user_accent": "#FF8800",
    "border": "#444",
}

_CSS_TEMPLATE = """
body {{
  margin: 0;
  padding: 24px;
  background: {bg};
  color: {text};
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
    sans-serif;
  line-height: 1.5;
}}
.session-meta {{
  border: 1px solid {border};
  padding: 8px 12px;
  border-radius: 6px;
  margin-bottom: 16px;
  color: {muted};
  font-size: 12px;
}}
.message {{
  background: {bg_alt};
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 12px;
  border: 1px solid {border};
}}
.message .role {{
  font-weight: bold;
  font-size: 12px;
  margin-bottom: 4px;
}}
.message.role-user .role {{
  color: {user_accent};
}}
.message.role-assistant .role {{
  color: {accent};
}}
.message .ts {{
  color: {muted};
  font-size: 11px;
  margin-left: 8px;
}}
.message .content {{
  white-space: pre-wrap;
  font-size: 13px;
}}
details {{
  margin-top: 6px;
}}
"""


def _format_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return str(ts)


def _render_message(message: ChatMessage) -> str:
    role = html.escape(message.role)
    ts = html.escape(_format_timestamp(message.timestamp))
    content = html.escape(message.content)
    if message.role == "tool":
        tool_name = html.escape(str(message.metadata.get("tool_name", "tool")))
        call_id = html.escape(message.tool_call_id or "")
        return (
            f'<div class="message role-tool">'
            f'<div class="role">{role}<span class="ts">{ts}</span></div>'
            f"<details><summary>Tool call: {tool_name} ({call_id})</summary>"
            f'<div class="content">{content}</div></details>'
            f"</div>"
        )
    return (
        f'<div class="message role-{role}">'
        f'<div class="role">{role}<span class="ts">{ts}</span></div>'
        f'<div class="content">{content}</div>'
        f"</div>"
    )


class HtmlExporter:
    """Render a :class:`ChatSession` as a single self-contained HTML file."""

    def export(
        self, session: ChatSession, request: ChatExportRequest
    ) -> ChatExportResult:
        if session.message_count == 0:
            raise ValueError("Cannot export an empty chat session")

        messages = list(session.messages)
        if request.redact_secrets:
            redactor = SecretRedactor()
            messages = [redactor.redact_message(m) for m in messages]

        css = _CSS_TEMPLATE.format(**_THEME)
        body_parts: list[str] = []
        if request.include_metadata:
            meta_lines = [
                f"session_id: {html.escape(session.session_id)}",
                f"message_count: {session.message_count}",
            ]
            for k, v in session.metadata.items():
                meta_lines.append(f"{html.escape(str(k))}: {html.escape(str(v))}")
            body_parts.append(
                '<div class="session-meta">' + "<br>".join(meta_lines) + "</div>"
            )
        body_parts.extend(_render_message(m) for m in messages)

        doc = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            f"<title>Chat export {html.escape(session.session_id)}</title>\n"
            f"<style>{css}</style>\n"
            "</head>\n"
            "<body>\n" + "\n".join(body_parts) + "\n</body>\n</html>\n"
        )

        out_path = Path(request.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")

        return ChatExportResult(
            path=request.output_path,
            byte_count=out_path.stat().st_size,
            message_count=len(messages),
        )


__all__ = ["HtmlExporter"]
