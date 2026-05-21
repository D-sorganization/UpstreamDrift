"""Notebook session model and persistence for Sidekick Phase 2 (Tools #2876).

Provides two public classes:

* :class:`NotebookSessionModel` — immutable dataclass describing an open
  notebook session.  Its :meth:`validate_path` method enforces Design-by-
  Contract path-containment: the notebook path must resolve inside the
  declared workspace root.

* :class:`NotebookSessionManager` — save/load session state to disk.  Only
  lightweight metadata is persisted (relative path + kernel env); the full
  notebook JSON is never embedded in the session file.

No Qt imports here.  This module is pure data / I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NotebookSessionModel:
    """Mutable session model for an open Sidekick notebook.

    Attributes:
        notebook_path: Absolute path to the ``.ipynb`` file.
        workspace_root: The root directory that ``notebook_path`` must be
            contained within.
        kernel_env: Virtual-environment name used by the kernel, or ``None``
            when no specific environment is selected.
        last_saved: UTC datetime of the most recent :meth:`save_session` call,
            or ``None`` if the session has not been persisted yet.

    Preconditions are enforced by :meth:`validate_path`.
    """

    notebook_path: Path
    workspace_root: Path
    kernel_env: str | None
    last_saved: datetime | None = field(default=None)
    _kernel_env_vars: dict[str, Any] = field(default_factory=dict, repr=False)

    def validate_path(self) -> None:
        """DbC: raise ValueError if *notebook_path* is outside *workspace_root*.

        Both paths are :meth:`~pathlib.Path.resolve`-d before comparison so
        that ``..`` components cannot be used to escape the root.

        Raises:
            ValueError: If the resolved notebook path is not relative to the
                resolved workspace root.
        """
        resolved = self.notebook_path.resolve()
        root = self.workspace_root.resolve()
        # is_relative_to correctly handles cross-platform separators and exact
        # matches (a file directly at the root is valid).
        if not resolved.is_relative_to(root):
            raise ValueError(f"Path {resolved} is outside workspace root {root}")

    def set_kernel_environment(self, env_vars: dict[str, Any]) -> None:
        """Store kernel environment variables for injection into the kernel.

        These variables are injected by :class:`WorkspaceBridge` before the
        kernel is launched.  This method simply stores the dict; no kernel
        communication happens here.

        Args:
            env_vars: Mapping of variable name to JSON-serializable value.
                Must be a dict.

        Raises:
            TypeError: If *env_vars* is not a dict.
        """
        if not isinstance(env_vars, dict):
            raise TypeError(f"env_vars must be a dict, got {type(env_vars).__name__}")
        self._kernel_env_vars = dict(env_vars)
        logger.debug("Kernel environment updated: %d variable(s)", len(env_vars))

    @property
    def kernel_env_vars(self) -> dict[str, Any]:
        """Return a copy of the kernel environment variables dict."""
        return dict(self._kernel_env_vars)


class NotebookSessionManager:
    """Persist and restore :class:`NotebookSessionModel` instances to disk.

    Session files are stored as JSON under *sessions_dir*.  Each file is named
    ``<session-id>.json`` where the session ID is a stable hash derived from
    the workspace root and the relative notebook path — so saving the same
    logical session twice is idempotent.

    The JSON schema is deliberately minimal::

        {
            "notebook_path": "notebooks/test.ipynb",   # relative to workspace_root
            "kernel_env": "py311" | null,
            "last_saved": "2026-05-16T21:00:00"        # ISO-8601 UTC
        }

    Full notebook JSON (cells, nbformat, …) is **never** embedded.

    Args:
        sessions_dir: Directory where session JSON files are written.  Created
            on first use if it does not exist.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = sessions_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_session(self, model: NotebookSessionModel) -> str:
        """Persist *model* and return its session ID.

        The session ID is derived deterministically from *workspace_root* and
        the relative notebook path, so repeated calls with the same logical
        session overwrite the previous file (idempotent).

        Args:
            model: The session to persist.

        Returns:
            A stable session-ID string (hex digest).
        """
        model.validate_path()
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

        rel_path = model.notebook_path.resolve().relative_to(
            model.workspace_root.resolve()
        )
        sid = self._session_id(model.workspace_root, rel_path)

        now = datetime.now(tz=UTC).replace(tzinfo=None)  # naive UTC for JSON compat
        payload = {
            "notebook_path": rel_path.as_posix(),
            "kernel_env": model.kernel_env,
            "last_saved": now.isoformat(),
            "workspace_root": str(model.workspace_root.resolve()),
        }
        session_file = self._sessions_dir / f"{sid}.json"
        session_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        model.last_saved = now
        logger.debug("Session %s saved to %s", sid, session_file)
        return sid

    def load_session(
        self, session_id: str, workspace_root: Path
    ) -> NotebookSessionModel:
        """Load and validate a persisted session.

        The restored :class:`NotebookSessionModel` is validated with
        :meth:`~NotebookSessionModel.validate_path` against *workspace_root*
        before it is returned — so loading a session from a different root
        raises :class:`ValueError`.

        Args:
            session_id: The session ID returned by :meth:`save_session`.
            workspace_root: The workspace root to validate the restored path
                against.  Must match the root used when the session was saved.

        Returns:
            The restored :class:`NotebookSessionModel`.

        Raises:
            FileNotFoundError: If no session with *session_id* exists.
            ValueError: If the restored path is outside *workspace_root*.
        """
        session_file = self._sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            raise FileNotFoundError(
                f"No session file found for ID '{session_id}' in {self._sessions_dir}"
            )

        data = json.loads(session_file.read_text(encoding="utf-8"))

        # Guard: the caller must supply the same workspace_root that was used
        # when the session was saved.  This prevents one workspace's session
        # from being loaded under a different workspace.
        saved_root = data.get("workspace_root")
        if saved_root is not None:
            saved_root_path = Path(saved_root).resolve()
            caller_root_path = workspace_root.resolve()
            if saved_root_path != caller_root_path:
                raise ValueError(
                    f"Session workspace root {saved_root_path} does not match "
                    f"supplied workspace root {caller_root_path}"
                )

        rel_path = Path(data["notebook_path"])
        notebook_path = workspace_root / rel_path

        last_saved: datetime | None = None
        if data.get("last_saved"):
            try:
                last_saved = datetime.fromisoformat(data["last_saved"])
            except ValueError:
                logger.warning("Could not parse last_saved: %s", data["last_saved"])

        model = NotebookSessionModel(
            notebook_path=notebook_path,
            workspace_root=workspace_root,
            kernel_env=data.get("kernel_env"),
            last_saved=last_saved,
        )
        model.validate_path()
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_id(workspace_root: Path, rel_path: Path) -> str:
        """Return a stable hex session ID from workspace root + relative path."""
        key = f"{workspace_root.resolve()!s}::{rel_path.as_posix()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
