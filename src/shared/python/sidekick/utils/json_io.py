"""Self-contained JSON I/O helpers for the sidekick library.

These were previously imported from a *foreign* application tree via the bare
top-level name ``utils.file_utils`` (``src/python/src/utils/``), which resolved
only by sys.path accident and collided with sidekick's own ``sidekick.utils``
package of the same short name (issue #3333). Vendoring the small, dependency-
free helpers here keeps the sidekick library self-contained and importable in an
installed (non-checkout) deployment.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
import contextlib

__all__ = ["safe_read_json", "safe_write_json"]

_logger = logging.getLogger(__name__)


def safe_read_json(file_path: Path | str, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` on missing/invalid/unreadable file.

    Args:
        file_path: Path to the JSON file. Must not be ``None``.
        default: Value returned when the file is absent or cannot be parsed.

    Returns:
        The parsed JSON document, or ``default``.

    Raises:
        ValueError: If ``file_path`` is ``None``.
    """
    if file_path is None:
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    if not path.exists():
        _logger.debug("JSON file not found: %s, using default", path)
        return default

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        _logger.error("Invalid JSON in %s: %s", path, e)
        return default
    except (PermissionError, OSError) as e:
        _logger.error("Error reading JSON file %s: %s", path, e)
        return default


def safe_write_json(
    file_path: Path | str,
    data: Any,
    indent: int = 2,
    create_parents: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> bool:
    """Atomically write ``data`` as JSON, returning success.

    The write is atomic (temp file + ``os.replace``) so an interruption leaves
    the previous file intact rather than a truncated, unparseable destination.

    Args:
        file_path: Destination path. Must not be ``None``.
        data: JSON-serialisable payload.
        indent: Indentation level passed to ``json.dump``.
        create_parents: Create missing parent directories when ``True``.
        default: Optional ``json.dump`` ``default=`` serialiser for
            non-natively-serialisable objects.

    Returns:
        ``True`` on success, ``False`` otherwise.

    Raises:
        ValueError: If ``file_path`` is ``None``.
    """
    if file_path is None:
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    try:
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        text = json.dumps(data, indent=indent, ensure_ascii=False, default=default)

        # Write to a temp sibling then os.replace (atomic on POSIX and Windows),
        # so an interruption leaves the previous file intact. ``open`` is used
        # deliberately (not ``os.fdopen``) so file-permission failures surface
        # the same way callers/tests expect.
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        return True
    except (PermissionError, OSError, TypeError, ValueError) as e:
        _logger.error("Error writing JSON file %s: %s", path, e)
        return False
