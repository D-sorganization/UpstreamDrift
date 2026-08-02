"""Preflight check for the ONNX Runtime shared library.

When ``ai_backend`` is built with ``--features local-embeddings`` the Rust
crate loads the ONNX Runtime at *run time* via the ``ORT_DYLIB_PATH``
environment variable.  If that variable is absent or points at an invalid file
the crate panics with a cryptic OS error.

This helper catches the problem early and raises a :class:`RuntimeError` with a
human-readable message and a link to the setup guide before any Rust code is
invoked.

Typical use:

    >>> from src.shared.python.ai._onnx_preflight import check_ort_loadable
    >>> check_ort_loadable()   # raises RuntimeError on failure, returns None on success

Command-line use (e.g. in CI or before launching an embedding-heavy workload):

    python -m src.shared.python.ai._onnx_preflight

Exits with code 0 on success, 1 on failure.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)

_SETUP_GUIDE = "docs/ai_backend_setup.md"
_RELEASES_URL = "https://github.com/microsoft/onnxruntime/releases"

_ENV_VAR = "ORT_DYLIB_PATH"


def check_ort_loadable(dylib_path: str | None = None) -> None:
    """Verify that the ONNX Runtime shared library can be loaded.

    Reads ``ORT_DYLIB_PATH`` from the environment (or accepts an explicit path
    via *dylib_path*) and attempts to open the library with
    :func:`ctypes.CDLL`.

    Args:
        dylib_path: Explicit path to the ONNX Runtime shared library. When
            ``None`` (default) the value of the ``ORT_DYLIB_PATH`` environment
            variable is used.

    Raises:
        RuntimeError: If ``ORT_DYLIB_PATH`` is unset and *dylib_path* is
            ``None``, or if the library file cannot be loaded by the OS.

    Returns:
        ``None`` on success.
    """
    path = dylib_path if dylib_path is not None else os.environ.get(_ENV_VAR)

    if not path:
        raise RuntimeError(
            f"ONNX Runtime not configured: {_ENV_VAR} is not set.\n"
            f"\n"
            f"The 'local-embeddings' feature requires the ONNX Runtime shared\n"
            f"library to be present on your system.\n"
            f"\n"
            f"Download the library from:\n"
            f"  {_RELEASES_URL}\n"
            f"\n"
            f"Then set the environment variable to point at the extracted file:\n"
            f"  Windows PowerShell:\n"
            f"    $env:{_ENV_VAR} = "
            f"'C:\\onnxruntime-win-x64-1.18.1\\lib\\onnxruntime.dll'\n"
            f"  Linux:\n"
            f"    export {_ENV_VAR}=/path/to/libonnxruntime.so\n"
            f"  macOS:\n"
            f"    export {_ENV_VAR}=/path/to/libonnxruntime.dylib\n"
            f"\n"
            f"Full setup guide: {_SETUP_GUIDE}\n"
            f"\n"
            f"ONNX runtime not loadable: see {_SETUP_GUIDE}"
        )

    try:
        ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError(
            f"ONNX Runtime not loadable: failed to open '{path}'.\n"
            f"\n"
            f"OS error: {exc}\n"
            f"\n"
            f"Common causes:\n"
            f"  - The path in {_ENV_VAR} does not exist or is misspelled.\n"
            f"  - The file is not a valid shared library for this platform.\n"
            "  - On Windows: the DLL's own dependencies"
            " (e.g. MSVC runtime) are missing.\n"
            f"  - On Linux: run `ldd {path}` to find missing dependencies.\n"
            f"  - Version mismatch: ort 2.0.0-rc.10 requires ONNX Runtime >= 1.17.\n"
            f"\n"
            f"Download a compatible release from:\n"
            f"  {_RELEASES_URL}\n"
            f"\n"
            f"Full setup guide: {_SETUP_GUIDE}\n"
            f"\n"
            f"ONNX runtime not loadable: see {_SETUP_GUIDE}"
        ) from exc

    logger.debug("ONNX Runtime loaded successfully from '%s'.", path)


def _main() -> int:  # pragma: no cover
    """Entry point for ``python -m src.shared.python.ai._onnx_preflight``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        check_ort_loadable()
        logger.info("OK: ONNX Runtime loaded from '%s'.", os.environ.get(_ENV_VAR))
        return 0
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
