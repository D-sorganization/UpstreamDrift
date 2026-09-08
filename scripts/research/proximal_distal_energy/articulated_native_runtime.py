"""Bootstrap the cmeel native library path for the authority runtime (#9607).

The hash-locked authority venv installs cmeel wheels whose shared libraries
live under ``site-packages/cmeel.prefix/lib``. The Linux dynamic loader reads
``LD_LIBRARY_PATH`` only at process start, so a running interpreter cannot
add it; the runtime therefore re-execs itself with the directory prepended
before importing ``pinocchio`` (#9607). DbC: when the cmeel prefix or the
sonames that ``libpinocchio`` links against are absent, the helper fails
with an explicit diagnostic instead of continuing toward a cryptic
``ImportError``.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable

CMEEL_PREFIX_DIRNAME = "cmeel.prefix"
CMEEL_LIB_DIRNAME = "lib"
LD_LIBRARY_PATH_SEPARATOR = os.pathsep
REQUIRED_CMEEL_SONAMES = (
    "liburdfdom_model.so.4.0",
    "liburdfdom_model_state.so.4.0",
    "liburdfdom_sensor.so.4.0",
    "liburdfdom_world.so.4.0",
)


def resolve_cmeel_prefix_lib(candidates: Iterable[Path]) -> Path:
    """Return the first candidate that contains ``cmeel.prefix/lib``.

    Precondition: candidates are candidate site-packages directories (they
    need not exist).
    Postcondition: the returned path is an existing directory.
    Raises:
        RuntimeError: no candidate contains ``cmeel.prefix/lib``; the
            message names every examined candidate for diagnosis.
    """
    examined: list[str] = []
    for candidate in candidates:
        lib_dir = Path(candidate) / CMEEL_PREFIX_DIRNAME / CMEEL_LIB_DIRNAME
        examined.append(str(lib_dir))
        if lib_dir.is_dir():
            return lib_dir
    raise RuntimeError(
        "cmeel.prefix/lib was not found under any candidate directory "
        f"(searched: {', '.join(examined)})"
    )


def require_cmeel_sonames(
    lib_dir: Path,
    sonames: tuple[str, ...] = REQUIRED_CMEEL_SONAMES,
) -> None:
    """Reject a cmeel prefix that lacks the shared libraries pinocchio needs.

    Precondition: ``lib_dir`` is an existing cmeel prefix library directory.
    Postcondition: every required soname exists as a file inside ``lib_dir``.
    Raises:
        RuntimeError: any required soname is absent; the message names the
            directory and every missing soname (lock/wheel ABI mismatch).
    """
    missing = [name for name in sonames if not (lib_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"cmeel prefix library directory {lib_dir} does not ship the "
            f"shared libraries libpinocchio links against (missing: "
            f"{', '.join(missing)}); the hash-locked wheel set is ABI "
            "incompatible and cmeel-urdfdom must be repinned (#9607)"
        )


def assemble_ld_library_path(lib_dir: Path, current: str | None) -> str:
    """Prepend ``lib_dir`` to an ``LD_LIBRARY_PATH`` value without duplicates.

    Precondition: ``lib_dir`` is a directory path.
    Postcondition: the returned value starts with ``lib_dir`` and contains
    no duplicate entries (order otherwise preserved).
    """
    lib_entry = str(lib_dir)
    entries = [
        entry for entry in (current or "").split(LD_LIBRARY_PATH_SEPARATOR) if entry
    ]
    assembled = [lib_entry]
    for entry in entries:
        if entry != lib_entry and entry not in assembled:
            assembled.append(entry)
    return LD_LIBRARY_PATH_SEPARATOR.join(assembled)


def _candidate_site_packages() -> list[Path]:
    """Return plausible site-packages roots for the running interpreter."""
    candidates = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ]
    for entry in sys.path:
        if entry and Path(entry).is_dir():
            candidates.append(Path(entry))
    return candidates


def ensure_cmeel_native_library_path(module_name: str | None = None) -> None:
    """Re-exec the runtime with ``cmeel.prefix/lib`` on ``LD_LIBRARY_PATH``.

    No-op on non-Linux platforms (the authority runtime is Linux-only) and
    when the cmeel prefix library directory is already on the loader path.
    Re-execution is loop-safe: after re-exec the directory is present and the
    function returns instead of re-executing again.
    Raises:
        RuntimeError: the cmeel prefix or a required soname is absent
            (see ``resolve_cmeel_prefix_lib`` / ``require_cmeel_sonames``).
    """
    if sys.platform != "linux":
        return
    lib_dir = resolve_cmeel_prefix_lib(_candidate_site_packages())
    require_cmeel_sonames(lib_dir)
    current = os.environ.get("LD_LIBRARY_PATH")
    if str(lib_dir) in (current or "").split(LD_LIBRARY_PATH_SEPARATOR):
        return
    os.environ["LD_LIBRARY_PATH"] = assemble_ld_library_path(lib_dir, current)
    target = ["-m", module_name] if module_name else [sys.argv[0]]
    # nosemgrep: python.lang.security.audit.dangerous-os-exec-tainted-env-args.dangerous-os-exec-tainted-env-args
    # Deliberate self re-exec (#9607): the interpreter is `sys.executable`, the
    # module target is a caller-supplied constant, and `sys.argv[1:]` is passed
    # through unchanged so the authority runtime resumes identically under the
    # prepended `LD_LIBRARY_PATH`. No new executable or argument surface is
    # introduced; the re-exec is loop-safe (returns once the lib dir is on the
    # loader path).
    os.execv(sys.executable, [sys.executable, *target, *sys.argv[1:]])
