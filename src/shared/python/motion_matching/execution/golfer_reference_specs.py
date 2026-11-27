"""Current-model golfer reference specs (#15648).

``src/config/golfer_reference_specs.json`` holds, per club, the overlay that
turns a committed legacy reference spec (capsule hands, loft body) into the
current golfer model (``src/config/golfer_model_versions.json``). The full
spec is not committed: the overlay is about twenty values, the legacy file
stays as the parent, and the result's canonical SHA-256 is recorded so a
changed parent or overlay fails loudly instead of drifting.

:func:`reference_spec` returns the current-model document for a club;
:func:`write_reference_spec` writes it for tools that take a spec path.
Regenerate the overlay with ``python3 scripts/generate_golfer_reference_specs.py``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.full_body_spec import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[5]
SETTINGS_PATH = REPO_ROOT / "src" / "config" / "golfer_reference_specs.json"

Key = str | int


def _settings(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))


def reference_clubs(path: Path | None = None) -> tuple[str, ...]:
    """Clubs with a current-model reference spec, sorted."""
    return tuple(sorted(_settings(path)["specs"]))


def _entry(club: str, path: Path | None) -> Mapping[str, Any]:
    specs = _settings(path)["specs"]
    if club not in specs:
        raise ValueError(f"no reference spec for {club!r}; known: {sorted(specs)}")
    return specs[club]


def overlay_between(
    parent: Any, child: Any, at: tuple[Key, ...] = ()
) -> list[dict[str, Any]]:
    """The ``set`` operations that turn ``parent`` into ``child``.

    Mappings and equal-length lists are compared item by item; where
    ``child`` drops a key or changes a list length, the whole value at that
    path is set (a chain shaft adds bodies and coordinates, #15684).

    Raises:
        ValueError: the root mapping drops a key (an empty path cannot be set).
    """
    if isinstance(parent, Mapping) and isinstance(child, Mapping):
        if set(parent) - set(child):
            if not at:
                raise ValueError("child drops keys at the root")
            return [{"path": list(at), "value": child}]
        ops: list[dict[str, Any]] = []
        for key in sorted(child):
            if key in parent:
                ops += overlay_between(parent[key], child[key], (*at, key))
            else:
                ops.append({"path": [*at, key], "value": child[key]})
        return ops
    if isinstance(parent, list) and isinstance(child, list):
        if len(parent) != len(child):
            return [{"path": list(at), "value": child}]
        return [
            op
            for i, (a, b) in enumerate(zip(parent, child, strict=True))
            for op in overlay_between(a, b, (*at, i))
        ]
    return [] if parent == child else [{"path": list(at), "value": child}]


def apply_overlay(
    document: Mapping[str, Any], ops: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """``document`` with each ``set`` operation applied; the input is not mutated.

    Raises ``ValueError`` for an empty path or a path whose parent is missing.
    """
    out = copy.deepcopy(dict(document))
    for op in ops:
        path = list(op["path"])
        if not path:
            raise ValueError("an overlay path must not be empty")
        node: Any = out
        try:
            for key in path[:-1]:
                node = node[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"overlay path {path} has no parent") from exc
        node[path[-1]] = copy.deepcopy(op["value"])
    return out


def reference_spec(club: str, *, path: Path | None = None) -> dict[str, Any]:
    """The current-model reference spec document for ``club``.

    Postcondition: its canonical SHA-256 equals the recorded one, else
    ``ValueError`` (the legacy parent or the overlay changed; regenerate).
    """
    entry = _entry(club, path)
    parent = json.loads((REPO_ROOT / entry["parent"]).read_text(encoding="utf-8"))
    if canonical_sha256(parent) != entry["parent_sha256"]:
        raise ValueError(f"the {club} legacy parent changed; regenerate the overlay")
    document = apply_overlay(parent, entry["overlay"])
    if canonical_sha256(document) != entry["sha256"]:
        raise ValueError(f"the {club} reference spec does not match its recorded hash")
    return document


def write_reference_spec(club: str, out: Path, *, path: Path | None = None) -> Path:
    """Write the ``club`` reference spec to ``out`` and return ``out``."""
    document = reference_spec(club, path=path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return out
